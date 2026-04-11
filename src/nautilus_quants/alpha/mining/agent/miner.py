# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""LLM-driven alpha factor mining orchestrator.

Drives a multi-round mining loop with parallel claude requests:
  1. Build prompts (with anti-duplication + performance feedback)
  2. Call ``claude -p`` in parallel via ThreadPoolExecutor
  3. Validate expressions via ``parse_expression()``
  4. Write factors YAML → trigger ``alpha analyze`` (serial)
  5. Collect IC/ICIR from registry → feed into next batch
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nautilus_quants.alpha.mining.agent.prompts import (
    build_generation_prompt,
    get_json_schema,
)
from nautilus_quants.factors.expression.complexity import ComplexityConstraints

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class DirectionConfig:
    """A single mining exploration direction."""

    name: str
    theme: str
    operators: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    rounds: int = 0


@dataclass(frozen=True)
class MiningConfig:
    """Configuration for a mining session."""

    analysis_config_path: Path
    output_dir: Path = Path("logs/alpha_mining")
    factors_per_round: int = 8
    bar_spec: str = "4h"
    model: str = "sonnet"
    source: str = "llm_claude"
    proxy: str = "http://127.0.0.1:8888"
    parallel: int = 3
    auto_analyze: bool = True
    hypothesis: str | None = None
    constraints: ComplexityConstraints = field(default_factory=ComplexityConstraints)
    directions: tuple[DirectionConfig, ...] = ()

    @staticmethod
    def from_yaml(
        path: Path,
        *,
        factors_per_round: int | None = None,
        model: str | None = None,
        hypothesis: str | None = None,
        auto_analyze: bool = True,
    ) -> MiningConfig:
        """Load mining config from analysis YAML's ``mining`` section."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        mining = raw.get("mining", {})
        bar_spec = raw.get("bar_spec", "4h")

        # Parse complexity constraints.
        c_raw = mining.get("constraints") or {}
        constraints = ComplexityConstraints(
            max_char_length=c_raw.get("max_char_length", 200),
            max_node_count=c_raw.get("max_node_count", 30),
            max_depth=c_raw.get("max_depth", 6),
            max_func_nesting=c_raw.get("max_func_nesting", 4),
            max_variables=c_raw.get("max_variables", 5),
            max_window=c_raw.get("max_window", 720),
            max_numeric_ratio=c_raw.get("max_numeric_ratio", 0.3),
        )

        # Parse exploration directions.
        d_raw = mining.get("directions") or {}
        directions: list[DirectionConfig] = []
        for name, d in d_raw.items():
            if not isinstance(d, dict) or "theme" not in d:
                raise ValueError(
                    f"Direction '{name}' must be a mapping with a 'theme' key"
                )
            directions.append(
                DirectionConfig(
                    name=name,
                    theme=d["theme"],
                    operators=tuple(d.get("operators") or ()),
                    variables=tuple(d.get("variables") or ()),
                    tags=tuple(d.get("tags") or ()),
                    rounds=d.get("rounds", 0),
                )
            )

        return MiningConfig(
            analysis_config_path=Path(path),
            output_dir=Path(mining.get("output_dir", "logs/alpha_mining")),
            factors_per_round=factors_per_round or mining.get("factors_per_round", 8),
            bar_spec=bar_spec,
            model=model or mining.get("model", "sonnet"),
            source=mining.get("source", "llm_claude"),
            proxy=mining.get("proxy", "http://127.0.0.1:8888"),
            parallel=mining.get("parallel", 3),
            auto_analyze=auto_analyze,
            hypothesis=hypothesis,
            constraints=constraints,
            directions=tuple(directions),
        )


@dataclass(frozen=True)
class ValidatedFactor:
    """A factor that passed expression parsing."""

    name: str
    expression: str
    hypothesis: str
    description: str
    tags: list[str]


@dataclass
class GenerateResult:
    """Result from a single generate round (thread-safe, no print)."""

    round_num: int
    round_dir: Path
    valid: list[ValidatedFactor]
    failed: list[dict]
    elapsed: float
    factor_yaml: Path | None = None
    error: str | None = None


# ── AlphaMiner ────────────────────────────────────────────────────────────


class AlphaMiner:
    """LLM-driven alpha factor mining orchestrator."""

    def __init__(self, config: MiningConfig) -> None:
        self._config = config
        self._mining_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._session_dir = config.output_dir / self._mining_id
        self._state: dict[str, Any] = {
            "mining_id": self._mining_id,
            "config": {
                "bar_spec": config.bar_spec,
                "model": config.model,
                "factors_per_round": config.factors_per_round,
                "parallel": config.parallel,
                "analysis_config": str(config.analysis_config_path),
            },
            "rounds": [],
            "all_expressions": [],
        }

    # ── Public API ────────────────────────────────────────────────────

    def run(self, rounds: int = 5) -> None:
        """Execute mining rounds.

        Dispatches to :meth:`_run_diversified` when ``directions`` are
        configured, otherwise runs the original single-direction loop.
        """
        self._session_dir.mkdir(parents=True, exist_ok=True)

        if self._config.directions:
            self._run_diversified(rounds)
        else:
            self._run_single(rounds)

    # ── Single-direction mode (original behaviour) ───────────────────

    def _run_single(self, rounds: int) -> None:
        """Run mining in single-direction mode (original behaviour)."""
        parallel = min(self._config.parallel, rounds)

        print(f"Mining session: {self._session_dir}")
        print(f"  Rounds: {rounds}, Factors/round: {self._config.factors_per_round}")
        print(f"  Model: {self._config.model}, Parallel: {parallel}")
        print()

        round_num = 1
        while round_num <= rounds:
            batch_size = min(parallel, rounds - round_num + 1)
            batch_rounds = list(range(round_num, round_num + batch_size))

            gen_results = self._run_generate_batch(batch_rounds)

            if self._config.auto_analyze:
                self._run_analyze_batch(gen_results)
            else:
                for gr in gen_results:
                    if gr.error is None:
                        self._record_round(gr.round_num, gr.valid, {})

            self._save_state()
            round_num += batch_size

        self._print_summary()

    # ── Multi-direction mode ─────────────────────────────────────────

    def _run_diversified(self, default_rounds: int) -> None:
        """Run mining across multiple exploration directions."""
        directions = self._config.directions

        print(f"Diversified mining session: {self._session_dir}")
        print(f"  Directions: {len(directions)}")
        for d in directions:
            dr = d.rounds or default_rounds
            ops = f"{len(d.operators)} ops" if d.operators else "all ops"
            vs = ", ".join(d.variables) if d.variables else "all vars"
            print(f"    {d.name}: theme={d.theme!r}, {ops}, vars=[{vs}], rounds={dr}")
        print()

        global_round = 0
        for direction in directions:
            dir_rounds = direction.rounds or default_rounds
            dir_output = self._session_dir / direction.name
            dir_output.mkdir(parents=True, exist_ok=True)

            print(f"{'=' * 60}")
            print(f"Direction: {direction.name} — {direction.theme}")
            print(f"{'=' * 60}")

            for local_round in range(1, dir_rounds + 1):
                global_round += 1
                round_dir = dir_output / f"round_{local_round:03d}"

                gr = self._generate_round(
                    global_round,
                    round_dir_override=round_dir,
                    direction=direction,
                )
                self._print_generate_result(gr, local_round, dir_rounds)

                if self._config.auto_analyze and gr.error is None and gr.valid:
                    analysis_yaml = self._write_analysis_yaml(
                        gr.factor_yaml, gr.round_dir,
                    )
                    print(f"  Analyzing ({len(gr.valid)} factors)...",
                          end="", flush=True)
                    t0 = time.time()
                    metrics = self._run_analysis(analysis_yaml)
                    print(f"  done ({time.time() - t0:.1f}s)")
                    self._record_round(
                        global_round, gr.valid, metrics,
                        direction_name=direction.name,
                    )
                else:
                    self._record_round(
                        global_round, gr.valid if gr.error is None else [],
                        {}, direction_name=direction.name,
                    )

                self._save_state()
            print()

        self._print_summary()

    # ── Batch orchestration ───────────────────────────────────────────

    def _run_generate_batch(self, batch_rounds: list[int]) -> list[GenerateResult]:
        """Run parallel claude requests for a batch of rounds."""
        n = len(batch_rounds)
        print(f"{'=' * 60}")
        if n > 1:
            print(f"Batch: Generating rounds {batch_rounds[0]}-{batch_rounds[-1]} "
                  f"in parallel ({n} threads)...")
        else:
            print(f"Round {batch_rounds[0]}")
        print(f"{'=' * 60}")

        if n == 1:
            # Single round — run directly, no thread overhead
            result = self._generate_round(batch_rounds[0])
            self._print_generate_result(result, 1, 1)
            return [result]

        # Parallel execution
        results: dict[int, GenerateResult] = {}
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(self._generate_round, r): r
                for r in batch_rounds
            }
            for future in as_completed(futures):
                rn = futures[future]
                try:
                    results[rn] = future.result()
                except Exception as e:
                    results[rn] = GenerateResult(
                        round_num=rn,
                        round_dir=self._session_dir / f"round_{rn:03d}",
                        valid=[], failed=[], elapsed=0, error=str(e),
                    )

        # Print results in order
        ordered = [results[r] for r in batch_rounds]
        for i, gr in enumerate(ordered, 1):
            self._print_generate_result(gr, i, n)

        # Batch dedup summary
        total_valid = sum(len(gr.valid) for gr in ordered if gr.error is None)
        total_failed = sum(len(gr.failed) for gr in ordered if gr.error is None)
        print(f"  Batch total: {total_valid} valid, {total_failed} failed")
        print()

        return ordered

    def _run_analyze_batch(self, gen_results: list[GenerateResult]) -> None:
        """Run serial analysis for each generate result."""
        for gr in gen_results:
            if gr.error is not None:
                self._record_round(gr.round_num, [], {})
                continue
            if not gr.valid:
                self._record_round(gr.round_num, gr.valid, {})
                continue

            # Write analysis yaml and run
            analysis_yaml = self._write_analysis_yaml(gr.factor_yaml, gr.round_dir)
            print(f"  Analyzing round {gr.round_num} "
                  f"({len(gr.valid)} factors)...", end="", flush=True)
            t0 = time.time()
            metrics = self._run_analysis(analysis_yaml)
            elapsed = time.time() - t0
            print(f"  done ({elapsed:.1f}s)")

            for fname, m in metrics.items():
                ic_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in m.get("ic_mean", {}).items()
                )
                print(f"    {fname}: IC=[{ic_str}]")

            self._record_round(gr.round_num, gr.valid, metrics)

    # ── Single round generation (thread-safe, no print) ───────────────

    def _generate_round(
        self,
        round_num: int,
        *,
        round_dir_override: Path | None = None,
        direction: DirectionConfig | None = None,
    ) -> GenerateResult:
        """Generate factors for one round. Safe to call from threads."""
        round_dir = round_dir_override or (self._session_dir / f"round_{round_num:03d}")
        round_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        try:
            # 1. Build prompt
            previous = self._get_previous_factors()
            top = self._get_top_factors()
            prompt = build_generation_prompt(
                round_num=round_num,
                num_factors=self._config.factors_per_round,
                bar_spec=self._config.bar_spec,
                previous_factors=previous,
                top_factors=top,
                hypothesis=self._config.hypothesis,
                theme=direction.theme if direction else None,
                operator_subset=direction.operators or None if direction else None,
                variable_subset=direction.variables or None if direction else None,
                constraints=self._config.constraints,
            )
            (round_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            # 2. Call Claude
            raw_stdout, raw_stderr, response = self._call_claude(prompt)

            # Save raw output
            (round_dir / "stdout.txt").write_text(raw_stdout, encoding="utf-8")
            if raw_stderr:
                (round_dir / "stderr.txt").write_text(raw_stderr, encoding="utf-8")
            (round_dir / "response.json").write_text(
                json.dumps(response, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # 3. Validate
            valid, failed = self._validate_factors(response)

            # Save validation results
            (round_dir / "validated.json").write_text(
                json.dumps({
                    "passed": [
                        {"name": f.name, "expression": f.expression,
                         "hypothesis": f.hypothesis}
                        for f in valid
                    ],
                    "failed": failed,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # 4. Write factor YAML
            factor_yaml = None
            if valid:
                factor_yaml = self._write_factor_yaml(valid, round_dir)

            elapsed = time.time() - t0
            return GenerateResult(
                round_num=round_num, round_dir=round_dir,
                valid=valid, failed=failed,
                elapsed=elapsed, factor_yaml=factor_yaml,
            )

        except Exception as e:
            elapsed = time.time() - t0
            (round_dir / "error.txt").write_text(str(e), encoding="utf-8")
            return GenerateResult(
                round_num=round_num, round_dir=round_dir,
                valid=[], failed=[], elapsed=elapsed, error=str(e),
            )

    def _print_generate_result(
        self, gr: GenerateResult, idx: int, total: int,
    ) -> None:
        """Print generation result (called after threads complete)."""
        prefix = f"  [{idx}/{total}]" if total > 1 else " "
        if gr.error:
            print(f"{prefix} Round {gr.round_num}: FAILED — {gr.error}")
        else:
            print(f"{prefix} Round {gr.round_num}: "
                  f"{len(gr.valid)} passed, {len(gr.failed)} failed "
                  f"({gr.elapsed:.1f}s)")

    # ── Claude CLI interaction ────────────────────────────────────────

    def _call_claude(self, prompt: str) -> tuple[str, str, dict]:
        """Call ``claude -p`` via subprocess.

        Returns ``(raw_stdout, raw_stderr, parsed_response)``.
        """
        env = os.environ.copy()
        proxy_url = self._config.proxy
        env.setdefault("http_proxy", proxy_url)
        env.setdefault("https_proxy", proxy_url)
        env.setdefault("HTTP_PROXY", proxy_url)
        env.setdefault("HTTPS_PROXY", proxy_url)

        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--json-schema", json.dumps(get_json_schema()),
            "--model", self._config.model,
            "--no-session-persistence",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            raise RuntimeError(
                f"claude -p failed (rc={result.returncode})\n"
                f"stderr: {stderr[:500]}\n"
                f"stdout: {stdout[:500]}"
            )

        if not stdout.strip():
            raise RuntimeError(
                f"claude -p returned empty stdout\n"
                f"stderr: {stderr[:500]}"
            )

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse claude JSON output: {e}\n"
                f"stdout (first 1000 chars): {stdout[:1000]}"
            ) from e

        # --output-format json returns:
        #   {"type": "result", "result": "..text..",
        #    "structured_output": {"factors": {...}}}
        if isinstance(parsed, dict):
            if "structured_output" in parsed and parsed["structured_output"]:
                return stdout, stderr, parsed["structured_output"]
            if "result" in parsed:
                inner = parsed["result"]
                if isinstance(inner, str):
                    try:
                        return stdout, stderr, json.loads(inner)
                    except json.JSONDecodeError:
                        return stdout, stderr, {"factors": {}}
                if isinstance(inner, dict):
                    return stdout, stderr, inner

        return stdout, stderr, parsed

    # ── Expression validation ─────────────────────────────────────────

    def _validate_factors(
        self, response: dict,
    ) -> tuple[list[ValidatedFactor], list[dict]]:
        """Validate each factor expression.

        Pipeline: char_length pre-check → parse → complexity → dedup.
        Returns (valid_factors, failed_details).
        """
        from nautilus_quants.factors.expression.complexity import check_constraints
        from nautilus_quants.factors.expression.normalize import expression_hash
        from nautilus_quants.factors.expression.parser import parse_expression

        existing_hashes = {
            expression_hash(e) for e in self._state["all_expressions"]
        }
        constraints = self._config.constraints

        raw_factors = response.get("factors", {})
        valid: list[ValidatedFactor] = []
        failed: list[dict] = []

        for name, data in raw_factors.items():
            expr = data.get("expression", "")

            # Fast pre-check: char length (no parsing needed).
            if len(expr) > constraints.max_char_length:
                failed.append({
                    "name": name, "expression": expr,
                    "error": (
                        f"complexity: char_length={len(expr)} "
                        f"> max {constraints.max_char_length}"
                    ),
                })
                continue

            # Syntax validation
            try:
                ast = parse_expression(expr)
            except Exception as e:
                failed.append({"name": name, "expression": expr, "error": f"parse: {e}"})
                continue

            # Complexity constraints (reuse parsed AST)
            violations = check_constraints(ast, constraints, expr_str=expr)
            if violations:
                failed.append({
                    "name": name, "expression": expr,
                    "error": f"complexity: {'; '.join(violations)}",
                })
                continue

            # Dedup check
            h = expression_hash(expr)
            if h in existing_hashes:
                failed.append({"name": name, "expression": expr, "error": "duplicate"})
                continue

            existing_hashes.add(h)
            valid.append(ValidatedFactor(
                name=name,
                expression=expr,
                hypothesis=data.get("hypothesis", ""),
                description=data.get("description", ""),
                tags=data.get("tags", []),
            ))

        return valid, failed

    # ── YAML generation ───────────────────────────────────────────────

    def _write_factor_yaml(
        self, factors: list[ValidatedFactor], round_dir: Path,
    ) -> Path:
        """Write validated factors as a standard FactorConfig YAML."""
        factors_section: dict[str, dict] = {}
        for f in factors:
            factors_section[f.name] = {
                "expression": f.expression,
                "description": f.description,
                "tags": f.tags,
            }

        doc = {
            "metadata": {
                "name": f"{self._config.source}_{round_dir.name}",
                "version": "1.0",
                "source": self._config.source,
                "description": f"LLM-generated factors ({len(factors)} factors)",
            },
            "variables": {
                "returns": "delta(close, 1) / replace_zero(delay(close, 1))",
            },
            "factors": factors_section,
        }

        path = round_dir / "factors.yaml"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated by AlphaMiner — do not edit manually\n")
            yaml.dump(doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return path

    def _write_analysis_yaml(
        self, factor_yaml_path: Path | None, round_dir: Path,
    ) -> Path:
        """Generate analysis config from base config with factor_config_path override."""
        with open(self._config.analysis_config_path, encoding="utf-8") as f:
            base = yaml.safe_load(f)

        base["factor_config_path"] = str(factor_yaml_path)
        base.pop("factors", None)
        base["output_dir"] = str(round_dir / "analysis")
        base.pop("mining", None)

        path = round_dir / "analysis.yaml"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated by AlphaMiner — do not edit manually\n")
            yaml.dump(base, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return path

    # ── Analysis execution ────────────────────────────────────────────

    def _run_analysis(self, analysis_yaml_path: Path) -> dict[str, dict]:
        """Run ``alpha analyze`` and retrieve metrics from registry."""
        result = subprocess.run(
            [sys.executable, "-m", "nautilus_quants.alpha", "analyze",
             str(analysis_yaml_path), "-q"],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.warning("alpha analyze failed: %s", result.stderr[:500])
            print(f"\n  Warning: alpha analyze failed: {result.stderr[:200]}")
            return {}

        return self._query_round_metrics()

    def _query_round_metrics(self) -> dict[str, dict]:
        """Query IC/ICIR from the registry for the latest round's factors."""
        try:
            from nautilus_quants.alpha.registry.database import RegistryDatabase
            from nautilus_quants.alpha.registry.environment import resolve_env
            from nautilus_quants.alpha.registry.repository import FactorRepository

            with open(self._config.analysis_config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            reg_cfg = raw.get("registry", {})
            env = resolve_env(reg_cfg.get("env", "test"))
            db_dir = reg_cfg.get("db_dir", "logs/registry")

            db = RegistryDatabase.for_environment(env, db_dir)
            repo = FactorRepository(db)

            latest_round = self._state["rounds"][-1] if self._state["rounds"] else None
            if not latest_round:
                db.close()
                return {}

            metrics_map: dict[str, dict] = {}
            for finfo in latest_round.get("factors", []):
                factor_id = f"{self._config.source}_{finfo['name']}"
                ms = repo.get_metrics(factor_id)
                if ms:
                    ic_mean: dict[str, float] = {}
                    icir: dict[str, float] = {}
                    for m in ms:
                        if m.ic_mean is not None:
                            ic_mean[m.period] = m.ic_mean
                        if m.icir is not None:
                            icir[m.period] = m.icir
                    metrics_map[finfo["name"]] = {"ic_mean": ic_mean, "icir": icir}

            db.close()
            return metrics_map
        except Exception as e:
            logger.warning("Failed to query registry metrics: %s", e)
            return {}

    # ── State management ──────────────────────────────────────────────

    def _record_round(
        self,
        round_num: int,
        valid: list[ValidatedFactor],
        metrics: dict[str, dict],
        *,
        direction_name: str | None = None,
    ) -> None:
        """Record round results into state."""
        factors_info = []
        for f in valid:
            info: dict[str, Any] = {
                "name": f.name,
                "expression": f.expression,
                "hypothesis": f.hypothesis,
            }
            if f.name in metrics:
                info["ic_mean"] = metrics[f.name].get("ic_mean", {})
                info["icir"] = metrics[f.name].get("icir", {})
            if direction_name:
                info["direction"] = direction_name
            factors_info.append(info)
            self._state["all_expressions"].append(f.expression)

        round_entry: dict[str, Any] = {
            "round": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generated": self._config.factors_per_round,
            "valid": len(valid),
            "factors": factors_info,
        }
        if direction_name:
            round_entry["direction"] = direction_name
        self._state["rounds"].append(round_entry)

    def _get_previous_factors(self) -> list[dict]:
        """Collect all previous factors for anti-duplication."""
        result = []
        for rnd in self._state["rounds"]:
            for f in rnd.get("factors", []):
                result.append(f)
        return result

    def _get_top_factors(self, top_n: int = 5) -> list[dict]:
        """Get top-N factors by average absolute ICIR."""
        all_factors = self._get_previous_factors()
        scored = []
        for f in all_factors:
            icir = f.get("icir", {})
            if icir:
                avg_icir = sum(abs(v) for v in icir.values()) / len(icir)
                scored.append((avg_icir, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_n]]

    def _save_state(self) -> None:
        """Persist state to JSON file."""
        path = self._session_dir / "state.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

    def _load_state(self, path: Path) -> None:
        """Load state from a previous session (for resumption)."""
        with open(path, encoding="utf-8") as f:
            self._state = json.load(f)

    # ── Summary ───────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        """Print mining session summary."""
        total_valid = sum(r["valid"] for r in self._state["rounds"])
        total_rounds = len(self._state["rounds"])
        top = self._get_top_factors()

        print()
        print("=" * 60)
        print("MINING SESSION SUMMARY")
        print("=" * 60)
        print(f"  Session: {self._mining_id}")
        print(f"  Rounds: {total_rounds}")
        print(f"  Total valid factors: {total_valid}")
        print(f"  Output: {self._session_dir}")

        if top:
            print()
            print(f"  Top {len(top)} factors by |ICIR|:")
            for i, f in enumerate(top, 1):
                icir = f.get("icir", {})
                icir_str = ", ".join(f"{k}={v:.3f}" for k, v in icir.items())
                print(f"    {i}. {f['name']}: {f['expression']}")
                print(f"       ICIR: [{icir_str}]")

        print()
        print("Next steps:")
        print(f"  python -m nautilus_quants.alpha list --env test --source {self._config.source}")
        print("  python -m nautilus_quants.alpha promote --source-env test --target-env dev")
        print("=" * 60)
