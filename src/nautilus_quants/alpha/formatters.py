# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Rich table formatters for alpha CLI output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from nautilus_quants.alpha.registry.models import (
        AnalysisMetrics,
        BacktestRunRecord,
        FactorRecord,
    )

console = Console()

# ── Style helpers ──

_STATUS_STYLES: dict[str, str] = {
    "candidate": "yellow",
    "active": "green",
    "archived": "dim",
}


def _status_markup(status: str) -> str:
    style = _STATUS_STYLES.get(status, "white")
    return f"[{style}]{status}[/{style}]"


def _color_float(
    value: float | None, fmt: str = ".4f", *, invert: bool = False,
) -> str:
    if value is None:
        return "[dim]-[/dim]"
    formatted = f"{value:{fmt}}"
    if value > 0:
        style = "red" if invert else "green"
    elif value < 0:
        style = "green" if invert else "red"
    else:
        style = "white"
    return f"[{style}]{formatted}[/{style}]"


def _color_pct(value: float | None, fmt: str = ".1f") -> str:
    if value is None:
        return "[dim]-[/dim]"
    formatted = f"{value * 100:{fmt}}%"
    style = "green" if value >= 0.5 else "red" if value < 0.4 else "yellow"
    return f"[{style}]{formatted}[/{style}]"


def _dim_or(value: str | None, fallback: str = "-") -> str:
    return value if value else f"[dim]{fallback}[/dim]"


# ── list ──


def print_factor_list(
    factors: list[FactorRecord],
    has_scores: bool,
    show_expression: bool = True,
) -> None:
    """Print factor list as a Rich table."""
    if has_scores:
        factors = sorted(
            factors,
            key=lambda f: f.parameters.get("promote_score", 0) or 0,
            reverse=True,
        )
        table = Table(
            title=f"Factors ({len(factors)})",
            show_header=True,
            header_style="bold cyan",
            show_lines=False,
            pad_edge=False,
        )
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("factor_id", style="white", no_wrap=True)
        table.add_column("score", justify="right", style="bold")
        table.add_column("status", justify="center")
        table.add_column("source", style="dim")
        table.add_column("tags", style="cyan")
        if show_expression:
            table.add_column("expression", style="dim")

        for i, f in enumerate(factors, 1):
            score = f.parameters.get("promote_score")
            score_str = _color_float(score)
            tags_str = ", ".join(f.tags) if f.tags else "-"
            row = [
                str(i), f.factor_id, score_str,
                _status_markup(f.status), _dim_or(f.source), tags_str,
            ]
            if show_expression:
                row.append(f.expression)
            table.add_row(*row)
    else:
        table = Table(
            title=f"Factors ({len(factors)})",
            show_header=True,
            header_style="bold cyan",
            show_lines=False,
            pad_edge=False,
        )
        table.add_column("factor_id", style="white", no_wrap=True)
        table.add_column("prototype", style="cyan")
        table.add_column("status", justify="center")
        table.add_column("source", style="dim")
        table.add_column("tags", style="cyan")
        if show_expression:
            table.add_column("expression", style="dim")

        for f in factors:
            tags_str = ", ".join(f.tags) if f.tags else "-"
            row = [
                f.factor_id, _dim_or(f.prototype),
                _status_markup(f.status), _dim_or(f.source), tags_str,
            ]
            if show_expression:
                row.append(f.expression)
            table.add_row(*row)

    console.print(table)


# ── inspect (single factor) ──


def print_factor_detail(
    f: FactorRecord,
    metrics: list[AnalysisMetrics] | None = None,
    backtests: list[tuple[Any, list[Any]]] | None = None,
) -> None:
    """Print factor details with optional metrics and backtests."""
    # Header panel
    info_lines = [
        f"[bold]Expression:[/bold] {f.expression}",
        f"[bold]Prototype:[/bold]  {_dim_or(f.prototype)}",
        f"[bold]Source:[/bold]     {_dim_or(f.source)}",
        f"[bold]Status:[/bold]     {_status_markup(f.status)}",
        f"[bold]Tags:[/bold]       {', '.join(f.tags) if f.tags else '[dim]-[/dim]'}",
    ]
    if f.parameters:
        info_lines.append(f"[bold]Parameters:[/bold] {f.parameters}")
    if f.variables:
        info_lines.append(f"[bold]Variables:[/bold]  {f.variables}")

    console.print(Panel(
        "\n".join(info_lines),
        title=f"[bold]{f.factor_id}[/bold]",
        border_style="cyan",
    ))

    # Metrics table
    if metrics:
        mt = Table(
            title=f"Analysis Metrics ({len(metrics)})",
            show_header=True,
            header_style="bold cyan",
            show_lines=False,
        )
        mt.add_column("run_id", style="dim", no_wrap=True)
        mt.add_column("period", style="white")
        mt.add_column("IC", justify="right")
        mt.add_column("ICIR", justify="right")
        mt.add_column("timeframe", style="dim")

        for m in metrics[:12]:
            mt.add_row(
                m.run_id,
                m.period,
                _color_float(m.ic_mean),
                _color_float(m.icir),
                m.timeframe or "-",
            )
        console.print(mt)

    # Backtests table
    if backtests:
        bt = Table(
            title=f"Backtests ({len(backtests)})",
            show_header=True,
            header_style="bold cyan",
            show_lines=False,
        )
        bt.add_column("backtest_id", style="dim", no_wrap=True)
        bt.add_column("sharpe", justify="right")
        bt.add_column("pnl%", justify="right")
        bt.add_column("max_dd", justify="right")
        bt.add_column("timeframe", style="dim")
        bt.add_column("instr", justify="right", style="dim")
        bt.add_column("factors", style="cyan", max_width=50, overflow="ellipsis")

        for run, factor_links in backtests:
            fids = ", ".join(bf.factor_id for bf in factor_links) if factor_links else "-"
            dd = _color_float(
                run.max_drawdown, ".2%", invert=True,
            ) if run.max_drawdown else "[dim]-[/dim]"
            bt.add_row(
                run.backtest_id,
                _color_float(run.sharpe_ratio),
                _color_float(run.total_pnl_pct, ".2f"),
                dd,
                run.timeframe or "-",
                str(run.instrument_count),
                fids,
            )
        console.print(bt)


# ── inspect --prototype ──


def print_prototype_group(
    proto_name: str,
    sources: set[str],
    template: str | None,
    factors: list[tuple[str, dict[str, float], str]],
    p_keys: list[str],
) -> None:
    """Print prototype group overview."""
    source_str = ", ".join(sorted(sources)) if sources else "(none)"
    header = f"[bold]{proto_name}[/bold]  ({len(factors)} factors, source: {source_str})"
    if template:
        header += f"\n[dim]Template: {template}[/dim]"

    table = Table(
        title=header,
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("factor_id", style="white", no_wrap=True)
    for k in p_keys:
        table.add_column(k, justify="right", style="yellow")
    table.add_column("status", justify="center")

    for name, params, st in factors:
        row = [name]
        for k in p_keys:
            val = params.get(k, "")
            row.append(str(val) if val != "" else "[dim]-[/dim]")
        row.append(_status_markup(st))
        table.add_row(*row)

    console.print(table)


# ── metrics ──


def print_metrics_table(factor_id: str, results: list[AnalysisMetrics]) -> None:
    """Print detailed metrics table for a factor."""
    table = Table(
        title=f"Metrics: {factor_id} ({len(results)} records)",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("run_id", style="dim", no_wrap=True)
    table.add_column("period", style="white")
    table.add_column("IC", justify="right")
    table.add_column("ICIR", justify="right")
    table.add_column("t(NW)", justify="right")
    table.add_column("p(NW)", justify="right")
    table.add_column("mono", justify="right")
    table.add_column("win%", justify="right")
    table.add_column("IC_lin", justify="right")
    table.add_column("IC_skew", justify="right")
    table.add_column("IC_kur", justify="right")
    table.add_column("AR1", justify="right")
    table.add_column("N", justify="right", style="dim")

    for m in results:
        table.add_row(
            m.run_id,
            m.period,
            _color_float(m.ic_mean),
            _color_float(m.icir),
            _color_float(m.t_stat_nw, ".2f"),
            f"{m.p_value_nw:.2e}" if m.p_value_nw is not None else "[dim]-[/dim]",
            _color_float(m.monotonicity, ".2f"),
            _color_pct(m.win_rate),
            _color_float(m.ic_linearity, ".3f"),
            _color_float(m.ic_skew),
            f"{m.ic_kurtosis:.4f}" if m.ic_kurtosis is not None else "[dim]-[/dim]",
            _color_float(m.ic_ar1, ".3f"),
            str(m.n_samples) if m.n_samples is not None else "[dim]-[/dim]",
        )

    console.print(table)


# ── backtests ──


def print_backtests_table(
    runs: list[tuple[Any, list[Any]]],
) -> None:
    """Print backtest runs table."""
    table = Table(
        title=f"Backtests ({len(runs)})",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("backtest_id", style="dim", no_wrap=True)
    table.add_column("strategy", style="white")
    table.add_column("timeframe", style="dim")
    table.add_column("instr", justify="right", style="dim")
    table.add_column("sharpe", justify="right")
    table.add_column("pnl%", justify="right")
    table.add_column("max_dd", justify="right")
    table.add_column("win_rate", justify="right")
    table.add_column("started_at", style="dim", no_wrap=True)
    table.add_column("dur(s)", justify="right", style="dim")
    table.add_column("factors", style="cyan", max_width=50, overflow="ellipsis")

    for run, factor_links in runs:
        fids = ", ".join(bf.factor_id for bf in factor_links) if factor_links else "-"
        dd = _color_float(
            run.max_drawdown, ".2%", invert=True,
        ) if run.max_drawdown else "[dim]-[/dim]"
        wr = _color_pct(run.win_rate) if run.win_rate else "[dim]-[/dim]"
        started = run.started_at[:16] if run.started_at else "-"
        table.add_row(
            run.backtest_id,
            run.strategy_name,
            run.timeframe or "-",
            str(run.instrument_count),
            _color_float(run.sharpe_ratio),
            _color_float(run.total_pnl_pct, ".2f"),
            dd,
            wr,
            started,
            f"{run.duration_seconds:.1f}",
            fids,
        )

    console.print(table)


# ── promote ──


def print_promote_header(
    source_env: str,
    target_env: str,
    config_path: str,
    periods: list[str],
    dry_run: bool,
    skip_corr: bool,
    max_factors: int,
) -> None:
    """Print promote pipeline header."""
    console.print(Panel(
        f"[bold]Source:[/bold]      {source_env}.duckdb\n"
        f"[bold]Target:[/bold]      {target_env}.duckdb\n"
        f"[bold]Config:[/bold]      {config_path}\n"
        f"[bold]Periods:[/bold]     {periods}\n"
        f"[bold]Dry run:[/bold]     {dry_run}\n"
        f"[bold]Skip corr:[/bold]   {skip_corr}\n"
        f"[bold]Max factors:[/bold] {max_factors}",
        title="[bold]Factor Promotion Pipeline[/bold]",
        border_style="blue",
    ))


def print_top_scores(df: Any, n: int = 30) -> None:
    """Print top scored factors table (accepts a pandas DataFrame)."""
    table = Table(
        title=f"Top {min(n, len(df))} Factors by Score",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("factor_id", style="white", no_wrap=True)
    table.add_column("score", justify="right", style="bold")
    table.add_column("avg_pp", justify="right")
    table.add_column("cons", justify="right")
    table.add_column("turn", justify="right")
    table.add_column("#pd", justify="right", style="dim")

    for fid, row in df.head(n).iterrows():
        table.add_row(
            str(fid),
            _color_float(row.get("final_score", 0)),
            _color_float(row.get("avg_period_score", 0)),
            _color_float(row.get("consistency", 0), ".3f"),
            _color_float(row.get("turnover_friendliness", 0), ".3f"),
            str(int(row.get("n_valid_periods", 0))),
        )

    console.print(table)


def print_selected_factors(selected_ids: list[str], df: Any) -> None:
    """Print final selected factors."""
    table = Table(
        title=f"Selected Factors ({len(selected_ids)})",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("#", style="dim", justify="right", width=4)
    table.add_column("factor_id", style="white", no_wrap=True)
    table.add_column("score", justify="right", style="bold")

    for i, fid in enumerate(selected_ids, 1):
        score = df.loc[fid, "final_score"] if fid in df.index else 0
        table.add_row(str(i), fid, _color_float(score))

    console.print(table)


def print_promote_summary(
    n_after_filter: int,
    n_after_dedup: int,
    n_selected: int,
    duration: float,
    dry_run: bool,
) -> None:
    """Print promotion pipeline summary."""
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]EXECUTED[/green]"
    console.print(Panel(
        f"[bold]After hard filter:[/bold]    {n_after_filter}\n"
        f"[bold]After fingerprint dedup:[/bold] {n_after_dedup}\n"
        f"[bold]Final selected:[/bold]       {n_selected}\n"
        f"[bold]Duration:[/bold]             {duration:.2f}s\n"
        f"[bold]Mode:[/bold]                {mode}",
        title="[bold]Promotion Summary[/bold]",
        border_style="green" if not dry_run else "yellow",
    ))


# ── audit ──


def print_audit_duplicates(dup_groups: list[list[Any]]) -> None:
    """Print expression duplicate groups."""
    if not dup_groups:
        console.print("[green]No expression duplicates found.[/green]")
        return

    table = Table(
        title=f"Expression Duplicates ({len(dup_groups)} groups)",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("group", style="dim", justify="right", width=5)
    table.add_column("factor_ids", style="white")
    table.add_column("expression", style="dim", max_width=70, overflow="ellipsis")

    for i, group in enumerate(dup_groups, 1):
        ids = " = ".join(f.factor_id for f in group)
        table.add_row(str(i), ids, group[0].expression[:80])

    console.print(table)


def print_audit_template_groups(groups: dict[str, list]) -> None:
    """Print template groups from audit."""
    if not groups:
        console.print("[dim]No template groups found.[/dim]")
        return

    table = Table(
        title=f"Template Groups ({len(groups)} groups)",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", style="dim", justify="right", width=4)
    table.add_column("template", style="white", max_width=65, overflow="ellipsis")
    table.add_column("members", style="cyan")

    for i, (tmpl, members) in enumerate(
        sorted(groups.items(), key=lambda x: -len(x[1])), 1,
    ):
        member_lines = []
        for fid, params in members[:5]:
            member_lines.append(f"{fid}  {params}")
        if len(members) > 5:
            member_lines.append(f"... +{len(members) - 5} more")
        table.add_row(str(i), tmpl[:70], "\n".join(member_lines))

    console.print(table)


# ── simple result outputs ──


def print_register_result(
    new: int, updated: int, unchanged: int, renamed: int = 0,
) -> None:
    """Print register command result."""
    total = new + updated + unchanged + renamed
    parts = [f"[green]{new} new[/green]", f"[yellow]{updated} updated[/yellow]",
             f"[dim]{unchanged} unchanged[/dim]"]
    if renamed:
        parts.append(f"[cyan]{renamed} renamed (collision)[/cyan]")
    console.print(f"Registered {total} factors ({', '.join(parts)})")


def print_backfill_result(counts: dict[str, int], dry_run: bool) -> None:
    """Print backfill command result."""
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]EXECUTED[/green]"
    console.print(f"[{mode}] Backfill results:")
    console.print(f"  expression_hash updated: [cyan]{counts['hash']}[/cyan]")
    console.print(f"  prototype fixed:         [cyan]{counts['prototype']}[/cyan]")
    console.print(f"  parameters extracted:    [cyan]{counts['parameters']}[/cyan]")
    if dry_run:
        console.print("\n[dim]Use --execute to apply changes.[/dim]")


def print_dedup_result(
    removed: list[tuple[str, str]], dry_run: bool,
) -> None:
    """Print dedup command result."""
    if not removed:
        console.print("[green]No duplicates found.[/green]")
        return

    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]EXECUTED[/green]"
    table = Table(
        title=f"[{mode}] Duplicates ({len(removed)})",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("action", style="red", width=8)
    table.add_column("factor_id", style="white")
    table.add_column("kept", style="green")

    for rm_id, keep_id in removed:
        table.add_row("DELETE", rm_id, keep_id)

    console.print(table)
    if dry_run:
        console.print("\n[dim]Use --execute to actually delete.[/dim]")


# ── tune command formatters ──


def _format_optional_float(value: float | None, precision: int = 4) -> str:
    """Render ``None``/``NaN`` as an em-dash, numbers with the given precision."""
    if value is None:
        return "—"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return str(value)
    import math

    if math.isnan(fv):
        return "—"
    return f"{fv:+.{precision}f}" if precision > 0 else f"{fv:.{precision}f}"


def print_tune_result(
    tune_result,
    *,
    label: str | None = None,
    registration=None,
) -> None:
    """Print a single ``TuneResult`` as a rich-formatted summary.

    Used by ``alpha tune`` for both prototype-grouped and single-factor
    modes. ``registration`` (a ``RegistrationSummary`` or ``None``) drives
    the "registered variants" footer.
    """
    header = label or tune_result.template
    title = Text(f"Tune result — {header}", style="bold cyan")
    console.print(Panel(title, border_style="cyan"))

    base_tbl = Table(show_header=False, box=None, pad_edge=False)
    base_tbl.add_column(style="dim", width=20)
    base_tbl.add_column(style="white")
    base_tbl.add_row("template", tune_result.template)
    base_tbl.add_row("original", tune_result.original_expression)
    base_tbl.add_row("best", f"[green]{tune_result.best_expression}[/green]")
    base_tbl.add_row(
        "CV ICIR",
        _format_optional_float(tune_result.best_icir_cv),
    )
    base_tbl.add_row(
        "holdout ICIR",
        _format_optional_float(tune_result.holdout_icir),
    )
    base_tbl.add_row(
        "holdout t(NW)",
        _format_optional_float(tune_result.holdout_t_stat_nw, precision=2),
    )
    base_tbl.add_row(
        "stability",
        _format_optional_float(tune_result.stability_score, precision=2),
    )
    base_tbl.add_row(
        "adjusted p",
        _format_optional_float(tune_result.adjusted_p_value, precision=4),
    )
    base_tbl.add_row(
        "trials",
        f"{tune_result.n_trials}  ([red]{tune_result.n_pruned} pruned[/red])",
    )
    console.print(base_tbl)

    if tune_result.top_k:
        top_tbl = Table(
            title="Top variants",
            show_header=True,
            header_style="bold cyan",
        )
        top_tbl.add_column("#", width=3, justify="right")
        top_tbl.add_column("CV ICIR", width=10, justify="right")
        top_tbl.add_column("Expression")
        for i, trial in enumerate(tune_result.top_k, start=1):
            top_tbl.add_row(
                str(i),
                _format_optional_float(trial.mean_icir),
                trial.expression,
            )
        console.print(top_tbl)

    if tune_result.param_importance:
        imp_tbl = Table(
            title="Parameter importance",
            show_header=True,
            header_style="bold cyan",
        )
        imp_tbl.add_column("parameter", style="white")
        imp_tbl.add_column("importance", justify="right", style="magenta")
        for name, value in sorted(
            tune_result.param_importance.items(),
            key=lambda kv: kv[1],
            reverse=True,
        ):
            imp_tbl.add_row(name, f"{value:.1%}")
        console.print(imp_tbl)

    if registration is not None:
        console.print()
        if registration.variants:
            console.print(
                f"[green]Registered {registration.n_registered} new[/green], "
                f"[yellow]{registration.n_updated} updated[/yellow], "
                f"[dim]{registration.n_skipped} skipped[/dim] "
                f"(source: {tune_result.best_expression})"
            )
            for v in registration.variants:
                marker = {
                    "new": "[green]+[/green]",
                    "updated": "[yellow]~[/yellow]",
                    "unchanged": "[dim]=[/dim]",
                    "skipped_duplicate": "[dim]-[/dim]",
                }.get(v.outcome, "?")
                console.print(f"  {marker} {v.factor_id}  ({v.outcome})")


def print_operator_comparison(
    operator_comparison: dict | None,
) -> None:
    """Render the per-slot operator comparison as a table.

    Passed the ``TuneResult.operator_comparison`` dict verbatim; skips quietly
    when operator tuning was disabled.
    """
    if not operator_comparison:
        return
    for slot_id, per_op in operator_comparison.items():
        if not per_op:
            continue
        tbl = Table(
            title=f"Operator comparison ({slot_id})",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("operator", style="white")
        tbl.add_column("best |ICIR|", justify="right", style="magenta")
        for name, score in sorted(per_op.items(), key=lambda kv: kv[1], reverse=True):
            tbl.add_row(name, f"{score:+.4f}")
        console.print(tbl)


def print_variable_comparison(
    variable_comparison: dict | None,
) -> None:
    """Render the per-slot variable comparison as a table."""
    if not variable_comparison:
        return
    for slot_id, per_var in variable_comparison.items():
        if not per_var:
            continue
        tbl = Table(
            title=f"Variable comparison ({slot_id})",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("variable", style="white")
        tbl.add_column("best |ICIR|", justify="right", style="magenta")
        for name, score in sorted(per_var.items(), key=lambda kv: kv[1], reverse=True):
            tbl.add_row(name, f"{score:+.4f}")
        console.print(tbl)


def print_eligibility_report(report) -> None:
    """Summarise the pre-tune eligibility pass."""
    tbl = Table(
        title="Tune eligibility",
        show_header=False,
        box=None,
        pad_edge=False,
    )
    tbl.add_column(style="dim", width=26)
    tbl.add_column(style="white")
    tbl.add_row("candidates", str(report.n_total))
    tbl.add_row("eligible", f"[green]{report.n_eligible}[/green]")
    tbl.add_row("rejected", f"[red]{report.n_rejected}[/red]")
    tbl.add_row("  no metrics", str(report.n_no_metrics))
    tbl.add_row("  |ICIR| below threshold", str(report.n_low_icir))
    tbl.add_row("  coverage below threshold", str(report.n_low_coverage))
    tbl.add_row("  n_samples below threshold", str(report.n_low_samples))
    tbl.add_row("  too few valid periods", str(report.n_too_few_periods))
    console.print(tbl)
