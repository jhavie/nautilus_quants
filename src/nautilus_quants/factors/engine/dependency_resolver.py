# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Dependency Resolver for Factor Computation.

Handles topological sorting of factors based on their dependencies
to ensure correct computation order.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nautilus_quants.factors.base.factor import Factor


class DependencyError(Exception):
    """Raised when dependency resolution fails."""
    pass


class CircularDependencyError(DependencyError):
    """Raised when circular dependencies are detected."""
    pass


@dataclass
class FactorNode:
    """Node in the dependency graph."""
    name: str
    factor: Factor | None = None
    expression: str = ""
    dependencies: set[str] = field(default_factory=set)
    is_variable: bool = False


class DependencyResolver:
    """
    Resolves factor dependencies using topological sort.
    
    Analyzes factor expressions to determine dependencies and
    produces an execution order that respects those dependencies.
    
    Example:
        ```python
        resolver = DependencyResolver()
        resolver.add_variable("returns", "delta(close, 1) / delay(close, 1)")
        resolver.add_factor("alpha1", "ts_mean(returns, 20)", factor_obj)
        
        order = resolver.resolve()
        # Returns: ["returns", "alpha1"]
        ```
    """
    
    # Known built-in variables (from input data)
    BUILTIN_VARIABLES = {
        "open", "high", "low", "close", "volume",
        "vwap", "returns", "adv",
    }
    
    # Known operators (not variables)
    KNOWN_OPERATORS = {
        # Time-series
        "ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max",
        "ts_rank", "ts_argmax", "ts_argmin",
        "delta", "delay", "correlation", "covariance",
        # Cross-sectional
        "cs_rank", "cs_zscore", "cs_scale", "cs_demean",
        "cs_max", "cs_min",
        # Math
        "log", "exp", "sqrt", "power", "abs", "sign",
        "floor", "ceil", "round", "max", "min",
    }
    
    def __init__(self) -> None:
        """Initialize the resolver."""
        self._nodes: dict[str, FactorNode] = {}
        self._resolved_order: list[str] | None = None
    
    def add_variable(self, name: str, expression: str) -> None:
        """
        Add a variable definition.
        
        Args:
            name: Variable name
            expression: Expression defining the variable
        """
        deps = self._extract_dependencies(expression)
        self._nodes[name] = FactorNode(
            name=name,
            expression=expression,
            dependencies=deps,
            is_variable=True,
        )
        self._resolved_order = None
    
    def add_factor(
        self, 
        name: str, 
        expression: str, 
        factor: Factor | None = None,
    ) -> None:
        """
        Add a factor definition.
        
        Args:
            name: Factor name
            expression: Factor expression
            factor: Optional Factor object
        """
        deps = self._extract_dependencies(expression)
        self._nodes[name] = FactorNode(
            name=name,
            factor=factor,
            expression=expression,
            dependencies=deps,
            is_variable=False,
        )
        self._resolved_order = None
    
    def _extract_dependencies(self, expression: str) -> set[str]:
        """
        Extract variable dependencies from an expression.
        
        Uses simple tokenization to find identifiers that are
        not operators or built-in variables.
        """
        import re
        
        # Find all identifiers
        identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression))
        
        # Filter out operators and builtins
        dependencies = identifiers - self.KNOWN_OPERATORS - self.BUILTIN_VARIABLES
        
        # Only keep dependencies that we know about
        return {dep for dep in dependencies if dep in self._nodes or dep not in self.BUILTIN_VARIABLES}
    
    def resolve(self) -> list[str]:
        """
        Resolve dependencies and return execution order.
        
        Returns:
            List of factor/variable names in execution order
            
        Raises:
            CircularDependencyError: If circular dependencies exist
        """
        if self._resolved_order is not None:
            return self._resolved_order
        
        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = defaultdict(int)
        graph: dict[str, list[str]] = defaultdict(list)
        
        # Build graph
        all_nodes = set(self._nodes.keys())
        for name, node in self._nodes.items():
            valid_deps = node.dependencies & all_nodes
            in_degree[name] = len(valid_deps)
            for dep in valid_deps:
                graph[dep].append(name)
        
        # Initialize queue with nodes having no dependencies
        queue = [name for name in all_nodes if in_degree[name] == 0]
        result: list[str] = []
        
        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(all_nodes):
            remaining = all_nodes - set(result)
            raise CircularDependencyError(
                f"Circular dependency detected involving: {remaining}"
            )
        
        self._resolved_order = result
        return result
    
    def get_factor_order(self) -> list[str]:
        """Get only factor names in execution order (excludes variables)."""
        order = self.resolve()
        return [name for name in order if not self._nodes[name].is_variable]
    
    def get_variable_order(self) -> list[str]:
        """Get only variable names in execution order."""
        order = self.resolve()
        return [name for name in order if self._nodes[name].is_variable]
    
    def get_dependencies(self, name: str) -> set[str]:
        """Get direct dependencies of a factor/variable."""
        if name not in self._nodes:
            return set()
        return self._nodes[name].dependencies.copy()
    
    def get_all_dependencies(self, name: str) -> set[str]:
        """Get all transitive dependencies of a factor/variable."""
        if name not in self._nodes:
            return set()
        
        all_deps: set[str] = set()
        to_process = list(self._nodes[name].dependencies)
        
        while to_process:
            dep = to_process.pop()
            if dep in all_deps or dep not in self._nodes:
                continue
            all_deps.add(dep)
            to_process.extend(self._nodes[dep].dependencies)
        
        return all_deps
    
    def clear(self) -> None:
        """Clear all registered factors and variables."""
        self._nodes.clear()
        self._resolved_order = None
