# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor base classes module (legacy).

The old Factor / ExpressionFactor / TimeSeriesFactor / CrossSectionalFactor
base classes have been removed in favour of FactorEngine, which evaluates
AST expressions directly on panel DataFrames without per-instrument Factor objects.
"""
