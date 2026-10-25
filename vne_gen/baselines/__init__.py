"""Adapters for external (learning-based) baselines evaluated with their official implementations."""

from .external import ExternalSolver, FunctionSolver

__all__ = ["ExternalSolver", "FunctionSolver"]
