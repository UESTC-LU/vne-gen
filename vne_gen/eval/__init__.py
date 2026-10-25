"""Evaluation protocols of Sec. VI: online, generalization, offline, time-aware, runtime and optimality."""

from .stats import mean_ci, paired_ci, summarize_runs
from .online_eval import evaluate_online, build_substrate, OnlineEvalConfig
from .offline_eval import evaluate_offline, OfflineEvalConfig
from .time_aware_eval import evaluate_time_aware, TimeAwareEvalConfig
from .runtime import measure_runtime, RuntimeConfig
from .optimality import full_space_optimum, evaluate_optimality, OptimalityConfig

__all__ = [
    "mean_ci",
    "paired_ci",
    "summarize_runs",
    "evaluate_online",
    "build_substrate",
    "OnlineEvalConfig",
    "evaluate_offline",
    "OfflineEvalConfig",
    "evaluate_time_aware",
    "TimeAwareEvalConfig",
    "measure_runtime",
    "RuntimeConfig",
    "full_space_optimum",
    "evaluate_optimality",
    "OptimalityConfig",
]
