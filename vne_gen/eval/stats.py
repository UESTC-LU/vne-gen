"""Confidence intervals across matched seeds (Student's t) and paired seed-wise differences."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` using the Student's t distribution across seeds."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    m = float(x.mean()) if n else float("nan")
    if n < 2:
        return m, m, m
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - half, m + half


def paired_ci(a: Sequence[float], b: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Mean and CI of the seed-wise differences ``a - b`` (``a`` and ``b`` matched by seed)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return mean_ci(d, confidence)


def summarize_runs(per_seed: Dict[str, List[float]], reference: str = None) -> Dict[str, dict]:
    """Summaries (mean/CI, gain over a reference method and paired CI) for ``{method: [values per seed]}``."""
    out: Dict[str, dict] = {}
    for name, vals in per_seed.items():
        m, lo, hi = mean_ci(vals)
        entry = {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(vals)}
        if reference is not None and reference in per_seed and name != reference:
            ref = per_seed[reference]
            ref_mean = float(np.mean(ref))
            entry["gain_over_reference_pct"] = 100.0 * (m - ref_mean) / ref_mean if ref_mean else float("nan")
            pm, plo, phi = paired_ci(vals, ref)
            entry.update({"paired_diff_mean": pm, "paired_ci_low": plo, "paired_ci_high": phi})
        out[name] = entry
    return out
