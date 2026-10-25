"""Time slots and activity indicators for time-aware offline VNE (Eq. 4).

The evaluation horizon is partitioned into *elementary* slots: the intervals between consecutive distinct
arrival/departure times of the request set.  A request is active in slot ``t`` iff the slot lies inside its
activity interval, :math:`\\alpha_{r,t}=\\mathbb I[a_r\\le t<e_r]`.  Enforcing the capacity constraint in every
elementary slot is equivalent to enforcing it at every instant of the horizon.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..problem.request import Request


def elementary_slots(requests: Sequence[Request]) -> List[Tuple[float, float]]:
    points = sorted({float(r.arrival) for r in requests} | {float(r.departure) for r in requests if np.isfinite(r.departure)})
    slots = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    if any(not np.isfinite(r.departure) for r in requests):
        slots.append((points[-1], float("inf")))
    return slots


def activity_matrix(requests: Sequence[Request], slots: Sequence[Tuple[float, float]]) -> np.ndarray:
    """``alpha[r, t] = 1`` iff request ``r`` is active during slot ``t``."""
    alpha = np.zeros((len(requests), len(slots)), dtype=np.int8)
    for i, r in enumerate(requests):
        for t, (s, e) in enumerate(slots):
            if r.arrival <= s and e <= r.departure:
                alpha[i, t] = 1
    return alpha
