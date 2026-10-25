"""Size-conditioned return targets :math:`R^\\star(n)` (Sec. IV-C).

For each request size ``n`` the deployment target is the largest initial return observed in the corpus,

.. math:: R^\\star(n)=\\max_{\\tau\\in\\mathcal D_{map}:|V_\\tau|=n} R_0^\\tau,\\qquad n\\in\\{2,\\dots,10\\}.

The table is stored with the checkpoint; online decoding uses :math:`R_0^{on}=R^\\star(|V_r|)` and offline
decoding :math:`R_0^{off}(\\rho)=\\rho R^\\star(|V_r|)`.
"""

from __future__ import annotations

from typing import Dict, Sequence

from ..env.mapping_env import AttemptRecord


def compute_return_table(records: Sequence[AttemptRecord]) -> Dict[int, float]:
    table: Dict[int, float] = {}
    for rec in records:
        n = rec.request.num_nodes
        r0 = rec.initial_return
        if n not in table or r0 > table[n]:
            table[n] = float(r0)
    return dict(sorted(table.items()))
