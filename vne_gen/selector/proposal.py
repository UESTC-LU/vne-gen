"""Proposal interface (Sec. V-A).

A proposal :math:`p_{r,m}` contains the request identifier, a complete mapping ``m``, the induced node and
link allocations, the revenue :math:`b_r` and the routed-resource cost :math:`c_{r,m}`.  Proposals with
identical allocation signatures :math:`\\sigma_r(m)` have identical coefficients in the capacity
constraints and identical objective coefficients, so each per-request pool keeps one representative per
signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Protocol, Sequence

import numpy as np

from ..problem.feasibility import is_feasible
from ..problem.mapping import Allocation, Mapping, mapping_cost
from ..problem.request import Request, request_revenue
from ..problem.substrate import CapacityProfile, Substrate


@dataclass
class Proposal:
    request_id: int
    mapping: Mapping
    allocation: Allocation
    revenue: float
    cost: float
    source: str = ""

    @property
    def signature(self) -> bytes:
        return self.allocation.signature()

    @classmethod
    def build(cls, request: Request, mapping: Mapping, substrate: Substrate, source: str = "") -> "Proposal":
        return cls(
            request_id=request.id,
            mapping=mapping,
            allocation=mapping.allocation(request, substrate),
            revenue=request_revenue(request),
            cost=mapping_cost(request, mapping),
            source=source,
        )


class ProposalPool:
    """Accumulated proposal pool :math:`\\mathcal M_r` of one request with allocation-signature deduplication."""

    def __init__(self, request_id: int) -> None:
        self.request_id = request_id
        self._by_signature: Dict[bytes, Proposal] = {}
        self.proposals: List[Proposal] = []

    def add(self, proposal: Proposal) -> bool:
        """Add a proposal if its allocation signature is new; returns ``True`` when it was added."""
        sig = proposal.signature
        if sig in self._by_signature:
            return False
        self._by_signature[sig] = proposal
        self.proposals.append(proposal)
        return True

    def __len__(self) -> int:
        return len(self.proposals)

    def __iter__(self) -> Iterator[Proposal]:
        return iter(self.proposals)


class ProposalSource(Protocol):
    """Anything that can construct one mapping proposal for a request under a supplied profile."""

    name: str

    def propose(self, request: Request, substrate: Substrate, profile: CapacityProfile, rho: float = 1.0,
                rng: Optional[np.random.Generator] = None) -> Optional[Mapping]:
        ...


class PortfolioSource:
    """A portfolio of proposal sources queried in turn (one call per source per generation call)."""

    def __init__(self, sources: Sequence[ProposalSource], name: str = "Portfolio") -> None:
        self.sources = list(sources)
        self.name = name

    def propose_all(self, request: Request, substrate: Substrate, profile: CapacityProfile, rho: float = 1.0,
                    rng: Optional[np.random.Generator] = None) -> List[Mapping]:
        out = []
        for s in self.sources:
            m = s.propose(request, substrate, profile, rho, rng)
            if m is not None:
                out.append(m)
        return out

    def propose(self, request, substrate, profile, rho: float = 1.0, rng=None) -> Optional[Mapping]:
        ms = self.propose_all(request, substrate, profile, rho, rng)
        return ms[0] if ms else None
