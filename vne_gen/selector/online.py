"""Online selector (Sec. V-C): one greedy generator call under the current residual profile, immediate commit."""

from __future__ import annotations

from typing import Optional

from ..problem.mapping import Mapping
from ..problem.request import Request
from ..problem.substrate import CapacityProfile, Substrate


class OnlineSelector:
    """Wraps any object with an ``embed`` method; the simulator commits every validated proposal at arrival."""

    def __init__(self, generator, name: Optional[str] = None) -> None:
        self.generator = generator
        self.name = name or getattr(generator, "name", "generator")

    def embed(self, request: Request, substrate: Substrate, profile: CapacityProfile) -> Optional[Mapping]:
        return self.generator.embed(request, substrate, profile)
