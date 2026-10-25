"""Generator-Selector framework (Sec. V): proposal interface, exact selectors and Algorithm 2."""

from .proposal import Proposal, ProposalPool, ProposalSource, PortfolioSource
from .ilp import SelectionResult, solve_selection
from .timeslots import elementary_slots, activity_matrix
from .online import OnlineSelector
from .offline import CandidatePoolSelector, SelectorConfig, OfflineRunResult

__all__ = [
    "Proposal",
    "ProposalPool",
    "ProposalSource",
    "PortfolioSource",
    "SelectionResult",
    "solve_selection",
    "elementary_slots",
    "activity_matrix",
    "OnlineSelector",
    "CandidatePoolSelector",
    "SelectorConfig",
    "OfflineRunResult",
]
