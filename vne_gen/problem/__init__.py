"""Problem model: substrate networks, virtual network requests, mappings, routing and feasibility (Sec. III)."""

from .substrate import RESOURCES, NUM_RESOURCES, Substrate, CapacityProfile
from .request import Request, request_revenue
from .mapping import Mapping, Allocation, mapping_cost, r2c, long_term_r2c
from .routing import min_hop_feasible_path
from .feasibility import (
    node_fits,
    feasible_virtual_nodes,
    feasible_substrate_nodes,
    validate_mapping,
    ValidationError,
)
from .generators import (
    load_geant,
    load_gml_substrate,
    waxman_substrate,
    sample_capacities,
    sample_request,
    generate_workload,
    WorkloadConfig,
)

__all__ = [
    "RESOURCES",
    "NUM_RESOURCES",
    "Substrate",
    "CapacityProfile",
    "Request",
    "request_revenue",
    "Mapping",
    "Allocation",
    "mapping_cost",
    "r2c",
    "long_term_r2c",
    "min_hop_feasible_path",
    "node_fits",
    "feasible_virtual_nodes",
    "feasible_substrate_nodes",
    "validate_mapping",
    "ValidationError",
    "load_geant",
    "load_gml_substrate",
    "waxman_substrate",
    "sample_capacities",
    "sample_request",
    "generate_workload",
    "WorkloadConfig",
]
