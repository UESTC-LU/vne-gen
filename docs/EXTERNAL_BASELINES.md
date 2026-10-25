# Running external baselines through the same protocols

The paper compares VNE-Gen with NEA-Rank, MCTS, A3C-GCN, GAL-VNE, HRL-ACRA and FlagVNE (and with GRC-Rank
and NRM-Rank in the offline studies), using the baselines' official implementations and training every
learning-based method on the same GEANT workload distribution.  This repository ships re-implementations of
the three ranking heuristics and a compact MCTS only; the learned baselines live in their own code bases:

| Method | Official code |
|---|---|
| A3C-GCN [9], FlagVNE [13], HRL-ACRA [12], MCTS [7] and the ranking heuristics | the `virne` benchmark library (`https://github.com/GeminiLight/virne`) |
| GAL-VNE [11] | `https://github.com/Thinklab-SJTU/GAL-VNE` |

## The solver interface

Every method takes part in the online, offline and time-aware protocols through one method:

```python
class MySolver:
    name = "MyMethod"

    def embed(self, request, substrate, profile):
        """Return a vne_gen.problem.Mapping feasible under `profile`, or None to reject the request."""

    # optional: used as a proposal source by the offline / time-aware selectors (Algorithm 2)
    def propose(self, request, substrate, profile, rho=1.0, rng=None):
        return self.embed(request, substrate, profile)
```

* `request.node_demand` is an `(|V_r|, 3)` integer array (CPU, GPU, RAM), `request.edges` a list of
  `(i, j)` virtual links with `request.link_demand[l]` the bandwidth of `request.edges[l]`.
* `profile.node` `(|V_S|, 3)` and `profile.link` `(|E_S|,)` are the residual capacities at arrival;
  `substrate.edges[e]` gives the end points of substrate edge `e` and `substrate.edge_id(u, v)` the inverse.
* `Mapping(node_map=[...], link_paths={l: [substrate nodes from node_map[i] to node_map[j]]})`.

`vne_gen.baselines.external.ExternalSolver` provides array conversions and `mapping_from_paths` for code
bases that return paths keyed by virtual-link end points.  Every admitted mapping is re-validated by
`vne_gen.problem.validate_mapping`, exactly as for VNE-Gen.

## Matching the workload

Use `vne_gen.eval.online_eval.build_substrate(kind, seed)` and `workload_for(seed, lam, num_arrivals)` to
obtain the substrate and request sequence of a given (seed, λ), convert them to the external format, and
feed the external solver's decisions back through `embed` (or run the external simulator on the exported
workload and compare admitted revenue on the same arrivals).  Keep the simulator's rule that departures are
processed before arrivals at equal time stamps and that accepted mappings are committed at arrival.
