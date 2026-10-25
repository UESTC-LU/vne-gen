"""Offline trajectory dataset :math:`\\mathcal D_{map}` and batch assembly for the K-block causal context.

Each training example is a pair (attempt :math:`\\tau`, decision index ``k``).  The example carries the
current, possibly incomplete decision block and the preceding ``K-1`` completed blocks: for every window
position ``w`` the state features :math:`s_w`, the return-to-go :math:`R_w`, the decision index ``w`` and,
for completed blocks, the selected placement pair.  The current block additionally carries the supervision
targets :math:`(v_k, u_k)` and the feasibility masks :math:`\\mathcal F_V(s_k)` and
:math:`\\mathcal F_S(s_k, v_k)`.

States are reconstructed on the fly by replaying the recorded action pairs through the mapping process, so
the corpus only stores requests, supplied profiles, actions and rewards.

:func:`collate_examples` packs all window states of a batch into two "packed" graph batches (request graphs
and substrate graphs) in the style of PyTorch Geometric, using plain numpy arrays so that the batching logic
is independent of the deep-learning framework.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..env.mapping_env import AttemptRecord, MappingAttempt, replay_attempt
from ..problem.substrate import Substrate
from .features import (
    NUM_EDGE_FEATURES,
    NUM_REQ_FEATURES,
    NUM_SCALARS,
    NUM_SUB_FEATURES,
    FeatureConfig,
    StateFeatures,
    encode_state,
)


# --------------------------------------------------------------------------- corpus io
@dataclass
class CorpusMeta:
    substrate_name: str = ""
    sources: Dict[str, int] = field(default_factory=dict)
    num_attempts: int = 0
    num_transitions: int = 0
    completed_fraction: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)


def save_corpus(path: str, substrate: Substrate, records: Sequence[AttemptRecord], meta: Optional[CorpusMeta] = None) -> None:
    payload = {
        "substrate": substrate.to_dict(),
        "attempts": [r.to_dict() for r in records],
        "meta": (meta or CorpusMeta()).__dict__,
    }
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_corpus(path: str) -> Tuple[Substrate, List[AttemptRecord], CorpusMeta]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        payload = pickle.load(f)
    substrate = Substrate.from_dict(payload["substrate"])
    records = [AttemptRecord.from_dict(d) for d in payload["attempts"]]
    meta = CorpusMeta(**payload.get("meta", {}))
    return substrate, records, meta


# --------------------------------------------------------------------------- examples
@dataclass
class Example:
    """One (attempt, decision) training example with its K-block window."""

    window: List[StateFeatures]  # states s_{k-L+1} .. s_k  (L <= K)
    rtg: np.ndarray  # (L,) returns-to-go R_w
    timesteps: np.ndarray  # (L,) decision indices w
    actions: np.ndarray  # (L, 2) selected (v_w, u_w) for completed blocks; last row = target (v_k, u_k)
    fs_mask: np.ndarray  # (|V_S|,) feasible substrate nodes for the target virtual node
    request_size: int
    success: bool
    num_steps: int = 1  # T_tau, number of emitted action pairs of the attempt


class TrajectoryDataset:
    """Index over all (attempt, decision) pairs of a corpus (``torch``-free; works with any DataLoader)."""

    def __init__(
        self,
        substrate: Substrate,
        records: Sequence[AttemptRecord],
        context_blocks: int = 3,
        feature_cfg: FeatureConfig = FeatureConfig(),
        completed_only: bool = False,
    ) -> None:
        self.substrate = substrate
        self.records = [r for r in records if (r.success or not completed_only)]
        self.K = int(context_blocks)
        self.feature_cfg = feature_cfg
        self.index: List[Tuple[int, int]] = []
        for a, rec in enumerate(self.records):
            for k in range(rec.num_steps):  # only emitted action pairs contribute to the losses
                self.index.append((a, k))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Example:
        a, k = self.index[idx]
        rec = self.records[a]
        rtg_all = rec.returns_to_go
        start = max(0, k - self.K + 1)
        attempt = MappingAttempt(self.substrate, rec.request, rec.profile)
        window: List[StateFeatures] = []
        for w in range(0, k + 1):
            if w >= start:
                window.append(encode_state(attempt, self.feature_cfg))
            if w < k:
                v, u = rec.actions[w]
                _, done = attempt.step(v, u)
                if done:
                    raise RuntimeError("attempt terminated before the indexed decision during replay")
        v_k, u_k = rec.actions[k]
        fs_mask = attempt.feasible_substrate_mask(v_k)
        actions = np.asarray(rec.actions[start : k + 1], dtype=np.int64).reshape(-1, 2)
        return Example(
            window=window,
            rtg=np.asarray(rtg_all[start : k + 1], dtype=np.float32),
            timesteps=np.arange(start, k + 1, dtype=np.int64),
            actions=actions,
            fs_mask=fs_mask,
            request_size=rec.request.num_nodes,
            success=rec.success,
            num_steps=rec.num_steps,
        )


# --------------------------------------------------------------------------- collation
def collate_examples(examples: Sequence[Example], context_blocks: int) -> Dict[str, np.ndarray]:
    """Pack a list of examples into numpy arrays (see module docstring for the layout).

    Window slots are right-aligned: slot ``K-1`` always holds the current decision block.
    """
    K = int(context_blocks)
    B = len(examples)
    G = sum(len(ex.window) for ex in examples)
    req_x, req_ei, req_ea, req_gid, req_placed = [], [], [], [], []
    sub_x, sub_ei, sub_ea, sub_gid, sub_used = [], [], [], [], []
    scalars = np.zeros((G, NUM_SCALARS), dtype=np.float32)
    req_offset = np.zeros(G, dtype=np.int64)
    sub_offset = np.zeros(G, dtype=np.int64)
    req_num = np.zeros(G, dtype=np.int64)
    sub_num = np.zeros(G, dtype=np.int64)
    win_graph = np.full((B, K), -1, dtype=np.int64)
    win_valid = np.zeros((B, K), dtype=bool)
    win_rtg = np.zeros((B, K), dtype=np.float32)
    win_t = np.zeros((B, K), dtype=np.int64)
    win_v = np.zeros((B, K), dtype=np.int64)  # packed request-node index of v_w
    win_u = np.zeros((B, K), dtype=np.int64)  # packed substrate-node index of u_w
    max_nv = max(ex.window[-1].num_req_nodes for ex in examples)
    max_ns = max(ex.window[-1].num_sub_nodes for ex in examples)
    fv_mask = np.zeros((B, max_nv), dtype=bool)
    fs_mask = np.zeros((B, max_ns), dtype=bool)
    cur_req_nodes = np.zeros((B, max_nv), dtype=np.int64)
    cur_sub_nodes = np.zeros((B, max_ns), dtype=np.int64)
    cur_req_valid = np.zeros((B, max_nv), dtype=bool)
    cur_sub_valid = np.zeros((B, max_ns), dtype=bool)
    cur_graph = np.zeros(B, dtype=np.int64)
    target_v = np.zeros(B, dtype=np.int64)
    target_u = np.zeros(B, dtype=np.int64)
    request_size = np.zeros(B, dtype=np.int64)
    success = np.zeros(B, dtype=bool)
    g = 0
    n_req = 0
    n_sub = 0
    for b, ex in enumerate(examples):
        L = len(ex.window)
        for w, feats in enumerate(ex.window):
            slot = K - L + w
            nv, ns = feats.num_req_nodes, feats.num_sub_nodes
            req_x.append(feats.req_x)
            req_ei.append(feats.req_edge_index + n_req)
            req_ea.append(feats.req_edge_attr)
            req_gid.append(np.full(nv, g, dtype=np.int64))
            req_placed.append(feats.req_placed)
            sub_x.append(feats.sub_x)
            sub_ei.append(feats.sub_edge_index + n_sub)
            sub_ea.append(feats.sub_edge_attr)
            sub_gid.append(np.full(ns, g, dtype=np.int64))
            sub_used.append(feats.sub_used)
            scalars[g] = feats.scalars
            req_offset[g] = n_req
            sub_offset[g] = n_sub
            req_num[g] = nv
            sub_num[g] = ns
            win_graph[b, slot] = g
            win_valid[b, slot] = True
            win_rtg[b, slot] = ex.rtg[w]
            win_t[b, slot] = ex.timesteps[w]
            win_v[b, slot] = n_req + int(ex.actions[w, 0])
            win_u[b, slot] = n_sub + int(ex.actions[w, 1])
            if w == L - 1:
                cur_graph[b] = g
                fv_mask[b, :nv] = feats.fv_mask
                fs_mask[b, :ns] = ex.fs_mask
                cur_req_nodes[b, :nv] = n_req + np.arange(nv)
                cur_sub_nodes[b, :ns] = n_sub + np.arange(ns)
                cur_req_valid[b, :nv] = True
                cur_sub_valid[b, :ns] = True
                target_v[b] = int(ex.actions[w, 0])
                target_u[b] = int(ex.actions[w, 1])
            n_req += nv
            n_sub += ns
            g += 1
        request_size[b] = ex.request_size
        success[b] = ex.success
    batch = {
        "req_x": np.concatenate(req_x, axis=0).astype(np.float32),
        "req_edge_index": np.concatenate(req_ei, axis=1).astype(np.int64),
        "req_edge_attr": np.concatenate(req_ea, axis=0).astype(np.float32),
        "req_graph": np.concatenate(req_gid, axis=0),
        "req_placed": np.concatenate(req_placed, axis=0),
        "sub_x": np.concatenate(sub_x, axis=0).astype(np.float32),
        "sub_edge_index": np.concatenate(sub_ei, axis=1).astype(np.int64),
        "sub_edge_attr": np.concatenate(sub_ea, axis=0).astype(np.float32),
        "sub_graph": np.concatenate(sub_gid, axis=0),
        "sub_used": np.concatenate(sub_used, axis=0),
        "scalars": scalars,
        "num_graphs": np.asarray(G, dtype=np.int64),
        "req_offset": req_offset,
        "sub_offset": sub_offset,
        "req_num": req_num,
        "sub_num": sub_num,
        "win_graph": win_graph,
        "win_valid": win_valid,
        "win_rtg": win_rtg,
        "win_t": win_t,
        "win_v": win_v,
        "win_u": win_u,
        "cur_graph": cur_graph,
        "fv_mask": fv_mask,
        "fs_mask": fs_mask,
        "cur_req_nodes": cur_req_nodes,
        "cur_sub_nodes": cur_sub_nodes,
        "cur_req_valid": cur_req_valid,
        "cur_sub_valid": cur_sub_valid,
        "target_v": target_v,
        "target_u": target_u,
        "request_size": request_size,
        "success": success,
    }
    assert np.all(fv_mask[np.arange(B), target_v]), "target virtual node must be feasible"
    assert np.all(fs_mask[np.arange(B), target_u]), "target substrate node must be feasible"
    return batch
