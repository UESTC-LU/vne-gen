"""The VNE-Gen model (Sec. IV-B).

Per decision ``k`` the model

1. encodes the request graph and the residual substrate state with two separate GNNs (node embeddings
   become the pointer keys, pooled embeddings and partial-mapping statistics form the state token);
2. assembles the structured token block ``[R_k, S_k, (A_k)]`` and the preceding ``K-1`` completed blocks;
3. runs the shared causal Transformer twice - first with the stage-V adapter to obtain :math:`h^V_k` at the
   state token, selects the virtual node ``v_k`` with the virtual-node pointer, then with the stage-S
   adapter after appending a half-action token for ``v_k`` to obtain :math:`h^S_k` and selects the
   substrate node ``u_k`` with the substrate-node pointer;
4. the pointer heads score the nodes of the current graphs directly through
   :math:`\\pi^z_\\theta(i\\mid h^z_k,R_k)\\propto\\exp(\\psi_z(h^z_k,R_k)^\\top\\phi_z(i)/\\sqrt{D_{ptr}})` restricted to
   the feasible set :math:`\\mathcal F_{z,k}`, so the output spaces adapt to :math:`|V_r|` and :math:`|V_S|`.

The same module serves training (batched teacher forcing over corpus transitions) and step-wise inference
(:class:`~vne_gen.generator.generator.VNEGenGenerator`), sharing the token assembly and pointer code paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.features import NUM_EDGE_FEATURES, NUM_REQ_FEATURES, NUM_SCALARS, NUM_SUB_FEATURES, StateFeatures
from .backbone import CausalBackbone
from .config import ModelConfig
from .gnn import GraphEncoder, masked_mean, scatter_max, scatter_mean

NEG = -1.0e9  # mask value for infeasible candidates

KIND_RETURN, KIND_STATE, KIND_ACTION, KIND_HALF = 0, 1, 2, 3


def batch_to_tensors(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        t = torch.as_tensor(np.asarray(v))
        if t.dtype == torch.float64:
            t = t.float()
        out[k] = t.to(device)
    return out


@dataclass
class EncodedStates:
    h_req: torch.Tensor  # (N_req_total, H)
    h_sub: torch.Tensor  # (N_sub_total, H)
    state_vec: torch.Tensor  # (G, state_in)
    state_tok: torch.Tensor  # (G, d)


class VNEGenModel(nn.Module):
    def __init__(self, cfg: ModelConfig, backbone_config: Optional[dict] = None) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.gnn_hidden
        self.req_encoder = GraphEncoder(NUM_REQ_FEATURES, NUM_EDGE_FEATURES, H, cfg.gnn_layers, cfg.dropout, cfg.message_passing)
        self.sub_encoder = GraphEncoder(NUM_SUB_FEATURES, NUM_EDGE_FEATURES, H, cfg.gnn_layers, cfg.dropout, cfg.message_passing)
        self.backbone = CausalBackbone(cfg, backbone_config)
        d = self.backbone.d_model
        self.d_model = d
        self.K = cfg.context_blocks
        state_in = 4 * H + 2 * H + NUM_SCALARS
        self.state_in = state_in
        # structured tokens
        self.return_proj = nn.Linear(1, d)
        self.null_return = nn.Parameter(torch.zeros(d))  # used when return conditioning is disabled
        self.state_proj = nn.Sequential(nn.Linear(state_in, d), nn.GELU(), nn.Linear(d, d))
        self.action_proj = nn.Linear(2 * H, d)
        self.half_action_proj = nn.Linear(H, d)
        self.kind_emb = nn.Embedding(4, d)
        self.time_emb = nn.Embedding(cfg.max_timesteps, d)
        self.token_norm = nn.LayerNorm(d)
        self.token_drop = nn.Dropout(cfg.dropout)
        # pointer heads (stage-specific query and key projections)
        P = cfg.pointer_dim
        rf = cfg.return_feat_dim
        self.return_feat = nn.Linear(1, rf)
        self.null_return_feat = nn.Parameter(torch.zeros(rf))
        self.query_V = nn.Sequential(nn.Linear(d + rf, P), nn.GELU(), nn.Linear(P, P))
        self.query_S = nn.Sequential(nn.Linear(d + rf, P), nn.GELU(), nn.Linear(P, P))
        self.key_V = nn.Linear(H, P)
        self.key_S = nn.Linear(H, P)
        # MLP (behaviour-cloning) path used when cfg.backbone == "mlp"
        if cfg.backbone == "mlp":
            self.mlp_V = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d))
            self.mlp_S = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d))
        # uncertainty weights of the two stage losses [30]
        self.log_sigma_V = nn.Parameter(torch.zeros(()))
        self.log_sigma_S = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------ parameter groups
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_counts(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    # ------------------------------------------------------------------ encoding
    def encode(self, batch: Dict[str, torch.Tensor]) -> EncodedStates:
        G = int(batch["num_graphs"])
        h_req = self.req_encoder(batch["req_x"], batch["req_edge_index"], batch["req_edge_attr"])
        h_sub = self.sub_encoder(batch["sub_x"], batch["sub_edge_index"], batch["sub_edge_attr"])
        rg, sg = batch["req_graph"], batch["sub_graph"]
        pool_req = torch.cat([scatter_mean(h_req, rg, G), scatter_max(h_req, rg, G)], dim=-1)
        pool_sub = torch.cat([scatter_mean(h_sub, sg, G), scatter_max(h_sub, sg, G)], dim=-1)
        pm_req = masked_mean(h_req, rg, batch["req_placed"].bool(), G)
        pm_sub = masked_mean(h_sub, sg, batch["sub_used"].bool(), G)
        state_vec = torch.cat([pool_req, pool_sub, pm_req, pm_sub, batch["scalars"]], dim=-1)
        state_tok = self.state_proj(state_vec)
        return EncodedStates(h_req, h_sub, state_vec, state_tok)

    # ------------------------------------------------------------------ tokens
    def return_token(self, rtg: torch.Tensor) -> torch.Tensor:
        """``rtg``: (...,) -> (..., d)."""
        if self.cfg.return_conditioning:
            return self.return_proj(rtg.unsqueeze(-1))
        return self.null_return.expand(*rtg.shape, self.d_model)

    def return_feature(self, rtg: torch.Tensor) -> torch.Tensor:
        if self.cfg.return_conditioning:
            return self.return_feat(rtg.unsqueeze(-1))
        return self.null_return_feat.expand(*rtg.shape, self.cfg.return_feat_dim)

    def assemble_tokens(
        self,
        rtg: torch.Tensor,  # (B, K)
        state_tok: torch.Tensor,  # (B, K, d)
        action_tok: torch.Tensor,  # (B, K, d) (ignored at the current slot K-1)
        valid: torch.Tensor,  # (B, K) bool
        timesteps: torch.Tensor,  # (B, K) long
        half_tok: Optional[torch.Tensor],  # (B, d) or None (stage V)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interleave ``[R, S, A]`` blocks into a (B, 3K, d) sequence with a (B, 3K) validity mask."""
        B, K = rtg.shape
        d = self.d_model
        dev = rtg.device
        t_emb = self.time_emb(timesteps.clamp(max=self.cfg.max_timesteps - 1))  # (B, K, d)
        r_tok = self.return_token(rtg) + self.kind_emb.weight[KIND_RETURN] + t_emb
        s_tok = state_tok + self.kind_emb.weight[KIND_STATE] + t_emb
        a_tok = action_tok + self.kind_emb.weight[KIND_ACTION] + t_emb
        tokens = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(B, 3 * K, d)
        mask = valid.unsqueeze(2).expand(B, K, 3).clone()
        mask[:, K - 1, 2] = False  # the current block has no completed action token
        if half_tok is not None:
            h = half_tok + self.kind_emb.weight[KIND_HALF] + t_emb[:, K - 1]
            tokens = tokens.clone()
            tokens[:, 3 * K - 1] = h
            mask[:, K - 1, 2] = True
        mask = mask.reshape(B, 3 * K)
        tokens = self.token_drop(self.token_norm(tokens))
        return tokens, mask

    # ------------------------------------------------------------------ pointer heads
    def pointer_logits(self, stage: str, h: torch.Tensor, rtg: torch.Tensor, keys: torch.Tensor,
                       feasible: torch.Tensor) -> torch.Tensor:
        """``h``: (B, d); ``rtg``: (B,); ``keys``: (B, M, H) node embeddings; ``feasible``: (B, M) bool."""
        q_net, k_net = (self.query_V, self.key_V) if stage == "V" else (self.query_S, self.key_S)
        q = q_net(torch.cat([h, self.return_feature(rtg)], dim=-1))  # (B, P)
        k = k_net(keys)  # (B, M, P)
        logits = torch.einsum("bmp,bp->bm", k, q) / math.sqrt(self.cfg.pointer_dim)
        return logits.masked_fill(~feasible, NEG)

    # ------------------------------------------------------------------ training forward
    def forward(self, batch: Dict[str, torch.Tensor], target_v: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Teacher-forced forward pass over a collated batch; returns stage-V and stage-S logits.

        ``target_v`` (B,) local index of the virtual node used to condition the substrate stage (defaults to
        the recorded ``batch['target_v']``).
        """
        enc = self.encode(batch)
        B, K = batch["win_rtg"].shape
        win_graph = batch["win_graph"].clamp(min=0)
        state_tok = enc.state_tok[win_graph]  # (B, K, d)
        a_in = torch.cat([enc.h_req[batch["win_v"]], enc.h_sub[batch["win_u"]]], dim=-1)  # (B, K, 2H)
        action_tok = self.action_proj(a_in)
        rtg = batch["win_rtg"]
        valid = batch["win_valid"].bool()
        timesteps = batch["win_t"]
        cur_rtg = rtg[:, K - 1]
        # candidate keys of the current graphs
        req_keys = enc.h_req[batch["cur_req_nodes"]]  # (B, max_nv, H)
        sub_keys = enc.h_sub[batch["cur_sub_nodes"]]  # (B, max_ns, H)
        fv = batch["fv_mask"].bool() & batch["cur_req_valid"].bool()
        fs = batch["fs_mask"].bool() & batch["cur_sub_valid"].bool()
        if target_v is None:
            target_v = batch["target_v"]
        v_packed = batch["cur_req_nodes"][torch.arange(B, device=target_v.device), target_v]
        half_in = enc.h_req[v_packed]  # (B, H)
        if self.cfg.backbone == "mlp":
            cur_state = enc.state_tok[batch["cur_graph"]]
            h_V = self.mlp_V(cur_state)
            h_S = self.mlp_S(torch.cat([cur_state, self.half_action_proj(half_in)], dim=-1))
        else:
            tokens, mask = self.assemble_tokens(rtg, state_tok, action_tok, valid, timesteps, None)
            out = self.backbone(tokens, mask, "V")
            h_V = out[:, 3 * K - 2]  # hidden state at the current state token
            half_tok = self.half_action_proj(half_in)
            tokens_s, mask_s = self.assemble_tokens(rtg, state_tok, action_tok, valid, timesteps, half_tok)
            out_s = self.backbone(tokens_s, mask_s, "S")
            h_S = out_s[:, 3 * K - 1]  # hidden state at the half-action token
        logits_V = self.pointer_logits("V", h_V, cur_rtg, req_keys, fv)
        logits_S = self.pointer_logits("S", h_S, cur_rtg, sub_keys, fs)
        return {"logits_V": logits_V, "logits_S": logits_S, "h_V": h_V, "h_S": h_S}

    def losses(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
               per_trajectory: bool = False) -> Dict[str, torch.Tensor]:
        """Negative log-likelihoods of the two stages and the uncertainty-weighted total (Sec. IV-C)."""
        nll_V = F.cross_entropy(out["logits_V"], batch["target_v"], reduction="none")
        nll_S = F.cross_entropy(out["logits_S"], batch["target_u"], reduction="none")
        if per_trajectory and "num_steps" in batch:
            w = 1.0 / batch["num_steps"].float().clamp(min=1.0)
            w = w / w.sum() * w.numel()
            nll_V, nll_S = nll_V * w, nll_S * w
        L_V, L_S = nll_V.mean(), nll_S.mean()
        total = (L_V / (2 * torch.exp(2 * self.log_sigma_V)) + L_S / (2 * torch.exp(2 * self.log_sigma_S))
                 + self.log_sigma_V + self.log_sigma_S)
        with torch.no_grad():
            acc_V = (out["logits_V"].argmax(-1) == batch["target_v"]).float().mean()
            acc_S = (out["logits_S"].argmax(-1) == batch["target_u"]).float().mean()
        return {"loss": total, "L_V": L_V, "L_S": L_S, "acc_V": acc_V, "acc_S": acc_S}

    # ------------------------------------------------------------------ step-wise inference API
    @torch.no_grad()
    def encode_single(self, feats: StateFeatures, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one state; returns ``(h_req (nv,H), h_sub (ns,H), state_tok (d,))``."""
        batch = {
            "num_graphs": torch.tensor(1),
            "req_x": torch.as_tensor(feats.req_x, device=device),
            "req_edge_index": torch.as_tensor(feats.req_edge_index, device=device),
            "req_edge_attr": torch.as_tensor(feats.req_edge_attr, device=device),
            "req_graph": torch.zeros(feats.num_req_nodes, dtype=torch.long, device=device),
            "req_placed": torch.as_tensor(feats.req_placed, device=device),
            "sub_x": torch.as_tensor(feats.sub_x, device=device),
            "sub_edge_index": torch.as_tensor(feats.sub_edge_index, device=device),
            "sub_edge_attr": torch.as_tensor(feats.sub_edge_attr, device=device),
            "sub_graph": torch.zeros(feats.num_sub_nodes, dtype=torch.long, device=device),
            "sub_used": torch.as_tensor(feats.sub_used, device=device),
            "scalars": torch.as_tensor(feats.scalars, device=device).unsqueeze(0),
        }
        enc = self.encode(batch)
        return enc.h_req, enc.h_sub, enc.state_tok[0]

    @torch.no_grad()
    def decode_step(
        self,
        stage: str,
        rtg_hist: List[float],
        state_hist: List[torch.Tensor],
        action_hist: List[torch.Tensor],
        t_hist: List[int],
        half_tok: Optional[torch.Tensor],
        keys: torch.Tensor,
        feasible: torch.Tensor,
    ) -> torch.Tensor:
        """Pointer logits for one decision given the cached window (last entry = current block).

        ``state_hist``/``action_hist`` hold the cached state and action tokens of the window (the action token of
        the current block is ignored and may be a zero vector); ``keys``: (M, H); ``feasible``: (M,) bool.
        """
        dev = keys.device
        L = len(state_hist)
        K = self.K
        d = self.d_model
        rtg = torch.zeros(1, K, device=dev)
        st = torch.zeros(1, K, d, device=dev)
        at = torch.zeros(1, K, d, device=dev)
        valid = torch.zeros(1, K, dtype=torch.bool, device=dev)
        ts = torch.zeros(1, K, dtype=torch.long, device=dev)
        for w in range(L):
            slot = K - L + w
            rtg[0, slot] = float(rtg_hist[w])
            st[0, slot] = state_hist[w]
            at[0, slot] = action_hist[w]
            valid[0, slot] = True
            ts[0, slot] = int(t_hist[w])
        cur_rtg = rtg[:, K - 1]
        if self.cfg.backbone == "mlp":
            cur_state = st[:, K - 1]
            if stage == "V":
                h = self.mlp_V(cur_state)
            else:
                h = self.mlp_S(torch.cat([cur_state, half_tok.unsqueeze(0)], dim=-1))
        else:
            tokens, mask = self.assemble_tokens(rtg, st, at, valid, ts, None if stage == "V" else half_tok.unsqueeze(0))
            out = self.backbone(tokens, mask, stage)
            h = out[:, 3 * K - 2] if stage == "V" else out[:, 3 * K - 1]
        logits = self.pointer_logits(stage, h, cur_rtg, keys.unsqueeze(0), feasible.unsqueeze(0))
        return logits[0]
