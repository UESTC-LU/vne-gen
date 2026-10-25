"""Request-level mapping with VNE-Gen (Algorithm 1).

Given a request, a supplied capacity profile and an initial return target :math:`R_0`, the generator
alternates learned decisions and deterministic routing:

1. encode the current state and the causal context;
2. construct :math:`\\mathcal F_V(s_k)`; if it is empty, roll back and return failure;
3. select :math:`v_k` from :math:`\\mathcal F_V(s_k)` with the virtual-node pointer;
4. condition the substrate stage on :math:`v_k`, construct :math:`\\mathcal F_S(s_k,v_k)` and select :math:`u_k`;
5. place :math:`(v_k,u_k)`, route the newly activated links (Eq. 5) and reserve bandwidth; on failure roll back
   and return failure;
6. update the partial mapping, the residual profile, the return-to-go :math:`R_{k+1}=R_k-g_k` and the context.

Online VNE uses greedy decoding with :math:`R_0^{on}=R^\\star(|V_r|)` (Eq. 8); offline and time-aware offline
VNE use stochastic decoding with :math:`R_0^{off}(\\rho)=\\rho R^\\star(|V_r|)` (Eq. 9).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.features import FeatureConfig, encode_state
from ..env.mapping_env import MappingAttempt
from ..model.config import ModelConfig
from ..model.vne_gen import VNEGenModel
from ..problem.mapping import Mapping
from ..problem.request import Request
from ..problem.substrate import CapacityProfile, Substrate

logger = logging.getLogger(__name__)

ORDERINGS = ("learned", "ascending_id", "degree_desc", "fixed_random")


class VNEGenGenerator:
    """Frozen VNE-Gen generator exposing the online ``embed`` and offline ``propose`` interfaces."""

    name = "VNE-Gen"

    def __init__(
        self,
        model: VNEGenModel,
        return_table: Dict[int, float],
        device: Optional[torch.device] = None,
        feature_cfg: Optional[FeatureConfig] = None,
        ordering: str = "learned",
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.return_table = {int(k): float(v) for k, v in return_table.items()}
        cfg = model.cfg
        self.feature_cfg = feature_cfg or FeatureConfig(cap_scale=cfg.cap_scale, demand_scale=cfg.demand_scale,
                                                        max_request_nodes=cfg.max_request_nodes)
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}")
        self.ordering = ordering
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.K = cfg.context_blocks
        self.last_attempt: Optional[MappingAttempt] = None

    # ------------------------------------------------------------------ return targets
    def return_target(self, num_nodes: int, rho: float = 1.0) -> float:
        if num_nodes in self.return_table:
            return rho * self.return_table[num_nodes]
        # unseen request size: fall back to the nearest tabulated size
        sizes = sorted(self.return_table)
        nearest = min(sizes, key=lambda n: abs(n - num_nodes))
        return rho * self.return_table[nearest]

    # ------------------------------------------------------------------ decoding helpers
    def _select(self, logits: torch.Tensor, feasible: np.ndarray, stochastic: bool) -> int:
        logits = logits.detach().float().cpu().numpy()
        logits = np.where(feasible, logits, -np.inf)
        if not stochastic:
            return int(np.argmax(logits))
        z = logits / max(self.temperature, 1e-6)
        z = z - z[feasible].max()
        p = np.exp(z)
        p[~feasible] = 0.0
        p = p / p.sum()
        return int(self.rng.choice(len(p), p=p))

    def _fixed_order_choice(self, request: Request, feasible: np.ndarray, fixed_order: List[int]) -> int:
        for v in fixed_order:
            if feasible[v]:
                return int(v)
        raise RuntimeError("no feasible virtual node in the fixed order")  # pragma: no cover

    def _fixed_order(self, request: Request) -> List[int]:
        n = request.num_nodes
        if self.ordering == "ascending_id":
            return list(range(n))
        if self.ordering == "degree_desc":
            deg = np.array([request.degree(i) for i in range(n)])
            return [int(i) for i in np.lexsort((np.arange(n), -deg))]
        if self.ordering == "fixed_random":
            return [int(i) for i in self.rng.permutation(n)]
        return []

    # ------------------------------------------------------------------ Algorithm 1
    @torch.no_grad()
    def generate(
        self,
        request: Request,
        substrate: Substrate,
        profile: CapacityProfile,
        R0: Optional[float] = None,
        stochastic: bool = False,
    ) -> Tuple[Optional[Mapping], MappingAttempt]:
        """Construct a complete validated mapping or fail; returns ``(mapping or None, attempt)``."""
        model = self.model
        dev = self.device
        attempt = MappingAttempt(substrate, request, profile)
        R = float(R0 if R0 is not None else self.return_target(request.num_nodes))
        fixed_order = self._fixed_order(request) if self.ordering != "learned" else None
        # cached window of structured tokens (return, state token, action token, timestep)
        rtg_hist: List[float] = []
        state_hist: List[torch.Tensor] = []
        action_hist: List[torch.Tensor] = []
        t_hist: List[int] = []
        unit = 1.0 / request.num_nodes
        for k in range(request.num_nodes):
            feats = encode_state(attempt, self.feature_cfg)
            fv = feats.fv_mask
            if not fv.any():
                attempt.terminate_empty()
                self.last_attempt = attempt
                return None, attempt
            h_req, h_sub, state_tok = model.encode_single(feats, dev)
            rtg_hist.append(R)
            state_hist.append(state_tok)
            action_hist.append(torch.zeros_like(state_tok))
            t_hist.append(k)
            if len(rtg_hist) > self.K:  # keep the current block and the K-1 preceding completed blocks
                rtg_hist.pop(0)
                state_hist.pop(0)
                action_hist.pop(0)
                t_hist.pop(0)
            fv_t = torch.as_tensor(fv, device=dev)
            if fixed_order is not None:
                v = self._fixed_order_choice(request, fv, fixed_order)
            else:
                logits_v = model.decode_step("V", rtg_hist, state_hist, action_hist, t_hist, None, h_req, fv_t)
                v = self._select(logits_v, fv, stochastic)
            fs = attempt.feasible_substrate_mask(v)
            if not fs.any():  # cannot happen when v in F_V (the residual state is unchanged between stages)
                attempt.terminate_empty()
                self.last_attempt = attempt
                return None, attempt
            half_tok = model.half_action_proj(h_req[v])
            logits_u = model.decode_step("S", rtg_hist, state_hist, action_hist, t_hist, half_tok, h_sub,
                                         torch.as_tensor(fs, device=dev))
            u = self._select(logits_u, fs, stochastic)
            # complete the current block's action token before the next decision
            action_hist[-1] = model.action_proj(torch.cat([h_req[v], h_sub[u]], dim=-1))
            reward, done = attempt.step(v, u)
            if done:
                break
            R = R - unit  # R_{k+1} = R_k - g_k with g_k = 1/|V_r| for a feasible non-terminal decision
        self.last_attempt = attempt
        if attempt.success:
            return attempt.mapping(), attempt
        return None, attempt

    # ------------------------------------------------------------------ selector interfaces
    def embed(self, request: Request, substrate: Substrate, profile: CapacityProfile) -> Optional[Mapping]:
        """Online VNE: one greedy call with the size-conditioned return target (Eq. 8)."""
        mapping, _ = self.generate(request, substrate, profile, R0=self.return_target(request.num_nodes), stochastic=False)
        return mapping

    def propose(self, request: Request, substrate: Substrate, profile: CapacityProfile, rho: float = 1.0,
                rng: Optional[np.random.Generator] = None) -> Optional[Mapping]:
        """Offline / time-aware VNE: one stochastic call with :math:`R_0^{off}(\\rho)` (Eq. 9)."""
        if rng is not None:
            self.rng = rng
        mapping, _ = self.generate(request, substrate, profile, R0=self.return_target(request.num_nodes, rho), stochastic=True)
        return mapping


def load_generator(checkpoint_path: str, device: Optional[str] = None, ordering: str = "learned",
                   temperature: float = 1.0, seed: int = 0) -> VNEGenGenerator:
    """Load a trained checkpoint (model weights, config and the :math:`R^\\star(n)` table) as a frozen generator."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    backbone_config = ckpt.get("backbone_config")
    if backbone_config is not None:
        cfg.pretrained = None  # all weights come from the checkpoint; avoid downloading the language model
    model = VNEGenModel(cfg, backbone_config=backbone_config)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    feature_cfg = FeatureConfig(**ckpt["feature_config"]) if "feature_config" in ckpt else None
    return VNEGenGenerator(model, ckpt["return_table"], dev, feature_cfg, ordering=ordering,
                           temperature=temperature, seed=seed)
