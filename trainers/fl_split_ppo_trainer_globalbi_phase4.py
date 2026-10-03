"""Approved F/M/G observation-scope variant of the full-privacy trainer.

The public rollout interface continues to carry three local normalized
observations. Only the local BI input is expanded for M/G. This leaves actor
inputs and the full-privacy reconstruction targets local in every variant.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch import optim

from models.fl_split_critic import LocalBIEncoder
from trainers.fl_split_ppo_trainer_privacy_full import PrivacyFullSplitFedPPOTrainerV3


LOCAL_DIMS = (7, 10, 8)
OFFSETS = (0, 7, 17)
VARIANTS = ("F", "M", "G")
CONTRACT = {
    "variants": VARIANTS,
    "actor_obs_dims": LOCAL_DIMS,
    "bi_obs_dims": {"F": LOCAL_DIMS, "M": (25, 25, 25), "G": (25, 25, 25)},
    "privacy_target_dims": (8, 11, 9),
    "mask": "post-local-ObsRMS foreign dimensions exactly float32 zero",
    "sidecar": "local normalized observation plus local previous action",
    "initialization": "fresh F common-state copy; mapped self/action projection; foreign zero",
}
CONTRACT_SHA256 = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()


def bi_observations(local: Sequence[torch.Tensor], variant: str) -> List[torch.Tensor]:
    """One mapping for online value, bootstrap and replay BI encoding."""
    if variant not in VARIANTS or len(local) != 3:
        raise ValueError("Expected F/M/G and three local observations")
    if tuple(int(x.shape[-1]) for x in local) != LOCAL_DIMS:
        raise ValueError("BI source must be the ordered 7/10/8 local observation slices")
    if any(x.dtype != torch.float32 for x in local):
        raise TypeError("BI source observations must be float32")
    if variant == "F":
        return list(local)
    joint = torch.cat(local, dim=-1)
    if variant == "G":
        return [joint for _ in range(3)]
    masked = []
    for i, d in enumerate(LOCAL_DIMS):
        item = torch.zeros_like(joint, dtype=torch.float32)
        item[..., OFFSETS[i]:OFFSETS[i] + d] = local[i]
        masked.append(item)
    return masked


class Phase4FullPrivacyTrainer(PrivacyFullSplitFedPPOTrainerV3):
    variant: str = "F"

    def _server_values_from_local_bi(self, obs_t_list, prev_act_t_list, critic_hidden_list=None):
        return super()._server_values_from_local_bi(
            bi_observations(obs_t_list, self.variant), prev_act_t_list, critic_hidden_list
        )

    def _encode_bi_local(self, obs_list_seq, prev_act_list_seq, done_list_seq=None):
        return super()._encode_bi_local(
            bi_observations(obs_list_seq, self.variant), prev_act_list_seq, done_list_seq=done_list_seq
        )

    def optimizer_state_dicts(self) -> Dict[str, Any]:
        out = super().optimizer_state_dicts()
        out["priv_opts"] = [opt.state_dict() for opt in self.priv_opts]
        return out

    def load_optimizer_state_dicts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        states = payload.get("priv_opts")
        if not isinstance(states, list) or len(states) != len(self.priv_opts):
            raise ValueError("Missing or mismatched full-privacy sidecar optimizer states")
        out = super().load_optimizer_state_dicts(payload)
        for opt, state in zip(self.priv_opts, states):
            opt.load_state_dict(state)
        out["privacy_optimizers_loaded"] = len(states)
        return out

    def checkpoint_state(self) -> Dict[str, Any]:
        out = super().checkpoint_state()
        out.update(phase4_variant=self.variant, phase4_contract_sha256=CONTRACT_SHA256,
                   priv_defenses=[defense.state_dict() for defense in self.priv_defenses],
                   last_privacy_logs=dict(getattr(self, "last_privacy_logs", {})))
        return out

    def load_checkpoint_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("phase4_variant") != self.variant:
            raise ValueError("Wrong Phase 4 model variant in checkpoint")
        if payload.get("phase4_contract_sha256") != CONTRACT_SHA256:
            raise ValueError("Wrong Phase 4 source contract hash in checkpoint")
        defenses = payload.get("priv_defenses")
        if not isinstance(defenses, list) or len(defenses) != len(self.priv_defenses):
            raise ValueError("Missing or mismatched privacy sidecars")
        out = super().load_checkpoint_state(payload)
        for module, state in zip(self.priv_defenses, defenses):
            module.load_state_dict(state, strict=True)
        self.last_privacy_logs = dict(payload.get("last_privacy_logs", {}))
        out["privacy_sidecars_loaded"] = len(defenses)
        return out


def build_variant(exp_cfg, variant: str, device: str = "cpu", writer=None) -> Phase4FullPrivacyTrainer:
    """Build a fresh F-compatible trainer, then widen only M/G local BI encoders."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    from scripts.experiment_fl_fedavg10_privacy_full import build_hparams
    from scripts.train_fl_split_shallow_hybrid import AGENT_CFGS
    cfg = exp_cfg.model_cfg
    trainer = Phase4FullPrivacyTrainer.build_default(
        obs_dims=LOCAL_DIMS, act_dims=tuple(len(a.mv_indices) for a in AGENT_CFGS),
        device=device, hp=build_hparams(exp_cfg),
        actor_local_out=cfg.actor_local_out, actor_shared_out=cfg.actor_shared_out,
        critic_local_pre_out=cfg.critic_local_pre_out, critic_bi_hidden=cfg.critic_bi_hidden,
        critic_gru_layers=cfg.critic_gru_layers, critic_gat_hidden=cfg.critic_gat_hidden,
        critic_readout_dim=cfg.critic_readout_dim,
        actor_local_hidden_dims=cfg.actor_local_hidden_dims,
        actor_shared_hidden_dims=cfg.actor_shared_hidden_dims,
        actor_action_hidden_dims=cfg.actor_action_hidden_dims,
        critic_local_pre_hidden_dims=cfg.critic_local_pre_hidden_dims,
        critic_readout_hidden_dims=cfg.critic_readout_hidden_dims,
        critic_value_hidden_dims=cfg.critic_value_hidden_dims,
        activate_feature_outputs=cfg.activate_feature_outputs,
        ablate_enhancer=cfg.ablate_enhancer, bi_extractor=cfg.bi_extractor,
        bi_post_gru_mlp=cfg.bi_post_gru_mlp, aggregation=cfg.aggregation,
        action_pathway=cfg.action_pathway, fuse=cfg.fuse,
    )
    trainer.variant = variant
    trainer.writer = writer
    if variant != "F":
        for client in trainer.clients:
            old = client.bi_encoder
            client.bi_encoder = LocalBIEncoder(
                obs_dim=25, act_dim=old.act_dim,
                local_pre_hidden_dims=cfg.critic_local_pre_hidden_dims,
                local_pre_out_dim=cfg.critic_local_pre_out,
                bi_hidden_dim=cfg.critic_bi_hidden,
                gru_num_layers=cfg.critic_gru_layers,
                activate_feature_outputs=cfg.activate_feature_outputs,
                bi_extractor=cfg.bi_extractor, bi_post_gru_mlp=cfg.bi_post_gru_mlp,
                action_pathway=cfg.action_pathway, fuse=cfg.fuse,
            ).to(trainer.device)
        # The old optimizer owned the discarded local encoders. Build fresh
        # optimizers against the final M/G parameters; baseline F is untouched.
        trainer.local_critic_opts = [
            optim.Adam(client.bi_encoder.parameters(), lr=trainer.hp.critic_lr)
            for client in trainer.clients
        ]
    return trainer


def copy_f_initial_state(f: Phase4FullPrivacyTrainer, target: Phase4FullPrivacyTrainer) -> None:
    """Copy every same-semantic state tensor and map the expanded input layer."""
    if f.variant != "F" or target.variant not in ("M", "G"):
        raise ValueError("Initialization requires fresh F source and M/G destination")
    if f.round_idx or target.round_idx:
        raise ValueError("Initialization must precede any training update")
    for src, dst in zip(f.clients, target.clients):
        dst.actor.load_state_dict(src.actor.state_dict(), strict=True)
    target.server.load_state_dict(f.server.state_dict(), strict=True)
    for src, dst in zip(f.priv_defenses, target.priv_defenses):
        dst.load_state_dict(src.state_dict(), strict=True)
    for i, (src, dst) in enumerate(zip(f.clients, target.clients)):
        old = src.bi_encoder.state_dict()
        new = dst.bi_encoder.state_dict()
        special = "obs_action_processor.0.weight"
        for key, value in old.items():
            if key == special:
                continue
            if key not in new or value.shape != new[key].shape:
                raise ValueError(f"Unexpected nonmatching BI state: agent={i} key={key}")
            new[key] = value.detach().clone()
        mapped = torch.zeros_like(new[special])
        d = LOCAL_DIMS[i]
        mapped[:, OFFSETS[i]:OFFSETS[i] + d] = old[special][:, :d]
        mapped[:, 25] = old[special][:, d]
        new[special] = mapped
        dst.bi_encoder.load_state_dict(new, strict=True)


def build_matched(exp_cfg, seed: int, device: str = "cpu") -> Dict[str, Phase4FullPrivacyTrainer]:
    """Create all arms without consuming the caller's torch RNG stream."""
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(int(seed))
        f = build_variant(exp_cfg, "F", device)
        m = build_variant(exp_cfg, "M", device)
        g = build_variant(exp_cfg, "G", device)
        copy_f_initial_state(f, m)
        copy_f_initial_state(f, g)
    return {"F": f, "M": m, "G": g}
