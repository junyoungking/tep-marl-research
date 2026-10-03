"""Split-FL PPO trainer with client-side DRAVL BI privacy defense.

DRAVL builds on the canonical ``SplitFedPPOTrainer`` fl_v2 mechanics.  Privacy
losses are attached to the local BI graph before the detach handoff, so
privacy gradients can update local BI encoders and reconstructor defenses but
cannot reach the server-side BI critic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.bi_privacy_defense import ClientBIPrivacyDefense
from trainers.fl_split_ppo_trainer import SplitFedPPOHyperParams
from trainers.fl_split_ppo_trainer_continue_rollout import SplitFedContinueRolloutPPOTrainer


@dataclass
class DRAVLSplitFedPPOHyperParams(SplitFedPPOHyperParams):
    privacy_enabled: bool = True
    priv_ar_coef: float = 0.01
    priv_nr_coef: float = 0.001
    priv_dcor_coef: float = 0.01
    grl_lambda: float = 1.0
    priv_noise_std: float = 1.0
    priv_lr: float = 1e-4
    priv_dcor_max_samples: int = 2048
    priv_recon_hidden_dims: Tuple[int, ...] = (128, 128)


class DRAVLSplitFedPPOTrainer(SplitFedContinueRolloutPPOTrainer):
    """Continue-rollout split-FL trainer with client-owned DRAVL defenses."""

    REQUIRED_PRIVACY_LOG_KEYS: Tuple[str, ...] = (
        "loss/priv_ar_raw",
        "loss/priv_nr_raw",
        "privacy/dcor_raw",
        "privacy/dcor_log",
        "loss/priv_nr_weighted",
        "loss/priv_dcor_weighted",
        "privacy/ar_lambda_eff",
        "loss/privacy_backward_total",
        "privacy/recon_mse",
    )

    def __init__(
        self,
        clients: Sequence[nn.Module],
        server: nn.Module,
        device: str = "cpu",
        hp: Optional[DRAVLSplitFedPPOHyperParams] = None,
        writer=None,
    ):
        resolved_hp = hp if hp is not None else DRAVLSplitFedPPOHyperParams()
        super().__init__(clients=clients, server=server, device=device, hp=resolved_hp, writer=writer)
        if not isinstance(self.hp, DRAVLSplitFedPPOHyperParams):
            # Allows build_default callers to pass a base dataclass-like object while still exposing fields.
            self.hp = DRAVLSplitFedPPOHyperParams(**self.hp.__dict__)
        self.priv_defenses = nn.ModuleList()
        self.priv_opts: List[optim.Optimizer] = []
        if self.hp.privacy_enabled:
            for client in self.clients:
                bi_encoder = client.bi_encoder
                bottleneck = getattr(bi_encoder, "bottleneck", None)
                privacy_bi_dim = int(getattr(bottleneck, "transmitted_dim", bi_encoder.bi_hidden_dim))
                defense = ClientBIPrivacyDefense(
                    bi_dim=privacy_bi_dim,
                    target_dim=int(bi_encoder.obs_dim) + int(bi_encoder.act_dim),
                    recon_hidden_dims=tuple(self.hp.priv_recon_hidden_dims),
                ).to(self.device)
                self.priv_defenses.append(defense)
                self.priv_opts.append(optim.Adam(defense.parameters(), lr=float(self.hp.priv_lr)))
        self.last_privacy_logs: Dict[str, float] = {}

    def optimizer_parameter_sets(self) -> Dict[str, set]:
        out = super().optimizer_parameter_sets()
        out["privacy"] = set(id(p) for defense in self.priv_defenses for p in defense.parameters())
        return out

    def save(self, save_dir: str):
        super().save(save_dir)
        if self.hp.privacy_enabled:
            for i, defense in enumerate(self.priv_defenses):
                torch.save(defense.state_dict(), os.path.join(save_dir, f"fl_bi_privacy_defense_{i}.pt"))

    @staticmethod
    def _as_float(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _grad_norm(parameters) -> float:
        total = 0.0
        for p in parameters:
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().cpu().item())
        return float(total ** 0.5)

    @staticmethod
    def _mean_log_dict(logs: List[Dict[str, float]]) -> Dict[str, float]:
        if not logs:
            return {}
        keys = sorted({key for log in logs for key in log})
        out: Dict[str, float] = {}
        for key in keys:
            vals = [float(log[key]) for log in logs if key in log and np.isfinite(float(log[key]))]
            if vals:
                out[key] = float(np.mean(vals))
        return out

    def _compute_privacy_terms(self, bi_local_list: List[torch.Tensor], obs_list_seq: List[torch.Tensor], prev_act_list_seq: List[torch.Tensor]):
        terms: List[torch.Tensor] = []
        logs: Dict[str, float] = {}
        if not self.hp.privacy_enabled:
            return terms, None, logs
        if len(self.priv_defenses) != self.num_agents:
            raise RuntimeError(f"Expected {self.num_agents} privacy defenses, got {len(self.priv_defenses)}")
        per_agent_total = []
        for i, defense in enumerate(self.priv_defenses):
            target_i = torch.cat([obs_list_seq[i], prev_act_list_seq[i]], dim=-1)
            bi_flat = bi_local_list[i].reshape(-1, bi_local_list[i].shape[-1])
            target_flat = target_i.reshape(-1, target_i.shape[-1])
            out = defense.compute(
                bi_flat,
                target_flat,
                ar_coef=self.hp.priv_ar_coef,
                nr_coef=self.hp.priv_nr_coef,
                dcor_coef=self.hp.priv_dcor_coef,
                grl_lambda=self.hp.grl_lambda,
                noise_std=self.hp.priv_noise_std,
                dcor_max_samples=self.hp.priv_dcor_max_samples,
            )
            terms.append(out["backward_loss"])
            per_agent_total.append(out["loss/privacy_backward_total"])
            for key in self.REQUIRED_PRIVACY_LOG_KEYS:
                if key in out:
                    logs[f"agent_{i}/{key}"] = self._as_float(out[key])
            # Short aliases match the spec's "agent_{i}/priv_ar_raw 등" wording
            # while the fully-qualified keys preserve the global scalar names.
            logs[f"agent_{i}/priv_ar_raw"] = self._as_float(out["loss/priv_ar_raw"])
            logs[f"agent_{i}/priv_nr_raw"] = self._as_float(out["loss/priv_nr_raw"])
            logs[f"agent_{i}/dcor_raw"] = self._as_float(out["privacy/dcor_raw"])
            logs[f"agent_{i}/dcor_log"] = self._as_float(out["privacy/dcor_log"])
        if terms:
            priv_scalar = torch.stack(terms).sum()
            for key in self.REQUIRED_PRIVACY_LOG_KEYS:
                vals = []
                for i in range(self.num_agents):
                    agent_key = f"agent_{i}/{key}"
                    if agent_key in logs:
                        vals.append(float(logs[agent_key]))
                if vals:
                    # For the true backward total, report the sum to match the scalar; for components use mean.
                    logs[key] = float(np.sum(vals) if key == "loss/privacy_backward_total" else np.mean(vals))
            return terms, priv_scalar, logs
        return terms, None, logs

    def critic_update_from_tensors(
        self,
        obs_list_seq: List[torch.Tensor],
        prev_act_list_seq: List[torch.Tensor],
        ret_list_seq: List[torch.Tensor],
        apply_local_handoff: bool = True,
        done_list_seq: Optional[List[torch.Tensor]] = None,
        done_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        self.debug_events = []
        # Keep actor purity explicit by clearing any stale actor gradients before critic/privacy phase.
        for opt in self.actor_opts:
            opt.zero_grad(set_to_none=True)

        batch_size, seq_len = obs_list_seq[0].shape[0], obs_list_seq[0].shape[1]
        done_list_seq = self._coerce_done_list_seq(done_list_seq, done_seq, batch_size=batch_size, seq_len=seq_len)
        bi_local_list = self._encode_bi_local(obs_list_seq, prev_act_list_seq, done_list_seq=done_list_seq)
        encoding_diag = dict(getattr(self, "last_bi_encoding_diagnostics", {}))

        priv_terms, priv_scalar, priv_logs = self._compute_privacy_terms(bi_local_list, obs_list_seq, prev_act_list_seq)
        if priv_scalar is not None and apply_local_handoff:
            priv_bi_grads = torch.autograd.grad(priv_scalar, bi_local_list, retain_graph=True, allow_unused=True)
            for i, grad in enumerate(priv_bi_grads):
                norm = 0.0 if grad is None else float(grad.detach().norm().cpu().item())
                priv_logs[f"privacy/bi_grad_norm_priv_{i}"] = norm
        else:
            for i in range(self.num_agents):
                priv_logs[f"privacy/bi_grad_norm_priv_{i}"] = 0.0

        # Baseline server phase: BI is detached before server forward/backward.
        bi_server_list = [bi.detach().clone().requires_grad_(True) for bi in bi_local_list]
        values_list, _ = self.server.forward_bi(bi_server_list)
        value_losses = [((values_list[i] - ret_list_seq[i]) ** 2).mean() for i in range(self.num_agents)]
        value_loss = torch.stack(value_losses).mean()

        self.debug_events.append("autograd_grad_before_server_step")
        dL_dBI = torch.autograd.grad(value_loss, bi_server_list, retain_graph=True)
        dL_dBI_local = [g.detach().clone() for g in dL_dBI]
        self.debug_events.append("dL_dBI_detached")
        for i, grad in enumerate(dL_dBI_local):
            td_norm = float(grad.detach().norm().cpu().item())
            priv_logs[f"privacy/bi_grad_norm_td_{i}"] = td_norm
            priv_norm = float(priv_logs.get(f"privacy/bi_grad_norm_priv_{i}", 0.0))
            priv_logs[f"privacy/bi_grad_ratio_{i}"] = float(priv_norm / (td_norm + 1e-12))

        server_before_step = [p.detach().clone() for p in self.server.parameters()]
        self.server_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.server.parameters(), self.hp.max_grad_norm)
        self.server_opt.step()
        self.debug_events.append("server_step")
        self.server_opt.zero_grad(set_to_none=True)
        server_after_step = [p.detach().clone() for p in self.server.parameters()]

        local_before = [[p.detach().clone() for p in client.bi_encoder.parameters()] for client in self.clients]
        server_before_local_backward = [p.detach().clone() for p in self.server.parameters()]
        server_grads_before_local = [None if p.grad is None else p.grad.detach().clone() for p in self.server.parameters()]

        if apply_local_handoff:
            for opt in self.local_critic_opts:
                opt.zero_grad(set_to_none=True)
            for opt in self.priv_opts:
                opt.zero_grad(set_to_none=True)
            backward_tensors: List[torch.Tensor] = list(bi_local_list)
            backward_grads: List[torch.Tensor] = list(dL_dBI_local)
            if priv_scalar is not None:
                backward_tensors.append(priv_scalar)
                backward_grads.append(torch.ones_like(priv_scalar))
            torch.autograd.backward(backward_tensors, backward_grads)
            self.debug_events.append("local_privacy_backward" if priv_scalar is not None else "local_backward")
            for i, client in enumerate(self.clients):
                grad_norm = nn.utils.clip_grad_norm_(client.bi_encoder.parameters(), self.hp.max_grad_norm)
                priv_logs[f"grad_norm/bi_encoder_{i}"] = float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm)
            for i, defense in enumerate(self.priv_defenses):
                grad_norm = nn.utils.clip_grad_norm_(defense.parameters(), self.hp.max_grad_norm)
                priv_logs[f"grad_norm/priv_defense_{i}"] = float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm)
            for opt in self.local_critic_opts:
                opt.step()
            for opt in self.priv_opts:
                opt.step()
            self.debug_events.append("local_privacy_step" if priv_scalar is not None else "local_step")
        else:
            for i in range(self.num_agents):
                priv_logs[f"grad_norm/bi_encoder_{i}"] = 0.0
                priv_logs[f"grad_norm/priv_defense_{i}"] = 0.0

        actor_grads_present = any(p.grad is not None for client in self.clients for p in client.actor.parameters())
        priv_logs["debug/actor_grads_present_after_critic_privacy"] = float(actor_grads_present)

        server_after_local_backward = [p.detach().clone() for p in self.server.parameters()]
        server_grads_after_local = [None if p.grad is None else p.grad.detach().clone() for p in self.server.parameters()]
        local_after = [[p.detach().clone() for p in client.bi_encoder.parameters()] for client in self.clients]

        def any_changed(before, after):
            return any(not torch.allclose(b, a) for b, a in zip(before, after))

        server_changed_on_server_step = any_changed(server_before_step, server_after_step)
        server_changed_during_local = any_changed(server_before_local_backward, server_after_local_backward)
        local_changed = any(any_changed(b, a) for b, a in zip(local_before, local_after))
        server_grads_created_during_local = any(
            (before is None and after is not None) or
            (before is not None and after is not None and not torch.allclose(before, after))
            for before, after in zip(server_grads_before_local, server_grads_after_local)
        )
        diagnostics = {
            "loss": float(value_loss.detach().cpu().item()),
            "value_losses": [float(v.detach().cpu().item()) for v in value_losses],
            "values_list": values_list,
            "dL_dBI": dL_dBI_local,
            "events": list(self.debug_events),
            "server_changed_on_server_step": server_changed_on_server_step,
            "server_changed_during_local_step": server_changed_during_local,
            "server_grads_created_during_local": server_grads_created_during_local,
            "local_changed_on_handoff": local_changed,
            "dL_dBI_requires_grad": [g.requires_grad for g in dL_dBI_local],
            "critic_gru_reset_enabled": bool(encoding_diag.get("critic_gru_reset_enabled", 0.0)),
            "critic_gru_done_resets": float(encoding_diag.get("critic_gru_done_resets", 0.0)),
            "hidden_resets": float(encoding_diag.get("hidden_resets", 0.0)),
            "num_segments": float(encoding_diag.get("num_segments", 0.0)),
            "mean_segment_len": float(encoding_diag.get("mean_segment_len", 0.0)),
            "privacy_logs": dict(priv_logs),
        }
        self.last_privacy_logs = dict(priv_logs)
        self.last_critic_diagnostics = diagnostics
        return diagnostics

    def update(
        self,
        client_rollouts: Optional[List[Dict[str, np.ndarray]]] = None,
        server_rollout: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
        *,
        agent_rollouts: Optional[List[Dict[str, np.ndarray]]] = None,
        last_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        if agent_rollouts is not None:
            if last_values is None:
                raise ValueError("last_values are required with legacy agent_rollouts")
            client_rollouts, server_rollout = self._legacy_rollouts_to_split(agent_rollouts, last_values)
        elif server_rollout is None and last_values is not None and client_rollouts is not None:
            client_rollouts, server_rollout = self._legacy_rollouts_to_split(client_rollouts, last_values)
        if client_rollouts is None or server_rollout is None:
            raise ValueError("update requires client_rollouts and server_rollout")

        T = self._validate_client_rollouts(client_rollouts)
        server_adv = self._server_compute_advantages(server_rollout)
        if any(adv.shape[0] != T for adv in server_adv.advantages):
            raise ValueError("Server advantage horizon must match client rollout horizon")

        done_list_np = self._expand_server_sequence(server_rollout["dones"], "dones")
        obs_list_np, act_list_np, prev_act_list_np = [], [], []
        old_logp_list_np, adv_list_np, ret_list_np = [], [], []
        for r in client_rollouts:
            obs_list_np.append(r["obs"])
            act_list_np.append(r["act"])
            prev_act_list_np.append(r["prev_act"])
            old_logp_list_np.append(r["logp"])
        for i in range(self.num_agents):
            adv_list_np.append(server_adv.advantages[i])
            ret_list_np.append(server_adv.returns[i])

        all_adv = np.concatenate(adv_list_np)
        adv_mean = all_adv.mean()
        adv_std = all_adv.std() + 1e-8
        adv_list_np = [(adv - adv_mean) / adv_std for adv in adv_list_np]
        per_agent_adv_std_normed = [float(adv.std()) for adv in adv_list_np]

        obs_seq = [torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device) for x in obs_list_np]
        prev_act_seq = [torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device) for x in prev_act_list_np]
        ret_seq = [torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device) for x in ret_list_np]
        done_seq = [torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device) for x in done_list_np]

        critic_losses = []
        per_agent_critic_losses = [[] for _ in range(self.num_agents)]
        privacy_epoch_logs: List[Dict[str, float]] = []
        actor_epoch_logs: List[Dict[str, Any]] = []
        for _ in range(max(1, int(self.hp.n_epochs))):
            diag = self.critic_update_from_tensors(
                obs_seq,
                prev_act_seq,
                ret_seq,
                apply_local_handoff=True,
                done_list_seq=done_seq,
            )
            critic_losses.append(diag["loss"])
            privacy_epoch_logs.append(dict(diag.get("privacy_logs", {})))
            for i, loss_i in enumerate(diag["value_losses"]):
                per_agent_critic_losses[i].append(float(loss_i))

            a_logs = self._actor_update_from_tensors(obs_list_np, act_list_np, old_logp_list_np, adv_list_np)
            actor_epoch_logs.append(a_logs)
            if self.hp.target_kl is not None and float(a_logs.get("approx_kl", 0.0)) > float(self.hp.target_kl):
                break

        actor_logs = self._aggregate_actor_epoch_logs(actor_epoch_logs)
        self.round_idx += 1
        fedavg_groups: List[str] = []
        if self.hp.fedavg_interval > 0 and self.round_idx % self.hp.fedavg_interval == 0:
            fedavg_groups = self.run_fedavg()

        critic_diag = self.last_critic_diagnostics
        logs = {
            "loss/actor": actor_logs["actor_loss"],
            "loss/critic": float(np.mean(critic_losses)),
            "loss/entropy": actor_logs["entropy"],
            "train/approx_kl": actor_logs["approx_kl"],
            "train/approx_kl_post": actor_logs["approx_kl_post"],
            "train/actor_epochs_used": float(actor_logs["actor_epochs_used"]),
            "train/approx_kl_epoch0": actor_logs["approx_kl_epoch0"],
            "train/approx_kl_last": actor_logs["approx_kl_last"],
            "train/clipfrac": actor_logs["clipfrac"],
            "fedavg/round": float(self.round_idx),
            "rollout/length": float(T),
            "rollout/num_episodes": float(critic_diag.get("num_segments", 0.0)),
            "rollout/mean_episode_len": float(critic_diag.get("mean_segment_len", 0.0)),
            "debug/actor_grad_norm": actor_logs["actor_grad_norm"],
            "debug/actor_param_delta": actor_logs["actor_param_delta"],
            "debug/actor_param_delta_min_epoch": actor_logs["actor_param_delta_min_epoch"],
            "debug/actor_logp_delta_post": actor_logs["actor_logp_delta_post"],
            "debug/actor_post_kl_available": actor_logs["actor_post_kl_available"],
            "debug/server_advantage_owner": 1.0,
            "debug/local_actor_reward_value_free": 1.0,
            "debug/critic_gru_reset_enabled": float(critic_diag.get("critic_gru_reset_enabled", False)),
            "debug/critic_gru_done_resets": float(critic_diag.get("critic_gru_done_resets", 0.0)),
            "debug/hidden_resets": float(critic_diag.get("hidden_resets", 0.0)),
            "fedavg/applied": float(bool(fedavg_groups)),
            "fedavg/scope_enhancer": float("enhancer" in fedavg_groups),
            "fedavg/scope_gru": float("gru" in fedavg_groups),
        }
        logs.update(self._safe_critic_scalar_logs(critic_diag))
        privacy_logs = self._mean_log_dict(privacy_epoch_logs)
        logs.update(privacy_logs)
        if "loss/privacy_backward_total" in logs:
            logs["ratio/privacy_backward_to_critic"] = float(logs["loss/privacy_backward_total"] / (logs["loss/critic"] + 1e-12))
            logs["loss/privacy_total"] = float(logs["loss/privacy_backward_total"])

        for i in range(self.num_agents):
            logs[f"agent_{i}/adv_mean_raw"] = server_adv.adv_mean_raw[i]
            logs[f"agent_{i}/adv_std_raw"] = server_adv.adv_std_raw[i]
            logs[f"agent_{i}/adv_std_normed"] = per_agent_adv_std_normed[i]
            logs[f"agent_{i}/loss_actor"] = actor_logs["per_agent_actor_loss"][i]
            logs[f"agent_{i}/loss_critic"] = float(np.mean(per_agent_critic_losses[i]))
            logs[f"agent_{i}/entropy"] = actor_logs["per_agent_entropy"][i]
            logs[f"agent_{i}/approx_kl"] = actor_logs["per_agent_approx_kl"][i]
            logs[f"agent_{i}/clipfrac"] = actor_logs["per_agent_clipfrac"][i]
            logs[f"agent_{i}/ratio_std"] = actor_logs["per_agent_ratio_std"][i]
            logs[f"agent_{i}/approx_kl_post"] = actor_logs["per_agent_approx_kl_post"][i]
            logs[f"agent_{i}/debug/actor_grad_norm"] = actor_logs["per_agent_actor_grad_norm"][i]
            logs[f"agent_{i}/debug/actor_param_delta"] = actor_logs["per_agent_actor_param_delta"][i]
        self.last_privacy_logs = privacy_logs
        if self.writer is not None and global_step is not None:
            for k, v in logs.items():
                self.writer.add_scalar(k, v, global_step)
        return logs
