"""Split-FL PPO trainer with full privacy gradients and decoupled critic epochs.

This variant keeps the continue-rollout / FedAvg10 rollout path from the
privacy-module-only v3 trainer, but changes the privacy contract:

- the reconstructor/attacker is trained first on detached BI features,
- the main BI privacy update uses the updated reconstructor in a frozen state,
- task gradients and privacy gradients both affect the BI encoder,
- the reconstructor is never stepped by the main privacy backward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bi_privacy_defense import ClientBIPrivacyDefense
from trainers.fl_split_ppo_trainer_dravl import DRAVLSplitFedPPOHyperParams
from trainers.fl_split_ppo_trainer_privacy_module_only_v3 import PrivacyModuleOnlySplitFedPPOTrainerV3

TRAINER_VERSION = "fl_full_v1"


@dataclass
class DRAVLSplitFedPPOHyperParamsFullV3(DRAVLSplitFedPPOHyperParams):
    trainer_version: str = TRAINER_VERSION
    critic_epochs: int = 16
    privacy_reconstructor_steps: int = 1
    privacy_encoder_steps: int = 1
    privacy_log_grad_components: bool = True


class PrivacyFullSplitFedPPOTrainerV3(PrivacyModuleOnlySplitFedPPOTrainerV3):
    """FedAvg10 split-FL trainer with detached reconstructor updates and main privacy gradients."""

    def _set_privacy_defense_requires_grad(self, requires_grad: bool) -> None:
        for defense in self.priv_defenses:
            for param in defense.parameters():
                param.requires_grad_(requires_grad)

    def _privacy_target_list(
        self,
        obs_list_seq: List[torch.Tensor],
        prev_act_list_seq: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        return [torch.cat([obs_list_seq[i], prev_act_list_seq[i]], dim=-1) for i in range(self.num_agents)]

    def _privacy_attacker_update(
        self,
        detached_bi_list: List[torch.Tensor],
        target_list: List[torch.Tensor],
        *,
        apply_local_handoff: bool,
    ) -> Dict[str, float]:
        logs: Dict[str, float] = {}
        if not self.hp.privacy_enabled or not self.priv_defenses:
            for i in range(self.num_agents):
                logs[f"grad_norm/priv_defense_{i}"] = 0.0
                logs[f"privacy/reconstructor_grad_{i}"] = 0.0
            logs["loss/privacy_recon_attacker_total"] = 0.0
            logs["loss/recon_attacker"] = 0.0
            logs["privacy/recon_mse"] = 0.0
            logs["privacy/attacker_applied"] = 0.0
            logs["debug/privacy_reconstructor_detached"] = 1.0
            return logs

        for opt in self.priv_opts:
            opt.zero_grad(set_to_none=True)

        attacker_losses: List[torch.Tensor] = []
        per_agent_mse: List[float] = []
        for i, defense in enumerate(self.priv_defenses):
            target_flat = target_list[i].reshape(-1, target_list[i].shape[-1]).detach()
            bi_flat = detached_bi_list[i].reshape(-1, detached_bi_list[i].shape[-1])
            recon = defense.reconstructor(bi_flat)
            loss_i = F.mse_loss(recon, target_flat)
            attacker_losses.append(loss_i)
            per_agent_mse.append(float(loss_i.detach().cpu().item()))
            logs[f"agent_{i}/loss/recon_attacker"] = per_agent_mse[-1]

        attacker_total = torch.stack(attacker_losses).sum()
        logs["loss/privacy_recon_attacker_total"] = float(attacker_total.detach().cpu().item())
        logs["loss/recon_attacker"] = float(np.mean(per_agent_mse)) if per_agent_mse else 0.0
        logs["privacy/recon_mse"] = logs["loss/recon_attacker"]
        logs["privacy/attacker_applied"] = float(bool(apply_local_handoff))
        logs["debug/privacy_reconstructor_detached"] = 1.0

        if apply_local_handoff:
            attacker_total.backward()
            for i, defense in enumerate(self.priv_defenses):
                grad_norm = nn.utils.clip_grad_norm_(defense.parameters(), self.hp.max_grad_norm)
                grad_norm_value = float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm)
                logs[f"grad_norm/priv_defense_{i}"] = grad_norm_value
                logs[f"privacy/reconstructor_grad_{i}"] = grad_norm_value
            for opt in self.priv_opts:
                opt.step()
        else:
            for i in range(self.num_agents):
                logs[f"grad_norm/priv_defense_{i}"] = 0.0
                logs[f"privacy/reconstructor_grad_{i}"] = 0.0

        self.last_privacy_logs = dict(logs)
        return logs

    def _privacy_main_update(
        self,
        bi_local_list: List[torch.Tensor],
        target_list: List[torch.Tensor],
        *,
        apply_local_handoff: bool,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        logs: Dict[str, float] = {}
        if not self.hp.privacy_enabled or not self.priv_defenses:
            for i in range(self.num_agents):
                logs[f"privacy/bi_grad_norm_priv_{i}"] = 0.0
            logs["loss/privacy_backward_total"] = 0.0
            logs["loss/privacy_total"] = 0.0
            logs["privacy/full_privacy_applied"] = 0.0
            logs["debug/privacy_reconstructor_detached"] = 1.0
            return None, logs

        self._set_privacy_defense_requires_grad(False)
        try:
            priv_terms: List[torch.Tensor] = []
            per_agent_total: List[float] = []
            for i, defense in enumerate(self.priv_defenses):
                bi_flat = bi_local_list[i].reshape(-1, bi_local_list[i].shape[-1])
                target_flat = target_list[i].reshape(-1, target_list[i].shape[-1])
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
                priv_terms.append(out["backward_loss"])
                per_agent_total.append(float(out["loss/privacy_backward_total"].detach().cpu().item()))

                for key in (
                    "loss/priv_ar_raw",
                    "loss/priv_nr_raw",
                    "privacy/dcor_raw",
                    "privacy/dcor_log",
                    "loss/priv_nr_weighted",
                    "loss/priv_dcor_weighted",
                    "privacy/ar_lambda_eff",
                    "loss/privacy_backward_total",
                    "privacy/recon_mse",
                ):
                    if key in out:
                        logs[f"agent_{i}/{key}"] = self._as_float(out[key])

            priv_scalar = torch.stack(priv_terms).sum()
            logs["loss/privacy_backward_total"] = float(priv_scalar.detach().cpu().item())
            logs["loss/privacy_total"] = float(priv_scalar.detach().cpu().item())
            logs["privacy/full_privacy_applied"] = float(bool(apply_local_handoff))
            logs["debug/privacy_reconstructor_detached"] = 1.0

            if apply_local_handoff:
                priv_grads = torch.autograd.grad(priv_scalar, bi_local_list, retain_graph=True, allow_unused=True)
                for i, grad in enumerate(priv_grads):
                    norm = 0.0 if grad is None else float(grad.detach().norm().cpu().item())
                    logs[f"privacy/bi_grad_norm_priv_{i}"] = norm
            else:
                for i in range(self.num_agents):
                    logs[f"privacy/bi_grad_norm_priv_{i}"] = 0.0

            for i in range(self.num_agents):
                logs[f"agent_{i}/loss/privacy_backward_total"] = per_agent_total[i]
            return priv_scalar, logs
        finally:
            self._set_privacy_defense_requires_grad(True)

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
        for opt in self.actor_opts:
            opt.zero_grad(set_to_none=True)

        batch_size, seq_len = obs_list_seq[0].shape[0], obs_list_seq[0].shape[1]
        done_list_seq = self._coerce_done_list_seq(done_list_seq, done_seq, batch_size=batch_size, seq_len=seq_len)
        bi_local_list = self._encode_bi_local(obs_list_seq, prev_act_list_seq, done_list_seq=done_list_seq)
        encoding_diag = dict(getattr(self, "last_bi_encoding_diagnostics", {}))

        target_list = self._privacy_target_list(obs_list_seq, prev_act_list_seq)
        detached_bi_list = [bi.detach() for bi in bi_local_list]

        bi_server_list = [bi.detach().clone().requires_grad_(True) for bi in bi_local_list]
        values_list, _ = self.server.forward_bi(bi_server_list)
        value_losses = [((values_list[i] - ret_list_seq[i]) ** 2).mean() for i in range(self.num_agents)]
        value_loss = torch.stack(value_losses).mean()

        self.debug_events.append("autograd_grad_before_server_step")
        dL_dBI = torch.autograd.grad(value_loss, bi_server_list, retain_graph=True)
        dL_dBI_local = [g.detach().clone() for g in dL_dBI]
        self.debug_events.append("dL_dBI_detached")

        attacker_logs = self._privacy_attacker_update(
            detached_bi_list,
            target_list,
            apply_local_handoff=apply_local_handoff,
        )

        priv_scalar, privacy_logs = self._privacy_main_update(
            bi_local_list,
            target_list,
            apply_local_handoff=apply_local_handoff,
        )

        for i, grad in enumerate(dL_dBI_local):
            td_norm = float(grad.detach().norm().cpu().item())
            privacy_logs[f"privacy/bi_grad_norm_td_{i}"] = td_norm
            privacy_logs[f"privacy/bi_grad_task_{i}"] = td_norm
            priv_norm = float(privacy_logs.get(f"privacy/bi_grad_norm_priv_{i}", 0.0))
            privacy_logs[f"privacy/bi_grad_ratio_{i}"] = float(priv_norm / (td_norm + 1e-12))

        self.debug_events.append("server_step")
        server_before_step = [p.detach().clone() for p in self.server.parameters()]
        self.server_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.server.parameters(), self.hp.max_grad_norm)
        self.server_opt.step()
        self.server_opt.zero_grad(set_to_none=True)
        server_after_step = [p.detach().clone() for p in self.server.parameters()]

        local_before = [[p.detach().clone() for p in client.bi_encoder.parameters()] for client in self.clients]
        server_before_local_backward = [p.detach().clone() for p in self.server.parameters()]
        server_grads_before_local = [None if p.grad is None else p.grad.detach().clone() for p in self.server.parameters()]

        if apply_local_handoff:
            for opt in self.local_critic_opts:
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
                privacy_logs[f"grad_norm/bi_encoder_{i}"] = float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm)
            for opt in self.local_critic_opts:
                opt.step()
            self.debug_events.append("local_privacy_step" if priv_scalar is not None else "local_step")
        else:
            for i in range(self.num_agents):
                privacy_logs[f"grad_norm/bi_encoder_{i}"] = 0.0

        server_after_local_backward = [p.detach().clone() for p in self.server.parameters()]
        server_grads_after_local = [None if p.grad is None else p.grad.detach().clone() for p in self.server.parameters()]
        local_after = [[p.detach().clone() for p in client.bi_encoder.parameters()] for client in self.clients]

        def any_changed(before, after):
            return any(not torch.allclose(b, a) for b, a in zip(before, after))

        server_changed_on_server_step = any_changed(server_before_step, server_after_step)
        server_changed_during_local = any_changed(server_before_local_backward, server_after_local_backward)
        local_changed = any(any_changed(b, a) for b, a in zip(local_before, local_after))
        server_grads_created_during_local = any(
            (before is None and after is not None)
            or (before is not None and after is not None and not torch.allclose(before, after))
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
            "privacy_logs": {**attacker_logs, **privacy_logs},
        }
        self.last_privacy_logs = dict(diagnostics["privacy_logs"])
        self.last_critic_diagnostics = diagnostics
        return diagnostics
