"""Split-FL PPO trainer with a detached privacy-module sidecar.

This variant keeps the fedavg10 main training path identical to the baseline
``SplitFedContinueRolloutPPOTrainer`` and trains the privacy-defense module as a
detached sidecar.  Privacy features are re-encoded after the baseline critic
update, detached from the main model graph, and only the privacy-defense
parameters receive gradients.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from trainers.fl_split_ppo_trainer_continue_rollout import SplitFedContinueRolloutPPOTrainer
from trainers.fl_split_ppo_trainer_dravl import DRAVLSplitFedPPOTrainer


class PrivacyModuleOnlySplitFedPPOTrainer(DRAVLSplitFedPPOTrainer):
    """Baseline-preserving split-FL trainer with detached privacy sidecar."""

    def _privacy_sidecar_update(
        self,
        bi_local_list: List[torch.Tensor],
        obs_list_seq: List[torch.Tensor],
        prev_act_list_seq: List[torch.Tensor],
        *,
        apply_local_handoff: bool,
        baseline_diag: Dict[str, Any],
    ) -> Dict[str, float]:
        logs: Dict[str, float] = {}
        if not self.hp.privacy_enabled or not self.priv_defenses:
            for i in range(self.num_agents):
                logs[f"privacy/bi_grad_norm_priv_{i}"] = 0.0
                logs[f"privacy/bi_grad_norm_td_{i}"] = 0.0
                logs[f"privacy/bi_grad_ratio_{i}"] = 0.0
                logs[f"grad_norm/priv_defense_{i}"] = 0.0
            logs["privacy/sidecar_applied"] = 0.0
            logs["debug/privacy_sidecar_detached"] = 1.0
            return logs

        detached_bi_list = [bi.detach().clone().requires_grad_(True) for bi in bi_local_list]
        _, priv_scalar, priv_logs = self._compute_privacy_terms(detached_bi_list, obs_list_seq, prev_act_list_seq)

        # Compute privacy gradients on detached BI tensors for logging only.
        if priv_scalar is not None:
            priv_bi_grads = torch.autograd.grad(priv_scalar, detached_bi_list, retain_graph=True, allow_unused=True)
        else:
            priv_bi_grads = [None for _ in range(self.num_agents)]

        td_grads = baseline_diag.get("dL_dBI") or []
        for i in range(self.num_agents):
            td_grad = td_grads[i] if i < len(td_grads) else None
            priv_grad = priv_bi_grads[i] if i < len(priv_bi_grads) else None
            td_norm = 0.0 if td_grad is None else float(td_grad.detach().norm().cpu().item())
            priv_norm = 0.0 if priv_grad is None else float(priv_grad.detach().norm().cpu().item())
            priv_logs[f"privacy/bi_grad_norm_td_{i}"] = td_norm
            priv_logs[f"privacy/bi_grad_norm_priv_{i}"] = priv_norm
            priv_logs[f"privacy/bi_grad_ratio_{i}"] = float(priv_norm / (td_norm + 1e-12))

        priv_total = float(priv_logs.get("loss/privacy_backward_total", 0.0))
        critic_loss = float(baseline_diag.get("loss", 0.0))
        priv_logs["loss/privacy_total"] = priv_total
        priv_logs["ratio/privacy_backward_to_critic"] = float(priv_total / (critic_loss + 1e-12))
        priv_logs["privacy/sidecar_applied"] = float(bool(apply_local_handoff))
        priv_logs["debug/privacy_sidecar_detached"] = 1.0

        if apply_local_handoff and priv_scalar is not None:
            for opt in self.priv_opts:
                opt.zero_grad(set_to_none=True)
            priv_scalar.backward()
            for i, defense in enumerate(self.priv_defenses):
                grad_norm = torch.nn.utils.clip_grad_norm_(defense.parameters(), self.hp.max_grad_norm)
                priv_logs[f"grad_norm/priv_defense_{i}"] = float(
                    grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm
                )
            for opt in self.priv_opts:
                opt.step()
        else:
            for i in range(self.num_agents):
                priv_logs[f"grad_norm/priv_defense_{i}"] = 0.0

        self.last_privacy_logs = dict(priv_logs)
        return priv_logs

    def critic_update_from_tensors(
        self,
        obs_list_seq: List[torch.Tensor],
        prev_act_list_seq: List[torch.Tensor],
        ret_list_seq: List[torch.Tensor],
        apply_local_handoff: bool = True,
        done_list_seq: Optional[List[torch.Tensor]] = None,
        done_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size, seq_len = obs_list_seq[0].shape[0], obs_list_seq[0].shape[1]
        done_list_seq = self._coerce_done_list_seq(done_list_seq, done_seq, batch_size=batch_size, seq_len=seq_len)

        # Capture the same rollout features the baseline critic update will see,
        # but keep them detached so the privacy sidecar cannot touch the main
        # model graph.
        with torch.no_grad():
            bi_local_list = self._encode_bi_local(obs_list_seq, prev_act_list_seq, done_list_seq=done_list_seq)

        # Run the canonical fedavg10 critic update first so the main model path
        # is bit-for-bit aligned with the baseline trainer.
        diagnostics = SplitFedContinueRolloutPPOTrainer.critic_update_from_tensors(
            self,
            obs_list_seq,
            prev_act_list_seq,
            ret_list_seq,
            apply_local_handoff=apply_local_handoff,
            done_list_seq=done_list_seq,
        )

        if self.hp.privacy_enabled:
            privacy_logs = self._privacy_sidecar_update(
                bi_local_list,
                obs_list_seq,
                prev_act_list_seq,
                apply_local_handoff=apply_local_handoff,
                baseline_diag=diagnostics,
            )
        else:
            privacy_logs = {
                **{f"privacy/bi_grad_norm_priv_{i}": 0.0 for i in range(self.num_agents)},
                **{f"privacy/bi_grad_norm_td_{i}": 0.0 for i in range(self.num_agents)},
                **{f"privacy/bi_grad_ratio_{i}": 0.0 for i in range(self.num_agents)},
                **{f"grad_norm/priv_defense_{i}": 0.0 for i in range(self.num_agents)},
                "loss/privacy_total": 0.0,
                "ratio/privacy_backward_to_critic": 0.0,
                "privacy/sidecar_applied": 0.0,
                "debug/privacy_sidecar_detached": 1.0,
            }

        diagnostics["privacy_logs"] = privacy_logs
        diagnostics["privacy_sidecar_detached"] = 1.0
        self.last_privacy_logs = dict(privacy_logs)
        self.last_critic_diagnostics = diagnostics
        return diagnostics
