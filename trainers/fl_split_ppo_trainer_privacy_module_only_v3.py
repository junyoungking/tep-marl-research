"""Split-FL PPO trainer with detached privacy sidecar and decoupled critic epochs.

This v3 variant keeps the privacy-module-only split-FL semantics from
``trainers.fl_split_ppo_trainer_privacy_module_only`` but separates critic
epochs from the actor KL early-stop loop.  Critic updates complete for the
configured ``critic_epochs`` regardless of whether the actor stops early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from trainers.fl_split_ppo_trainer_dravl import DRAVLSplitFedPPOHyperParams
from trainers.fl_split_ppo_trainer_privacy_module_only import PrivacyModuleOnlySplitFedPPOTrainer
from utils.training_metrics_logger import TrainingMetricsLogger

TRAINER_VERSION = "fl_v3"


@dataclass
class DRAVLSplitFedPPOHyperParamsV3(DRAVLSplitFedPPOHyperParams):
    trainer_version: str = TRAINER_VERSION
    critic_epochs: int = 4


class PrivacyModuleOnlySplitFedPPOTrainerV3(PrivacyModuleOnlySplitFedPPOTrainer):
    """Privacy-sidecar trainer with critic epochs decoupled from actor KL."""

    def _critic_epochs(self) -> int:
        critic_epochs = int(getattr(self.hp, "critic_epochs", 1))
        if critic_epochs <= 0:
            raise ValueError(f"critic_epochs must be positive, got {critic_epochs}")
        return critic_epochs

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
        for i, r in enumerate(client_rollouts):
            obs_list_np.append(r["obs"])
            act_list_np.append(r["act"])
            prev_act_list_np.append(r["prev_act"])
            old_logp_list_np.append(r["logp"])
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

        critic_epochs = self._critic_epochs()
        critic_epoch_logs: List[Dict[str, Any]] = []
        privacy_epoch_logs: List[Dict[str, float]] = []
        for _ in range(critic_epochs):
            diag = self.critic_update_from_tensors(
                obs_seq,
                prev_act_seq,
                ret_seq,
                apply_local_handoff=True,
                done_list_seq=done_seq,
            )
            critic_epoch_logs.append(diag)
            privacy_epoch_logs.append(dict(diag.get("privacy_logs", {})))

        if not critic_epoch_logs:
            raise RuntimeError("critic_epoch_logs unexpectedly empty")

        actor_epochs = max(1, int(self.hp.n_epochs))
        actor_epoch_logs: List[Dict[str, Any]] = []
        for _ in range(actor_epochs):
            a_logs = self._actor_update_from_tensors(obs_list_np, act_list_np, old_logp_list_np, adv_list_np)
            actor_epoch_logs.append(a_logs)
            if self.hp.target_kl is not None and float(a_logs.get("approx_kl", 0.0)) > float(self.hp.target_kl):
                break

        critic_diag = critic_epoch_logs[-1]

        self.round_idx += 1
        fedavg_groups: List[str] = []
        if self.hp.fedavg_interval > 0 and self.round_idx % self.hp.fedavg_interval == 0:
            fedavg_groups = self.run_fedavg()

        logger = getattr(self, "metrics_logger", None)
        if logger is None:
            logger = TrainingMetricsLogger(self.writer, self.hp.log_every, marl=True, fedavg=True, privacy=True)
            self.metrics_logger = logger
        logs = logger.compose_split_update(
            actor_epochs=actor_epoch_logs,
            critic_epochs=critic_epoch_logs,
            server_advantage=server_adv,
            normalized_adv_stds=per_agent_adv_std_normed,
            round_idx=self.round_idx,
            fedavg_groups=fedavg_groups,
            privacy_epochs=privacy_epoch_logs,
        )
        # Keep the historical inspection state, but derive it from the shared
        # composition rather than maintaining a second privacy reduction.
        privacy_logs = {key: value for key, value in logs.items() if logger.mode_for_tag(key) == "PRIVACY"}
        self.last_privacy_logs = privacy_logs
        self.last_critic_diagnostics = critic_diag
        if self.writer is not None and global_step is not None:
            logger.record_update(logs, step=global_step)
        return logs
