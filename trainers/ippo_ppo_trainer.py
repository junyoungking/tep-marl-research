from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.fl_split_actor import SplitLocalActor
from models.ippo_critic import IPPOCritic


TRAINER_VERSION = "ippo_architecture_matched_v1"


@dataclass
class IPPOHyperParams:
    trainer_version: str = TRAINER_VERSION
    actor_lr: float = 1.2e-4
    critic_lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05
    max_grad_norm: float = 0.5
    actor_epochs: int = 4
    critic_epochs: int = 16
    target_kl: float = 0.03
    rollout_horizon: int = 1024
    total_steps: int = 1024
    log_every: int = 1000


class IPPOTrainer(nn.Module):
    """Architecture-matched recurrent IPPO trainer.

    Each agent owns an independent actor, critic, and optimizer pair. The
    trainer intentionally avoids any server aggregation, FedAvg, or privacy
    sidecar semantics.
    """

    def __init__(
        self,
        actors: Sequence[SplitLocalActor],
        critics: Sequence[IPPOCritic],
        device: str = "cpu",
        hp: Optional[IPPOHyperParams] = None,
        writer: Optional[Any] = None,
    ):
        super().__init__()
        if len(actors) != len(critics):
            raise ValueError(f"actors and critics must have the same length, got {len(actors)} vs {len(critics)}")
        if len(actors) != 3:
            raise ValueError(f"IPPOTrainer expects exactly 3 agents, got {len(actors)}")
        self.device = torch.device(device)
        self.actors = nn.ModuleList(list(actors))
        self.critics = nn.ModuleList(list(critics))
        self.to(self.device)
        self.hp = hp if hp is not None else IPPOHyperParams()
        self.writer = writer
        self.num_agents = len(self.actors)
        self.round_idx = 0
        self.last_actor_diagnostics: List[Dict[str, Any]] = []
        self.last_critic_diagnostics: List[Dict[str, Any]] = []

        self.actor_opts = [optim.Adam(actor.parameters(), lr=self.hp.actor_lr) for actor in self.actors]
        self.critic_opts = [optim.Adam(critic.parameters(), lr=self.hp.critic_lr) for critic in self.critics]

    @classmethod
    def build_default(
        cls,
        obs_dims: Sequence[int],
        act_dims: Sequence[int],
        device: str = "cpu",
        hp: Optional[IPPOHyperParams] = None,
    ) -> "IPPOTrainer":
        if len(obs_dims) != 3 or len(act_dims) != 3:
            raise ValueError("IPPOTrainer.build_default expects exactly three heterogeneous agents")
        actors: List[SplitLocalActor] = []
        critics: List[IPPOCritic] = []
        for obs_dim, act_dim in zip(obs_dims, act_dims):
            actor = SplitLocalActor(
                obs_dim=obs_dim,
                act_dim=act_dim,
                local_obs_hidden_dims=(),
                local_obs_out_dim=64,
                shared_obs_hidden_dims=(),
                shared_obs_out_dim=64,
                local_action_hidden_dims=(),
                log_std_min=-5.0,
                log_std_max=2.0,
                activate_feature_outputs=True,
                ablate_enhancer=False,
            )
            critic = IPPOCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                local_pre_hidden_dims=(),
                local_pre_out_dim=64,
                bi_hidden_dim=64,
                gru_num_layers=1,
                activate_feature_outputs=True,
            )
            actors.append(actor)
            critics.append(critic)
        return cls(actors=actors, critics=critics, device=device, hp=hp)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _scalar_float(value: Any) -> Optional[float]:
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, Number):
            return float(value)
        if isinstance(value, np.ndarray) and value.shape == ():
            return float(value.item())
        return None

    @staticmethod
    def _parameter_delta_norm(before: Sequence[torch.Tensor], module: nn.Module) -> float:
        total = 0.0
        for old, new in zip(before, module.parameters()):
            total += float((new.detach() - old).pow(2).sum().cpu().item())
        return float(total ** 0.5)

    @staticmethod
    def _as_float_array(value: Any, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array, got shape={arr.shape}")
        return arr

    @staticmethod
    def _compute_gae(
        rew: np.ndarray,
        val: np.ndarray,
        done: np.ndarray,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = rew.shape[0]
        next_values = np.empty_like(val)
        next_values[:-1] = val[1:]
        next_values[-1] = last_value
        next_values = next_values * (1.0 - done)
        deltas = rew + gamma * next_values - val
        adv = np.zeros_like(rew, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - done[t]
            gae = deltas[t] + gamma * gae_lambda * mask * gae
            adv[t] = gae
        return adv, adv + val

    @staticmethod
    def _segment_bounds(done_seq: torch.Tensor) -> List[Tuple[int, int]]:
        """Return half-open episode segment bounds, including terminal steps."""
        if done_seq.dim() == 2:
            if done_seq.shape[0] != 1:
                raise NotImplementedError("Done-boundary segmentation currently supports batch_size=1")
            flat_done = done_seq[0]
        elif done_seq.dim() == 1:
            flat_done = done_seq
        else:
            raise ValueError(f"done_seq must have shape (T,) or (1, T), got {tuple(done_seq.shape)}")
        flat_done = flat_done.reshape(-1)
        seq_len = int(flat_done.numel())
        if seq_len == 0:
            return []
        ends = (flat_done > 0.5).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        bounds: List[Tuple[int, int]] = []
        start = 0
        for end_idx in ends:
            end = int(end_idx) + 1
            if end > start:
                bounds.append((start, end))
            start = end
        if start < seq_len:
            bounds.append((start, seq_len))
        return bounds

    @staticmethod
    def _internal_done_reset_count(done_seq: Optional[torch.Tensor]) -> int:
        """Count internal resets between consecutive steps, excluding the final terminal step."""
        if done_seq is None:
            return 0
        if done_seq.dim() == 1:
            flat_done = done_seq.reshape(1, -1)
        elif done_seq.dim() == 2:
            flat_done = done_seq
        else:
            raise ValueError(f"done_seq must have shape (T,) or (B, T), got {tuple(done_seq.shape)}")
        if flat_done.shape[1] <= 1:
            return 0
        return int((flat_done[:, :-1] > 0.5).sum().detach().cpu().item())

    def _validate_done_list_seq(
        self,
        done_list_seq: Optional[List[torch.Tensor]],
        batch_size: int,
        seq_len: int,
    ) -> Optional[List[torch.Tensor]]:
        if done_list_seq is None:
            return None
        if len(done_list_seq) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} done tensors, got {len(done_list_seq)}")
        validated: List[torch.Tensor] = []
        for i, done_seq in enumerate(done_list_seq):
            if not torch.is_tensor(done_seq):
                raise TypeError(f"done_list_seq[{i}] must be a torch.Tensor")
            if done_seq.dim() == 1:
                done_seq = done_seq.reshape(1, -1)
            if tuple(done_seq.shape) != (batch_size, seq_len):
                raise ValueError(
                    f"done_list_seq[{i}] shape mismatch: expected {(batch_size, seq_len)}, got {tuple(done_seq.shape)}"
                )
            validated.append(done_seq.to(device=self.device, dtype=torch.float32))
        return validated

    def _coerce_done_list_seq(
        self,
        done_list_seq: Optional[List[torch.Tensor]],
        done_seq: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[List[torch.Tensor]]:
        if done_seq is not None:
            if done_list_seq is not None:
                raise ValueError("Pass either done_seq or done_list_seq, not both")
            if done_seq.dim() == 1:
                done_seq = done_seq.reshape(1, -1)
            done_list_seq = [done_seq for _ in range(self.num_agents)]
        return self._validate_done_list_seq(done_list_seq, batch_size=batch_size, seq_len=seq_len)

    def _done_segmentation_diagnostics(
        self,
        done_list_seq: Optional[List[torch.Tensor]],
        seq_len: int,
    ) -> Dict[str, float]:
        if done_list_seq is None:
            return {
                "critic_gru_reset_enabled": 0.0,
                "critic_gru_done_resets": 0.0,
                "hidden_resets": float(self.num_agents),
                "num_segments": 1.0 if seq_len > 0 else 0.0,
                "mean_segment_len": float(seq_len),
            }
        bounds = self._segment_bounds(done_list_seq[0])
        segment_lengths = [end - start for start, end in bounds]
        return {
            "critic_gru_reset_enabled": 1.0,
            "critic_gru_done_resets": float(self._internal_done_reset_count(done_list_seq[0])),
            "hidden_resets": float(len(bounds) * self.num_agents),
            "num_segments": float(len(bounds)),
            "mean_segment_len": float(np.mean(segment_lengths)) if segment_lengths else 0.0,
        }

    def _tensor_list(self, xs: Sequence[Any]) -> List[torch.Tensor]:
        return [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in xs]

    def _critic_values_from_sequence(
        self,
        agent_idx: int,
        obs_seq: torch.Tensor,
        prev_act_seq: torch.Tensor,
        done_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        critic = self.critics[agent_idx]
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
        if prev_act_seq.dim() == 2:
            prev_act_seq = prev_act_seq.unsqueeze(0)
        if obs_seq.dim() != 3 or prev_act_seq.dim() != 3:
            raise ValueError(
                f"critic expects sequence tensors with shape (B, T, D), got obs={tuple(obs_seq.shape)} prev_act={tuple(prev_act_seq.shape)}"
            )
        if obs_seq.shape[:2] != prev_act_seq.shape[:2]:
            raise ValueError(f"obs/prev_act sequence mismatch: {tuple(obs_seq.shape)} vs {tuple(prev_act_seq.shape)}")
        if obs_seq.shape[-1] != critic.obs_dim:
            raise ValueError(f"Expected obs dim {critic.obs_dim}, got {obs_seq.shape[-1]}")
        if prev_act_seq.shape[-1] != critic.act_dim:
            raise ValueError(f"Expected action dim {critic.act_dim}, got {prev_act_seq.shape[-1]}")

        batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
        done_list_seq = self._validate_done_list_seq([done_seq] * self.num_agents, batch_size, seq_len) if done_seq is not None else None
        diagnostics = self._done_segmentation_diagnostics(done_list_seq, seq_len=seq_len)

        if done_list_seq is None:
            hidden = critic.init_hidden(batch_size=batch_size, device=self.device, dtype=obs_seq.dtype)
            values, hidden_out, local_features = critic(obs_seq, prev_act_seq, hidden)
            return values, hidden_out, local_features, diagnostics

        if batch_size != 1:
            raise NotImplementedError("Done-boundary critic segmentation currently supports batch_size=1")
        bounds = self._segment_bounds(done_list_seq[0])
        value_parts: List[torch.Tensor] = []
        feature_parts: List[torch.Tensor] = []
        hidden_out: Optional[torch.Tensor] = None
        for start, end in bounds:
            hidden = critic.init_hidden(batch_size=batch_size, device=self.device, dtype=obs_seq.dtype)
            values_part, hidden_out, local_features = critic(
                obs_seq[:, start:end, :],
                prev_act_seq[:, start:end, :],
                hidden,
            )
            value_parts.append(values_part)
            feature_parts.append(local_features)
        if not value_parts:
            raise ValueError("No critic segments were produced from the rollout sequence")
        values = torch.cat(value_parts, dim=1)
        local_features = torch.cat(feature_parts, dim=1)
        assert hidden_out is not None
        return values, hidden_out, local_features, diagnostics

    @torch.no_grad()
    def init_hidden(self, batch_size: int = 1) -> List[torch.Tensor]:
        return [critic.init_hidden(batch_size=batch_size, device=self.device) for critic in self.critics]

    @torch.no_grad()
    def act(
        self,
        obs_list: List[np.ndarray],
        prev_act_list: List[np.ndarray],
        critic_hidden_list: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[np.ndarray], List[float], List[float], List[torch.Tensor]]:
        obs_t_list = self._tensor_list(obs_list)
        prev_act_t_list = self._tensor_list(prev_act_list)
        if critic_hidden_list is None:
            critic_hidden_list = self.init_hidden(batch_size=1)
        actions_t, logps_t, values_t, new_hidden_list = [], [], [], []
        for i, actor in enumerate(self.actors):
            action_i, logp_i = actor.act(obs_t_list[i], deterministic=deterministic)
            value_i, hidden_i, _ = self.critics[i](obs_t_list[i], prev_act_t_list[i], critic_hidden_list[i])
            actions_t.append(action_i)
            logps_t.append(float(logp_i.detach().cpu().item()))
            values_t.append(float(value_i.reshape(-1)[-1].detach().cpu().item()))
            new_hidden_list.append(hidden_i.detach())
        actions_np = [a.detach().cpu().numpy().astype(np.float32) for a in actions_t]
        return actions_np, logps_t, values_t, new_hidden_list

    @torch.no_grad()
    def value(
        self,
        obs_list: List[np.ndarray],
        prev_act_list: List[np.ndarray],
        critic_hidden_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[float], List[torch.Tensor]]:
        obs_t_list = self._tensor_list(obs_list)
        prev_act_t_list = self._tensor_list(prev_act_list)
        if critic_hidden_list is None:
            critic_hidden_list = self.init_hidden(batch_size=1)
        values_t, new_hidden_list = [], []
        for i in range(self.num_agents):
            value_i, hidden_i, _ = self.critics[i](obs_t_list[i], prev_act_t_list[i], critic_hidden_list[i])
            values_t.append(float(value_i.reshape(-1)[-1].detach().cpu().item()))
            new_hidden_list.append(hidden_i.detach())
        return values_t, new_hidden_list

    def critic_update_agent(
        self,
        agent_idx: int,
        obs_seq: torch.Tensor,
        prev_act_seq: torch.Tensor,
        ret_seq: torch.Tensor,
        done_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        critic = self.critics[agent_idx]
        opt = self.critic_opts[agent_idx]

        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
        if prev_act_seq.dim() == 2:
            prev_act_seq = prev_act_seq.unsqueeze(0)
        if ret_seq.dim() == 1:
            ret_seq = ret_seq.unsqueeze(0)
        if done_seq is not None and done_seq.dim() == 1:
            done_seq = done_seq.unsqueeze(0)
        obs_seq = obs_seq.to(self.device, dtype=torch.float32)
        prev_act_seq = prev_act_seq.to(self.device, dtype=torch.float32)
        ret_seq = ret_seq.to(self.device, dtype=torch.float32)
        done_seq = None if done_seq is None else done_seq.to(self.device, dtype=torch.float32)
        if ret_seq.dim() != 2:
            raise ValueError(f"ret_seq must have shape (B, T), got {tuple(ret_seq.shape)}")
        if obs_seq.shape[:2] != prev_act_seq.shape[:2] or obs_seq.shape[:2] != ret_seq.shape[:2]:
            raise ValueError(
                f"critic rollout sequence mismatch: obs={tuple(obs_seq.shape)} prev_act={tuple(prev_act_seq.shape)} ret={tuple(ret_seq.shape)}"
            )
        if done_seq is not None and tuple(done_seq.shape) != tuple(ret_seq.shape):
            raise ValueError(f"done_seq shape mismatch: expected {tuple(ret_seq.shape)}, got {tuple(done_seq.shape)}")

        before = [p.detach().clone() for p in critic.parameters()]
        epoch_losses: List[float] = []
        epoch_grad_norms: List[float] = []
        epoch_pred_means: List[float] = []
        diagnostics: Dict[str, float] = {}
        n_epochs = max(1, int(self.hp.critic_epochs))
        for _ in range(n_epochs):
            opt.zero_grad(set_to_none=True)
            pred_values, hidden_out, local_features, diagnostics = self._critic_values_from_sequence(
                agent_idx,
                obs_seq,
                prev_act_seq,
                done_seq,
            )
            pred_values = pred_values.reshape_as(ret_seq)
            value_loss = ((pred_values - ret_seq) ** 2).mean()
            value_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), self.hp.max_grad_norm)
            opt.step()
            epoch_losses.append(float(value_loss.detach().cpu().item()))
            epoch_grad_norms.append(float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm))
            epoch_pred_means.append(float(pred_values.detach().mean().cpu().item()))

        after = [p.detach().clone() for p in critic.parameters()]
        return {
            "loss": float(epoch_losses[-1]),
            "loss_mean": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "grad_norm": float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0,
            "pred_value_mean": float(np.mean(epoch_pred_means)) if epoch_pred_means else 0.0,
            "critic_epochs_used": int(n_epochs),
            "param_delta": self._parameter_delta_norm(before, critic),
            "critic_gru_reset_enabled": bool(diagnostics.get("critic_gru_reset_enabled", 0.0)),
            "critic_gru_done_resets": float(diagnostics.get("critic_gru_done_resets", 0.0)),
            "hidden_resets": float(diagnostics.get("hidden_resets", 0.0)),
            "num_segments": float(diagnostics.get("num_segments", 0.0)),
            "mean_segment_len": float(diagnostics.get("mean_segment_len", 0.0)),
            "value_loss_trace": epoch_losses,
        }

    def actor_update_agent(
        self,
        agent_idx: int,
        obs_np: np.ndarray,
        act_np: np.ndarray,
        old_logp_np: np.ndarray,
        adv_np: np.ndarray,
    ) -> Dict[str, Any]:
        actor = self.actors[agent_idx]
        opt = self.actor_opts[agent_idx]
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act_np, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(old_logp_np, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device).detach()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        if old_logp.dim() == 0:
            old_logp = old_logp.unsqueeze(0)
        if adv.dim() == 0:
            adv = adv.unsqueeze(0)
        if obs.shape[0] != act.shape[0] or obs.shape[0] != old_logp.shape[0] or obs.shape[0] != adv.shape[0]:
            raise ValueError(
                f"actor rollout sequence mismatch: obs={tuple(obs.shape)} act={tuple(act.shape)} old_logp={tuple(old_logp.shape)} adv={tuple(adv.shape)}"
            )

        before = [p.detach().clone() for p in actor.parameters()]
        epoch_logs: List[Dict[str, Any]] = []
        n_epochs = max(1, int(self.hp.actor_epochs))
        target_kl = None if self.hp.target_kl is None else float(self.hp.target_kl)
        for _ in range(n_epochs):
            opt.zero_grad(set_to_none=True)
            new_logp, entropy, _, _ = actor.evaluate_actions(obs, act)
            log_ratio = torch.clamp(new_logp - old_logp, -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.hp.clip_range, 1.0 + self.hp.clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = entropy.mean()
            total_loss = policy_loss - self.hp.ent_coef * entropy_loss
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), self.hp.max_grad_norm)
            opt.step()
            approx_kl = float((old_logp - new_logp).mean().detach().cpu().item())
            epoch_logs.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "entropy": float(entropy_loss.detach().cpu().item()),
                    "approx_kl": approx_kl,
                    "clipfrac": float((torch.abs(ratio - 1.0) > self.hp.clip_range).float().mean().detach().cpu().item()),
                    "ratio_std": float(ratio.std(unbiased=False).detach().cpu().item()),
                    "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                }
            )
            if target_kl is not None and approx_kl > target_kl:
                break

        with torch.no_grad():
            post_logp, _, _, _ = actor.evaluate_actions(obs, act)
        after = [p.detach().clone() for p in actor.parameters()]
        epoch0 = epoch_logs[0] if epoch_logs else {}
        last = epoch_logs[-1] if epoch_logs else {}
        return {
            "policy_loss": float(last.get("policy_loss", 0.0)),
            "total_loss": float(last.get("total_loss", 0.0)),
            "entropy": float(last.get("entropy", 0.0)),
            "approx_kl": float(last.get("approx_kl", 0.0)),
            "approx_kl_post": float((old_logp - post_logp).mean().detach().cpu().item()),
            "clipfrac": float(last.get("clipfrac", 0.0)),
            "ratio_std": float(last.get("ratio_std", 0.0)),
            "grad_norm": float(np.mean([log.get("grad_norm", 0.0) for log in epoch_logs])) if epoch_logs else 0.0,
            "actor_epochs_used": int(len(epoch_logs)),
            "approx_kl_epoch0": float(epoch0.get("approx_kl", 0.0)),
            "approx_kl_last": float(last.get("approx_kl", 0.0)),
            "param_delta": self._parameter_delta_norm(before, actor),
            "epoch_logs": epoch_logs,
        }

    def update(
        self,
        client_rollouts: Optional[List[Dict[str, np.ndarray]]] = None,
        global_step: Optional[int] = None,
        *,
        agent_rollouts: Optional[List[Dict[str, np.ndarray]]] = None,
        last_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        if agent_rollouts is not None:
            client_rollouts = agent_rollouts
        if client_rollouts is None:
            raise ValueError("update requires client_rollouts or agent_rollouts")
        if last_values is None:
            raise ValueError("update requires last_values")
        if len(client_rollouts) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} client rollouts, got {len(client_rollouts)}")
        if len(last_values) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} last values, got {len(last_values)}")

        required = {"obs", "act", "prev_act", "logp", "val", "rew", "done"}
        T = None
        done_ref: Optional[np.ndarray] = None
        obs_np_list: List[np.ndarray] = []
        act_np_list: List[np.ndarray] = []
        prev_act_np_list: List[np.ndarray] = []
        old_logp_np_list: List[np.ndarray] = []
        adv_np_list: List[np.ndarray] = []
        ret_np_list: List[np.ndarray] = []
        adv_mean_raw: List[float] = []
        adv_std_raw: List[float] = []
        done_np_list: List[np.ndarray] = []

        for i, rollout in enumerate(client_rollouts):
            missing = sorted(required - set(rollout))
            if missing:
                raise KeyError(f"client_rollouts[{i}] missing keys {missing}")
            obs = np.asarray(rollout["obs"], dtype=np.float32)
            act = np.asarray(rollout["act"], dtype=np.float32)
            prev_act = np.asarray(rollout["prev_act"], dtype=np.float32)
            old_logp = np.asarray(rollout["logp"], dtype=np.float32)
            val = self._as_float_array(rollout["val"], f"client_rollouts[{i}]['val']")
            rew = self._as_float_array(rollout["rew"], f"client_rollouts[{i}]['rew']")
            done = self._as_float_array(rollout["done"], f"client_rollouts[{i}]['done']")
            if obs.shape[0] != act.shape[0] or obs.shape[0] != prev_act.shape[0]:
                raise ValueError(f"client_rollouts[{i}] rollout horizon mismatch")
            if T is None:
                T = obs.shape[0]
            elif obs.shape[0] != T:
                raise ValueError("All client rollouts must share the same horizon length")
            if val.shape[0] != T or rew.shape[0] != T or done.shape[0] != T:
                raise ValueError(f"client_rollouts[{i}] sequence length mismatch")
            if done_ref is None:
                done_ref = done
            elif not np.allclose(done_ref, done):
                raise ValueError("All agent rollouts must share the same done sequence in this environment")

            adv_i, ret_i = self._compute_gae(
                rew=rew,
                val=val,
                done=done,
                last_value=float(last_values[i]),
                gamma=self.hp.gamma,
                gae_lambda=self.hp.gae_lambda,
            )
            adv_mean_raw.append(float(adv_i.mean()))
            adv_std_raw.append(float(adv_i.std()))
            adv_norm = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

            obs_np_list.append(obs)
            act_np_list.append(act)
            prev_act_np_list.append(prev_act)
            old_logp_np_list.append(old_logp)
            adv_np_list.append(adv_norm)
            ret_np_list.append(ret_i)
            done_np_list.append(done)

        assert T is not None
        obs_seq_t = [torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device) for x in obs_np_list]
        prev_act_seq_t = [torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device) for x in prev_act_np_list]
        ret_seq_t = [torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device) for x in ret_np_list]
        done_seq_t = [torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device) for x in done_np_list]

        critic_logs: List[Dict[str, Any]] = []
        for i in range(self.num_agents):
            critic_logs.append(
                self.critic_update_agent(
                    i,
                    obs_seq_t[i],
                    prev_act_seq_t[i],
                    ret_seq_t[i],
                    done_seq_t[i],
                )
            )

        actor_logs: List[Dict[str, Any]] = []
        for i in range(self.num_agents):
            actor_logs.append(
                self.actor_update_agent(
                    i,
                    obs_np_list[i],
                    act_np_list[i],
                    old_logp_np_list[i],
                    adv_np_list[i],
                )
            )

        self.last_critic_diagnostics = critic_logs
        self.last_actor_diagnostics = actor_logs
        self.round_idx += 1

        done_count = float(np.sum(done_np_list[0] > 0.5))
        critic_loss_mean = self._mean([log["loss"] for log in critic_logs])
        actor_total_loss_mean = self._mean([log["total_loss"] for log in actor_logs])
        actor_policy_loss_mean = self._mean([log["policy_loss"] for log in actor_logs])
        entropy_mean = self._mean([log["entropy"] for log in actor_logs])
        approx_kl_mean = self._mean([log["approx_kl"] for log in actor_logs])
        approx_kl_post_mean = self._mean([log["approx_kl_post"] for log in actor_logs])
        actor_epochs_used_mean = self._mean([float(log["actor_epochs_used"]) for log in actor_logs])
        critic_epochs_used_mean = self._mean([float(log["critic_epochs_used"]) for log in critic_logs])
        actor_grad_norm_mean = self._mean([log["grad_norm"] for log in actor_logs])
        critic_grad_norm_mean = self._mean([log["grad_norm"] for log in critic_logs])

        logs: Dict[str, Any] = {
            "loss/actor": actor_total_loss_mean,
            "loss/actor_policy": actor_policy_loss_mean,
            "loss/critic": critic_loss_mean,
            "loss/entropy": entropy_mean,
            "train/approx_kl": approx_kl_mean,
            "train/approx_kl_post": approx_kl_post_mean,
            "train/actor_epochs_used": actor_epochs_used_mean,
            "train/critic_epochs_used": critic_epochs_used_mean,
            "train/approx_kl_epoch0": self._mean([log["approx_kl_epoch0"] for log in actor_logs]),
            "train/approx_kl_last": self._mean([log["approx_kl_last"] for log in actor_logs]),
            "train/clipfrac": self._mean([log["clipfrac"] for log in actor_logs]),
            "train/actor_grad_norm": actor_grad_norm_mean,
            "train/critic_grad_norm": critic_grad_norm_mean,
            "rollout/length": float(T),
            "rollout/episodes_per_update": done_count,
            "rollout/num_episodes": done_count,
            "debug/no_fedavg": 1.0,
            "debug/no_server_critic": 1.0,
            "debug/no_server_gat": 1.0,
            "debug/no_privacy": 1.0,
            "debug/adv_mean_raw": self._mean(adv_mean_raw),
            "debug/adv_std_raw": self._mean(adv_std_raw),
            "debug/adv_std_normed": self._mean([float(np.std(adv)) for adv in adv_np_list]),
            "debug/critic_done_resets": self._mean([log["critic_gru_done_resets"] for log in critic_logs]),
            "debug/hidden_resets": self._mean([log["hidden_resets"] for log in critic_logs]),
            "debug/critic_gru_reset_enabled": float(any(log["critic_gru_reset_enabled"] for log in critic_logs)),
            "debug/mean_segment_len": self._mean([log["mean_segment_len"] for log in critic_logs]),
        }

        for i in range(self.num_agents):
            logs[f"agent_{i}/adv_mean_raw"] = adv_mean_raw[i]
            logs[f"agent_{i}/adv_std_raw"] = adv_std_raw[i]
            logs[f"agent_{i}/adv_std_normed"] = float(np.std(adv_np_list[i]))
            logs[f"agent_{i}/loss_actor_total"] = actor_logs[i]["total_loss"]
            logs[f"agent_{i}/loss_actor_policy"] = actor_logs[i]["policy_loss"]
            logs[f"agent_{i}/loss_critic"] = critic_logs[i]["loss"]
            logs[f"agent_{i}/entropy"] = actor_logs[i]["entropy"]
            logs[f"agent_{i}/approx_kl"] = actor_logs[i]["approx_kl"]
            logs[f"agent_{i}/approx_kl_post"] = actor_logs[i]["approx_kl_post"]
            logs[f"agent_{i}/actor_epochs_used"] = float(actor_logs[i]["actor_epochs_used"])
            logs[f"agent_{i}/critic_epochs_used"] = float(critic_logs[i]["critic_epochs_used"])
            logs[f"agent_{i}/clipfrac"] = actor_logs[i]["clipfrac"]
            logs[f"agent_{i}/ratio_std"] = actor_logs[i]["ratio_std"]
            logs[f"agent_{i}/debug/actor_grad_norm"] = actor_logs[i]["grad_norm"]
            logs[f"agent_{i}/debug/critic_grad_norm"] = critic_logs[i]["grad_norm"]
            logs[f"agent_{i}/debug/actor_param_delta"] = actor_logs[i]["param_delta"]
            logs[f"agent_{i}/debug/critic_param_delta"] = critic_logs[i]["param_delta"]
            logs[f"agent_{i}/debug/critic_done_resets"] = critic_logs[i]["critic_gru_done_resets"]
            logs[f"agent_{i}/debug/hidden_resets"] = critic_logs[i]["hidden_resets"]

        if self.writer is not None and global_step is not None:
            for key, value in logs.items():
                if isinstance(value, (float, int, np.floating, np.integer)):
                    self.writer.add_scalar(key, float(value), global_step)
        return logs

    def parameter_id_sets(self) -> Dict[str, set]:
        out: Dict[str, set] = {}
        for i, actor in enumerate(self.actors):
            out[f"agent_{i}_actor"] = {id(p) for p in actor.parameters()}
        for i, critic in enumerate(self.critics):
            out[f"agent_{i}_critic"] = {id(p) for p in critic.parameters()}
        return out

    def optimizer_parameter_sets(self) -> Dict[str, set]:
        out: Dict[str, set] = {}
        for i, opt in enumerate(self.actor_opts):
            out[f"agent_{i}_actor_opt"] = {id(p) for group in opt.param_groups for p in group["params"]}
        for i, opt in enumerate(self.critic_opts):
            out[f"agent_{i}_critic_opt"] = {id(p) for group in opt.param_groups for p in group["params"]}
        return out

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(save_dir, f"agent_{i}_actor.pt"))
        for i, critic in enumerate(self.critics):
            torch.save(critic.state_dict(), os.path.join(save_dir, f"agent_{i}_critic.pt"))


__all__ = ["IPPOHyperParams", "IPPOTrainer", "TRAINER_VERSION"]
