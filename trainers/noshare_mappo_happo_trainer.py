from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.centralized_recurrent_critic import CentralizedRecurrentCritic
from models.fl_split_actor import SplitLocalActor


TRAINER_VERSION = "noshare_mappo_happo_v1"
SUPPORTED_MODES = ("mappo", "happo")


@dataclass
class NoShareMARLHyperParams:
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


class NoShareMAPPOHAPPOTrainer(nn.Module):
    """No-parameter-sharing MAPPO/HAPPO with one centralized GRU critic."""

    def __init__(
        self,
        actors: Sequence[SplitLocalActor],
        centralized_critic: CentralizedRecurrentCritic,
        mode: str,
        device: str = "cpu",
        hp: Optional[NoShareMARLHyperParams] = None,
        writer: Optional[Any] = None,
        permutation_seed: int = 0,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}, got {mode!r}")
        if len(actors) != 3:
            raise ValueError(f"Exactly three independent actors are required, got {len(actors)}")
        self.mode = mode
        self.device = torch.device(device)
        self.actors = nn.ModuleList(list(actors))
        self.centralized_critic = centralized_critic
        self.hp = hp if hp is not None else NoShareMARLHyperParams()
        self.writer = writer
        self.num_agents = len(self.actors)
        self.to(self.device)
        self.actor_opts = [optim.Adam(actor.parameters(), lr=self.hp.actor_lr) for actor in self.actors]
        self.critic_opt = optim.Adam(self.centralized_critic.parameters(), lr=self.hp.critic_lr)
        self.permutation_seed = int(permutation_seed)
        self.permutation_rng = np.random.default_rng(self.permutation_seed)
        self.round_idx = 0
        self.last_update_diagnostics: Dict[str, Any] = {}

    @classmethod
    def build_default(
        cls,
        obs_dims: Sequence[int],
        act_dims: Sequence[int],
        mode: str,
        device: str = "cpu",
        hp: Optional[NoShareMARLHyperParams] = None,
        writer: Optional[Any] = None,
        seed: int = 0,
        permutation_seed: Optional[int] = None,
    ) -> "NoShareMAPPOHAPPOTrainer":
        if list(map(int, obs_dims)) != [7, 10, 8]:
            raise ValueError(f"Verified local observation dimensions are [7, 10, 8], got {list(obs_dims)}")
        if list(map(int, act_dims)) != [1, 1, 1]:
            raise ValueError(f"Verified local action dimensions are [1, 1, 1], got {list(act_dims)}")
        torch.manual_seed(int(seed))
        actors: List[SplitLocalActor] = []
        for obs_dim, act_dim in zip(obs_dims, act_dims):
            actors.append(
                SplitLocalActor(
                    obs_dim=int(obs_dim),
                    act_dim=int(act_dim),
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
            )
        critic = CentralizedRecurrentCritic(joint_input_dim=sum(obs_dims) + sum(act_dims), hidden_dim=64)
        return cls(
            actors=actors,
            centralized_critic=critic,
            mode=mode,
            device=device,
            hp=hp,
            writer=writer,
            permutation_seed=int(seed if permutation_seed is None else permutation_seed),
        )

    @staticmethod
    def count_parameters(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    @staticmethod
    def _parameter_delta_norm(before: Sequence[torch.Tensor], module: nn.Module) -> float:
        squared = 0.0
        for old, new in zip(before, module.parameters()):
            squared += float((new.detach() - old).pow(2).sum().cpu().item())
        return float(squared ** 0.5)

    @staticmethod
    def _as_vector(value: Any, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.ndim != 1:
            raise ValueError(f"{name} must have shape (T,), got {result.shape}")
        return result

    @staticmethod
    def _hash_array(value: np.ndarray) -> str:
        stable = np.ascontiguousarray(value, dtype=np.float32)
        return hashlib.sha256(stable.tobytes()).hexdigest()

    @staticmethod
    def build_joint_input(
        obs_list: Sequence[Any],
        prev_action_list: Sequence[Any],
    ) -> np.ndarray:
        """Concatenate ``obs0,obs1,obs2,prev_action0,prev_action1,prev_action2``."""
        if len(obs_list) != 3 or len(prev_action_list) != 3:
            raise ValueError("Joint critic input requires three observations and three previous actions")
        arrays = [np.asarray(value, dtype=np.float32) for value in list(obs_list) + list(prev_action_list)]
        leading = arrays[0].shape[:-1]
        for index, array in enumerate(arrays):
            if array.ndim < 1:
                raise ValueError(f"joint input component {index} must have at least one dimension")
            if array.shape[:-1] != leading:
                raise ValueError(
                    f"joint input leading-shape mismatch: expected {leading}, got {array.shape[:-1]} at component {index}"
                )
        result = np.concatenate(arrays, axis=-1).astype(np.float32, copy=False)
        if result.shape[-1] != 28:
            raise ValueError(f"Joint critic input must be dimension 28, got {result.shape[-1]}")
        return result

    @staticmethod
    def compute_gae(
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        reward = np.asarray(reward, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)
        if reward.ndim != 1 or value.shape != reward.shape or done.shape != reward.shape:
            raise ValueError("reward, value, and done must share shape (T,)")
        next_values = np.empty_like(value)
        next_values[:-1] = value[1:]
        next_values[-1] = np.float32(last_value)
        next_values *= 1.0 - done
        delta = reward + float(gamma) * next_values - value
        advantage = np.zeros_like(reward)
        running = 0.0
        for t in range(reward.shape[0] - 1, -1, -1):
            running = float(delta[t]) + float(gamma) * float(gae_lambda) * (1.0 - float(done[t])) * running
            advantage[t] = running
        return advantage, advantage + value

    @staticmethod
    def segment_bounds(done: torch.Tensor) -> List[Tuple[int, int]]:
        flat = done.reshape(-1)
        bounds: List[Tuple[int, int]] = []
        start = 0
        for index in (flat > 0.5).nonzero(as_tuple=False).flatten().detach().cpu().tolist():
            end = int(index) + 1
            if end > start:
                bounds.append((start, end))
            start = end
        if start < int(flat.numel()):
            bounds.append((start, int(flat.numel())))
        return bounds

    @torch.no_grad()
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return self.centralized_critic.init_hidden(batch_size=batch_size, device=self.device)

    def _joint_tensor(self, obs_list: Sequence[Any], prev_action_list: Sequence[Any]) -> torch.Tensor:
        return torch.as_tensor(
            self.build_joint_input(obs_list, prev_action_list),
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def act(
        self,
        obs_list: Sequence[np.ndarray],
        prev_action_list: Sequence[np.ndarray],
        critic_hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[List[np.ndarray], List[float], float, torch.Tensor]:
        if critic_hidden is None:
            critic_hidden = self.init_hidden(batch_size=1)
        joint_input = self._joint_tensor(obs_list, prev_action_list)
        values, hidden_out, _ = self.centralized_critic(joint_input, critic_hidden)
        actions: List[np.ndarray] = []
        logps: List[float] = []
        for actor, obs in zip(self.actors, obs_list):
            action, logp = actor.act(torch.as_tensor(obs, dtype=torch.float32, device=self.device), deterministic)
            actions.append(action.detach().cpu().numpy().astype(np.float32))
            logps.append(float(logp.detach().cpu().item()))
        return actions, logps, float(values.reshape(-1)[-1].cpu().item()), hidden_out.detach()

    @torch.no_grad()
    def value(
        self,
        obs_list: Sequence[np.ndarray],
        prev_action_list: Sequence[np.ndarray],
        critic_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        if critic_hidden is None:
            critic_hidden = self.init_hidden(batch_size=1)
        values, hidden_out, _ = self.centralized_critic(
            self._joint_tensor(obs_list, prev_action_list), critic_hidden
        )
        return float(values.reshape(-1)[-1].cpu().item()), hidden_out.detach()

    def centralized_values_from_sequence(
        self,
        joint_input: torch.Tensor,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if joint_input.dim() == 2:
            joint_input = joint_input.unsqueeze(0)
        if joint_input.dim() != 3 or joint_input.shape[-1] != 28:
            raise ValueError(f"joint_input must have shape (B,T,28), got {tuple(joint_input.shape)}")
        if joint_input.shape[0] != 1:
            raise NotImplementedError("Recurrent rollout updates currently support batch size 1")
        sequence_length = int(joint_input.shape[1])
        if done is None:
            bounds = [(0, sequence_length)]
            done_resets = 0
        else:
            if done.dim() == 2:
                done = done.squeeze(0)
            if tuple(done.shape) != (sequence_length,):
                raise ValueError(f"done must have shape ({sequence_length},), got {tuple(done.shape)}")
            bounds = self.segment_bounds(done)
            done_resets = int((done[:-1] > 0.5).sum().detach().cpu().item()) if sequence_length > 1 else 0
        parts: List[torch.Tensor] = []
        for start, end in bounds:
            hidden = self.centralized_critic.init_hidden(1, self.device, joint_input.dtype)
            values, _, _ = self.centralized_critic(joint_input[:, start:end, :], hidden)
            parts.append(values.reshape(1, -1))
        if not parts:
            raise ValueError("Cannot evaluate an empty centralized critic sequence")
        return torch.cat(parts, dim=1), {
            "num_segments": len(bounds),
            "hidden_resets": len(bounds),
            "critic_gru_done_resets": done_resets,
            "segment_bounds": bounds,
        }

    def actor_update_agent(
        self,
        agent_idx: int,
        obs_np: np.ndarray,
        act_np: np.ndarray,
        collection_old_logp_np: np.ndarray,
        effective_advantage_np: np.ndarray,
    ) -> Dict[str, Any]:
        actor = self.actors[agent_idx]
        optimizer = self.actor_opts[agent_idx]
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(act_np, dtype=torch.float32, device=self.device)
        collection_old_logp = torch.as_tensor(
            collection_old_logp_np, dtype=torch.float32, device=self.device
        ).reshape(-1).detach()
        advantage = torch.as_tensor(
            effective_advantage_np, dtype=torch.float32, device=self.device
        ).reshape(-1).detach()
        if obs.shape[0] != action.shape[0] or obs.shape[0] != advantage.shape[0]:
            raise ValueError("Actor rollout inputs must share the same time dimension")
        before = [parameter.detach().clone() for parameter in actor.parameters()]
        epochs: List[Dict[str, float]] = []
        for epoch in range(max(1, int(self.hp.actor_epochs))):
            optimizer.zero_grad(set_to_none=True)
            new_logp, entropy, _, _ = actor.evaluate_actions(obs, action)
            log_ratio = torch.clamp(new_logp - collection_old_logp, -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            surrogate_a = ratio * advantage
            surrogate_b = torch.clamp(
                ratio, 1.0 - self.hp.clip_range, 1.0 + self.hp.clip_range
            ) * advantage
            policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss - self.hp.ent_coef * entropy_mean
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), self.hp.max_grad_norm)
            optimizer.step()
            approx_kl = float((collection_old_logp - new_logp).mean().detach().cpu().item())
            epochs.append(
                {
                    "epoch": float(epoch),
                    "loss_actor": float(loss.detach().cpu().item()),
                    "loss_policy": float(policy_loss.detach().cpu().item()),
                    "entropy": float(entropy_mean.detach().cpu().item()),
                    "approx_kl": approx_kl,
                    "clipfrac": float(
                        (torch.abs(ratio - 1.0) > self.hp.clip_range).float().mean().detach().cpu().item()
                    ),
                    "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                }
            )
            if self.hp.target_kl is not None and approx_kl > float(self.hp.target_kl):
                break
        with torch.no_grad():
            final_logp, _, _, _ = actor.evaluate_actions(obs, action)
            # HAPPO's multiplicative factor is defined against the immutable
            # rollout-collection policy, not an intermediate actor snapshot.
            final_ratio = torch.exp(final_logp - collection_old_logp).detach()
        last = epochs[-1]
        return {
            **last,
            "actor_epochs_used": len(epochs),
            "param_delta": self._parameter_delta_norm(before, actor),
            "final_logp": final_logp.detach(),
            "final_ratio": final_ratio,
            "ratio_denominator": "rollout_collection_old_logp",
            "effective_advantage_requires_grad": bool(advantage.requires_grad),
            "epoch_logs": epochs,
        }

    def centralized_critic_update(
        self,
        joint_input_np: np.ndarray,
        return_np: np.ndarray,
        done_np: np.ndarray,
    ) -> Dict[str, Any]:
        joint_input = torch.as_tensor(joint_input_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        target = torch.as_tensor(return_np, dtype=torch.float32, device=self.device).reshape(1, -1).detach()
        done = torch.as_tensor(done_np, dtype=torch.float32, device=self.device).reshape(-1)
        before = [parameter.detach().clone() for parameter in self.centralized_critic.parameters()]
        epochs: List[Dict[str, float]] = []
        reset_diagnostics: Dict[str, Any] = {}
        for epoch in range(max(1, int(self.hp.critic_epochs))):
            self.critic_opt.zero_grad(set_to_none=True)
            prediction, reset_diagnostics = self.centralized_values_from_sequence(joint_input, done)
            loss = ((prediction - target) ** 2).mean()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.centralized_critic.parameters(), self.hp.max_grad_norm
            )
            self.critic_opt.step()
            epochs.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach().cpu().item()),
                    "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                }
            )
        return {
            "loss": epochs[-1]["loss"],
            "grad_norm": float(np.mean([entry["grad_norm"] for entry in epochs])),
            "critic_epochs_used": len(epochs),
            "param_delta": self._parameter_delta_norm(before, self.centralized_critic),
            "epoch_logs": epochs,
            **reset_diagnostics,
        }

    def _validate_rollout(self, rollout: Dict[str, Any]) -> Dict[str, Any]:
        required = {"joint_input", "value", "reward", "done"}
        for agent_idx in range(self.num_agents):
            required.update({f"obs_{agent_idx}", f"action_{agent_idx}", f"old_logp_{agent_idx}"})
        missing = sorted(required - set(rollout))
        if missing:
            raise KeyError(f"rollout is missing keys: {missing}")
        parsed: Dict[str, Any] = {
            "joint_input": np.asarray(rollout["joint_input"], dtype=np.float32),
            "value": self._as_vector(rollout["value"], "value"),
            "reward": self._as_vector(rollout["reward"], "reward"),
            "done": self._as_vector(rollout["done"], "done"),
        }
        if parsed["joint_input"].ndim != 2 or parsed["joint_input"].shape[1] != 28:
            raise ValueError(f"joint_input must have shape (T,28), got {parsed['joint_input'].shape}")
        horizon = parsed["joint_input"].shape[0]
        for key in ("value", "reward", "done"):
            if parsed[key].shape[0] != horizon:
                raise ValueError(f"{key} horizon does not match joint_input")
        for agent_idx in range(self.num_agents):
            obs = np.asarray(rollout[f"obs_{agent_idx}"], dtype=np.float32)
            action = np.asarray(rollout[f"action_{agent_idx}"], dtype=np.float32)
            old_logp = self._as_vector(rollout[f"old_logp_{agent_idx}"], f"old_logp_{agent_idx}")
            if obs.shape[0] != horizon or action.shape[0] != horizon or old_logp.shape[0] != horizon:
                raise ValueError(f"agent {agent_idx} rollout horizon mismatch")
            parsed[f"obs_{agent_idx}"] = obs
            parsed[f"action_{agent_idx}"] = action
            parsed[f"old_logp_{agent_idx}"] = old_logp
        return parsed

    def update(
        self,
        rollout: Dict[str, Any],
        last_value: float,
        global_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        data = self._validate_rollout(rollout)
        raw_advantage, returns = self.compute_gae(
            data["reward"], data["value"], data["done"], last_value,
            self.hp.gamma, self.hp.gae_lambda,
        )
        advantage = ((raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)).astype(np.float32)
        base_hash = self._hash_array(advantage)
        phase_sequence: List[str] = []
        actor_logs_by_index: Dict[int, Dict[str, Any]] = {}
        factor_trace: List[Dict[str, Any]] = []

        if self.mode == "mappo":
            update_order = [0, 1, 2]
            for agent_idx in update_order:
                phase_sequence.append(f"actor_{agent_idx}_all_epochs")
                actor_logs_by_index[agent_idx] = self.actor_update_agent(
                    agent_idx,
                    data[f"obs_{agent_idx}"], data[f"action_{agent_idx}"],
                    data[f"old_logp_{agent_idx}"], advantage,
                )
        else:
            update_order = self.permutation_rng.permutation(self.num_agents).astype(int).tolist()
            factor = torch.ones(advantage.shape[0], dtype=torch.float32, device=self.device).detach()
            base_advantage = torch.as_tensor(advantage, dtype=torch.float32, device=self.device).detach()
            for position, agent_idx in enumerate(update_order):
                factor_before = factor.detach().clone()
                effective_advantage = (factor_before * base_advantage).detach()
                phase_sequence.append(f"actor_{agent_idx}_all_epochs")
                actor_log = self.actor_update_agent(
                    agent_idx,
                    data[f"obs_{agent_idx}"], data[f"action_{agent_idx}"],
                    data[f"old_logp_{agent_idx}"], effective_advantage.cpu().numpy(),
                )
                actor_logs_by_index[agent_idx] = actor_log
                factor = (factor_before * actor_log["final_ratio"].detach()).detach()
                factor_trace.append(
                    {
                        "position": position,
                        "agent_idx": agent_idx,
                        "actor_epochs_used": actor_log["actor_epochs_used"],
                        "factor_before_mean": float(factor_before.mean().cpu().item()),
                        "ratio_mean": float(actor_log["final_ratio"].mean().cpu().item()),
                        "factor_after_mean": float(factor.mean().cpu().item()),
                        "factor_after_std": float(factor.std(unbiased=False).cpu().item()),
                        "factor_after_min": float(factor.min().cpu().item()),
                        "factor_after_max": float(factor.max().cpu().item()),
                        "factor_requires_grad": bool(factor.requires_grad),
                        "factor_updated_after_actor_epochs": True,
                        "ratio_denominator": actor_log["ratio_denominator"],
                    }
                )

        phase_sequence.append("centralized_critic_all_epochs")
        critic_log = self.centralized_critic_update(
            data["joint_input"], returns, data["done"]
        )
        actor_logs = [actor_logs_by_index[index] for index in range(self.num_agents)]
        factor_final = factor if self.mode == "happo" else torch.ones(
            advantage.shape[0], dtype=torch.float32, device=self.device
        )
        logs: Dict[str, Any] = {
            "loss/actor": float(np.mean([entry["loss_actor"] for entry in actor_logs])),
            "loss/critic": critic_log["loss"],
            "loss/entropy": float(np.mean([entry["entropy"] for entry in actor_logs])),
            "train/approx_kl": float(np.mean([entry["approx_kl"] for entry in actor_logs])),
            "train/clipfrac": float(np.mean([entry["clipfrac"] for entry in actor_logs])),
            "train/actor_epochs_used": float(np.mean([entry["actor_epochs_used"] for entry in actor_logs])),
            "train/critic_epochs_used": float(critic_log["critic_epochs_used"]),
            "train/actor_grad_norm": float(np.mean([entry["grad_norm"] for entry in actor_logs])),
            "train/critic_grad_norm": float(critic_log["grad_norm"]),
            "rollout/length": float(advantage.shape[0]),
            "rollout/episodes_per_update": float(np.sum(data["done"] > 0.5)),
            "debug/no_parameter_sharing": 1.0,
            "debug/no_fedavg": 1.0,
            "debug/no_gat": 1.0,
            "debug/no_privacy": 1.0,
            "debug/adv_mean_raw": float(raw_advantage.mean()),
            "debug/adv_std_raw": float(raw_advantage.std()),
            "debug/adv_mean_normed": float(advantage.mean()),
            "debug/adv_std_normed": float(advantage.std()),
            "debug/critic_done_resets": float(critic_log["critic_gru_done_resets"]),
            "central_critic/loss": float(critic_log["loss"]),
            "central_critic/grad_norm": float(critic_log["grad_norm"]),
        }
        for agent_idx, actor_log in enumerate(actor_logs):
            logs[f"agent_{agent_idx}/loss_actor"] = float(actor_log["loss_actor"])
            logs[f"agent_{agent_idx}/approx_kl"] = float(actor_log["approx_kl"])
            logs[f"agent_{agent_idx}/entropy"] = float(actor_log["entropy"])
            logs[f"agent_{agent_idx}/actor_epochs_used"] = float(actor_log["actor_epochs_used"])
            logs[f"agent_{agent_idx}/debug/actor_grad_norm"] = float(actor_log["grad_norm"])
        if self.mode == "happo":
            logs.update(
                {
                    "happo/factor_mean": float(factor_final.mean().cpu().item()),
                    "happo/factor_std": float(factor_final.std(unbiased=False).cpu().item()),
                    "happo/factor_min": float(factor_final.min().cpu().item()),
                    "happo/factor_max": float(factor_final.max().cpu().item()),
                }
            )
        if self.writer is not None and global_step is not None:
            for tag, value in logs.items():
                if isinstance(value, (Number, np.number)):
                    self.writer.add_scalar(tag, float(value), int(global_step))
        diagnostics = {
            "mode": self.mode,
            "update_order": update_order,
            "phase_sequence": phase_sequence,
            "base_advantage_hash": base_hash,
            "raw_advantage": raw_advantage,
            "normalized_advantage": advantage,
            "returns": returns,
            "actor_logs": actor_logs,
            "critic_log": critic_log,
            "factor_trace": factor_trace,
            "factor_final": factor_final.detach().cpu().numpy(),
            "critic_updated_after_all_actors": phase_sequence[-1] == "centralized_critic_all_epochs",
        }
        self.last_update_diagnostics = diagnostics
        self.round_idx += 1
        return {**logs, "diagnostics": diagnostics}

    def parameter_id_sets(self) -> Dict[str, set]:
        result = {
            f"agent_{index}_actor": {id(parameter) for parameter in actor.parameters()}
            for index, actor in enumerate(self.actors)
        }
        result["centralized_critic"] = {
            id(parameter) for parameter in self.centralized_critic.parameters()
        }
        return result

    def optimizer_parameter_sets(self) -> Dict[str, set]:
        result = {
            f"agent_{index}_actor_optimizer": {
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            }
            for index, optimizer in enumerate(self.actor_opts)
        }
        result["centralized_critic_optimizer"] = {
            id(parameter) for group in self.critic_opt.param_groups for parameter in group["params"]
        }
        return result

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for index, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(save_dir, f"agent_{index}_actor.pt"))
        torch.save(
            self.centralized_critic.state_dict(),
            os.path.join(save_dir, "centralized_critic.pt"),
        )

    def load(self, save_dir: str, map_location: Optional[str] = None) -> None:
        location = map_location if map_location is not None else str(self.device)
        for index, actor in enumerate(self.actors):
            actor.load_state_dict(
                torch.load(os.path.join(save_dir, f"agent_{index}_actor.pt"), map_location=location)
            )
        self.centralized_critic.load_state_dict(
            torch.load(os.path.join(save_dir, "centralized_critic.pt"), map_location=location)
        )


__all__ = [
    "NoShareMARLHyperParams",
    "NoShareMAPPOHAPPOTrainer",
    "SUPPORTED_MODES",
    "TRAINER_VERSION",
]
