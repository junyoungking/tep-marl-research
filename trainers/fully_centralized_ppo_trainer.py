from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.centralized_recurrent_critic import CentralizedRecurrentCritic
from models.fully_centralized_actor import FullyCentralizedActor


TRAINER_VERSION = "fully_centralized_ppo_v1"


@dataclass
class FullyCentralizedPPOHyperParams:
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


class FullyCentralizedPPOTrainer(nn.Module):
    """One joint actor and one recurrent centralized critic."""

    def __init__(
        self,
        actor: FullyCentralizedActor,
        centralized_critic: CentralizedRecurrentCritic,
        device: str = "cpu",
        hp: Optional[FullyCentralizedPPOHyperParams] = None,
        writer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.actor = actor
        self.centralized_critic = centralized_critic
        self.hp = hp if hp is not None else FullyCentralizedPPOHyperParams()
        self.writer = writer
        self.to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.hp.actor_lr)
        self.critic_opt = optim.Adam(self.centralized_critic.parameters(), lr=self.hp.critic_lr)
        self.last_update_diagnostics: Dict[str, Any] = {}

    @classmethod
    def build_default(
        cls,
        obs_dims: Sequence[int] = (7, 10, 8),
        act_dims: Sequence[int] = (1, 1, 1),
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 64,
        device: str = "cpu",
        hp: Optional[FullyCentralizedPPOHyperParams] = None,
        writer: Optional[Any] = None,
        seed: int = 0,
    ) -> "FullyCentralizedPPOTrainer":
        if list(map(int, obs_dims)) != [7, 10, 8]:
            raise ValueError(f"Verified observation dimensions are [7, 10, 8], got {list(obs_dims)}")
        if list(map(int, act_dims)) != [1, 1, 1]:
            raise ValueError(f"Verified action dimensions are [1, 1, 1], got {list(act_dims)}")
        torch.manual_seed(int(seed))
        return cls(
            actor=FullyCentralizedActor(sum(obs_dims), sum(act_dims), hidden_dim=int(actor_hidden_dim)),
            centralized_critic=CentralizedRecurrentCritic(
                sum(obs_dims) + sum(act_dims), hidden_dim=int(critic_hidden_dim)
            ),
            device=device,
            hp=hp,
            writer=writer,
        )

    @staticmethod
    def count_parameters(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    @staticmethod
    def build_joint_observation(obs_list: Sequence[Any]) -> np.ndarray:
        if len(obs_list) != 3:
            raise ValueError("Joint actor input requires exactly three normalized observations")
        arrays = [np.asarray(value, dtype=np.float32) for value in obs_list]
        leading = arrays[0].shape[:-1]
        if any(array.ndim < 1 or array.shape[:-1] != leading for array in arrays):
            raise ValueError("Joint observation components must share their leading shape")
        result = np.concatenate(arrays, axis=-1).astype(np.float32, copy=False)
        if result.shape[-1] != 25:
            raise ValueError(f"Joint actor observation must be dimension 25, got {result.shape[-1]}")
        return result

    @classmethod
    def build_joint_input(cls, obs_list: Sequence[Any], previous_action: Any) -> np.ndarray:
        obs = cls.build_joint_observation(obs_list)
        previous = np.asarray(previous_action, dtype=np.float32)
        if previous.shape[:-1] != obs.shape[:-1] or previous.shape[-1] != 3:
            raise ValueError(f"Previous joint action must end in dimension 3, got {previous.shape}")
        result = np.concatenate([obs, previous], axis=-1).astype(np.float32, copy=False)
        if result.shape[-1] != 28:
            raise AssertionError("Central critic input dimension drift")
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
        next_value = np.empty_like(value)
        next_value[:-1] = value[1:]
        next_value[-1] = np.float32(last_value)
        next_value *= 1.0 - done
        delta = reward + float(gamma) * next_value - value
        advantage = np.zeros_like(reward)
        running = 0.0
        for index in range(reward.shape[0] - 1, -1, -1):
            running = float(delta[index]) + float(gamma) * float(gae_lambda) * (1.0 - float(done[index])) * running
            advantage[index] = running
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
        if start < flat.numel():
            bounds.append((start, int(flat.numel())))
        return bounds

    @staticmethod
    def _delta_norm(before: Sequence[torch.Tensor], module: nn.Module) -> float:
        return float(sum(float((new.detach() - old).pow(2).sum().cpu()) for old, new in zip(before, module.parameters())) ** 0.5)

    @staticmethod
    def _hash_array(array: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()

    @torch.no_grad()
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return self.centralized_critic.init_hidden(batch_size, self.device)

    @torch.no_grad()
    def act(
        self,
        obs_list: Sequence[np.ndarray],
        previous_action: np.ndarray,
        critic_hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, torch.Tensor]:
        if critic_hidden is None:
            critic_hidden = self.init_hidden(1)
        joint_obs = torch.as_tensor(self.build_joint_observation(obs_list), dtype=torch.float32, device=self.device)
        joint_input = torch.as_tensor(self.build_joint_input(obs_list, previous_action), dtype=torch.float32, device=self.device)
        action, log_prob = self.actor.act(joint_obs, deterministic=deterministic)
        value, hidden_out, _ = self.centralized_critic(joint_input, critic_hidden)
        return (
            action.detach().cpu().numpy().astype(np.float32),
            float(log_prob.detach().reshape(-1)[-1].cpu()),
            float(value.detach().reshape(-1)[-1].cpu()),
            hidden_out.detach(),
        )

    @torch.no_grad()
    def value(
        self,
        obs_list: Sequence[np.ndarray],
        previous_action: np.ndarray,
        critic_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        if critic_hidden is None:
            critic_hidden = self.init_hidden(1)
        joint_input = torch.as_tensor(self.build_joint_input(obs_list, previous_action), dtype=torch.float32, device=self.device)
        value, hidden_out, _ = self.centralized_critic(joint_input, critic_hidden)
        return float(value.detach().reshape(-1)[-1].cpu()), hidden_out.detach()

    def centralized_values_from_sequence(self, joint_input: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if joint_input.dim() == 2:
            joint_input = joint_input.unsqueeze(0)
        done = done.reshape(-1)
        if joint_input.dim() != 3 or joint_input.shape[0] != 1 or joint_input.shape[-1] != 28:
            raise ValueError(f"Expected joint input shape (1,T,28), got {tuple(joint_input.shape)}")
        bounds = self.segment_bounds(done)
        parts = []
        for start, end in bounds:
            hidden = self.centralized_critic.init_hidden(1, self.device, joint_input.dtype)
            values, _, _ = self.centralized_critic(joint_input[:, start:end, :], hidden)
            parts.append(values.reshape(1, -1))
        if not parts:
            raise ValueError("Cannot evaluate an empty rollout")
        return torch.cat(parts, dim=1), {
            "segment_bounds": bounds,
            "hidden_resets": len(bounds),
            "critic_gru_done_resets": int((done[:-1] > 0.5).sum().detach().cpu()) if done.numel() > 1 else 0,
        }

    def critic_update(self, joint_input_np: np.ndarray, return_np: np.ndarray, done_np: np.ndarray) -> Dict[str, Any]:
        joint_input = torch.as_tensor(joint_input_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        target = torch.as_tensor(return_np, dtype=torch.float32, device=self.device).reshape(1, -1).detach()
        done = torch.as_tensor(done_np, dtype=torch.float32, device=self.device).reshape(-1)
        before = [parameter.detach().clone() for parameter in self.centralized_critic.parameters()]
        rows: List[Dict[str, float]] = []
        reset: Dict[str, Any] = {}
        for epoch in range(max(1, int(self.hp.critic_epochs))):
            self.critic_opt.zero_grad(set_to_none=True)
            prediction, reset = self.centralized_values_from_sequence(joint_input, done)
            loss = ((prediction - target) ** 2).mean()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.hp.max_grad_norm)
            self.critic_opt.step()
            rows.append({"epoch": float(epoch), "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu())})
        return {**rows[-1], **reset, "critic_epochs_used": len(rows), "param_delta": self._delta_norm(before, self.centralized_critic), "epoch_logs": rows}

    def actor_update(self, joint_obs_np: np.ndarray, action_np: np.ndarray, old_log_prob_np: np.ndarray, advantage_np: np.ndarray) -> Dict[str, Any]:
        obs = torch.as_tensor(joint_obs_np, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action_np, dtype=torch.float32, device=self.device)
        old_log_prob = torch.as_tensor(old_log_prob_np, dtype=torch.float32, device=self.device).reshape(-1).detach()
        advantage = torch.as_tensor(advantage_np, dtype=torch.float32, device=self.device).reshape(-1).detach()
        before = [parameter.detach().clone() for parameter in self.actor.parameters()]
        rows: List[Dict[str, float]] = []
        for epoch in range(max(1, int(self.hp.actor_epochs))):
            self.actor_opt.zero_grad(set_to_none=True)
            new_log_prob, entropy, _, _ = self.actor.evaluate_actions(obs, action)
            ratio = torch.exp(torch.clamp(new_log_prob - old_log_prob, -20.0, 20.0))
            policy_loss = -torch.minimum(
                ratio * advantage,
                torch.clamp(ratio, 1.0 - self.hp.clip_range, 1.0 + self.hp.clip_range) * advantage,
            ).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss - self.hp.ent_coef * entropy_mean
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
            self.actor_opt.step()
            approx_kl = float((old_log_prob - new_log_prob).mean().detach().cpu())
            rows.append({
                "epoch": float(epoch), "loss": float(loss.detach().cpu()), "policy_loss": float(policy_loss.detach().cpu()),
                "entropy": float(entropy_mean.detach().cpu()), "approx_kl": approx_kl,
                "clipfrac": float((torch.abs(ratio - 1.0) > self.hp.clip_range).float().mean().detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
            })
            if self.hp.target_kl is not None and approx_kl > float(self.hp.target_kl):
                break
        return {**rows[-1], "actor_epochs_used": len(rows), "param_delta": self._delta_norm(before, self.actor), "epoch_logs": rows}

    def update(self, rollout: Dict[str, Any], last_value: float, global_step: Optional[int] = None) -> Dict[str, Any]:
        required = {"joint_obs", "joint_input", "joint_action", "old_log_prob", "value", "reward", "done"}
        missing = sorted(required - set(rollout))
        if missing:
            raise KeyError(f"Rollout missing keys: {missing}")
        data = {key: np.asarray(rollout[key], dtype=np.float32) for key in required}
        horizon = data["joint_obs"].shape[0]
        if data["joint_obs"].shape != (horizon, 25) or data["joint_input"].shape != (horizon, 28) or data["joint_action"].shape != (horizon, 3):
            raise ValueError("Centralized rollout tensor dimensions are invalid")
        for key in ("old_log_prob", "value", "reward", "done"):
            data[key] = data[key].reshape(-1)
            if data[key].shape != (horizon,):
                raise ValueError(f"{key} must have shape ({horizon},)")
        raw_advantage, returns = self.compute_gae(data["reward"], data["value"], data["done"], last_value, self.hp.gamma, self.hp.gae_lambda)
        advantage = ((raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)).astype(np.float32)
        phase_sequence = ["centralized_critic_all_epochs", "centralized_actor_all_epochs"]
        critic_log = self.critic_update(data["joint_input"], returns, data["done"])
        actor_log = self.actor_update(data["joint_obs"], data["joint_action"], data["old_log_prob"], advantage)
        logs: Dict[str, Any] = {
            "loss/actor": actor_log["loss"], "loss/critic": critic_log["loss"], "loss/entropy": actor_log["entropy"],
            "train/approx_kl": actor_log["approx_kl"], "train/clipfrac": actor_log["clipfrac"],
            "train/actor_epochs_used": float(actor_log["actor_epochs_used"]), "train/critic_epochs_used": float(critic_log["critic_epochs_used"]),
            "train/actor_grad_norm": actor_log["grad_norm"], "train/critic_grad_norm": critic_log["grad_norm"],
            "central_critic/loss": critic_log["loss"], "central_critic/grad_norm": critic_log["grad_norm"],
            "rollout/length": float(horizon), "rollout/episodes_per_update": float(np.sum(data["done"] > 0.5)),
            "debug/fully_centralized_actor": 1.0, "debug/no_local_actor_copies": 1.0, "debug/no_fedavg": 1.0,
            "debug/no_bi": 1.0, "debug/no_gat": 1.0, "debug/no_privacy": 1.0,
            "debug/adv_mean_raw": float(raw_advantage.mean()), "debug/adv_std_raw": float(raw_advantage.std()),
            "debug/adv_mean_normed": float(advantage.mean()), "debug/adv_std_normed": float(advantage.std()),
            "debug/critic_done_resets": float(critic_log["critic_gru_done_resets"]),
        }
        diagnostics = {
            "phase_sequence": phase_sequence, "advantage_hash": self._hash_array(advantage),
            "actor": actor_log, "critic": critic_log,
        }
        self.last_update_diagnostics = diagnostics
        logs["diagnostics"] = diagnostics
        if self.writer is not None and global_step is not None:
            logger = getattr(self, "metrics_logger", None)
            if logger is not None:
                logger.record_update(logs, step=int(global_step))
            else:
                for key, value in logs.items():
                    if key != "diagnostics" and isinstance(value, (int, float)):
                        self.writer.add_scalar(key, float(value), int(global_step))
        return logs

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, "centralized_actor.pt"))
        torch.save(self.centralized_critic.state_dict(), os.path.join(directory, "centralized_critic.pt"))

    def load(self, directory: str, map_location: Optional[str] = None) -> None:
        location = map_location if map_location is not None else str(self.device)
        self.actor.load_state_dict(torch.load(os.path.join(directory, "centralized_actor.pt"), map_location=location))
        self.centralized_critic.load_state_dict(torch.load(os.path.join(directory, "centralized_critic.pt"), map_location=location))


__all__ = ["TRAINER_VERSION", "FullyCentralizedPPOHyperParams", "FullyCentralizedPPOTrainer"]
