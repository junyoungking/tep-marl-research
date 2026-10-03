from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class FullyCentralizedActor(nn.Module):
    """Joint Gaussian actor for the three TEP controllers."""

    def __init__(
        self,
        joint_obs_dim: int = 25,
        joint_action_dim: int = 3,
        hidden_dim: int = 64,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.joint_obs_dim = int(joint_obs_dim)
        self.joint_action_dim = int(joint_action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.body = nn.Sequential(
            nn.Linear(self.joint_obs_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(int(hidden_dim), self.joint_action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.joint_action_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in self.body.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.constant_(self.log_std, 0.0)

    @staticmethod
    def _ensure_batch(value: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if value.dim() == 1:
            return value.unsqueeze(0), True
        if value.dim() == 2:
            return value, False
        raise ValueError(f"Expected 1D or 2D actor input, got {tuple(value.shape)}")

    def forward(self, joint_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, squeeze = self._ensure_batch(joint_obs)
        if obs.shape[-1] != self.joint_obs_dim:
            raise ValueError(f"Expected joint observation dimension {self.joint_obs_dim}, got {obs.shape[-1]}")
        mu = self.mu_head(self.body(obs))
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mu)
        if squeeze:
            return mu.squeeze(0), std.squeeze(0)
        return mu, std

    def get_distribution(self, joint_obs: torch.Tensor) -> Normal:
        mu, std = self.forward(joint_obs)
        return Normal(mu, std)

    @torch.no_grad()
    def act(self, joint_obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.get_distribution(joint_obs)
        action = distribution.mean if deterministic else distribution.sample()
        action = torch.clamp(action, -1.0, 1.0)
        joint_log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, joint_log_prob

    def evaluate_actions(
        self, joint_obs: torch.Tensor, joint_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.get_distribution(joint_obs)
        action, _ = self._ensure_batch(joint_action)
        joint_log_prob = distribution.log_prob(action).sum(dim=-1)
        joint_entropy = distribution.entropy().sum(dim=-1)
        return joint_log_prob, joint_entropy, distribution.mean, distribution.stddev


__all__ = ["FullyCentralizedActor"]
