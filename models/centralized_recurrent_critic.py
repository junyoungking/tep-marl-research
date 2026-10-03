from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CentralizedRecurrentCritic(nn.Module):
    """Central value function used by the no-share MAPPO/HAPPO baselines.

    The input at time ``t`` is the concatenation of all local observations and
    all *previous* local actions.  Current actions never enter the critic.
    """

    def __init__(self, joint_input_dim: int = 28, hidden_dim: int = 64) -> None:
        super().__init__()
        self.joint_input_dim = int(joint_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_layer = nn.Linear(self.joint_input_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.input_layer.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(self.input_layer.bias, 0.0)
        for name, parameter in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.constant_(parameter, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(
        self,
        joint_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if joint_input.dim() == 1:
            joint_input = joint_input.reshape(1, 1, -1)
            squeeze_batch = True
        elif joint_input.dim() == 2:
            joint_input = joint_input.unsqueeze(0)
            squeeze_batch = True
        elif joint_input.dim() != 3:
            raise ValueError(
                "joint_input must have shape (D,), (T,D), or (B,T,D), "
                f"got {tuple(joint_input.shape)}"
            )
        if joint_input.shape[-1] != self.joint_input_dim:
            raise ValueError(
                f"Expected joint input dimension {self.joint_input_dim}, "
                f"got {joint_input.shape[-1]}"
            )
        features_in = self.activation(self.input_layer(joint_input))
        features, hidden_out = self.gru(features_in, hidden)
        values = self.value_head(features).squeeze(-1)
        if squeeze_batch:
            values = values.squeeze(0)
            features = features.squeeze(0)
        return values, hidden_out, features

    @torch.no_grad()
    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, int(batch_size), self.hidden_dim, device=device, dtype=dtype)


__all__ = ["CentralizedRecurrentCritic"]
