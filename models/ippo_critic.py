from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.fl_split_critic import LocalBIEncoder


class IPPOCritic(nn.Module):
    """Local recurrent IPPO critic.

    The module reuses the canonical local preprocessing + GRU path and adds a
    fixed linear value head with no additional hidden layers or activations.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        local_pre_hidden_dims: Sequence[int] = (),
        local_pre_out_dim: int = 64,
        bi_hidden_dim: int = 64,
        gru_num_layers: int = 1,
        activate_feature_outputs: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.local_pre_out_dim = int(local_pre_out_dim)
        self.bi_hidden_dim = int(bi_hidden_dim)
        self.gru_num_layers = int(gru_num_layers)

        self.local_encoder = LocalBIEncoder(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            local_pre_hidden_dims=tuple(local_pre_hidden_dims),
            local_pre_out_dim=self.local_pre_out_dim,
            bi_hidden_dim=self.bi_hidden_dim,
            gru_num_layers=self.gru_num_layers,
            activate_feature_outputs=activate_feature_outputs,
            bi_extractor="gru",
            action_pathway="on",
            fuse="learned",
            bottleneck_dim=None,
        )
        self.value_head = nn.Linear(self.bi_hidden_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(
        self,
        obs: torch.Tensor,
        prev_act: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return value estimates, next hidden state, and local features."""
        local_features, hidden_out = self.local_encoder.encode_bi(obs, prev_act, hidden)
        values = self.value_head(local_features).squeeze(-1)
        return values, hidden_out, local_features

    def predict_values(
        self,
        obs: torch.Tensor,
        prev_act: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values, hidden_out, _ = self.forward(obs, prev_act, hidden)
        return values, hidden_out

    @torch.no_grad()
    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.local_encoder.init_hidden(batch_size=batch_size, device=device, dtype=dtype)


def build_ippo_critic(
    obs_dim: int,
    act_dim: int,
    *,
    local_pre_hidden_dims: Sequence[int] = (),
    local_pre_out_dim: int = 64,
    bi_hidden_dim: int = 64,
    gru_num_layers: int = 1,
    activate_feature_outputs: bool = True,
) -> IPPOCritic:
    return IPPOCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        local_pre_hidden_dims=local_pre_hidden_dims,
        local_pre_out_dim=local_pre_out_dim,
        bi_hidden_dim=bi_hidden_dim,
        gru_num_layers=gru_num_layers,
        activate_feature_outputs=activate_feature_outputs,
    )


__all__ = ["IPPOCritic", "build_ippo_critic"]
