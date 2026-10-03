import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Sequence, Tuple


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation=nn.ReLU,
    output_activation=None,
) -> nn.Sequential:
    layers = []
    prev_dim = int(input_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(h)))
        layers.append(activation())
        prev_dim = int(h)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def init_linear_weights(module: nn.Module, final_gain: float = 1.0) -> nn.Module:
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for m in linear_layers:
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(m.bias, 0.0)
    if linear_layers:
        nn.init.orthogonal_(linear_layers[-1].weight, gain=final_gain)
        nn.init.constant_(linear_layers[-1].bias, 0.0)
    return module


class SplitLocalActor(nn.Module):
    """One local actor for a heterogeneous split-federated agent.

    The module mirrors one slice of ``HybridActor``: local obs processing,
    a per-client obs enhancer copy, then a local action head. Only
    ``obs_enhancer`` is intended for FedAvg.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        local_obs_hidden_dims: Sequence[int] = (128,),
        local_obs_out_dim: int = 128,
        shared_obs_hidden_dims: Sequence[int] = (128,),
        shared_obs_out_dim: int = 128,
        local_action_hidden_dims: Sequence[int] = (128,),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        activate_feature_outputs: bool = False,
        ablate_enhancer: bool = False,
        tanh_squashed_policy: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.ablate_enhancer = bool(ablate_enhancer)
        # Historical Split-FL/A0 behavior is deliberately the default.  The
        # opt-in transform is used only by the D1 validity experiment.
        self.tanh_squashed_policy = bool(tanh_squashed_policy)
        feature_output_activation = nn.ReLU if activate_feature_outputs else None

        self.obs_processor = build_mlp(
            input_dim=self.obs_dim,
            hidden_dims=local_obs_hidden_dims,
            output_dim=local_obs_out_dim,
            output_activation=feature_output_activation,
        )
        init_linear_weights(self.obs_processor)

        if self.ablate_enhancer:
            if int(local_obs_out_dim) != int(shared_obs_out_dim):
                raise ValueError(
                    "ablate_enhancer=True requires local_obs_out_dim == shared_obs_out_dim "
                    f"for identity passthrough, got {local_obs_out_dim} vs {shared_obs_out_dim}"
                )
            self.obs_enhancer = nn.Identity()
        else:
            self.obs_enhancer = build_mlp(
                input_dim=local_obs_out_dim,
                hidden_dims=shared_obs_hidden_dims,
                output_dim=shared_obs_out_dim,
                output_activation=feature_output_activation,
            )
            init_linear_weights(self.obs_enhancer)

        if len(local_action_hidden_dims) > 0:
            action_body_out_dim = int(local_action_hidden_dims[-1])
            self.action_body = build_mlp(
                input_dim=shared_obs_out_dim,
                hidden_dims=local_action_hidden_dims,
                output_dim=action_body_out_dim,
                output_activation=feature_output_activation,
            )
        else:
            action_body_out_dim = int(shared_obs_out_dim)
            self.action_body = nn.Identity()
        init_linear_weights(self.action_body)

        self.mu_head = nn.Linear(action_body_out_dim, self.act_dim)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if x.dim() == 1:
            return x.unsqueeze(0), True
        if x.dim() == 2:
            return x, False
        raise ValueError(f"Expected 1D or 2D actor input, got shape={tuple(x.shape)}")

    def forward(self, obs: torch.Tensor):
        obs_b, squeeze_back = self._ensure_batch(obs)
        h_local = self.obs_processor(obs_b)
        h_shared = self.obs_enhancer(h_local)
        h_action = self.action_body(h_shared)
        mu = self.mu_head(h_action)
        log_std = self.log_std.unsqueeze(0).expand_as(mu)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if squeeze_back:
            mu = mu.squeeze(0)
            std = std.squeeze(0)
        return mu, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mu, std = self.forward(obs)
        return Normal(mu, std)

    def set_tanh_squashed_policy(self, enabled: bool) -> None:
        """Select the action-law without changing any trainable parameter."""
        self.tanh_squashed_policy = bool(enabled)

    @staticmethod
    def _squashed_log_prob(dist: Normal, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Stable log density of ``tanh(z)`` evaluated using its stored z."""
        # This form avoids cancellation near |tanh(z)| == 1 and is equivalent
        # to log N(z) - log(1 - tanh(z)^2).
        log_det = 2.0 * (torch.log(torch.tensor(2.0, device=pre_tanh_action.device, dtype=pre_tanh_action.dtype))
                         - pre_tanh_action
                         - torch.nn.functional.softplus(-2.0 * pre_tanh_action))
        return (dist.log_prob(pre_tanh_action) - log_det).sum(dim=-1)

    @torch.no_grad()
    def sample_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Return action, action-law log-probability, and the sampled latent."""
        dist = self.get_distribution(obs)
        latent = dist.mean if deterministic else dist.sample()
        if self.tanh_squashed_policy:
            action = torch.tanh(latent)
            logp = self._squashed_log_prob(dist, latent)
        else:
            action = torch.clamp(latent, -1.0, 1.0)
            logp = dist.log_prob(action).sum(dim=-1)
        return action, logp, latent

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        action, logp, _ = self.sample_action(obs, deterministic=deterministic)
        return action, logp

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        pre_tanh_action: torch.Tensor | None = None,
    ):
        dist = self.get_distribution(obs)
        action_b, _ = self._ensure_batch(action)
        if self.tanh_squashed_policy:
            if pre_tanh_action is None:
                # This fallback is for inspection/evaluation only. PPO uses
                # stored latents so it never loses precision near the bounds.
                eps = torch.finfo(action_b.dtype).eps
                pre_tanh_action = torch.atanh(action_b.clamp(-1.0 + eps, 1.0 - eps))
            latent_b, _ = self._ensure_batch(pre_tanh_action)
            if latent_b.shape != action_b.shape:
                raise ValueError(
                    f"pre_tanh_action shape {tuple(latent_b.shape)} must match action shape {tuple(action_b.shape)}"
                )
            logp = self._squashed_log_prob(dist, latent_b)
        else:
            logp = dist.log_prob(action_b).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy, dist.mean, dist.stddev

    def fedavg_actor_module(self) -> nn.Module:
        return self.obs_enhancer
