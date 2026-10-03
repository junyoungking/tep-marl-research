import os
from dataclasses import asdict, dataclass
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.fl_split_actor import SplitLocalActor
from models.fl_split_critic import GlobalBIServerCritic, LocalBIEncoder
from utils.fedavg import fedavg_modules, parameter_ids


TRAINER_VERSION = "fl_v2"


@dataclass
class SplitFedPPOHyperParams:
    trainer_version: str = TRAINER_VERSION
    actor_lr: float = 5e-5
    critic_lr: float = 1e-4
    server_lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 3
    batch_size: int = 0
    rollout_horizon: int = 512
    total_steps: int = 200_000
    log_every: int = 1000
    target_kl: Optional[float] = 0.01
    fedavg_interval: int = 1
    fedavg_scope: str = "both"
    bi_extractor_sharing_mode: str = "independent"
    bi_extractor_fedavg_interval: int = 10
    bi_type_grl: bool = False
    bi_type_loss_coef: float = 0.0
    bi_grl_alpha: float = 1.0


@dataclass
class ServerAdvantageBatch:
    """Server-side GAE output returned to local actor/critic update phases."""

    advantages: List[np.ndarray]
    returns: List[np.ndarray]
    adv_mean_raw: List[float]
    adv_std_raw: List[float]


class SplitFedLocalClient(nn.Module):
    """Single local client for split-federated actor/critic pieces."""

    def __init__(self, actor: SplitLocalActor, bi_encoder: LocalBIEncoder, agent_id: int):
        super().__init__()
        self.actor = actor
        self.bi_encoder = bi_encoder
        self.agent_id = int(agent_id)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        return self.actor.act(obs, deterministic=deterministic)

    def evaluate_actor(self, obs: torch.Tensor, action: torch.Tensor):
        return self.actor.evaluate_actions(obs, action)

    def encode_bi(
        self,
        obs: torch.Tensor,
        prev_act: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        *,
        return_transmitted: bool = False,
    ):
        return self.bi_encoder.encode_bi(
            obs,
            prev_act,
            hidden=hidden,
            return_transmitted=return_transmitted,
        )

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        return self.bi_encoder.init_hidden(batch_size=batch_size, device=device)

    def actor_fedavg_module(self) -> nn.Module:
        return self.actor.fedavg_actor_module()

    def critic_fedavg_module(self) -> nn.Module:
        return self.bi_encoder.fedavg_bi_module()

    def critic_extractor_modules(self) -> List[nn.Module]:
        return self.bi_encoder.extractor_parameter_modules()

    def critic_local_modules(self) -> List[nn.Module]:
        return self.bi_encoder.local_parameter_modules()


class SplitFedPPOTrainer:
    """Single-process split-learning-style federated MARL PPO trainer.

    This class is still a single-process logical simulation. Its public update
    boundary separates local client actor rollouts from server-side value /
    reward / done metadata so PPO advantages are explicitly server-owned.
    """

    def __init__(
        self,
        clients: Sequence[SplitFedLocalClient],
        server: GlobalBIServerCritic,
        device: str = "cpu",
        hp: Optional[SplitFedPPOHyperParams] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        if len(clients) != 3:
            raise ValueError(f"First-pass SplitFedPPOTrainer expects exactly 3 clients, got {len(clients)}")
        self.device = torch.device(device)
        self.clients = nn.ModuleList(clients).to(self.device)
        self.server = server.to(self.device)
        self.hp = hp if hp is not None else SplitFedPPOHyperParams()
        self.writer = writer
        self.num_agents = 3
        self.round_idx = 0
        self.debug_events: List[str] = []
        self.last_critic_diagnostics: Dict[str, Any] = {}
        self.bi_extractor_sharing_mode = str(getattr(self.hp, "bi_extractor_sharing_mode", "independent"))
        self.bi_extractor_fedavg_interval = int(getattr(self.hp, "bi_extractor_fedavg_interval", 10))
        if self.bi_extractor_sharing_mode not in {"independent", "true_shared", "fedavg10"}:
            raise ValueError(
                f"Unsupported bi_extractor_sharing_mode={self.bi_extractor_sharing_mode!r}; expected independent/true_shared/fedavg10"
            )
        if self.bi_extractor_sharing_mode == "fedavg10" and self.bi_extractor_fedavg_interval <= 0:
            raise ValueError("bi_extractor_fedavg_interval must be positive when bi_extractor_sharing_mode='fedavg10'")
        self.shared_critic_opt: Optional[optim.Optimizer] = None
        self.actor_opts = [optim.Adam(client.actor.parameters(), lr=self.hp.actor_lr) for client in self.clients]
        if self.bi_extractor_sharing_mode == "true_shared":
            self.local_critic_opts = []
            for client in self.clients:
                local_params = self._modules_parameters(client.bi_encoder.local_parameter_modules())
                if local_params:
                    self.local_critic_opts.append(optim.Adam(local_params, lr=self.hp.critic_lr))
            shared_params = self._unique_parameters(self.clients[0].bi_encoder.extractor_parameter_modules())
            self.shared_critic_opt = optim.Adam(shared_params, lr=self.hp.critic_lr) if shared_params else None
        else:
            self.local_critic_opts = [optim.Adam(client.bi_encoder.parameters(), lr=self.hp.critic_lr) for client in self.clients]
        self.server_opt = optim.Adam(self.server.parameters(), lr=self.hp.server_lr)

    @staticmethod
    def _parameter_delta_norm(before: Sequence[torch.Tensor], module: nn.Module) -> float:
        total = 0.0
        for old, new in zip(before, module.parameters()):
            total += float((new.detach() - old).pow(2).sum().cpu().item())
        return float(total ** 0.5)

    @staticmethod
    def _modules_parameters(modules: Sequence[nn.Module]) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    @staticmethod
    def _unique_parameters(modules: Sequence[nn.Module]) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set[int] = set()
        for module in modules:
            for param in module.parameters():
                ident = id(param)
                if ident in seen:
                    continue
                seen.add(ident)
                params.append(param)
        return params

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

    @classmethod
    def _safe_critic_scalar_logs(cls, diagnostics: Dict[str, Any]) -> Dict[str, float]:
        logs: Dict[str, float] = {}
        key_map = {
            "server_changed_on_server_step": "debug/critic_server_changed_on_server_step",
            "server_changed_during_local_step": "debug/critic_server_changed_during_local_step",
            "server_grads_created_during_local": "debug/critic_server_grads_created_during_local",
            "local_changed_on_handoff": "debug/critic_local_changed_on_handoff",
        }
        for src_key, dst_key in key_map.items():
            scalar = cls._scalar_float(diagnostics.get(src_key))
            if scalar is not None:
                logs[dst_key] = scalar

        value_losses = diagnostics.get("value_losses")
        if isinstance(value_losses, list):
            for i, loss in enumerate(value_losses):
                scalar = cls._scalar_float(loss)
                if scalar is not None:
                    logs[f"critic/value_loss_agent_{i}"] = scalar

        type_loss = cls._scalar_float(diagnostics.get("type_loss"))
        if type_loss is not None:
            logs["critic/type_loss"] = type_loss

        weighted_type_loss = cls._scalar_float(diagnostics.get("type_loss_weighted"))
        if weighted_type_loss is not None:
            logs["critic/type_loss_weighted"] = weighted_type_loss

        type_branch_active = cls._scalar_float(diagnostics.get("type_branch_active"))
        if type_branch_active is not None:
            logs["debug/critic_type_branch_active"] = type_branch_active

        requires_grad = diagnostics.get("dL_dBI_requires_grad")
        if isinstance(requires_grad, list) and requires_grad:
            bool_scalars = [cls._scalar_float(x) for x in requires_grad]
            bool_scalars = [x for x in bool_scalars if x is not None]
            if bool_scalars:
                logs["debug/critic_dL_dBI_requires_grad_rate"] = float(np.mean(bool_scalars))
        return logs

    @classmethod
    def build_default(
        cls,
        obs_dims: Sequence[int],
        act_dims: Sequence[int],
        device: str = "cpu",
        hp: Optional[SplitFedPPOHyperParams] = None,
        actor_local_hidden: int = 128,
        actor_local_out: int = 128,
        actor_shared_hidden: int = 128,
        actor_shared_out: int = 128,
        actor_action_hidden: int = 128,
        critic_local_pre_hidden: int = 128,
        critic_local_pre_out: int = 128,
        critic_bi_hidden: int = 128,
        critic_gru_layers: int = 1,
        critic_gat_hidden: int = 128,
        critic_readout_hidden: int = 128,
        critic_readout_dim: int = 128,
        critic_value_hidden: int = 64,
        actor_local_hidden_dims: Optional[Sequence[int]] = None,
        actor_shared_hidden_dims: Optional[Sequence[int]] = None,
        actor_action_hidden_dims: Optional[Sequence[int]] = None,
        critic_local_pre_hidden_dims: Optional[Sequence[int]] = None,
        critic_readout_hidden_dims: Optional[Sequence[int]] = None,
        critic_value_hidden_dims: Optional[Sequence[int]] = None,
        activate_feature_outputs: bool = False,
        ablate_enhancer: bool = False,
        bi_extractor: str = "gru",
        bi_post_gru_mlp: bool = False,
        aggregation: str = "gat",
        action_pathway: str = "on",
        fuse: str = "learned",
        critic_bi_bottleneck_dim: Optional[int] = None,
        critic_bi_bottleneck_seed: int = 0,
    ):
        if len(obs_dims) != 3 or len(act_dims) != 3:
            raise ValueError("First-pass split-fed trainer expects exactly three heterogeneous agents")
        resolved_actor_local_hidden_dims = cls._resolve_hidden_dims(
            actor_local_hidden_dims,
            actor_local_hidden,
            "actor_local_hidden_dims",
        )
        resolved_actor_shared_hidden_dims = cls._resolve_hidden_dims(
            actor_shared_hidden_dims,
            actor_shared_hidden,
            "actor_shared_hidden_dims",
        )
        resolved_actor_action_hidden_dims = cls._resolve_hidden_dims(
            actor_action_hidden_dims,
            actor_action_hidden,
            "actor_action_hidden_dims",
        )
        resolved_critic_local_pre_hidden_dims = cls._resolve_hidden_dims(
            critic_local_pre_hidden_dims,
            critic_local_pre_hidden,
            "critic_local_pre_hidden_dims",
        )
        resolved_critic_readout_hidden_dims = cls._resolve_hidden_dims(
            critic_readout_hidden_dims,
            critic_readout_hidden,
            "critic_readout_hidden_dims",
        )
        resolved_critic_value_hidden_dims = cls._resolve_hidden_dims(
            critic_value_hidden_dims,
            critic_value_hidden,
            "critic_value_hidden_dims",
        )
        resolved_hp = hp if hp is not None else SplitFedPPOHyperParams()
        bi_sharing_mode = str(getattr(resolved_hp, "bi_extractor_sharing_mode", "independent"))
        if bi_sharing_mode not in {"independent", "true_shared", "fedavg10"}:
            raise ValueError(
                f"Unsupported bi_extractor_sharing_mode={bi_sharing_mode!r}; expected independent/true_shared/fedavg10"
            )
        if bi_sharing_mode == "fedavg10":
            bi_fedavg_interval = int(getattr(resolved_hp, "bi_extractor_fedavg_interval", 10))
            if bi_fedavg_interval <= 0:
                raise ValueError("bi_extractor_fedavg_interval must be positive when bi_extractor_sharing_mode='fedavg10'")
        else:
            bi_fedavg_interval = int(getattr(resolved_hp, "bi_extractor_fedavg_interval", 10))
        bi_init_seed = int(torch.initial_seed()) if bi_sharing_mode in {"true_shared", "fedavg10"} else None
        shared_bi_extractor = None
        shared_f_B = None
        if bi_sharing_mode == "true_shared":
            shared_bi_extractor, shared_f_B = LocalBIEncoder.build_bi_extractor_modules(
                local_pre_out_dim=critic_local_pre_out,
                bi_hidden_dim=critic_bi_hidden,
                gru_num_layers=critic_gru_layers,
                bi_extractor=bi_extractor,
                bi_post_gru_mlp=bi_post_gru_mlp,
                activate_feature_outputs=activate_feature_outputs,
                init_seed=bi_init_seed,
            )
        clients = []
        for i, (obs_dim, act_dim) in enumerate(zip(obs_dims, act_dims)):
            actor = SplitLocalActor(
                obs_dim=obs_dim,
                act_dim=act_dim,
                local_obs_hidden_dims=resolved_actor_local_hidden_dims,
                local_obs_out_dim=actor_local_out,
                shared_obs_hidden_dims=resolved_actor_shared_hidden_dims,
                shared_obs_out_dim=actor_shared_out,
                local_action_hidden_dims=resolved_actor_action_hidden_dims,
                activate_feature_outputs=activate_feature_outputs,
                ablate_enhancer=ablate_enhancer,
            )
            encoder = LocalBIEncoder(
                obs_dim=obs_dim,
                act_dim=act_dim,
                local_pre_hidden_dims=resolved_critic_local_pre_hidden_dims,
                local_pre_out_dim=critic_local_pre_out,
                bi_hidden_dim=critic_bi_hidden,
                gru_num_layers=critic_gru_layers,
                activate_feature_outputs=activate_feature_outputs,
                bi_extractor=bi_extractor,
                bi_post_gru_mlp=bi_post_gru_mlp,
                action_pathway=action_pathway,
                fuse=fuse,
                bottleneck_dim=critic_bi_bottleneck_dim,
                bottleneck_seed=critic_bi_bottleneck_seed,
                shared_bi_extractor=shared_bi_extractor,
                shared_f_B=shared_f_B,
                bi_init_seed=bi_init_seed if bi_sharing_mode == "fedavg10" else None,
            )
            clients.append(SplitFedLocalClient(actor=actor, bi_encoder=encoder, agent_id=i))
        server = GlobalBIServerCritic(
            num_agents=3,
            bi_dim=critic_bi_hidden,
            gat_hidden_dim=critic_gat_hidden,
            readout_hidden_dims=resolved_critic_readout_hidden_dims,
            readout_dim=critic_readout_dim,
            value_hidden_dims=resolved_critic_value_hidden_dims,
            activate_feature_outputs=activate_feature_outputs,
            aggregation=aggregation,
            bi_type_grl=getattr(resolved_hp, "bi_type_grl", False),
            bi_type_loss_coef=getattr(resolved_hp, "bi_type_loss_coef", 0.0),
            bi_grl_alpha=getattr(resolved_hp, "bi_grl_alpha", 1.0),
        )
        trainer = cls(clients=clients, server=server, device=device, hp=resolved_hp)
        trainer.bi_extractor_sharing_mode = bi_sharing_mode
        trainer.bi_extractor_fedavg_interval = bi_fedavg_interval
        trainer.shared_bi_extractor = shared_bi_extractor
        trainer.shared_f_B = shared_f_B
        return trainer

    @staticmethod
    def _resolve_hidden_dims(hidden_dims: Optional[Sequence[int]], fallback_dim: int, name: str) -> Tuple[int, ...]:
        if hidden_dims is None:
            if isinstance(fallback_dim, bool) or not isinstance(fallback_dim, Number):
                raise TypeError(f"{name} fallback must be a positive integer, got {fallback_dim!r}")
            fallback_int = int(fallback_dim)
            if float(fallback_dim) != float(fallback_int) or fallback_int <= 0:
                raise ValueError(f"{name} fallback must be a positive integer, got {fallback_dim!r}")
            return (fallback_int,)
        if isinstance(hidden_dims, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of positive integers, not a string")
        try:
            dims = tuple(hidden_dims)
        except TypeError as exc:
            raise TypeError(f"{name} must be a sequence of positive integers") from exc
        resolved = []
        for dim in dims:
            if isinstance(dim, bool) or not isinstance(dim, Number):
                raise TypeError(f"{name} entries must be positive integers, got {dim!r}")
            dim_int = int(dim)
            if float(dim) != float(dim_int) or dim_int <= 0:
                raise ValueError(f"{name} entries must be positive integers, got {dim!r}")
            resolved.append(dim_int)
        return tuple(resolved)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for i, client in enumerate(self.clients):
            torch.save(client.actor.state_dict(), os.path.join(save_dir, f"split_actor_{i}.pt"))
            torch.save(client.bi_encoder.state_dict(), os.path.join(save_dir, f"split_bi_encoder_{i}.pt"))
        torch.save(self.server.state_dict(), os.path.join(save_dir, "split_global_bi_server_critic.pt"))

    def optimizer_state_dicts(self) -> Dict[str, Any]:
        return {
            "actor_opts": [opt.state_dict() for opt in self.actor_opts],
            "local_critic_opts": [opt.state_dict() for opt in self.local_critic_opts],
            "shared_critic_opt": None if self.shared_critic_opt is None else self.shared_critic_opt.state_dict(),
            "server_opt": self.server_opt.state_dict(),
        }

    def load_optimizer_state_dicts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"loaded": False}
        actor_states = list(payload.get("actor_opts", []))
        if len(actor_states) != len(self.actor_opts):
            raise ValueError(
                f"actor optimizer state count mismatch: expected {len(self.actor_opts)}, got {len(actor_states)}"
            )
        for opt, state in zip(self.actor_opts, actor_states):
            opt.load_state_dict(state)

        local_states = list(payload.get("local_critic_opts", []))
        if len(local_states) != len(self.local_critic_opts):
            raise ValueError(
                f"local critic optimizer state count mismatch: expected {len(self.local_critic_opts)}, got {len(local_states)}"
            )
        for opt, state in zip(self.local_critic_opts, local_states):
            opt.load_state_dict(state)

        shared_state = payload.get("shared_critic_opt")
        if self.shared_critic_opt is None:
            if shared_state is not None:
                summary["shared_critic_opt_ignored"] = True
        else:
            if shared_state is None:
                raise ValueError("Missing shared_critic_opt state for trainer with shared critic optimizer")
            self.shared_critic_opt.load_state_dict(shared_state)

        server_state = payload.get("server_opt")
        if server_state is None:
            raise ValueError("Missing server_opt state")
        self.server_opt.load_state_dict(server_state)
        summary["loaded"] = True
        return summary

    def checkpoint_state(self) -> Dict[str, Any]:
        return {
            "trainer_version": TRAINER_VERSION,
            "hp": asdict(self.hp),
            "round_idx": int(self.round_idx),
            "num_agents": int(self.num_agents),
            "bi_extractor_sharing_mode": self.bi_extractor_sharing_mode,
            "bi_extractor_fedavg_interval": int(self.bi_extractor_fedavg_interval),
            "clients": [
                {
                    "actor": client.actor.state_dict(),
                    "bi_encoder": client.bi_encoder.state_dict(),
                }
                for client in self.clients
            ],
            "server": self.server.state_dict(),
            "optimizers": self.optimizer_state_dicts(),
            "debug_events": list(self.debug_events),
            "last_critic_diagnostics": dict(self.last_critic_diagnostics),
        }

    def load_checkpoint_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if str(payload.get("trainer_version", "")) not in {TRAINER_VERSION, ""}:
            raise ValueError(
                f"Unsupported trainer_version {payload.get('trainer_version')!r}; expected {TRAINER_VERSION!r}"
            )
        clients_payload = list(payload.get("clients", []))
        if len(clients_payload) != len(self.clients):
            raise ValueError(f"client state count mismatch: expected {len(self.clients)}, got {len(clients_payload)}")
        for client, client_payload in zip(self.clients, clients_payload):
            actor_state = client_payload.get("actor")
            bi_state = client_payload.get("bi_encoder")
            if actor_state is None or bi_state is None:
                raise ValueError("client checkpoint payload missing actor or bi_encoder state")
            client.actor.load_state_dict(actor_state)
            client.bi_encoder.load_state_dict(bi_state)

        server_state = payload.get("server")
        if server_state is None:
            raise ValueError("checkpoint payload missing server state")
        self.server.load_state_dict(server_state)

        optimizer_summary = self.load_optimizer_state_dicts(dict(payload.get("optimizers", {})))
        self.round_idx = int(payload.get("round_idx", self.round_idx))
        self.debug_events = list(payload.get("debug_events", self.debug_events))
        self.last_critic_diagnostics = dict(payload.get("last_critic_diagnostics", self.last_critic_diagnostics))
        return {
            "trainer_version": str(payload.get("trainer_version", TRAINER_VERSION)),
            "round_idx": self.round_idx,
            "num_agents": int(payload.get("num_agents", self.num_agents)),
            "bi_extractor_sharing_mode": self.bi_extractor_sharing_mode,
            "bi_extractor_fedavg_interval": int(self.bi_extractor_fedavg_interval),
            "optimizer_summary": optimizer_summary,
        }

    def init_hidden(self, batch_size: int = 1) -> List[torch.Tensor]:
        return [client.init_hidden(batch_size=batch_size, device=self.device) for client in self.clients]

    @staticmethod
    def _compute_gae(rew: np.ndarray, val: np.ndarray, done: np.ndarray, last_value: float, gamma: float, gae_lambda: float):
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
    def _as_float_array(value: Any, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array, got shape={arr.shape}")
        return arr

    def _server_compute_advantages(self, server_rollout: Dict[str, Any]) -> ServerAdvantageBatch:
        """Compute GAE/returns from server-visible value metadata only.

        Required inputs are values, rewards, dones, and last_values. Raw
        observations, actions, and old log-probabilities are intentionally not
        accepted by this server-side advantage boundary.
        """

        forbidden = {"obs", "act", "action", "prev_act", "logp", "old_logp"}
        leaked = forbidden.intersection(server_rollout)
        if leaked:
            raise ValueError(f"server_rollout must not contain raw actor data: {sorted(leaked)}")
        for key in ("values", "rewards", "dones", "last_values"):
            if key not in server_rollout:
                raise KeyError(f"server_rollout missing key {key}")

        values = [self._as_float_array(v, f"server_rollout['values'][{i}]") for i, v in enumerate(server_rollout["values"])]
        last_values = [float(v) for v in server_rollout["last_values"]]
        if len(values) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} value sequences, got {len(values)}")
        if len(last_values) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} last values, got {len(last_values)}")

        rewards_raw = server_rollout["rewards"]
        dones_raw = server_rollout["dones"]
        rewards_list = self._expand_server_sequence(rewards_raw, "rewards")
        dones_list = self._expand_server_sequence(dones_raw, "dones")

        advantages, returns, adv_mean_raw, adv_std_raw = [], [], [], []
        T = values[0].shape[0]
        for i in range(self.num_agents):
            if values[i].shape[0] != T:
                raise ValueError("All server value sequences must share one horizon length")
            if rewards_list[i].shape[0] != T:
                raise ValueError(f"reward horizon mismatch for agent {i}: {rewards_list[i].shape[0]} vs {T}")
            if dones_list[i].shape[0] != T:
                raise ValueError(f"done horizon mismatch for agent {i}: {dones_list[i].shape[0]} vs {T}")
            adv_i, ret_i = self._compute_gae(
                rew=rewards_list[i],
                val=values[i],
                done=dones_list[i],
                last_value=last_values[i],
                gamma=self.hp.gamma,
                gae_lambda=self.hp.gae_lambda,
            )
            advantages.append(adv_i)
            returns.append(ret_i)
            adv_mean_raw.append(float(adv_i.mean()))
            adv_std_raw.append(float(adv_i.std()))
        return ServerAdvantageBatch(
            advantages=advantages,
            returns=returns,
            adv_mean_raw=adv_mean_raw,
            adv_std_raw=adv_std_raw,
        )

    def _expand_server_sequence(self, value: Any, name: str) -> List[np.ndarray]:
        if isinstance(value, (list, tuple)) and len(value) == self.num_agents:
            arrays = [self._as_float_array(v, f"server_rollout['{name}'][{i}]") for i, v in enumerate(value)]
            lengths = {arr.shape[0] for arr in arrays}
            if len(lengths) != 1:
                raise ValueError(f"All per-agent {name} sequences must share one horizon length")
            return arrays
        common = self._as_float_array(value, f"server_rollout['{name}']")
        return [common.copy() for _ in range(self.num_agents)]

    def _validate_client_rollouts(self, client_rollouts: List[Dict[str, np.ndarray]]) -> int:
        if len(client_rollouts) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} client rollouts, got {len(client_rollouts)}")
        required = {"obs", "act", "prev_act", "logp"}
        forbidden = {"rew", "reward", "done", "val", "value", "return", "ret", "adv"}
        T = None
        for i, rollout in enumerate(client_rollouts):
            missing = sorted(required - set(rollout))
            if missing:
                raise KeyError(f"client_rollouts[{i}] missing keys {missing}")
            leaked = sorted(forbidden.intersection(rollout))
            if leaked:
                raise ValueError(f"client_rollouts[{i}] contains server-owned keys: {leaked}")
            horizon = np.asarray(rollout["obs"]).shape[0]
            if T is None:
                T = horizon
            elif horizon != T:
                raise ValueError("All client rollouts must share same horizon length")
        return int(T or 0)

    def _legacy_rollouts_to_split(
        self,
        agent_rollouts: List[Dict[str, np.ndarray]],
        last_values: List[float],
    ) -> tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
        if len(agent_rollouts) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} rollouts, got {len(agent_rollouts)}")
        client_rollouts, value_list = [], []
        rewards, dones = None, None
        for i, r in enumerate(agent_rollouts):
            for key in ("obs", "act", "prev_act", "logp", "val", "rew", "done"):
                if key not in r:
                    raise KeyError(f"agent_rollouts[{i}] missing key {key}")
            client_rollouts.append({key: r[key] for key in ("obs", "act", "prev_act", "logp")})
            value_list.append(r["val"])
            if rewards is None:
                rewards = r["rew"]
                dones = r["done"]
            elif not np.allclose(rewards, r["rew"]) or not np.allclose(dones, r["done"]):
                raise ValueError("Legacy split adapter expects shared reward/done sequences across agents")
        server_rollout = {
            "values": value_list,
            "rewards": rewards,
            "dones": dones,
            "last_values": last_values,
        }
        return client_rollouts, server_rollout

    def _tensor_list(self, xs: Sequence[Any]) -> List[torch.Tensor]:
        return [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in xs]

    def _server_values_from_local_bi(
        self,
        obs_t_list: List[torch.Tensor],
        prev_act_t_list: List[torch.Tensor],
        hidden_list: Optional[List[torch.Tensor]] = None,
    ):
        if hidden_list is None:
            hidden_list = [None] * self.num_agents
        bi_list, new_hidden_list = [], []
        for i, client in enumerate(self.clients):
            bi_i, h_i = client.encode_bi(obs_t_list[i], prev_act_t_list[i], hidden_list[i])
            bi_list.append(bi_i)
            new_hidden_list.append(h_i)
        values, aux = self.server.forward_bi(bi_list)
        return values, new_hidden_list, aux

    @staticmethod
    def _segment_bounds(done_seq: torch.Tensor) -> List[Tuple[int, int]]:
        """Return half-open episode segment bounds, including terminal transitions."""
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
        """Number of resets between consecutive steps, excluding terminal-at-final-step."""
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

    def _done_segmentation_diagnostics(self, done_list_seq: Optional[List[torch.Tensor]], seq_len: int) -> Dict[str, float]:
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

    def _encode_bi_local(
        self,
        obs_list_seq: List[torch.Tensor],
        prev_act_list_seq: List[torch.Tensor],
        done_list_seq: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Encode local BI tensors, resetting hidden state at done-defined episode boundaries."""
        batch_size, seq_len = obs_list_seq[0].shape[0], obs_list_seq[0].shape[1]
        done_list_seq = self._validate_done_list_seq(done_list_seq, batch_size=batch_size, seq_len=seq_len)
        diagnostics = self._done_segmentation_diagnostics(done_list_seq, seq_len=seq_len)

        bi_local_list: List[torch.Tensor] = []
        for i, client in enumerate(self.clients):
            if done_list_seq is None:
                hidden_i = client.init_hidden(batch_size=batch_size, device=self.device)
                bi_i, _ = client.encode_bi(obs_list_seq[i], prev_act_list_seq[i], hidden_i)
                bi_local_list.append(bi_i)
                continue

            if batch_size != 1:
                raise NotImplementedError("Done-boundary BI segmentation currently supports batch_size=1")
            bounds = self._segment_bounds(done_list_seq[i])
            parts: List[torch.Tensor] = []
            for start, end in bounds:
                hidden_i = client.init_hidden(batch_size=batch_size, device=self.device)
                bi_part, _ = client.encode_bi(
                    obs_list_seq[i][:, start:end, :],
                    prev_act_list_seq[i][:, start:end, :],
                    hidden_i,
                )
                parts.append(bi_part)
            if not parts:
                raise ValueError("No BI segments were produced from non-empty rollout")
            bi_local_list.append(torch.cat(parts, dim=1))
        self.last_bi_encoding_diagnostics = diagnostics
        return bi_local_list

    @torch.no_grad()
    def act(
        self,
        obs_list: List[np.ndarray],
        prev_act_list: List[np.ndarray],
        critic_hidden_list: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ):
        obs_t_list = self._tensor_list(obs_list)
        prev_act_t_list = self._tensor_list(prev_act_list)
        actions_t, logps_t = [], []
        for i, client in enumerate(self.clients):
            action_i, logp_i = client.act(obs_t_list[i], deterministic=deterministic)
            actions_t.append(action_i)
            logps_t.append(logp_i)
        values_t, new_hidden_list, _ = self._server_values_from_local_bi(obs_t_list, prev_act_t_list, critic_hidden_list)
        actions_np = [a.detach().cpu().numpy().astype(np.float32) for a in actions_t]
        logps_np = [float(lp.detach().cpu().item()) for lp in logps_t]
        values_np = [float(v.detach().reshape(-1)[-1].cpu().item()) for v in values_t]
        new_hidden_list = [h.detach() for h in new_hidden_list]
        return actions_np, logps_np, values_np, new_hidden_list

    @torch.no_grad()
    def value(
        self,
        obs_list: List[np.ndarray],
        prev_act_list: List[np.ndarray],
        critic_hidden_list: Optional[List[torch.Tensor]] = None,
    ):
        obs_t_list = self._tensor_list(obs_list)
        prev_act_t_list = self._tensor_list(prev_act_list)
        values_t, new_hidden_list, _ = self._server_values_from_local_bi(obs_t_list, prev_act_t_list, critic_hidden_list)
        values_np = [float(v.detach().reshape(-1)[-1].cpu().item()) for v in values_t]
        new_hidden_list = [h.detach() for h in new_hidden_list]
        return values_np, new_hidden_list

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
        batch_size, seq_len = obs_list_seq[0].shape[0], obs_list_seq[0].shape[1]
        done_list_seq = self._coerce_done_list_seq(done_list_seq, done_seq, batch_size=batch_size, seq_len=seq_len)
        bi_local_list = self._encode_bi_local(obs_list_seq, prev_act_list_seq, done_list_seq=done_list_seq)
        encoding_diag = dict(getattr(self, "last_bi_encoding_diagnostics", {}))
        bi_server_list = [bi.detach().clone().requires_grad_(True) for bi in bi_local_list]
        values_list, _ = self.server.forward_bi(bi_server_list)
        value_losses = [((values_list[i] - ret_list_seq[i]) ** 2).mean() for i in range(self.num_agents)]
        value_loss = torch.stack(value_losses).mean()
        type_branch_diag = None
        type_loss = None
        type_loss_weighted = None
        if self.server.type_predictor is not None and self.hp.bi_type_loss_coef != 0.0:
            type_branch_diag = self.server.forward_type_branch(bi_server_list)
            type_loss = type_branch_diag["loss"]
            type_loss_weighted = type_loss * float(self.hp.bi_type_loss_coef)
        total_server_loss = value_loss if type_loss_weighted is None else value_loss + type_loss_weighted

        self.debug_events.append("autograd_grad_before_server_step")
        dL_dBI = torch.autograd.grad(total_server_loss, bi_server_list, retain_graph=True)
        dL_dBI_local = [g.detach().clone() for g in dL_dBI]
        self.debug_events.append("dL_dBI_detached")

        server_before_step = [p.detach().clone() for p in self.server.parameters()]
        self.server_opt.zero_grad(set_to_none=True)
        total_server_loss.backward()
        nn.utils.clip_grad_norm_(self.server.parameters(), self.hp.max_grad_norm)
        self.server_opt.step()
        self.debug_events.append("server_step")
        self.server_opt.zero_grad(set_to_none=True)
        server_after_step = [p.detach().clone() for p in self.server.parameters()]

        local_before = [[p.detach().clone() for p in client.bi_encoder.parameters()] for client in self.clients]
        server_before_local_backward = [p.detach().clone() for p in self.server.parameters()]
        server_grads_before_local = [None if p.grad is None else p.grad.detach().clone() for p in self.server.parameters()]
        shared_before = None
        if self.shared_critic_opt is not None:
            shared_before = [p.detach().clone() for p in self._unique_parameters(self.clients[0].bi_encoder.extractor_parameter_modules())]

        if apply_local_handoff:
            for opt in self.local_critic_opts:
                opt.zero_grad(set_to_none=True)
            if self.shared_critic_opt is not None:
                self.shared_critic_opt.zero_grad(set_to_none=True)
            torch.autograd.backward(bi_local_list, dL_dBI_local)
            self.debug_events.append("local_backward")
            if self.shared_critic_opt is not None:
                for client in self.clients:
                    nn.utils.clip_grad_norm_(
                        self._modules_parameters(client.bi_encoder.local_parameter_modules()),
                        self.hp.max_grad_norm,
                    )
                shared_params = self._unique_parameters(self.clients[0].bi_encoder.extractor_parameter_modules())
                if shared_params:
                    nn.utils.clip_grad_norm_(shared_params, self.hp.max_grad_norm)
            else:
                for client in self.clients:
                    nn.utils.clip_grad_norm_(client.bi_encoder.parameters(), self.hp.max_grad_norm)
            for opt in self.local_critic_opts:
                opt.step()
            if self.shared_critic_opt is not None:
                self.shared_critic_opt.step()
            self.debug_events.append("local_step")

        server_after_local_backward = [p.detach().clone() for p in self.server.parameters()]
        server_grads_after_local = [None if p.grad is None else p.grad.detach().clone() for p in self.server.parameters()]
        local_after = [[p.detach().clone() for p in client.bi_encoder.parameters()] for client in self.clients]

        def any_changed(before, after):
            return any(not torch.allclose(b, a) for b, a in zip(before, after))

        server_changed_on_server_step = any_changed(server_before_step, server_after_step)
        server_changed_during_local = any_changed(server_before_local_backward, server_after_local_backward)
        local_changed = any(any_changed(b, a) for b, a in zip(local_before, local_after))
        server_grads_created_during_local = any(
            (before is None and after is not None) or
            (before is not None and after is not None and not torch.allclose(before, after))
            for before, after in zip(server_grads_before_local, server_grads_after_local)
        )
        diagnostics = {
            "loss": float(value_loss.detach().cpu().item()),
            "total_server_loss": float(total_server_loss.detach().cpu().item()),
            "value_losses": [float(v.detach().cpu().item()) for v in value_losses],
            "type_loss": None if type_loss is None else float(type_loss.detach().cpu().item()),
            "type_loss_weighted": None if type_loss_weighted is None else float(type_loss_weighted.detach().cpu().item()),
            "type_branch_active": float(type_branch_diag is not None),
            "values_list": values_list,
            "dL_dBI": dL_dBI_local,
            "type_branch": type_branch_diag,
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
        }
        self.last_critic_diagnostics = diagnostics
        return diagnostics

    def _actor_update_from_tensors(
        self,
        obs_list_np: List[np.ndarray],
        act_list_np: List[np.ndarray],
        old_logp_list_np: List[np.ndarray],
        adv_list_np: List[np.ndarray],
    ) -> Dict[str, Any]:
        actor_losses, entropies, approx_kls, clipfracs, ratio_stds = [], [], [], [], []
        obs_tensors, act_tensors, old_logp_tensors, pre_logp_tensors = [], [], [], []
        actor_before = [[p.detach().clone() for p in client.actor.parameters()] for client in self.clients]
        for opt in self.actor_opts:
            opt.zero_grad(set_to_none=True)
        for i, client in enumerate(self.clients):
            obs_flat = torch.as_tensor(obs_list_np[i], dtype=torch.float32, device=self.device)
            act_flat = torch.as_tensor(act_list_np[i], dtype=torch.float32, device=self.device)
            old_logp = torch.as_tensor(old_logp_list_np[i], dtype=torch.float32, device=self.device)
            adv = torch.as_tensor(adv_list_np[i], dtype=torch.float32, device=self.device).detach()
            new_logp, entropy, _, _ = client.evaluate_actor(obs_flat, act_flat)
            obs_tensors.append(obs_flat)
            act_tensors.append(act_flat)
            old_logp_tensors.append(old_logp)
            pre_logp_tensors.append(new_logp.detach())
            log_ratio = torch.clamp(new_logp - old_logp, -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.hp.clip_range, 1.0 + self.hp.clip_range) * adv
            loss_i = -torch.min(surr1, surr2).mean()
            actor_losses.append(loss_i)
            entropies.append(entropy.mean())
            with torch.no_grad():
                approx_kls.append((old_logp - new_logp).mean().item())
                clipfracs.append((torch.abs(ratio - 1.0) > self.hp.clip_range).float().mean().item())
                ratio_stds.append(ratio.std(unbiased=False).item())
        actor_loss = torch.stack(actor_losses).mean()
        entropy = torch.stack(entropies).mean()
        actor_total_loss = actor_loss - self.hp.ent_coef * entropy
        actor_total_loss.backward()
        per_agent_grad_norm = []
        for client in self.clients:
            grad_norm = nn.utils.clip_grad_norm_(client.actor.parameters(), self.hp.max_grad_norm)
            per_agent_grad_norm.append(float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm))
        for opt in self.actor_opts:
            opt.step()

        per_agent_param_delta = [
            self._parameter_delta_norm(before, client.actor)
            for before, client in zip(actor_before, self.clients)
        ]
        approx_kls_post, post_logp_deltas = [], []
        post_kl_available = 1.0
        with torch.no_grad():
            for i, client in enumerate(self.clients):
                post_logp, _, _, _ = client.evaluate_actor(obs_tensors[i], act_tensors[i])
                approx_kls_post.append((old_logp_tensors[i] - post_logp).mean().item())
                post_logp_deltas.append((post_logp - pre_logp_tensors[i]).abs().mean().item())
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "approx_kl": self._mean(approx_kls),
            "clipfrac": self._mean(clipfracs),
            "approx_kl_post": self._mean(approx_kls_post),
            "actor_grad_norm": self._mean(per_agent_grad_norm),
            "actor_param_delta": self._mean(per_agent_param_delta),
            "actor_logp_delta_post": self._mean(post_logp_deltas),
            "actor_post_kl_available": post_kl_available,
            "per_agent_actor_loss": [float(x.detach().cpu().item()) for x in actor_losses],
            "per_agent_entropy": [float(x.detach().cpu().item()) for x in entropies],
            "per_agent_approx_kl": approx_kls,
            "per_agent_approx_kl_post": approx_kls_post,
            "per_agent_clipfrac": clipfracs,
            "per_agent_ratio_std": ratio_stds,
            "per_agent_actor_grad_norm": per_agent_grad_norm,
            "per_agent_actor_param_delta": per_agent_param_delta,
        }

    @classmethod
    def _aggregate_actor_epoch_logs(cls, actor_epoch_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not actor_epoch_logs:
            raise ValueError("Cannot aggregate empty actor_epoch_logs")

        last_log = actor_epoch_logs[-1]
        out: Dict[str, Any] = {}
        last_scalar_keys = {"approx_kl", "approx_kl_post"}
        for key in (
            "actor_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
            "approx_kl_post",
            "actor_grad_norm",
            "actor_param_delta",
            "actor_logp_delta_post",
            "actor_post_kl_available",
        ):
            vals = [cls._scalar_float(log.get(key)) for log in actor_epoch_logs]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if not vals:
                continue
            out[key] = float(last_log[key]) if key in last_scalar_keys else float(np.mean(vals))

        list_last_keys = {"per_agent_approx_kl", "per_agent_approx_kl_post"}
        list_keys = (
            "per_agent_actor_loss",
            "per_agent_entropy",
            "per_agent_approx_kl",
            "per_agent_approx_kl_post",
            "per_agent_clipfrac",
            "per_agent_ratio_std",
            "per_agent_actor_grad_norm",
            "per_agent_actor_param_delta",
        )
        for key in list_keys:
            if key not in last_log:
                continue
            if key in list_last_keys:
                out[key] = [float(x) for x in last_log[key]]
                continue
            stacked = np.asarray([log[key] for log in actor_epoch_logs if key in log], dtype=np.float64)
            if stacked.size:
                out[key] = [float(x) for x in stacked.mean(axis=0)]

        out["actor_epochs_used"] = int(len(actor_epoch_logs))
        out["approx_kl_epoch0"] = float(actor_epoch_logs[0].get("approx_kl", 0.0))
        out["approx_kl_last"] = float(last_log.get("approx_kl", 0.0))
        param_deltas = [
            cls._scalar_float(log.get("actor_param_delta"))
            for log in actor_epoch_logs
        ]
        param_deltas = [v for v in param_deltas if v is not None and np.isfinite(v)]
        out["actor_param_delta_min_epoch"] = float(min(param_deltas)) if param_deltas else 0.0
        return out

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

        critic_losses = []
        per_agent_critic_losses = [[] for _ in range(self.num_agents)]
        actor_epoch_logs: List[Dict[str, Any]] = []
        for _ in range(max(1, int(self.hp.n_epochs))):
            diag = self.critic_update_from_tensors(
                obs_seq,
                prev_act_seq,
                ret_seq,
                apply_local_handoff=True,
                done_list_seq=done_seq,
            )
            critic_losses.append(diag["loss"])
            for i, loss_i in enumerate(diag["value_losses"]):
                per_agent_critic_losses[i].append(float(loss_i))

            a_logs = self._actor_update_from_tensors(obs_list_np, act_list_np, old_logp_list_np, adv_list_np)
            actor_epoch_logs.append(a_logs)
            if self.hp.target_kl is not None and float(a_logs.get("approx_kl", 0.0)) > float(self.hp.target_kl):
                break

        actor_logs = self._aggregate_actor_epoch_logs(actor_epoch_logs)
        self.round_idx += 1
        fedavg_groups: List[str] = []
        if self.hp.fedavg_interval > 0 and self.round_idx % self.hp.fedavg_interval == 0:
            fedavg_groups = self.run_fedavg()
        bi_fedavg_applied = False
        if self.bi_extractor_sharing_mode == "fedavg10" and self.bi_extractor_fedavg_interval > 0:
            if self.round_idx % self.bi_extractor_fedavg_interval == 0:
                bi_fedavg_applied = bool(self.run_bi_extractor_fedavg())

        critic_diag = self.last_critic_diagnostics
        logs = {
            "loss/actor": actor_logs["actor_loss"],
            "loss/critic": float(np.mean(critic_losses)),
            "loss/entropy": actor_logs["entropy"],
            "train/approx_kl": actor_logs["approx_kl"],
            "train/approx_kl_post": actor_logs["approx_kl_post"],
            "train/actor_epochs_used": float(actor_logs["actor_epochs_used"]),
            "train/approx_kl_epoch0": actor_logs["approx_kl_epoch0"],
            "train/approx_kl_last": actor_logs["approx_kl_last"],
            "train/clipfrac": actor_logs["clipfrac"],
            "fedavg/round": float(self.round_idx),
            "rollout/length": float(T),
            "rollout/num_episodes": float(critic_diag.get("num_segments", 0.0)),
            "rollout/mean_episode_len": float(critic_diag.get("mean_segment_len", 0.0)),
            "debug/actor_grad_norm": actor_logs["actor_grad_norm"],
            "debug/actor_param_delta": actor_logs["actor_param_delta"],
            "debug/actor_param_delta_min_epoch": actor_logs["actor_param_delta_min_epoch"],
            "debug/actor_logp_delta_post": actor_logs["actor_logp_delta_post"],
            "debug/actor_post_kl_available": actor_logs["actor_post_kl_available"],
            "debug/server_advantage_owner": 1.0,
            "debug/local_actor_reward_value_free": 1.0,
            "debug/critic_gru_reset_enabled": float(critic_diag.get("critic_gru_reset_enabled", False)),
            "debug/critic_gru_done_resets": float(critic_diag.get("critic_gru_done_resets", 0.0)),
            "debug/hidden_resets": float(critic_diag.get("hidden_resets", 0.0)),
            "fedavg/applied": float(bool(fedavg_groups)),
            "fedavg/scope_enhancer": float("enhancer" in fedavg_groups),
            "fedavg/scope_gru": float("gru" in fedavg_groups),
            "bi_fedavg/applied": float(bi_fedavg_applied),
            "bi_fedavg/mode": 1.0 if self.bi_extractor_sharing_mode == "fedavg10" else 0.0,
        }
        logs.update(self._safe_critic_scalar_logs(critic_diag))
        for i in range(self.num_agents):
            logs[f"agent_{i}/adv_mean_raw"] = server_adv.adv_mean_raw[i]
            logs[f"agent_{i}/adv_std_raw"] = server_adv.adv_std_raw[i]
            logs[f"agent_{i}/adv_std_normed"] = per_agent_adv_std_normed[i]
            logs[f"agent_{i}/loss_actor"] = actor_logs["per_agent_actor_loss"][i]
            logs[f"agent_{i}/loss_critic"] = float(np.mean(per_agent_critic_losses[i]))
            logs[f"agent_{i}/entropy"] = actor_logs["per_agent_entropy"][i]
            logs[f"agent_{i}/approx_kl"] = actor_logs["per_agent_approx_kl"][i]
            logs[f"agent_{i}/clipfrac"] = actor_logs["per_agent_clipfrac"][i]
            logs[f"agent_{i}/ratio_std"] = actor_logs["per_agent_ratio_std"][i]
            logs[f"agent_{i}/approx_kl_post"] = actor_logs["per_agent_approx_kl_post"][i]
            logs[f"agent_{i}/debug/actor_grad_norm"] = actor_logs["per_agent_actor_grad_norm"][i]
            logs[f"agent_{i}/debug/actor_param_delta"] = actor_logs["per_agent_actor_param_delta"][i]
        if self.writer is not None and global_step is not None:
            sink = getattr(self, "metrics_logger", None)
            if sink is not None:
                sink.record_update(logs, step=global_step)
            else:
                for k, v in logs.items():
                    self.writer.add_scalar(k, v, global_step)
        return logs

    @staticmethod
    def _has_state(module: nn.Module) -> bool:
        return any(torch.is_tensor(value) for value in module.state_dict().values())

    def fedavg_group_names(self) -> List[str]:
        scope = getattr(self.hp, "fedavg_scope", "both")
        if scope not in {"both", "gru", "enhancer", "none"}:
            raise ValueError(f"Unsupported fedavg_scope={scope!r}; expected both/gru/enhancer/none")
        if scope == "both":
            return ["enhancer", "gru"]
        if scope == "enhancer":
            return ["enhancer"]
        if scope == "gru":
            return ["gru"]
        return []

    def run_fedavg(self) -> List[str]:
        applied: List[str] = []
        for group in self.fedavg_group_names():
            if group == "enhancer":
                modules = [client.actor_fedavg_module() for client in self.clients]
            elif group == "gru":
                modules = [client.critic_fedavg_module() for client in self.clients]
            else:
                raise ValueError(f"Unknown FedAvg group {group!r}")
            if not all(self._has_state(module) for module in modules):
                continue
            fedavg_modules(modules)
            applied.append(group)
        return applied

    def run_bi_extractor_fedavg(self) -> List[str]:
        if self.bi_extractor_sharing_mode != "fedavg10":
            return []
        applied: List[str] = []
        extractor_modules = [client.bi_encoder.bi_extractor for client in self.clients]
        if all(self._has_state(module) for module in extractor_modules):
            fedavg_modules(extractor_modules)
            applied.append("gru")
        fb_modules = [client.bi_encoder.f_B for client in self.clients]
        if all(self._has_state(module) for module in fb_modules):
            fedavg_modules(fb_modules)
            applied.append("f_B")
        return applied

    def optimizer_parameter_sets(self) -> Dict[str, set]:
        actor_ids = set(parameter_ids(client.actor for client in self.clients))
        server_ids = set(id(p) for p in self.server.parameters())
        if self.bi_extractor_sharing_mode == "true_shared":
            local_critic_ids = set()
            for client in self.clients:
                local_critic_ids.update(id(p) for p in self._modules_parameters(client.bi_encoder.local_parameter_modules()))
            shared_critic_ids = set(id(p) for p in self._unique_parameters(self.clients[0].bi_encoder.extractor_parameter_modules()))
        else:
            local_critic_ids = set(parameter_ids(client.bi_encoder for client in self.clients))
            shared_critic_ids = set()
        return {
            "actor": actor_ids,
            "local_critic": local_critic_ids,
            "shared_critic": shared_critic_ids,
            "server": server_ids,
        }
