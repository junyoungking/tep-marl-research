"""G2-only central scalar value treatment over the unchanged G local actors.

The original G trainer is built first under its normal seed and supplies the
three Actor modules, Actor optimizers, action law, and PPO actor update method.
Its BI/GAT modules are inert in this treatment. The active value optimizer
owns only CentralizedRecurrentCritic. No baseline trainer is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from models.centralized_recurrent_critic import CentralizedRecurrentCritic
from trainers.fl_split_ppo_trainer import ServerAdvantageBatch
from utils.fedavg import fedavg_modules
from utils.training_metrics_logger import TrainingMetricsLogger


CENTRAL_INIT_STREAM_SEED = 420028


def _moments(values: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {f"{prefix}/mean": float(x.mean()), f"{prefix}/std": float(x.std()),
            f"{prefix}/min": float(x.min()), f"{prefix}/max": float(x.max())}


def _value_diagnostics(values: np.ndarray, targets: np.ndarray, advantages: np.ndarray,
                       norm_adv: np.ndarray, done: np.ndarray, prefix: str,
                       rewards: np.ndarray, last_value: float, gamma: float) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    r = np.asarray(targets, dtype=np.float64).reshape(-1)
    d = np.asarray(done, dtype=np.float64).reshape(-1)
    if not (len(v) == len(r) == len(d)):
        raise ValueError("Value diagnostic horizon mismatch")
    rew=np.asarray(rewards,dtype=np.float64).reshape(-1)
    if len(rew)!=len(v):
        raise ValueError("Value diagnostic reward horizon mismatch")
    next_v=np.concatenate((v[1:],[float(last_value)]))
    td=rew+float(gamma)*(1-d)*next_v-v
    out = {**_moments(v, f"{prefix}/prediction"),
           **_moments(r, f"{prefix}/target"),
           **_moments(td, f"{prefix}/td_residual"),
           **_moments(advantages, f"{prefix}/advantage_raw"),
           **_moments(norm_adv, f"{prefix}/advantage_normalized"),
           f"{prefix}/mse_rollout": float(np.mean((v-r)**2)),
           f"{prefix}/explained_variance": float(1-np.var(r-v)/np.var(r)) if np.var(r)>0 else float("nan"),
           f"{prefix}/v_target_pearson": float(np.corrcoef(v,r)[0,1]) if np.std(v)>0 and np.std(r)>0 else float("nan"),
           f"{prefix}/terminal_abs_error": float(np.mean(np.abs(v[d>0.5]-r[d>0.5]))) if np.any(d>0.5) else float("nan"),
           f"{prefix}/terminal_count": float(np.count_nonzero(d>0.5)),
           f"{prefix}/trace_mask_zero_count": float(np.count_nonzero(d>0.5)),
           f"{prefix}/bootstrap_mask_zero_count": float(np.count_nonzero(d>0.5))}
    for k in (1,5,10,20):
        idx = np.flatnonzero(d>0.5)-k
        idx = idx[idx>=0]
        out[f"{prefix}/preterminal_{k}_abs_error"] = float(np.mean(np.abs(v[idx]-r[idx]))) if len(idx) else float("nan")
        out[f"{prefix}/preterminal_{k}_count"] = float(len(idx))
    return out


class G2ControlTrainer:
    """Observation-only wrapper around the unchanged G training computation."""

    variant = "G"

    def __init__(self, original_g):
        if original_g.variant != "G":
            raise ValueError("Control must be the G trainer")
        self.base = original_g

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def writer(self):
        return self.base.writer

    @writer.setter
    def writer(self, value):
        self.base.writer = value

    @property
    def metrics_logger(self):
        return getattr(self.base, "metrics_logger", None)

    @metrics_logger.setter
    def metrics_logger(self, value):
        self.base.metrics_logger = value

    def checkpoint_state(self):
        return self.base.checkpoint_state()

    def load_checkpoint_state(self, payload):
        return self.base.load_checkpoint_state(payload)

    def update(self, *, agent_rollouts, last_values, global_step=None):
        client, server = self.base._legacy_rollouts_to_split(agent_rollouts, last_values)
        advantage = self.base._server_compute_advantages(server)
        raw = np.concatenate(advantage.advantages)
        mean, std = raw.mean(), raw.std()+1e-8
        done = np.asarray(agent_rollouts[0]["done"], dtype=np.float32)
        values = [np.asarray(r["val"], dtype=np.float32) for r in agent_rollouts]
        logs = self.base.update(agent_rollouts=agent_rollouts, last_values=last_values,
                                global_step=global_step)
        diagnostic = {}
        for i in range(3):
            diagnostic.update(_value_diagnostics(values[i], advantage.returns[i],
                           advantage.advantages[i], (advantage.advantages[i]-mean)/std,
                           done, f"g2/agent_{i}",
                           np.asarray(agent_rollouts[i]["rew"],dtype=np.float32),
                           float(last_values[i]),self.hp.gamma))
        stack_v = np.stack(values)
        stack_adv = np.stack(advantage.advantages)
        diagnostic["g2/three_v_dispersion_mean"] = float(np.std(stack_v, axis=0).mean())
        diagnostic["g2/three_gae_dispersion_mean"] = float(np.std(stack_adv, axis=0).mean())
        diagnostic["g2/rollout_start_replay_hidden_zero"] = 1.0
        clean = {k:v for k,v in diagnostic.items() if np.isfinite(v)}
        return {**logs, **clean}


class G2CentralScalarTrainer:
    """Central V treatment; Actor calculation delegates to the original G code."""

    variant = "G"

    def __init__(self, original_g):
        if original_g.variant != "G":
            raise ValueError("G2 treatment must originate from the G trainer")
        self.base = original_g
        self.clients = original_g.clients
        self.actor_opts = original_g.actor_opts
        self.hp = original_g.hp
        self.device = original_g.device
        self.num_agents = 3
        self.writer = None
        self.metrics_logger = None
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(CENTRAL_INIT_STREAM_SEED)
            self.central = CentralizedRecurrentCritic(joint_input_dim=28, hidden_dim=64).to(self.device)
        self.central_opt = optim.Adam(self.central.parameters(), lr=self.hp.critic_lr)
        # Phase4 accounting names the value optimizer group "bi". The G2
        # report maps this group to central_opt for the treatment arm.
        self.local_critic_opts = [self.central_opt]
        self.server_opt = original_g.server_opt
        self.priv_opts = original_g.priv_opts
        self.last_hidden_norm = 0.0
        self.last_critic_diagnostics: dict[str, Any] = {}

    @property
    def round_idx(self) -> int:
        return self.base.round_idx

    def init_hidden(self, batch_size: int = 1):
        return [self.central.init_hidden(batch_size=batch_size, device=self.device), None, None]

    @staticmethod
    def _joint(local, previous):
        if [int(x.shape[-1]) for x in local] != [7,10,8]:
            raise ValueError("Actor-local observation order changed")
        if [int(x.shape[-1]) for x in previous] != [1,1,1]:
            raise ValueError("Previous action order changed")
        return torch.cat([*local,*previous], dim=-1)

    @torch.no_grad()
    def act(self, obs_list, prev_act_list, critic_hidden_list=None, deterministic: bool = False):
        obs = self.base._tensor_list(obs_list)
        prev = self.base._tensor_list(prev_act_list)
        actions, logps = [], []
        for client, item in zip(self.clients, obs):
            action, logp = client.act(item, deterministic=deterministic)
            actions.append(action.detach().cpu().numpy().astype(np.float32))
            logps.append(float(logp.detach().cpu().item()))
        h = None if critic_hidden_list is None else critic_hidden_list[0]
        value, new_h, _ = self.central(self._joint(obs, prev), h)
        self.last_hidden_norm = float(new_h.norm().detach().cpu())
        scalar = float(value.detach().reshape(-1)[-1].cpu())
        return actions, logps, [scalar]*3, [new_h.detach(), None, None]

    @torch.no_grad()
    def value(self, obs_list, prev_act_list, critic_hidden_list=None):
        obs = self.base._tensor_list(obs_list)
        prev = self.base._tensor_list(prev_act_list)
        h = None if critic_hidden_list is None else critic_hidden_list[0]
        value, new_h, _ = self.central(self._joint(obs, prev), h)
        scalar = float(value.detach().reshape(-1)[-1].cpu())
        return [scalar]*3, [new_h.detach(), None, None]

    def _replay_values(self, joint: torch.Tensor, done: np.ndarray) -> tuple[torch.Tensor, int]:
        # Matches the existing G value replay lifecycle: zero at rollout start,
        # and zero after each done boundary. Collector hidden carries across
        # rollout boundaries, so the known mismatch remains in both arms.
        ends = np.flatnonzero(done > 0.5).tolist()
        start = 0
        chunks = []
        for end in [x+1 for x in ends] + ([len(done)] if not ends or ends[-1]+1<len(done) else []):
            if end <= start:
                continue
            values, _, _ = self.central(joint[:, start:end, :], self.central.init_hidden(device=self.device))
            chunks.append(values)
            start = end
        if start != len(done):
            raise ValueError("Central replay segmentation did not consume all transitions")
        return torch.cat(chunks, dim=1), len(chunks)

    def update(self, *, agent_rollouts, last_values, global_step=None):
        if len(agent_rollouts) != 3 or len(last_values) != 3:
            raise ValueError("Expected three local Actor rollout channels")
        T = len(agent_rollouts[0]["obs"])
        if any(len(r["obs"]) != T for r in agent_rollouts):
            raise ValueError("Rollout horizon mismatch")
        value_channels = [np.asarray(r["val"], dtype=np.float32) for r in agent_rollouts]
        if not all(np.array_equal(value_channels[0], x) for x in value_channels[1:]):
            raise ValueError("Scalar V collection channels differ")
        if not all(float(last_values[0]) == float(x) for x in last_values[1:]):
            raise ValueError("Scalar V bootstrap channels differ")
        rewards = np.asarray(agent_rollouts[0]["rew"], dtype=np.float32)
        done = np.asarray(agent_rollouts[0]["done"], dtype=np.float32)
        if any(not np.array_equal(rewards, np.asarray(r["rew"], dtype=np.float32)) or
               not np.array_equal(done, np.asarray(r["done"], dtype=np.float32)) for r in agent_rollouts[1:]):
            raise ValueError("Shared team reward/done mismatch")
        advantage, returns = self.base._compute_gae(rewards, value_channels[0], done,
                                                     float(last_values[0]), self.hp.gamma, self.hp.gae_lambda)
        norm_adv = (advantage-advantage.mean())/(advantage.std()+1e-8)
        obs = [np.asarray(r["obs"], dtype=np.float32) for r in agent_rollouts]
        acts = [np.asarray(r["act"], dtype=np.float32) for r in agent_rollouts]
        prev = [np.asarray(r["prev_act"], dtype=np.float32) for r in agent_rollouts]
        old_logp = [np.asarray(r["logp"], dtype=np.float32) for r in agent_rollouts]
        joint = self._joint([torch.as_tensor(x[None], dtype=torch.float32, device=self.device) for x in obs],
                            [torch.as_tensor(x[None], dtype=torch.float32, device=self.device) for x in prev])
        target = torch.as_tensor(returns[None], dtype=torch.float32, device=self.device)
        critic_epoch_logs = []
        for _ in range(int(self.hp.critic_epochs)):
            prediction, segments = self._replay_values(joint, done)
            loss = ((prediction-target)**2).mean()
            self.central_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(self.central.parameters(), self.hp.max_grad_norm)
            self.central_opt.step()
            critic_epoch_logs.append({"loss": float(loss.detach().cpu()),
                                      "value_losses": [float(loss.detach().cpu())]*3,
                                      "num_segments": float(segments),
                                      "mean_segment_len": float(T/segments),
                                      "critic_gru_reset_enabled": True,
                                      "critic_gru_done_resets": float(max(0,segments-1)),
                                      "hidden_resets": float(max(0,segments-1)),
                                      "central_grad_norm": float(grad.detach().cpu())})
        actor_epochs = []
        for _ in range(max(1,int(self.hp.n_epochs))):
            log = self.base._actor_update_from_tensors(obs, acts, old_logp, [norm_adv]*3)
            actor_epochs.append(log)
            if self.hp.target_kl is not None and float(log["approx_kl"]) > float(self.hp.target_kl):
                break
        self.base.round_idx += 1
        groups = []
        if self.hp.fedavg_interval > 0 and self.base.round_idx % self.hp.fedavg_interval == 0:
            fedavg_modules([client.actor_fedavg_module() for client in self.clients])
            groups = ["enhancer"]
        batch = ServerAdvantageBatch([advantage]*3,[returns]*3,
                                     [float(advantage.mean())]*3,[float(advantage.std())]*3)
        logger = self.metrics_logger or TrainingMetricsLogger(self.writer, self.hp.log_every,
                            marl=True, fedavg=True, privacy=True)
        logs = logger.compose_split_update(actor_epochs=actor_epochs,
                    critic_epochs=critic_epoch_logs, server_advantage=batch,
                    normalized_adv_stds=[float(norm_adv.std())]*3,
                    round_idx=self.base.round_idx, fedavg_groups=groups,
                    privacy_epochs=[{} for _ in critic_epoch_logs])
        if self.writer is not None and global_step is not None:
            logger.record_update(logs, step=global_step)
        logs.update(_value_diagnostics(value_channels[0], returns, advantage, norm_adv, done,
                                      "g2/scalar",rewards,float(last_values[0]),self.hp.gamma))
        joint_np=joint.detach().cpu().numpy().reshape(T,28)
        logs.update(_moments(joint_np[:, :25],"g2/scalar/joint25_input"))
        logs.update(_moments(joint_np[:, 25:],"g2/scalar/previous_joint_action3"))
        logs["g2/scalar/central_grad_norm_last"] = critic_epoch_logs[-1]["central_grad_norm"]
        logs["g2/scalar/hidden_norm_last"] = self.last_hidden_norm
        logs["g2/scalar/value_loss_first_epoch"] = critic_epoch_logs[0]["loss"]
        logs["g2/scalar/value_loss_last_epoch"] = critic_epoch_logs[-1]["loss"]
        logs["g2/scalar/rollout_start_replay_hidden_zero"] = 1.0
        self.last_critic_diagnostics = critic_epoch_logs[-1]
        return {k:v for k,v in logs.items() if isinstance(v,(int,float)) and np.isfinite(float(v))}

    def checkpoint_state(self) -> dict:
        return {"g2_format": "central_scalar_v1", "base": self.base.checkpoint_state(),
                "central": self.central.state_dict(), "central_optimizer": self.central_opt.state_dict(),
                "last_hidden_norm": self.last_hidden_norm}

    def load_checkpoint_state(self, payload: dict) -> dict:
        if payload.get("g2_format") != "central_scalar_v1":
            raise ValueError("Wrong G2 central scalar checkpoint")
        result = self.base.load_checkpoint_state(payload["base"])
        self.central.load_state_dict(payload["central"], strict=True)
        self.central_opt.load_state_dict(payload["central_optimizer"])
        self.last_hidden_norm = float(payload["last_hidden_norm"])
        return result
