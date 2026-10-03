from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_fl_split_shallow_hybrid import (  # noqa: E402
    AGENT_CFGS,
    RunningMeanStd,
    build_joint_mv_indices,
    concat_joint_action,
    make_env,
    make_local_obs,
    mean_or_none,
    resolve_device,
    save_obs_rms,
    set_global_seed,
)
from scripts.ippo_tensorboard_compatibility import (  # noqa: E402
    REQUIRED_COMMON_TAGS,
    RawMetricWindow,
    append_raw_window_csv,
    assert_finite_values,
    write_raw_window_to_tensorboard,
)
from scripts.ippo_actor_lr_schedule import actor_lr_at_step, apply_actor_lr  # noqa: E402
from trainers.ippo_ppo_trainer import IPPOHyperParams, IPPOTrainer, TRAINER_VERSION  # noqa: E402


@dataclass
class IPPOModelConfig:
    actor_local_hidden_dims: Tuple[int, ...] = ()
    actor_local_out: int = 64
    actor_shared_hidden_dims: Tuple[int, ...] = ()
    actor_shared_out: int = 64
    actor_action_hidden_dims: Tuple[int, ...] = ()
    critic_local_pre_hidden_dims: Tuple[int, ...] = ()
    critic_local_pre_out: int = 64
    critic_bi_hidden: int = 64
    critic_gru_layers: int = 1
    critic_value_hidden_dims: Tuple[int, ...] = ()
    activate_feature_outputs: bool = True


@dataclass
class ExperimentConfig:
    name: str = "ippo_architecture_matched_smoke"
    actor_lr: float = 1.2e-4
    actor_lr_regime: str = "fixed"
    actor_lr_final: Optional[float] = None
    actor_lr_hold_steps: Optional[int] = None
    actor_lr_schedule_total_steps: int = 200_000
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
    smoke: bool = True

    max_steps: int = 300
    rl_dt_seconds: int = 60
    delta_mv_max: float = 0.2
    production_index: int = 17
    baseline: float = -11.0
    scale: float = 5.0
    reset_reward: bool = False

    device: str = "cpu"
    timestamp_run_dir: bool = False
    run_root: str = os.path.join("runs", "ippo_architecture_matched_smoke")
    stage_a_tuning: bool = False
    model_cfg: IPPOModelConfig = field(default_factory=IPPOModelConfig)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def append_transition(
    buffers: List[Dict[str, list]],
    local_obs_list: List[np.ndarray],
    actions_list: List[np.ndarray],
    prev_actions: List[np.ndarray],
    logps_list: List[float],
    values_list: List[float],
    reward: float,
    done: bool,
) -> None:
    for i in range(len(buffers)):
        buffers[i]["obs"].append(np.asarray(local_obs_list[i], dtype=np.float32))
        buffers[i]["act"].append(np.asarray(actions_list[i], dtype=np.float32))
        buffers[i]["prev_act"].append(np.asarray(prev_actions[i], dtype=np.float32))
        buffers[i]["logp"].append(np.float32(logps_list[i]))
        buffers[i]["val"].append(np.float32(values_list[i]))
        buffers[i]["rew"].append(np.float32(reward))
        buffers[i]["done"].append(np.float32(done))


def buffers_to_rollouts(buffers: List[Dict[str, list]]) -> List[Dict[str, np.ndarray]]:
    rollouts = []
    for buffer in buffers:
        if len(buffer["obs"]) == 0:
            raise ValueError("Cannot update from an empty rollout")
        rollouts.append({key: np.asarray(values, dtype=np.float32) for key, values in buffer.items()})
    return rollouts


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def trainer_parameter_counts(trainer: IPPOTrainer) -> Dict[str, Any]:
    actor_counts = [count_parameters(actor) for actor in trainer.actors]
    critic_counts = [count_parameters(critic) for critic in trainer.critics]
    unique_parameter_ids = {id(p) for module in list(trainer.actors) + list(trainer.critics) for p in module.parameters()}
    return {
        "actors": actor_counts,
        "critics": critic_counts,
        "actor_total": int(sum(actor_counts)),
        "critic_total": int(sum(critic_counts)),
        "total_unique": int(sum(actor_counts) + sum(critic_counts)),
        "unique_parameter_ids": int(len(unique_parameter_ids)),
    }


def make_default_experiment() -> ExperimentConfig:
    return ExperimentConfig()


def build_hparams(exp_cfg: ExperimentConfig) -> IPPOHyperParams:
    return IPPOHyperParams(
        actor_lr=exp_cfg.actor_lr,
        critic_lr=exp_cfg.critic_lr,
        gamma=exp_cfg.gamma,
        gae_lambda=exp_cfg.gae_lambda,
        clip_range=exp_cfg.clip_range,
        ent_coef=exp_cfg.ent_coef,
        max_grad_norm=exp_cfg.max_grad_norm,
        actor_epochs=exp_cfg.actor_epochs,
        critic_epochs=exp_cfg.critic_epochs,
        target_kl=exp_cfg.target_kl,
        rollout_horizon=exp_cfg.rollout_horizon,
        total_steps=exp_cfg.total_steps,
        log_every=exp_cfg.log_every,
    )


def build_trainer(exp_cfg: ExperimentConfig, device: str, writer=None) -> IPPOTrainer:
    obs_dims = [len(cfg.obs_indices) for cfg in AGENT_CFGS]
    act_dims = [len(cfg.mv_indices) for cfg in AGENT_CFGS]
    trainer = IPPOTrainer.build_default(
        obs_dims=obs_dims,
        act_dims=act_dims,
        device=device,
        hp=build_hparams(exp_cfg),
    )
    trainer.writer = writer
    return trainer


def validate_experiment(exp_cfg: ExperimentConfig) -> None:
    if exp_cfg.rollout_horizon != 1024:
        raise ValueError(f"IPPO baseline requires rollout_horizon=1024, got {exp_cfg.rollout_horizon}")
    if exp_cfg.gamma != 0.99:
        raise ValueError(f"IPPO baseline requires gamma=0.99, got {exp_cfg.gamma}")
    if exp_cfg.gae_lambda != 0.95:
        raise ValueError(f"IPPO baseline requires gae_lambda=0.95, got {exp_cfg.gae_lambda}")
    if exp_cfg.clip_range != 0.2:
        raise ValueError(f"IPPO baseline requires clip_range=0.2, got {exp_cfg.clip_range}")
    if exp_cfg.max_grad_norm != 0.5:
        raise ValueError(f"IPPO baseline requires max_grad_norm=0.5, got {exp_cfg.max_grad_norm}")
    if exp_cfg.actor_epochs != 4:
        raise ValueError(f"IPPO baseline requires actor_epochs=4, got {exp_cfg.actor_epochs}")
    if exp_cfg.critic_epochs != 16:
        raise ValueError(f"IPPO baseline requires critic_epochs=16, got {exp_cfg.critic_epochs}")
    if exp_cfg.target_kl != 0.03:
        raise ValueError(f"IPPO baseline requires target_kl=0.03, got {exp_cfg.target_kl}")
    m = exp_cfg.model_cfg
    if m.actor_local_hidden_dims != ():
        raise ValueError(f"IPPO actor requires actor_local_hidden_dims=(), got {m.actor_local_hidden_dims}")
    if m.actor_local_out != 64 or m.actor_shared_out != 64 or m.critic_local_pre_out != 64 or m.critic_bi_hidden != 64:
        raise ValueError("IPPO baseline requires all local widths to be 64")
    if m.actor_shared_hidden_dims != ():
        raise ValueError(f"IPPO actor requires actor_shared_hidden_dims=(), got {m.actor_shared_hidden_dims}")
    if m.actor_action_hidden_dims != ():
        raise ValueError(f"IPPO actor requires actor_action_hidden_dims=(), got {m.actor_action_hidden_dims}")
    if m.critic_local_pre_hidden_dims != ():
        raise ValueError(f"IPPO critic requires critic_local_pre_hidden_dims=(), got {m.critic_local_pre_hidden_dims}")
    if m.critic_gru_layers != 1:
        raise ValueError(f"IPPO critic requires critic_gru_layers=1, got {m.critic_gru_layers}")
    if m.critic_value_hidden_dims != ():
        raise ValueError(f"IPPO critic requires critic_value_hidden_dims=(), got {m.critic_value_hidden_dims}")
    if not m.activate_feature_outputs:
        raise ValueError("IPPO baseline requires activate_feature_outputs=True")
    if exp_cfg.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if exp_cfg.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    actor_lr_at_step(
        regime=exp_cfg.actor_lr_regime,
        start_lr=exp_cfg.actor_lr,
        final_lr=exp_cfg.actor_lr_final,
        hold_steps=exp_cfg.actor_lr_hold_steps,
        total_steps=exp_cfg.actor_lr_schedule_total_steps,
        step=0,
    )
    if exp_cfg.smoke and exp_cfg.total_steps != exp_cfg.rollout_horizon:
        raise ValueError("smoke mode requires total_steps to equal rollout_horizon")
    if exp_cfg.stage_a_tuning:
        if not math.isfinite(exp_cfg.actor_lr) or exp_cfg.actor_lr <= 0:
            raise ValueError(f"Stage A tuning requires a positive finite actor_lr, got {exp_cfg.actor_lr}")
        if not math.isfinite(exp_cfg.critic_lr) or exp_cfg.critic_lr <= 0:
            raise ValueError(f"Stage A tuning requires a positive finite critic_lr, got {exp_cfg.critic_lr}")
        if not math.isfinite(exp_cfg.ent_coef) or exp_cfg.ent_coef < 0:
            raise ValueError(f"Stage A tuning requires a non-negative finite ent_coef, got {exp_cfg.ent_coef}")
    else:
        if exp_cfg.actor_lr != 1.2e-4:
            raise ValueError(f"IPPO baseline requires actor_lr=1.2e-4, got {exp_cfg.actor_lr}")
        if exp_cfg.critic_lr != 1e-4:
            raise ValueError(f"IPPO baseline requires critic_lr=1e-4, got {exp_cfg.critic_lr}")
        if exp_cfg.ent_coef != 0.05:
            raise ValueError(f"IPPO baseline requires ent_coef=0.05, got {exp_cfg.ent_coef}")


def resolve_experiment_run_root(exp_cfg: ExperimentConfig, run_root_override: Optional[str]) -> Path:
    if run_root_override:
        root = Path(run_root_override)
    else:
        root = Path(exp_cfg.run_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if exp_cfg.timestamp_run_dir:
        root = root / f"ippo_architecture_matched_{time.strftime('%Y%m%d_%H%M%S')}"
    return root


def write_config(
    run_root: Path,
    exp_cfg: ExperimentConfig,
    hp: IPPOHyperParams,
    joint_mv_indices: Tuple[int, ...],
    device: str,
    seed: int,
    parameter_counts: Optional[Dict[str, Any]] = None,
) -> None:
    config = {
        "trainer_version": getattr(hp, "trainer_version", TRAINER_VERSION),
        "seed": seed,
        "device": device,
        "experiment": json_safe(asdict(exp_cfg)),
        "ppo_hp": json_safe(asdict(hp)),
        "model_cfg": json_safe(asdict(exp_cfg.model_cfg)),
        "agents": [json_safe(asdict(cfg)) for cfg in AGENT_CFGS],
        "joint_mv_indices": list(joint_mv_indices),
        "architecture_contract": {
            "type": "architecture-matched recurrent IPPO",
            "actor": {
                "agent_0": "obs[7] -> Linear(7->64) -> ReLU -> Linear(64->64) -> ReLU -> Identity -> Linear(64->1)",
                "agent_1": "obs[10] -> Linear(10->64) -> ReLU -> Linear(64->64) -> ReLU -> Identity -> Linear(64->1)",
                "agent_2": "obs[8] -> Linear(8->64) -> ReLU -> Linear(64->64) -> ReLU -> Identity -> Linear(64->1)",
                "log_std": "learnable Parameter(initial=0, clamp=[-5, 2], std=exp(clamped_log_std))",
            },
            "critic": {
                "agent_0": "concat(obs[7], prev_action[1]) -> Linear(8->64) -> ReLU -> GRU(64->64, layers=1, batch_first=True, bidirectional=False) -> Linear(64->1)",
                "agent_1": "concat(obs[10], prev_action[1]) -> Linear(11->64) -> ReLU -> GRU(64->64, layers=1, batch_first=True, bidirectional=False) -> Linear(64->1)",
                "agent_2": "concat(obs[8], prev_action[1]) -> Linear(9->64) -> ReLU -> GRU(64->64, layers=1, batch_first=True, bidirectional=False) -> Linear(64->1)",
            },
        },
        "rollout_collection": {
            "mode": "fixed_horizon_continue_across_done",
            "horizon": int(exp_cfg.rollout_horizon),
            "done_behavior": "On done, log episode, reset env, reset critic hidden, reset prev_action, and continue filling the same rollout buffer.",
        },
        "update_contract": {
            "gamma": exp_cfg.gamma,
            "gae_lambda": exp_cfg.gae_lambda,
            "clip_range": exp_cfg.clip_range,
            "entropy_coef": exp_cfg.ent_coef,
            "actor_epochs": exp_cfg.actor_epochs,
            "critic_epochs": exp_cfg.critic_epochs,
            "target_kl": exp_cfg.target_kl,
            "max_grad_norm": exp_cfg.max_grad_norm,
            "actor_lr": exp_cfg.actor_lr,
            "actor_lr_regime": exp_cfg.actor_lr_regime,
            "actor_lr_final": exp_cfg.actor_lr_final,
            "actor_lr_hold_steps": exp_cfg.actor_lr_hold_steps,
            "actor_lr_schedule_total_steps": exp_cfg.actor_lr_schedule_total_steps,
            "critic_lr": exp_cfg.critic_lr,
            "value_clipping": None,
            "scheduler": None,
            "actor_scheduler": exp_cfg.actor_lr_regime,
            "critic_scheduler": None,
        },
        "normalization_contract": {
            "obs_rms": "per-agent RunningMeanStd over local observations; update in training, freeze in eval",
            "reward": f"(profit - {exp_cfg.baseline}) / {exp_cfg.scale}",
        },
        "checkpoint_contract": {
            "config.json": True,
            "agent_0_actor.pt": True,
            "agent_1_actor.pt": True,
            "agent_2_actor.pt": True,
            "agent_0_critic.pt": True,
            "agent_1_critic.pt": True,
            "agent_2_critic.pt": True,
            "agent_0_obs_rms.npz": True,
            "agent_1_obs_rms.npz": True,
            "agent_2_obs_rms.npz": True,
            "result_summary.json": True,
        },
        "logging_contract": {
            "required_tags": list(REQUIRED_COMMON_TAGS) + [
                "train/actor_lr",
                "agent_0/train/actor_lr",
                "agent_1/train/actor_lr",
                "agent_2/train/actor_lr",
                "loss/actor_policy",
                "train/approx_kl_post",
                "agent_0/loss_actor_total",
                "agent_1/loss_actor_total",
                "agent_2/loss_actor_total",
            ],
            "forbidden_tags": ["fedavg/*", "privacy/*"],
        },
        "non_goals": [
            "full_200k_training",
            "hyperparameter_tuning",
            "multi_seed_benchmarking",
            "server_critic",
            "GAT",
            "FedAvg",
            "privacy_modules",
        ],
        "parameter_counts": json_safe(parameter_counts or {}),
    }
    write_json(run_root / "config.json", config)


def run_one_experiment(
    exp_cfg: ExperimentConfig,
    seed: int = 42,
    run_root_override: Optional[str] = None,
    writer=None,
) -> Dict[str, Any]:
    validate_experiment(exp_cfg)
    device = resolve_device(exp_cfg.device)
    set_global_seed(seed)
    run_root = resolve_experiment_run_root(exp_cfg, run_root_override)
    run_root.mkdir(parents=True, exist_ok=True)

    joint_mv_indices = build_joint_mv_indices(AGENT_CFGS)
    env = None
    try:
        env, full_obs, _ = make_env(
            mv_indices=joint_mv_indices,
            seed=seed,
            rl_dt_seconds=exp_cfg.rl_dt_seconds,
            max_steps=exp_cfg.max_steps,
            delta_mv_max=exp_cfg.delta_mv_max,
            production_index=exp_cfg.production_index,
            baseline=exp_cfg.baseline,
            scale=exp_cfg.scale,
            reset_reward=exp_cfg.reset_reward,
        )
        trainer = build_trainer(exp_cfg, device=device, writer=writer)
        parameter_counts = trainer_parameter_counts(trainer)
        write_config(run_root, exp_cfg, trainer.hp, joint_mv_indices, device, seed, parameter_counts=parameter_counts)

        obs_rms_list = [RunningMeanStd(shape=(len(cfg.obs_indices),)) for cfg in AGENT_CFGS]
        prev_actions = [np.zeros(len(cfg.mv_indices), dtype=np.float32) for cfg in AGENT_CFGS]
        critic_hidden_list = trainer.init_hidden(batch_size=1)

        step_count = 0
        update_count = 0
        skipped_short_count = 0
        episode_seed = seed
        episode_return = 0.0
        episode_len = 0
        raw_profit_values: List[float] = []
        shutdown_values: List[float] = []
        episode_returns: List[float] = []
        final_logs: Dict[str, Any] = {}
        actor_lr_trace: List[Dict[str, Any]] = []
        raw_window = RawMetricWindow()
        raw_window_path = run_root / "tensorboard_raw_metric_windows.csv"
        if raw_window_path.exists():
            raw_window_path.unlink()

        while step_count < exp_cfg.total_steps:
            remaining = exp_cfg.total_steps - step_count
            target_horizon = min(exp_cfg.rollout_horizon, remaining)
            buffers = [{key: [] for key in ("obs", "act", "prev_act", "logp", "val", "rew", "done")} for _ in AGENT_CFGS]
            last_transition_done = False
            episodes_in_rollout = 0

            for _ in range(target_horizon):
                local_obs_list = make_local_obs(np.asarray(full_obs, dtype=np.float32), obs_rms_list, update_rms=True)
                actions_list, logps_list, values_list, critic_hidden_list = trainer.act(
                    obs_list=local_obs_list,
                    prev_act_list=prev_actions,
                    critic_hidden_list=critic_hidden_list,
                    deterministic=False,
                )
                assert_finite_values("actions", actions_list)
                assert_finite_values("log probabilities", logps_list)
                assert_finite_values("values", values_list)
                joint_action = concat_joint_action(actions_list)
                next_full_obs, reward, terminated, truncated, info = env.step(joint_action)
                done = bool(terminated or truncated)

                append_transition(
                    buffers=buffers,
                    local_obs_list=local_obs_list,
                    actions_list=actions_list,
                    prev_actions=prev_actions,
                    logps_list=logps_list,
                    values_list=values_list,
                    reward=float(reward),
                    done=done,
                )

                prev_actions = actions_list
                full_obs = np.asarray(next_full_obs, dtype=np.float32)
                step_count += 1
                episode_return += float(reward)
                episode_len += 1

                raw_prod = info.get("raw_prod", None)
                raw_cost = info.get("raw_cost", None)
                if raw_prod is not None and raw_cost is not None:
                    raw_profit = float(raw_prod) - float(raw_cost)
                else:
                    raw_profit = float(float(reward) * float(exp_cfg.scale) + float(exp_cfg.baseline))
                assert_finite_values("raw metrics", [raw_prod, raw_cost, raw_profit])
                raw_profit_values.append(raw_profit)
                shutdown = float(bool(info.get("shutdown", False)))
                shutdown_values.append(shutdown)
                raw_window.append(step=step_count, raw_prod=raw_prod, raw_cost=raw_cost, shutdown=shutdown)

                if step_count % exp_cfg.log_every == 0:
                    raw_row = raw_window.flush(end_step=step_count)
                    if raw_row is not None:
                        append_raw_window_csv(raw_window_path, raw_row)
                        if writer is not None:
                            write_raw_window_to_tensorboard(writer, raw_row)
                            if hasattr(writer, "flush"):
                                writer.flush()

                if done:
                    episode_returns.append(float(episode_return))
                    episodes_in_rollout += 1
                    if writer is not None:
                        writer.add_scalar("episode/return", float(episode_return), step_count)
                        writer.add_scalar("episode/length", float(episode_len), step_count)
                    episode_return = 0.0
                    episode_len = 0
                    episode_seed += 1
                    last_transition_done = True
                    if step_count >= exp_cfg.total_steps:
                        break
                    full_obs, _ = env.reset(seed=episode_seed)
                    full_obs = np.asarray(full_obs, dtype=np.float32)
                    critic_hidden_list = trainer.init_hidden(batch_size=1)
                    prev_actions = [np.zeros(len(cfg.mv_indices), dtype=np.float32) for cfg in AGENT_CFGS]
                    continue
                if step_count >= exp_cfg.total_steps:
                    break

            collected = len(buffers[0]["obs"])
            if collected == 0:
                raise RuntimeError("No rollout steps were collected")

            if last_transition_done:
                last_values = [0.0 for _ in AGENT_CFGS]
            else:
                last_local_obs_list = make_local_obs(np.asarray(full_obs, dtype=np.float32), obs_rms_list, update_rms=False)
                last_values, _ = trainer.value(
                    obs_list=last_local_obs_list,
                    prev_act_list=prev_actions,
                    critic_hidden_list=critic_hidden_list,
                )

            scheduled_actor_lr = actor_lr_at_step(
                regime=exp_cfg.actor_lr_regime,
                start_lr=exp_cfg.actor_lr,
                final_lr=exp_cfg.actor_lr_final,
                hold_steps=exp_cfg.actor_lr_hold_steps,
                total_steps=exp_cfg.actor_lr_schedule_total_steps,
                step=step_count,
            )
            lr_state = apply_actor_lr(
                trainer,
                actor_lr=scheduled_actor_lr,
                expected_critic_lr=exp_cfg.critic_lr,
            )
            actor_agent_lrs = [float(groups[0]) for groups in lr_state["actor_optimizer_lrs"]]
            actor_lr_trace.append(
                {
                    "step": int(step_count),
                    "scheduled_actor_lr": float(scheduled_actor_lr),
                    "actor_optimizer_lrs": actor_agent_lrs,
                    "critic_optimizer_lrs": [
                        float(groups[0]) for groups in lr_state["critic_optimizer_lrs_after"]
                    ],
                }
            )
            if writer is not None:
                writer.add_scalar("train/actor_lr", float(scheduled_actor_lr), step_count)
                for agent_idx, actual_lr in enumerate(actor_agent_lrs):
                    writer.add_scalar(f"agent_{agent_idx}/train/actor_lr", actual_lr, step_count)

            agent_rollouts = buffers_to_rollouts(buffers)
            logs = trainer.update(agent_rollouts=agent_rollouts, last_values=last_values, global_step=step_count)
            logs["train/actor_lr"] = float(scheduled_actor_lr)
            for agent_idx, actual_lr in enumerate(actor_agent_lrs):
                logs[f"agent_{agent_idx}/train/actor_lr"] = actual_lr
            logs["rollout/episodes_per_update"] = float(episodes_in_rollout)
            update_count += 1
            logs["train/update_count"] = float(update_count)
            logs["train/updates_per_env_step"] = float(update_count) / float(step_count)
            assert_finite_values(
                "training logs",
                [
                    logs["loss/actor"],
                    logs["loss/critic"],
                    logs["loss/entropy"],
                    logs["train/approx_kl"],
                    logs["train/clipfrac"],
                    logs["train/actor_epochs_used"],
                    logs["train/critic_epochs_used"],
                    logs["rollout/length"],
                    logs["rollout/episodes_per_update"],
                    logs["train/update_count"],
                    logs["train/updates_per_env_step"],
                    logs["train/actor_lr"],
                ],
            )
            if writer is not None:
                writer.add_scalar("train/update_count", float(update_count), step_count)
                writer.add_scalar("train/updates_per_env_step", logs["train/updates_per_env_step"], step_count)
            final_logs = dict(logs)
            if exp_cfg.smoke:
                break

        raw_row = raw_window.flush(end_step=step_count)
        if raw_row is not None:
            append_raw_window_csv(raw_window_path, raw_row)
            if writer is not None:
                write_raw_window_to_tensorboard(writer, raw_row)

        trainer.save(str(run_root))
        save_obs_rms(str(run_root), AGENT_CFGS, obs_rms_list)
        if writer is not None and hasattr(writer, "flush"):
            try:
                writer.flush()
            except Exception:
                pass
        result = {
            "variant": "ippo_architecture_matched",
            "trainer_version": getattr(trainer.hp, "trainer_version", TRAINER_VERSION),
            "run_root": str(run_root),
            "seed": seed,
            "total_steps": step_count,
            "updates": update_count,
            "skipped_short": skipped_short_count,
            "smoke": bool(exp_cfg.smoke),
            "final_logs": json_safe(final_logs),
            "recent_profit_mean": mean_or_none(raw_profit_values[-100:]),
            "episode_return_mean": mean_or_none(episode_returns[-20:]),
            "shutdown_rate": mean_or_none(shutdown_values),
            "rollout_collection": {
                "mode": "fixed_horizon_continue_across_done",
                "horizon": int(exp_cfg.rollout_horizon),
                "last_update_rollout_T": int(final_logs.get("rollout/length", 0) or 0),
                "last_update_episodes_per_update": float(final_logs.get("rollout/episodes_per_update", 0.0) or 0.0),
            },
            "training_hps": json_safe(asdict(trainer.hp)),
            "actor_lr_contract": {
                "regime": exp_cfg.actor_lr_regime,
                "start_lr": exp_cfg.actor_lr,
                "final_lr": exp_cfg.actor_lr if exp_cfg.actor_lr_final is None else exp_cfg.actor_lr_final,
                "hold_steps": exp_cfg.actor_lr_hold_steps,
                "schedule_total_steps": exp_cfg.actor_lr_schedule_total_steps,
                "update_timing": "after_rollout_before_critic_then_actor_update",
                "critic_scheduler": None,
            },
            "actor_lr_trace": actor_lr_trace,
            "model_cfg": json_safe(asdict(exp_cfg.model_cfg)),
            "parameter_counts": json_safe(parameter_counts),
            "tensorboard_raw_metric_windows": str(raw_window_path),
            "checkpoint_files_exist": {
                "config.json": (run_root / "config.json").exists(),
                "agent_0_actor.pt": (run_root / "agent_0_actor.pt").exists(),
                "agent_1_actor.pt": (run_root / "agent_1_actor.pt").exists(),
                "agent_2_actor.pt": (run_root / "agent_2_actor.pt").exists(),
                "agent_0_critic.pt": (run_root / "agent_0_critic.pt").exists(),
                "agent_1_critic.pt": (run_root / "agent_1_critic.pt").exists(),
                "agent_2_critic.pt": (run_root / "agent_2_critic.pt").exists(),
                "agent_0_obs_rms.npz": (run_root / "agent_0_obs_rms.npz").exists(),
                "agent_1_obs_rms.npz": (run_root / "agent_1_obs_rms.npz").exists(),
                "agent_2_obs_rms.npz": (run_root / "agent_2_obs_rms.npz").exists(),
                "result_summary.json": True,
            },
            "checkpoint_contract": {
                "required_files": [
                    "config.json",
                    "agent_0_actor.pt",
                    "agent_1_actor.pt",
                    "agent_2_actor.pt",
                    "agent_0_critic.pt",
                    "agent_1_critic.pt",
                    "agent_2_critic.pt",
                    "agent_0_obs_rms.npz",
                    "agent_1_obs_rms.npz",
                    "agent_2_obs_rms.npz",
                    "result_summary.json",
                ]
            },
            "metric_windows": {
                "recent_profit_mean": "last 100 raw profit samples",
                "episode_return_mean": "last 20 completed episodes",
                "shutdown_rate": "all collected rollout steps",
            },
        }
        if writer is not None:
            result["tensorboard_log_dir"] = str(getattr(writer, "log_dir", run_root / "tensorboard"))
        write_json(run_root / "result_summary.json", result)
        return result
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the architecture-matched recurrent IPPO baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--stage-a-tuning", action="store_true", default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--actor-lr-regime", choices=("fixed", "hold_then_linear"), default=None)
    parser.add_argument("--actor-lr-final", type=float, default=None)
    parser.add_argument("--actor-lr-hold-steps", type=int, default=None)
    parser.add_argument("--actor-lr-schedule-total-steps", type=int, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-horizon", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--reset-reward", action="store_true", default=None)
    return parser.parse_args(argv)


def experiment_from_args(args: argparse.Namespace) -> ExperimentConfig:
    exp_cfg = make_default_experiment()
    overrides: Dict[str, Any] = {}
    for arg_name, field_name in [
        ("actor_lr", "actor_lr"),
        ("actor_lr_regime", "actor_lr_regime"),
        ("actor_lr_final", "actor_lr_final"),
        ("actor_lr_hold_steps", "actor_lr_hold_steps"),
        ("actor_lr_schedule_total_steps", "actor_lr_schedule_total_steps"),
        ("critic_lr", "critic_lr"),
        ("ent_coef", "ent_coef"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    for arg_name, field_name in [
        ("total_steps", "total_steps"),
        ("rollout_horizon", "rollout_horizon"),
        ("log_every", "log_every"),
        ("max_steps", "max_steps"),
        ("baseline", "baseline"),
        ("scale", "scale"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    if args.name is not None:
        overrides["name"] = args.name
    if args.run_root is not None:
        overrides["run_root"] = args.run_root
    if args.smoke is True:
        overrides["smoke"] = True
    if args.no_smoke:
        overrides["smoke"] = False
    if args.stage_a_tuning is True:
        overrides["stage_a_tuning"] = True
    if args.reset_reward is not None:
        overrides["reset_reward"] = bool(args.reset_reward)
    if overrides:
        exp_cfg = replace(exp_cfg, **overrides)
    exp_cfg = replace(exp_cfg, device=args.device)
    return exp_cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        exp_cfg = experiment_from_args(args)
        result = run_one_experiment(exp_cfg, seed=args.seed)
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ippo-train] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
