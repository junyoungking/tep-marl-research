from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.ippo_actor_lr_schedule import actor_lr_at_step  # noqa: E402
from scripts.ippo_tensorboard_compatibility import (  # noqa: E402
    REQUIRED_COMMON_TAGS,
    assert_finite_values,
)
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
from trainers.fully_centralized_ppo_trainer import (  # noqa: E402
    TRAINER_VERSION,
    FullyCentralizedPPOHyperParams,
    FullyCentralizedPPOTrainer,
)
from utils.training_metrics_logger import TrainingMetricsLogger  # noqa: E402


@dataclass
class ExperimentConfig:
    name: str = "fully_centralized_ppo_smoke"
    candidate_id: str = "smoke"
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
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64
    concurrent_heavy_job_status: str = "not_instrumented"
    max_grad_norm: float = 0.5
    actor_epochs: int = 4
    critic_epochs: int = 16
    target_kl: float = 0.03
    rollout_horizon: int = 1024
    total_steps: int = 1024
    log_every: int = 1000
    smoke: bool = True
    stage_c1_tuning: bool = False
    max_steps: int = 300
    rl_dt_seconds: int = 60
    delta_mv_max: float = 0.2
    production_index: int = 17
    baseline: float = -11.0
    scale: float = 5.0
    reset_reward: bool = False
    device: str = "cpu"
    run_root: str = os.path.join("runs", "fully_centralized_ppo", "smoke")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if key != "diagnostics"}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        result = value.detach().cpu()
        return result.item() if result.ndim == 0 else result.tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def build_hparams(config: ExperimentConfig) -> FullyCentralizedPPOHyperParams:
    return FullyCentralizedPPOHyperParams(
        actor_lr=config.actor_lr, critic_lr=config.critic_lr, gamma=config.gamma,
        gae_lambda=config.gae_lambda, clip_range=config.clip_range, ent_coef=config.ent_coef,
        max_grad_norm=config.max_grad_norm, actor_epochs=config.actor_epochs,
        critic_epochs=config.critic_epochs, target_kl=config.target_kl,
        rollout_horizon=config.rollout_horizon, total_steps=config.total_steps,
        log_every=config.log_every,
    )


def validate_experiment(config: ExperimentConfig) -> None:
    exact = {
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
        "max_grad_norm": 0.5, "actor_epochs": 4, "critic_epochs": 16,
        "target_kl": 0.03, "rollout_horizon": 1024, "baseline": -11.0, "scale": 5.0,
    }
    for name, expected in exact.items():
        if getattr(config, name) != expected:
            raise ValueError(f"Centralized fairness contract requires {name}={expected}")
    if config.total_steps <= 0 or config.actor_lr <= 0 or config.critic_lr <= 0 or config.ent_coef < 0:
        raise ValueError("Invalid centralized PPO optimization configuration")
    if config.actor_hidden_dim <= 0 or config.critic_hidden_dim <= 0:
        raise ValueError("actor_hidden_dim and critic_hidden_dim must be positive")
    if config.smoke and config.total_steps != 1024:
        raise ValueError("Smoke test requires one exact 1024-step rollout")
    actor_lr_at_step(
        regime=config.actor_lr_regime, start_lr=config.actor_lr,
        final_lr=config.actor_lr_final, hold_steps=config.actor_lr_hold_steps,
        total_steps=config.actor_lr_schedule_total_steps, step=0,
    )


def build_trainer(config: ExperimentConfig, device: str, seed: int, writer: Optional[Any] = None) -> FullyCentralizedPPOTrainer:
    return FullyCentralizedPPOTrainer.build_default(
        obs_dims=[len(agent.obs_indices) for agent in AGENT_CFGS],
        act_dims=[len(agent.mv_indices) for agent in AGENT_CFGS],
        actor_hidden_dim=config.actor_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        device=device, hp=build_hparams(config), writer=writer, seed=seed,
    )


def parameter_counts(trainer: FullyCentralizedPPOTrainer) -> Dict[str, int]:
    actor = trainer.count_parameters(trainer.actor)
    critic = trainer.count_parameters(trainer.centralized_critic)
    return {"centralized_actor": actor, "centralized_critic": critic, "total": actor + critic}


def empty_rollout_buffer() -> Dict[str, List[Any]]:
    return {key: [] for key in ("joint_obs", "joint_input", "joint_action", "old_log_prob", "value", "reward", "done")}


def freeze_rollout(buffer: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(values, dtype=np.float32) for key, values in buffer.items()}


def split_joint_action(joint_action: np.ndarray) -> List[np.ndarray]:
    flat = np.asarray(joint_action, dtype=np.float32).reshape(3)
    return [flat[index:index + 1] for index in range(3)]


def write_config(run_root: Path, config: ExperimentConfig, trainer: FullyCentralizedPPOTrainer, device: str, seed: int) -> None:
    actor_parameters = trainer.count_parameters(trainer.actor)
    critic_parameters = trainer.count_parameters(trainer.centralized_critic)
    actor_hidden_dim = int(config.actor_hidden_dim)
    critic_hidden_dim = int(config.critic_hidden_dim)
    payload = {
        "trainer_version": TRAINER_VERSION, "seed": int(seed), "device": device,
        "experiment": asdict(config), "ppo_hp": asdict(trainer.hp),
        "architecture_contract": {
            "actor": f"concat(normalized obs0[7],obs1[10],obs2[8]) -> Linear(25,{actor_hidden_dim}) -> ReLU -> Linear({actor_hidden_dim},{actor_hidden_dim}) -> ReLU -> mu(3), diagonal Gaussian",
            "actor_hidden_dims": [actor_hidden_dim, actor_hidden_dim],
            "actor_parameter_count": actor_parameters,
            "critic": f"concat(joint_obs[25],previous_joint_action[3]) -> Linear(28,{critic_hidden_dim}) -> ReLU -> GRU({critic_hidden_dim},{critic_hidden_dim}) -> Linear({critic_hidden_dim},1)",
            "critic_hidden_dim": critic_hidden_dim,
            "critic_parameter_count": critic_parameters,
            "execution": "fully centralized",
            "forbidden": ["local actor copies", "FedAvg", "BI", "GAT", "privacy", "current action critic input"],
        },
        "update_contract": {
            "phase_sequence": ["centralized_critic_all_epochs", "centralized_actor_all_epochs"],
            "advantage": "one centralized GAE normalized once over full rollout",
            "joint_log_probability": "sum over all three action dimensions",
            "actor_lr_regime": config.actor_lr_regime,
            "actor_lr_start": config.actor_lr,
            "actor_lr_final": config.actor_lr if config.actor_lr_final is None else config.actor_lr_final,
            "actor_lr_hold_steps": config.actor_lr_hold_steps,
            "actor_lr_schedule_total_steps": config.actor_lr_schedule_total_steps,
        },
        "rollout_contract": {"horizon": 1024, "continue_across_done": True, "terminal_bootstrap": False},
        "checkpoint_contract": ["centralized_actor.pt", "centralized_critic.pt", *[f"agent_{i}_obs_rms.npz" for i in range(3)]],
        "tensorboard_contract": list(REQUIRED_COMMON_TAGS) + [
            "loss/actor", "loss/critic", "loss/entropy", "train/approx_kl", "train/clipfrac",
            "train/actor_epochs_used", "train/critic_epochs_used", "train/actor_lr", "train/critic_lr",
            "central_critic/loss", "central_critic/grad_norm",
        ],
        "parameter_counts": parameter_counts(trainer),
    }
    write_json(run_root / "config.json", payload)


def run_one_experiment(
    config: ExperimentConfig,
    seed: int = 42,
    run_root_override: Optional[str] = None,
    writer: Optional[Any] = None,
) -> Dict[str, Any]:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()
    validate_experiment(config)
    set_global_seed(seed)
    device = resolve_device(config.device)
    run_root = Path(run_root_override or config.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(config, device, seed, writer)
    metrics_logger = TrainingMetricsLogger(writer, config.log_every, marl=False, fedavg=False, privacy=False)
    trainer.metrics_logger = metrics_logger
    write_config(run_root, config, trainer, device, seed)
    env = None
    try:
        env, full_obs, _ = make_env(
            mv_indices=build_joint_mv_indices(AGENT_CFGS), seed=seed,
            rl_dt_seconds=config.rl_dt_seconds, max_steps=config.max_steps,
            delta_mv_max=config.delta_mv_max, production_index=config.production_index,
            baseline=config.baseline, scale=config.scale, reset_reward=config.reset_reward,
        )
        obs_rms = [RunningMeanStd(shape=(len(agent.obs_indices),)) for agent in AGENT_CFGS]
        previous_action = np.zeros(3, dtype=np.float32)
        critic_hidden = trainer.init_hidden(1)
        step_count = update_count = 0
        episode_seed = seed
        episode_returns: List[float] = []
        raw_profits: List[float] = []
        shutdowns: List[float] = []
        final_logs: Dict[str, Any] = {}
        actor_lr_trace: List[Dict[str, Any]] = []

        while step_count < config.total_steps:
            buffer = empty_rollout_buffer()
            last_transition_done = False
            episodes_in_rollout = 0
            target = min(config.rollout_horizon, config.total_steps - step_count)
            for _ in range(target):
                local_obs = make_local_obs(np.asarray(full_obs, dtype=np.float32), obs_rms, update_rms=True)
                joint_obs = trainer.build_joint_observation(local_obs)
                joint_input = trainer.build_joint_input(local_obs, previous_action)
                joint_action, log_prob, value, critic_hidden = trainer.act(
                    local_obs, previous_action, critic_hidden, deterministic=False
                )
                assert_finite_values("centralized joint action", [joint_action])
                assert_finite_values("centralized joint log probability", [log_prob])
                assert_finite_values("centralized value", [value])
                next_obs, reward, terminated, truncated, info = env.step(concat_joint_action(split_joint_action(joint_action)))
                done = bool(terminated or truncated)
                buffer["joint_obs"].append(joint_obs)
                buffer["joint_input"].append(joint_input)
                buffer["joint_action"].append(joint_action)
                buffer["old_log_prob"].append(np.float32(log_prob))
                buffer["value"].append(np.float32(value))
                buffer["reward"].append(np.float32(reward))
                buffer["done"].append(np.float32(done))
                previous_action = np.asarray(joint_action, dtype=np.float32).reshape(3)
                full_obs = np.asarray(next_obs, dtype=np.float32)
                step_count += 1
                raw_prod, raw_cost = info.get("raw_prod"), info.get("raw_cost")
                raw_profit = float(raw_prod) - float(raw_cost) if raw_prod is not None and raw_cost is not None else float(reward) * config.scale + config.baseline
                raw_profits.append(raw_profit)
                shutdown = float(bool(info.get("shutdown", False)))
                shutdowns.append(shutdown)
                completed_episode = metrics_logger.record_transition(
                    step=step_count, reward=float(reward), raw_prod=raw_prod,
                    raw_cost=raw_cost, shutdown=shutdown, done=done,
                )
                last_transition_done = done
                if done:
                    if completed_episode is None:
                        raise AssertionError("terminal transition missing from shared logger")
                    episode_returns.append(completed_episode.return_value)
                    episodes_in_rollout += 1
                    episode_seed += 1
                    if step_count < config.total_steps:
                        full_obs, _ = env.reset(seed=episode_seed)
                        full_obs = np.asarray(full_obs, dtype=np.float32)
                    previous_action = np.zeros(3, dtype=np.float32)
                    critic_hidden = trainer.init_hidden(1)
                if step_count >= config.total_steps:
                    break

            if last_transition_done:
                last_value = 0.0
            else:
                last_obs = make_local_obs(np.asarray(full_obs, dtype=np.float32), obs_rms, update_rms=False)
                last_value, _ = trainer.value(last_obs, previous_action, critic_hidden)
            scheduled_actor_lr = actor_lr_at_step(
                regime=config.actor_lr_regime, start_lr=config.actor_lr,
                final_lr=config.actor_lr_final, hold_steps=config.actor_lr_hold_steps,
                total_steps=config.actor_lr_schedule_total_steps, step=step_count,
            )
            for group in trainer.actor_opt.param_groups:
                group["lr"] = float(scheduled_actor_lr)
            if abs(float(trainer.critic_opt.param_groups[0]["lr"]) - config.critic_lr) > 1e-15:
                raise AssertionError("Critic optimizer LR changed unexpectedly")
            actor_lr_trace.append({"global_step": step_count, "actor_lr": scheduled_actor_lr, "critic_lr": config.critic_lr})
            logs = trainer.update(freeze_rollout(buffer), last_value, step_count)
            update_count += 1
            logs["rollout/episodes_per_update"] = float(episodes_in_rollout)
            logs["train/update_count"] = float(update_count)
            logs["train/updates_per_env_step"] = float(update_count) / float(step_count)
            logs["train/actor_lr"] = float(scheduled_actor_lr)
            logs["train/critic_lr"] = float(config.critic_lr)
            metrics_logger.record_update({key: logs[key] for key in ("rollout/episodes_per_update", "train/update_count", "train/updates_per_env_step", "train/actor_lr", "train/critic_lr")}, step=step_count)
            final_logs = {key: value for key, value in logs.items() if key != "diagnostics"}
            if config.smoke:
                break

        metrics_logger.flush(step=max(step_count, 1))
        trainer.save(str(run_root))
        save_obs_rms(str(run_root), AGENT_CFGS, obs_rms)
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()
        required = ["config.json", "centralized_actor.pt", "centralized_critic.pt", *[f"agent_{i}_obs_rms.npz" for i in range(3)]]
        result = {
            "variant": "fully_centralized_ppo", "trainer_version": TRAINER_VERSION,
            "candidate_id": config.candidate_id, "run_root": str(run_root), "seed": seed,
            "total_steps": step_count, "updates": update_count, "smoke": config.smoke,
            "final_logs": final_logs, "recent_profit_mean": mean_or_none(raw_profits[-100:]),
            "episode_return_mean": mean_or_none(episode_returns[-20:]), "shutdown_rate": mean_or_none(shutdowns),
            "parameter_counts": parameter_counts(trainer), "actor_lr_trace": actor_lr_trace,
            "checkpoint_files_exist": {name: (run_root / name).exists() for name in required},
            "tensorboard_raw_metric_windows": None,
        }
        ended_perf = time.perf_counter()
        ended_at_utc = datetime.now(timezone.utc).isoformat()
        training_wall_seconds = float(ended_perf - started_perf)
        timing = {
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
            "training_wall_seconds": training_wall_seconds,
            "environment_steps": int(step_count),
            "steps_per_second": float(step_count / training_wall_seconds) if training_wall_seconds > 0 else None,
            "resolved_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if str(device).startswith("cuda") and torch.cuda.is_available() else None,
            "cpu_identity": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "concurrent_heavy_job_status": config.concurrent_heavy_job_status,
            "elapsed_clock": "time.perf_counter",
        }
        result["timing"] = timing
        result["checkpoint_files_exist"]["result_summary.json"] = True
        write_json(run_root / "result_summary.json", result)
        write_json(run_root / "training_timing.json", timing)
        return result
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def load_obs_rms(run_root: str | Path) -> List[RunningMeanStd]:
    root = Path(run_root)
    result = []
    for agent in AGENT_CFGS:
        payload = np.load(root / f"{agent.name}_obs_rms.npz")
        rms = RunningMeanStd(shape=(len(agent.obs_indices),))
        rms.mean = np.asarray(payload["mean"], dtype=np.float64)
        rms.var = np.asarray(payload["var"], dtype=np.float64)
        rms.count = float(payload["count"])
        result.append(rms)
    return result


def load_frozen_trainer(run_root: str | Path, device: str = "cpu") -> tuple[FullyCentralizedPPOTrainer, ExperimentConfig, List[RunningMeanStd]]:
    root = Path(run_root)
    payload = json.loads((root / "config.json").read_text(encoding="utf-8-sig"))
    allowed = {field.name for field in fields(ExperimentConfig)}
    config = ExperimentConfig(**{key: value for key, value in payload["experiment"].items() if key in allowed})
    trainer = build_trainer(config, device, int(payload["seed"]))
    trainer.load(str(root), map_location=device)
    trainer.eval()
    for parameter in trainer.parameters():
        parameter.requires_grad_(False)
    return trainer, config, load_obs_rms(root)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fully Centralized PPO on TEP")
    parser.add_argument("--candidate-id", default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=1.2e-4)
    parser.add_argument("--actor-lr-regime", choices=("fixed", "hold_then_linear"), default="fixed")
    parser.add_argument("--actor-lr-final", type=float, default=None)
    parser.add_argument("--actor-lr-hold-steps", type=int, default=None)
    parser.add_argument("--actor-lr-schedule-total-steps", type=int, default=200_000)
    parser.add_argument("--entropy", type=float, default=0.05)
    parser.add_argument("--actor-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--concurrent-heavy-job-status", default="not_instrumented")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig(
        name=f"fully_centralized_ppo_{args.candidate_id}_seed{args.seed}", candidate_id=args.candidate_id,
        actor_lr=args.actor_lr, actor_lr_regime=args.actor_lr_regime,
        actor_lr_final=args.actor_lr_final, actor_lr_hold_steps=args.actor_lr_hold_steps,
        actor_lr_schedule_total_steps=args.actor_lr_schedule_total_steps,
        ent_coef=args.entropy, actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        concurrent_heavy_job_status=args.concurrent_heavy_job_status,
        total_steps=args.total_steps, smoke=args.smoke,
        stage_c1_tuning=not args.smoke, device=args.device, run_root=args.run_root,
    )
    writer = None
    try:
        if not args.no_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            root = Path(args.run_root)
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            writer = SummaryWriter(log_dir=str(root / "tensorboard"))
        result = run_one_experiment(config, seed=args.seed, writer=writer)
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[fully-centralized-ppo] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
