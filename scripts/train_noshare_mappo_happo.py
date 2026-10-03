from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fl_split_actor import SplitLocalActor  # noqa: E402
from scripts.ippo_tensorboard_compatibility import (  # noqa: E402
    REQUIRED_COMMON_TAGS,
    RawMetricWindow,
    append_raw_window_csv,
    assert_finite_values,
    write_raw_window_to_tensorboard,
)
from scripts.ippo_actor_lr_schedule import actor_lr_at_step  # noqa: E402
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
from trainers.noshare_mappo_happo_trainer import (  # noqa: E402
    TRAINER_VERSION,
    NoShareMAPPOHAPPOTrainer,
    NoShareMARLHyperParams,
)


@dataclass
class ExperimentConfig:
    mode: str = "mappo"
    name: str = "noshare_mappo_smoke"
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
    stage_b1_tuning: bool = False
    max_steps: int = 300
    rl_dt_seconds: int = 60
    delta_mv_max: float = 0.2
    production_index: int = 17
    baseline: float = -11.0
    scale: float = 5.0
    reset_reward: bool = False
    device: str = "cpu"
    run_root: str = os.path.join("runs", "noshare_mappo_smoke")


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
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def build_hparams(config: ExperimentConfig) -> NoShareMARLHyperParams:
    return NoShareMARLHyperParams(
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        max_grad_norm=config.max_grad_norm,
        actor_epochs=config.actor_epochs,
        critic_epochs=config.critic_epochs,
        target_kl=config.target_kl,
        rollout_horizon=config.rollout_horizon,
        total_steps=config.total_steps,
        log_every=config.log_every,
    )


def validate_experiment(config: ExperimentConfig) -> None:
    exact = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "max_grad_norm": 0.5,
        "actor_epochs": 4,
        "critic_epochs": 16,
        "target_kl": 0.03,
        "rollout_horizon": 1024,
        "baseline": -11.0,
        "scale": 5.0,
    }
    for name, expected in exact.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(f"Fairness contract requires {name}={expected}, got {actual}")
    if config.mode not in ("mappo", "happo"):
        raise ValueError("mode must be mappo or happo")
    if config.stage_b1_tuning:
        if config.actor_lr <= 0 or config.critic_lr <= 0 or config.ent_coef < 0:
            raise ValueError("Stage B1 learning rates must be positive and entropy must be non-negative")
        if config.actor_lr_regime not in ("fixed", "hold_then_linear"):
            raise ValueError(f"Unsupported Stage B1 actor LR regime: {config.actor_lr_regime}")
        if config.actor_lr_regime == "hold_then_linear":
            if config.actor_lr_final is None or config.actor_lr_hold_steps is None:
                raise ValueError("hold_then_linear requires actor_lr_final and actor_lr_hold_steps")
            actor_lr_at_step(
                regime=config.actor_lr_regime,
                start_lr=config.actor_lr,
                final_lr=config.actor_lr_final,
                hold_steps=config.actor_lr_hold_steps,
                total_steps=config.actor_lr_schedule_total_steps,
                step=0,
            )
    else:
        if config.actor_lr != 1.2e-4 or config.critic_lr != 1e-4 or config.ent_coef != 0.05:
            raise ValueError("Verified baseline requires actor_lr=1.2e-4, critic_lr=1e-4, ent_coef=0.05")
    if config.smoke and config.total_steps != config.rollout_horizon:
        raise ValueError("smoke mode requires exactly one 1024-step rollout/update")
    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive")


def build_trainer(
    config: ExperimentConfig,
    device: str,
    seed: int,
    writer: Optional[Any] = None,
) -> NoShareMAPPOHAPPOTrainer:
    return NoShareMAPPOHAPPOTrainer.build_default(
        obs_dims=[len(agent.obs_indices) for agent in AGENT_CFGS],
        act_dims=[len(agent.mv_indices) for agent in AGENT_CFGS],
        mode=config.mode,
        device=device,
        hp=build_hparams(config),
        writer=writer,
        seed=seed,
        permutation_seed=seed,
    )


def parameter_counts(trainer: NoShareMAPPOHAPPOTrainer) -> Dict[str, Any]:
    return {
        "actors": [trainer.count_parameters(actor) for actor in trainer.actors],
        "centralized_critic": trainer.count_parameters(trainer.centralized_critic),
        "total": sum(trainer.count_parameters(actor) for actor in trainer.actors)
        + trainer.count_parameters(trainer.centralized_critic),
    }


def empty_rollout_buffer() -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {
        "joint_input": [], "value": [], "reward": [], "done": [],
    }
    for index in range(3):
        result[f"obs_{index}"] = []
        result[f"action_{index}"] = []
        result[f"old_logp_{index}"] = []
    return result


def append_transition(
    buffer: Dict[str, List[Any]],
    trainer: NoShareMAPPOHAPPOTrainer,
    local_obs: Sequence[np.ndarray],
    previous_actions: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    logps: Sequence[float],
    value: float,
    reward: float,
    done: bool,
) -> None:
    buffer["joint_input"].append(trainer.build_joint_input(local_obs, previous_actions))
    buffer["value"].append(np.float32(value))
    buffer["reward"].append(np.float32(reward))
    buffer["done"].append(np.float32(done))
    for index in range(3):
        buffer[f"obs_{index}"].append(np.asarray(local_obs[index], dtype=np.float32))
        buffer[f"action_{index}"].append(np.asarray(actions[index], dtype=np.float32))
        buffer[f"old_logp_{index}"].append(np.float32(logps[index]))


def freeze_rollout(buffer: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(values, dtype=np.float32) for key, values in buffer.items()}


def write_config(
    run_root: Path,
    config: ExperimentConfig,
    trainer: NoShareMAPPOHAPPOTrainer,
    device: str,
    seed: int,
) -> None:
    payload = {
        "trainer_version": TRAINER_VERSION,
        "mode": config.mode,
        "seed": seed,
        "device": device,
        "experiment": asdict(config),
        "ppo_hp": asdict(trainer.hp),
        "architecture_contract": {
            "actors": "three independent architecture-matched SplitLocalActor instances; no actor GRU",
            "actor_parameter_counts": [4738, 4930, 4802],
            "centralized_critic": "concat(obs0[7],obs1[10],obs2[8],prev_action0[1],prev_action1[1],prev_action2[1]) -> Linear(28,64) -> ReLU -> GRU(64,64,1,batch_first=True,bidirectional=False) -> Linear(64,1)",
            "centralized_critic_parameter_count": 26881,
            "forbidden": ["GAT", "bidirectional GRU", "FedAvg", "privacy", "current action critic input"],
        },
        "reward_contract": "shared team reward = (raw_profit + 11) / 5",
        "rollout_contract": {
            "horizon": 1024,
            "continue_across_done": True,
            "done_reset": ["environment", "centralized critic hidden", "all previous actions"],
        },
        "update_contract": {
            "base_advantage": "one centralized GAE, normalized once over the full rollout",
            "actor_phase_before_critic": True,
            "mappo_order": [0, 1, 2],
            "happo_order": "new seed-reproducible random permutation per rollout",
            "happo_factor": "factor *= exp(updated_logp - rollout_collection_old_logp), detached after full actor update",
            "actor_lr_regime": config.actor_lr_regime,
            "actor_lr_start": config.actor_lr,
            "actor_lr_final": config.actor_lr if config.actor_lr_final is None else config.actor_lr_final,
            "actor_lr_hold_steps": config.actor_lr_hold_steps,
            "actor_lr_schedule_total_steps": config.actor_lr_schedule_total_steps,
            "critic_lr": config.critic_lr,
            "entropy_coef": config.ent_coef,
        },
        "checkpoint_contract": [
            "agent_0_actor.pt", "agent_1_actor.pt", "agent_2_actor.pt",
            "centralized_critic.pt", "agent_0_obs_rms.npz", "agent_1_obs_rms.npz",
            "agent_2_obs_rms.npz", "config.json", "result_summary.json",
        ],
        "tensorboard_contract": list(REQUIRED_COMMON_TAGS) + [
            "train/actor_lr", "train/critic_lr",
            "agent_0/loss_actor", "agent_1/loss_actor", "agent_2/loss_actor",
            "agent_0/approx_kl", "agent_1/approx_kl", "agent_2/approx_kl",
            "central_critic/loss", "central_critic/grad_norm",
            "happo/factor_mean", "happo/factor_std", "happo/factor_min", "happo/factor_max",
        ],
        "parameter_counts": parameter_counts(trainer),
        "non_goals": (
            ["multi-seed confirmation", "1000-episode final evaluation", "final benchmark"]
            if config.stage_b1_tuning
            else ["full_200k", "tuning", "multi-seed", "final benchmark"]
        ),
    }
    write_json(run_root / "config.json", payload)


@torch.no_grad()
def actor_only_actions(
    actors: Sequence[SplitLocalActor],
    local_obs: Sequence[np.ndarray],
    deterministic: bool = True,
) -> List[np.ndarray]:
    """Decentralized evaluation path; deliberately has no critic argument."""
    actions: List[np.ndarray] = []
    for actor, observation in zip(actors, local_obs):
        action, _ = actor.act(
            torch.as_tensor(observation, dtype=torch.float32, device=next(actor.parameters()).device),
            deterministic=deterministic,
        )
        actions.append(action.detach().cpu().numpy().astype(np.float32))
    return actions


def load_actor_only_checkpoints(
    run_root: str | Path,
    device: str = "cpu",
) -> List[SplitLocalActor]:
    root = Path(run_root)
    actors: List[SplitLocalActor] = []
    for index, (obs_dim, act_dim) in enumerate(zip((7, 10, 8), (1, 1, 1))):
        actor = SplitLocalActor(
            obs_dim=obs_dim, act_dim=act_dim, local_obs_hidden_dims=(), local_obs_out_dim=64,
            shared_obs_hidden_dims=(), shared_obs_out_dim=64, local_action_hidden_dims=(),
            log_std_min=-5.0, log_std_max=2.0, activate_feature_outputs=True,
        ).to(device)
        actor.load_state_dict(torch.load(root / f"agent_{index}_actor.pt", map_location=device))
        actor.eval()
        actors.append(actor)
    return actors


def _append_happo_order(path: Path, update: int, step: int, order: Sequence[int]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "global_step", "position", "agent_idx"])
        if not exists:
            writer.writeheader()
        for position, agent_idx in enumerate(order):
            writer.writerow({"update": update, "global_step": step, "position": position, "agent_idx": agent_idx})


def run_one_experiment(
    config: ExperimentConfig,
    seed: int = 42,
    run_root_override: Optional[str] = None,
    writer: Optional[Any] = None,
) -> Dict[str, Any]:
    validate_experiment(config)
    set_global_seed(seed)
    device = resolve_device(config.device)
    run_root = Path(run_root_override or config.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(config, device, seed, writer)
    write_config(run_root, config, trainer, device, seed)
    env = None
    try:
        joint_mv_indices = build_joint_mv_indices(AGENT_CFGS)
        env, full_obs, _ = make_env(
            mv_indices=joint_mv_indices, seed=seed, rl_dt_seconds=config.rl_dt_seconds,
            max_steps=config.max_steps, delta_mv_max=config.delta_mv_max,
            production_index=config.production_index, baseline=config.baseline,
            scale=config.scale, reset_reward=config.reset_reward,
        )
        obs_rms = [RunningMeanStd(shape=(len(agent.obs_indices),)) for agent in AGENT_CFGS]
        previous_actions = [np.zeros(len(agent.mv_indices), dtype=np.float32) for agent in AGENT_CFGS]
        critic_hidden = trainer.init_hidden(1)
        step_count = update_count = 0
        episode_seed = seed
        episode_return = 0.0
        episode_length = 0
        episode_returns: List[float] = []
        raw_profits: List[float] = []
        shutdowns: List[float] = []
        final_logs: Dict[str, Any] = {}
        actor_lr_trace: List[Dict[str, Any]] = []
        raw_window = RawMetricWindow()
        raw_window_path = run_root / "tensorboard_raw_metric_windows.csv"
        if raw_window_path.exists():
            raw_window_path.unlink()
        order_path = run_root / "happo_update_order.csv"
        if order_path.exists():
            order_path.unlink()

        while step_count < config.total_steps:
            buffer = empty_rollout_buffer()
            last_transition_done = False
            episodes_in_rollout = 0
            target = min(config.rollout_horizon, config.total_steps - step_count)
            for _ in range(target):
                local_obs = make_local_obs(np.asarray(full_obs, dtype=np.float32), obs_rms, update_rms=True)
                actions, logps, value, critic_hidden = trainer.act(
                    local_obs, previous_actions, critic_hidden, deterministic=False
                )
                assert_finite_values("actions", actions)
                assert_finite_values("logps", logps)
                assert_finite_values("central value", [value])
                next_obs, reward, terminated, truncated, info = env.step(concat_joint_action(actions))
                done = bool(terminated or truncated)
                append_transition(
                    buffer, trainer, local_obs, previous_actions, actions, logps,
                    value, float(reward), done,
                )
                previous_actions = actions
                full_obs = np.asarray(next_obs, dtype=np.float32)
                step_count += 1
                episode_return += float(reward)
                episode_length += 1
                raw_prod, raw_cost = info.get("raw_prod"), info.get("raw_cost")
                raw_profit = (
                    float(raw_prod) - float(raw_cost)
                    if raw_prod is not None and raw_cost is not None
                    else float(reward) * config.scale + config.baseline
                )
                raw_profits.append(raw_profit)
                shutdown = float(bool(info.get("shutdown", False)))
                shutdowns.append(shutdown)
                raw_window.append(step=step_count, raw_prod=raw_prod, raw_cost=raw_cost, shutdown=shutdown)
                if step_count % config.log_every == 0:
                    row = raw_window.flush(end_step=step_count)
                    if row is not None:
                        append_raw_window_csv(raw_window_path, row)
                        if writer is not None:
                            write_raw_window_to_tensorboard(writer, row)
                last_transition_done = done
                if done:
                    episode_returns.append(episode_return)
                    episodes_in_rollout += 1
                    if writer is not None:
                        writer.add_scalar("episode/return", episode_return, step_count)
                        writer.add_scalar("episode/length", episode_length, step_count)
                    episode_return = 0.0
                    episode_length = 0
                    episode_seed += 1
                    if step_count < config.total_steps:
                        full_obs, _ = env.reset(seed=episode_seed)
                        full_obs = np.asarray(full_obs, dtype=np.float32)
                    critic_hidden = trainer.init_hidden(1)
                    previous_actions = [np.zeros(len(agent.mv_indices), dtype=np.float32) for agent in AGENT_CFGS]
                if step_count >= config.total_steps:
                    break

            if last_transition_done:
                last_value = 0.0
            else:
                last_local_obs = make_local_obs(np.asarray(full_obs, dtype=np.float32), obs_rms, update_rms=False)
                last_value, _ = trainer.value(last_local_obs, previous_actions, critic_hidden)
            scheduled_actor_lr = actor_lr_at_step(
                regime=config.actor_lr_regime,
                start_lr=config.actor_lr,
                final_lr=config.actor_lr_final,
                hold_steps=config.actor_lr_hold_steps,
                total_steps=config.actor_lr_schedule_total_steps,
                step=step_count,
            )
            for optimizer in trainer.actor_opts:
                for group in optimizer.param_groups:
                    group["lr"] = float(scheduled_actor_lr)
            actor_optimizer_lrs = [float(optimizer.param_groups[0]["lr"]) for optimizer in trainer.actor_opts]
            critic_optimizer_lr = float(trainer.critic_opt.param_groups[0]["lr"])
            if not all(abs(value - scheduled_actor_lr) <= 1e-15 for value in actor_optimizer_lrs):
                raise AssertionError("Actor optimizer LR application mismatch")
            if abs(critic_optimizer_lr - config.critic_lr) > 1e-15:
                raise AssertionError("Critic optimizer LR changed unexpectedly")
            actor_lr_trace.append(
                {
                    "global_step": int(step_count),
                    "scheduled_actor_lr": float(scheduled_actor_lr),
                    "actor_optimizer_lrs": actor_optimizer_lrs,
                    "critic_optimizer_lr": critic_optimizer_lr,
                }
            )
            logs = trainer.update(freeze_rollout(buffer), last_value, step_count)
            update_count += 1
            logs["rollout/episodes_per_update"] = float(episodes_in_rollout)
            logs["train/update_count"] = float(update_count)
            logs["train/updates_per_env_step"] = float(update_count) / float(step_count)
            logs["train/actor_lr"] = float(scheduled_actor_lr)
            logs["train/critic_lr"] = critic_optimizer_lr
            if writer is not None:
                writer.add_scalar("rollout/episodes_per_update", episodes_in_rollout, step_count)
                writer.add_scalar("train/update_count", update_count, step_count)
                writer.add_scalar("train/updates_per_env_step", logs["train/updates_per_env_step"], step_count)
                writer.add_scalar("train/actor_lr", scheduled_actor_lr, step_count)
                writer.add_scalar("train/critic_lr", critic_optimizer_lr, step_count)
            if config.mode == "happo":
                _append_happo_order(
                    order_path, update_count, step_count, logs["diagnostics"]["update_order"]
                )
            final_logs = {key: value for key, value in logs.items() if key != "diagnostics"}
            if config.smoke:
                break

        row = raw_window.flush(end_step=step_count)
        if row is not None:
            append_raw_window_csv(raw_window_path, row)
            if writer is not None:
                write_raw_window_to_tensorboard(writer, row)
        trainer.save(str(run_root))
        save_obs_rms(str(run_root), AGENT_CFGS, obs_rms)
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()
        required = [
            "config.json", "agent_0_actor.pt", "agent_1_actor.pt", "agent_2_actor.pt",
            "centralized_critic.pt", "agent_0_obs_rms.npz", "agent_1_obs_rms.npz",
            "agent_2_obs_rms.npz",
        ]
        result = {
            "variant": f"noshare_{config.mode}",
            "trainer_version": TRAINER_VERSION,
            "run_root": str(run_root),
            "seed": seed,
            "total_steps": step_count,
            "updates": update_count,
            "smoke": config.smoke,
            "final_logs": final_logs,
            "recent_profit_mean": mean_or_none(raw_profits[-100:]),
            "episode_return_mean": mean_or_none(episode_returns[-20:]),
            "shutdown_rate": mean_or_none(shutdowns),
            "parameter_counts": parameter_counts(trainer),
            "actor_lr_trace": actor_lr_trace,
            "checkpoint_files_exist": {name: (run_root / name).exists() for name in required},
            "tensorboard_raw_metric_windows": str(raw_window_path),
            "happo_update_order_csv": str(order_path) if config.mode == "happo" else None,
        }
        result["checkpoint_files_exist"]["result_summary.json"] = True
        write_json(run_root / "result_summary.json", result)
        return result
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train no-share MAPPO or HAPPO on TEP")
    parser.add_argument("--mode", required=True, choices=("mappo", "happo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root or os.path.join("runs", f"noshare_{args.mode}_smoke")
    config = ExperimentConfig(
        mode=args.mode, name=f"noshare_{args.mode}_smoke", run_root=run_root,
        total_steps=args.total_steps, device=args.device,
    )
    writer = None
    try:
        if not args.no_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            resolved = Path(run_root)
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / resolved
            writer = SummaryWriter(log_dir=str(resolved / "tensorboard"))
        result = run_one_experiment(config, args.seed, writer=writer)
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[noshare-{args.mode}] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
