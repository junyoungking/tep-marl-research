import argparse
import json
import os
import pickle
import random
import sys
import time
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.checkpointing_utils import (
    file_set_hash,
    load_json,
    load_checkpoint_state,
    nested_max_abs_diff,
    rng_state_payload,
    restore_rng_state,
    save_checkpoint_state,
    write_json as write_json_file,
)
from trainers.fl_split_ppo_trainer import TRAINER_VERSION, SplitFedPPOHyperParams, SplitFedPPOTrainer
from trainers.fl_split_ppo_trainer_continue_rollout import SplitFedContinueRolloutPPOTrainer
from utils.training_metrics_logger import TrainingMetricsLogger


@dataclass
class SubAgentConfig:
    name: str
    obs_indices: Tuple[int, ...]
    mv_indices: Tuple[int, ...]


@dataclass
class SplitFLModelConfig:
    actor_local_hidden_dims: Tuple[int, ...] = ()
    actor_local_out: int = 64
    actor_shared_hidden_dims: Tuple[int, ...] = ()
    actor_shared_out: int = 64
    actor_action_hidden_dims: Tuple[int, ...] = ()

    critic_local_pre_hidden_dims: Tuple[int, ...] = ()
    critic_local_pre_out: int = 64
    critic_bi_hidden: int = 64
    critic_gru_layers: int = 1
    critic_gat_hidden: int = 64
    critic_readout_hidden_dims: Tuple[int, ...] = ()
    critic_readout_dim: int = 64
    critic_value_hidden_dims: Tuple[int, ...] = ()
    activate_feature_outputs: bool = True
    ablate_enhancer: bool = False
    bi_extractor: str = "gru"
    bi_post_gru_mlp: bool = False
    aggregation: str = "gat"
    action_pathway: str = "on"
    fuse: str = "learned"


SHALLOW_HIDDEN_DIM_PRESET = asdict(SplitFLModelConfig())


@dataclass
class ExperimentConfig:
    name: str
    actor_lr: float = 5e-5
    critic_lr: float = 1e-4
    server_lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 6
    batch_size: int = 256
    rollout_horizon: int = 1024
    total_steps: int = 200000
    log_every: int = 1000
    target_kl: Optional[float] = 0.05
    fedavg_interval: int = 1
    fedavg_scope: str = "both"
    bi_type_grl: bool = False
    bi_type_loss_coef: float = 0.0
    bi_grl_alpha: float = 1.0

    max_steps: int = 300
    rl_dt_seconds: int = 60
    delta_mv_max: float = 0.2
    production_index: int = 17
    baseline: float = -11.0
    scale: float = 5.0
    reset_reward: bool = False

    device: str = "cpu"
    smoke: bool = True
    min_rollout_steps: int = 16
    allow_short_final_rollout: bool = False
    timestamp_run_dir: bool = False
    checkpoint_steps: Tuple[int, ...] = field(default_factory=tuple)
    run_root: str = os.path.join("runs", "smoke_fl_split_shallow_hybrid")
    model_cfg: SplitFLModelConfig = field(default_factory=SplitFLModelConfig)


AGENT_CFGS: List[SubAgentConfig] = [
    SubAgentConfig("agent_0", (1, 2, 3, 6, 7, 8, 9), (1,)),
    SubAgentConfig("agent_1", (21, 23, 24, 25, 26, 27, 28, 11, 12, 13), (8,)),
    SubAgentConfig("agent_2", (14, 22, 4, 15, 16, 17, 18, 19), (6,)),
]


FRESH600K_VARIANTS: Dict[str, Dict[str, str]] = {
    "A0": {"label": "A0_both_fedavg", "fedavg_scope": "both"},
    "A1": {"label": "A1_no_actor_fedavg", "fedavg_scope": "gru"},
    "A2": {"label": "A2_no_critic_fedavg", "fedavg_scope": "enhancer"},
    "A3": {"label": "A3_no_both_fedavg", "fedavg_scope": "none"},
}

FRESH600K_CHECKPOINT_STEPS: Tuple[int, ...] = (200_000, 400_000, 500_000, 600_000)

# Approved Round-1 HP screen from .omx/plans/shallow-fl-hp-optimization.md.
# Keep the shallow architecture fixed and vary only active HPs/conditions.
# NOTE: fl_v2 enforces target_kl in the actor epoch loop; this sweep keeps the
# historical HP grid unchanged unless explicitly overridden.
ROUND1_HP_SWEEP: List[Dict[str, Any]] = [
    dict(
        id="R1_control_logged_best",
        fedavg_interval=10,
        actor_lr=1e-4,
        critic_lr=1e-4,
        server_lr=1e-4,
        clip_range=0.10,
        ent_coef=0.00,
        n_epochs=6,
        delta_mv_max=0.2,
    ),
    dict(
        id="R1_ctde_like_f10_d03",
        fedavg_interval=10,
        actor_lr=5e-5,
        critic_lr=1e-4,
        server_lr=1e-4,
        clip_range=0.20,
        ent_coef=0.00,
        n_epochs=5,
        delta_mv_max=0.3,
    ),
    dict(
        id="R1_ctde_like_f10_d02",
        fedavg_interval=10,
        actor_lr=5e-5,
        critic_lr=1e-4,
        server_lr=1e-4,
        clip_range=0.20,
        ent_coef=0.00,
        n_epochs=5,
        delta_mv_max=0.2,
    ),
    dict(
        id="R1_ctde_like_f1_d03",
        fedavg_interval=1,
        actor_lr=5e-5,
        critic_lr=1e-4,
        server_lr=1e-4,
        clip_range=0.20,
        ent_coef=0.00,
        n_epochs=5,
        delta_mv_max=0.3,
    ),
    dict(
        id="R1_shallow_hpo_entropy",
        fedavg_interval=10,
        actor_lr=7e-5,
        critic_lr=1e-4,
        server_lr=1e-4,
        clip_range=0.20,
        ent_coef=0.10,
        n_epochs=4,
        delta_mv_max=0.2,
    ),
    dict(
        id="R1_mid_epoch4",
        fedavg_interval=10,
        actor_lr=7e-5,
        critic_lr=1e-4,
        server_lr=1e-4,
        clip_range=0.15,
        ent_coef=0.05,
        n_epochs=4,
        delta_mv_max=0.2,
    ),
    dict(
        id="R1_critic_fast",
        fedavg_interval=10,
        actor_lr=7e-5,
        critic_lr=2e-4,
        server_lr=2e-4,
        clip_range=0.15,
        ent_coef=0.05,
        n_epochs=4,
        delta_mv_max=0.2,
    ),
    dict(
        id="R1_mid_epoch6",
        fedavg_interval=10,
        actor_lr=7e-5,
        critic_lr=1e-4,
        server_lr=1e-4,
        clip_range=0.15,
        ent_coef=0.05,
        n_epochs=6,
        delta_mv_max=0.2,
    ),
]


def _sweep_run_root(sweep_id: str, hp: Dict[str, Any]) -> str:
    return os.path.join(
        "runs",
        "fl_split_shallow_hp",
        (
            f"{sweep_id}_fedavg{hp['fedavg_interval']}"
            f"_ac{hp['actor_lr']}_cr{hp['critic_lr']}_srv{hp['server_lr']}"
            f"_clip{hp['clip_range']}_ent{hp['ent_coef']}"
            f"_ep{hp['n_epochs']}_dmv{hp['delta_mv_max']}"
        ),
    )


def build_round1_sweep_experiment(hp: Dict[str, Any]) -> ExperimentConfig:
    hp = dict(hp)
    sweep_id = str(hp.pop("id"))
    return ExperimentConfig(
        name=f"fl_split_shallow_hp_{sweep_id}",
        smoke=False,
        timestamp_run_dir=True,
        run_root=_sweep_run_root(sweep_id, hp),
        **hp,
    )


EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="fl_split_shallow_hybrid_smoke_parity",
        rollout_horizon=16,
        total_steps=16,
        max_steps=16,
        min_rollout_steps=16,
        log_every=16,
        n_epochs=1,
    ),
    *[build_round1_sweep_experiment(hp) for hp in ROUND1_HP_SWEEP],
]


# =========================================================
# PyCharm Run Configuration
# =========================================================
# PyCharm에서 CLI 인자 없이 Run 버튼으로 실행할 때 쓰는 script-local 기본값입니다.
# - 512-step 실행: (1,)
# - smoke test 실행: (0,)
# PyCharm 설정은 Script path와 Working directory만 잡고 Parameters는 비워두면 됩니다.
PYCHARM_EXPERIMENT_INDICES: Tuple[int, ...] = tuple(
    i for i, e in enumerate(EXPERIMENTS) if not e.smoke
)
PYCHARM_SEED: int = 42


def make_default_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="fl_split_shallow_hybrid_default",
        smoke=False,
        checkpoint_steps=tuple(),
    )


def apply_common_overrides(exp_cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    overrides: Dict[str, Any] = {}
    for arg_name, field_name in [
        ("total_steps", "total_steps"),
        ("rollout_horizon", "rollout_horizon"),
        ("max_steps", "max_steps"),
        ("min_rollout_steps", "min_rollout_steps"),
        ("fedavg_interval", "fedavg_interval"),
        ("n_epochs", "n_epochs"),
        ("batch_size", "batch_size"),
        ("actor_lr", "actor_lr"),
        ("critic_lr", "critic_lr"),
        ("server_lr", "server_lr"),
        ("gamma", "gamma"),
        ("gae_lambda", "gae_lambda"),
        ("clip_range", "clip_range"),
        ("ent_coef", "ent_coef"),
        ("vf_coef", "vf_coef"),
        ("max_grad_norm", "max_grad_norm"),
        ("target_kl", "target_kl"),
        ("log_every", "log_every"),
        ("rl_dt_seconds", "rl_dt_seconds"),
        ("delta_mv_max", "delta_mv_max"),
        ("production_index", "production_index"),
        ("baseline", "baseline"),
        ("scale", "scale"),
        ("bi_type_grl", "bi_type_grl"),
        ("bi_type_loss_coef", "bi_type_loss_coef"),
        ("bi_grl_alpha", "bi_grl_alpha"),
        ("device", "device"),
        ("reset_reward", "reset_reward"),
        ("timestamp_run_dir", "timestamp_run_dir"),
        ("allow_short_final_rollout", "allow_short_final_rollout"),
        ("fedavg_scope", "fedavg_scope"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[field_name] = value
    if getattr(args, "checkpoint_steps", None) is not None:
        overrides["checkpoint_steps"] = tuple(int(item) for item in getattr(args, "checkpoint_steps"))
    if getattr(args, "name", None):
        overrides["name"] = str(args.name)
    if getattr(args, "run_root", None) is not None:
        overrides["run_root"] = args.run_root
    if getattr(args, "smoke", None) is True:
        overrides["smoke"] = True
    if getattr(args, "no_smoke", False):
        overrides["smoke"] = False
    model_cfg = exp_cfg.model_cfg
    model_overrides: Dict[str, Any] = {}
    for arg_name, field_name in [
        ("ablate_enhancer", "ablate_enhancer"),
        ("bi_extractor", "bi_extractor"),
        ("bi_post_gru_mlp", "bi_post_gru_mlp"),
        ("aggregation", "aggregation"),
        ("action_pathway", "action_pathway"),
        ("fuse", "fuse"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            model_overrides[field_name] = value
    if model_overrides:
        overrides["model_cfg"] = replace(model_cfg, **model_overrides)
    return replace(exp_cfg, **overrides)


MODEL_STRUCTURE_DELTAS = [
    "actor: SplitLocalActor keeps FL local obs/enhancer/action heads but removes extra hidden MLP layers via empty hidden dims",
    "critic: LocalBIEncoder keeps local BI GRU/FedAvg copies but removes extra local pre MLP hidden layers via empty hidden dims",
    "shallow_preset: feature-output ReLU is enabled after existing actor/critic projections without adding trainable layers",
    "shallow_preset: actor/critic projection widths are 64 and optional action/value bodies remain Identity/direct paths",
    "server: GAT/readout/value remain global/server-side; server receives BI tensors only",
    "fedavg: shallow script rejects fedavg_interval <= 0 so the corrected shallow path cannot disable FedAvg",
    "rollout: fixed-horizon collection stitches episodes within one update and resets env/critic hidden on done",
]


BASELINE_TB_TAGS = [
    "raw/prod",
    "raw/cost",
    "raw/profit",
    "raw/shutdown_rate",
    "episode/return",
    "episode/length",
    "loss/actor",
    "loss/critic",
    "loss/entropy",
    "train/approx_kl",
    "train/clipfrac",
    "agent_{i}/adv_mean_raw",
    "agent_{i}/adv_std_raw",
    "agent_{i}/adv_std_normed",
    "agent_{i}/loss_actor",
    "agent_{i}/loss_critic",
    "agent_{i}/entropy",
    "agent_{i}/approx_kl",
    "agent_{i}/clipfrac",
    "agent_{i}/ratio_std",
]


SPLIT_FL_TB_TAGS = [
    "fedavg/round",
    "rollout/length",
    "rollout/episodes_per_update",
    "rollout/skipped_short",
    "debug/actor_grad_norm",
    "debug/actor_param_delta",
    "debug/actor_logp_delta_post",
    "debug/actor_post_kl_available",
    "train/approx_kl_post",
    "train/actor_epochs_used",
    "train/approx_kl_epoch0",
    "train/approx_kl_last",
    "train/update_count",
    "train/updates_per_env_step",
    "debug/critic_server_changed_on_server_step",
    "debug/critic_local_changed_on_handoff",
    "debug/critic_gru_reset_enabled",
    "debug/critic_gru_done_resets",
    "debug/hidden_resets",
    "rollout/num_episodes",
    "rollout/mean_episode_len",
    "agent_{i}/approx_kl_post",
]


class TEPUnavailableError(RuntimeError):
    """Raised when the real TEP simulator cannot be imported or reset."""


class RunningMeanStd:
    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0):
        x = np.asarray(x, dtype=np.float64)
        x_norm = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(x_norm, -clip, clip).astype(np.float32)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    return value


def mean_or_none(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def trainer_parameter_counts(trainer: SplitFedPPOTrainer) -> Dict[str, Any]:
    actor_counts = [count_parameters(client.actor) for client in trainer.clients]
    bi_counts = [count_parameters(client.bi_encoder) for client in trainer.clients]
    server_count = count_parameters(trainer.server)
    return {
        "actors": actor_counts,
        "bi_encoders": bi_counts,
        "server": server_count,
        "total": int(sum(actor_counts) + sum(bi_counts) + server_count),
    }


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        # Some PyTorch/backend combinations cannot enable strict deterministic
        # kernels; keep the rest of the explicit seed/cudnn determinism setup.
        pass


def build_joint_mv_indices(agent_cfgs: List[SubAgentConfig]) -> Tuple[int, ...]:
    all_mv = []
    for cfg in agent_cfgs:
        all_mv.extend(cfg.mv_indices)
    if len(set(all_mv)) != len(all_mv):
        raise ValueError(f"Overlapping mv_indices are not allowed: {all_mv}")
    return tuple(all_mv)


def concat_joint_action(action_parts: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(a, dtype=np.float32).reshape(-1) for a in action_parts],
        axis=0,
    ).astype(np.float32)


def save_obs_rms(run_root: str, cfgs: List[SubAgentConfig], obs_rms_list: List[RunningMeanStd]):
    for cfg, rms in zip(cfgs, obs_rms_list):
        save_path = os.path.join(run_root, f"{cfg.name}_obs_rms.npz")
        np.savez(save_path, mean=rms.mean, var=rms.var, count=rms.count)


def load_obs_rms(
    run_root: str,
    cfgs: Optional[List[SubAgentConfig]] = None,
    allow_missing: bool = False,
) -> List[RunningMeanStd]:
    cfg_list = list(cfgs or AGENT_CFGS)
    obs_rms_list: List[RunningMeanStd] = []
    for cfg in cfg_list:
        load_path = os.path.join(run_root, f"{cfg.name}_obs_rms.npz")
        rms: Optional[RunningMeanStd] = None
        if os.path.exists(load_path):
            with np.load(load_path) as data:
                files = set(data.files)
                missing = {"mean", "var", "count"} - files
                if missing:
                    raise KeyError(f"{load_path} is missing obs RMS keys: {sorted(missing)}")
                mean = np.asarray(data["mean"], dtype=np.float64)
                var = np.asarray(data["var"], dtype=np.float64)
                count = float(np.asarray(data["count"]).reshape(()))
                inferred_shape = mean.shape
                if inferred_shape != var.shape:
                    raise ValueError(
                        f"{load_path} has inconsistent ObsRMS shapes: mean={mean.shape}, var={var.shape}"
                    )
                expected_shape = (len(cfg.obs_indices),)
                if cfgs is not None and inferred_shape != expected_shape:
                    raise ValueError(
                        f"{load_path} has incompatible ObsRMS shape: mean={mean.shape}, var={var.shape}, expected={expected_shape}"
                    )
                rms = RunningMeanStd(shape=inferred_shape)
                rms.mean = mean
                rms.var = var
                rms.count = count
        elif not allow_missing:
            raise FileNotFoundError(f"Missing obs normalization artifact {load_path}")
        if rms is None:
            rms = RunningMeanStd(shape=(len(cfg.obs_indices),))
        obs_rms_list.append(rms)
    return obs_rms_list


def capture_obs_rms_state(
    obs_rms_list: List[RunningMeanStd],
    cfgs: Optional[List[SubAgentConfig]] = None,
) -> List[Dict[str, Any]]:
    cfg_list = list(cfgs or AGENT_CFGS)
    if len(obs_rms_list) != len(cfg_list):
        raise ValueError(f"ObsRMS count mismatch: expected {len(cfg_list)}, got {len(obs_rms_list)}")
    return [
        {
            "agent": cfg.name,
            "obs_indices": list(cfg.obs_indices),
            "mv_indices": list(cfg.mv_indices),
            "state": {
                "mean": np.asarray(rms.mean, dtype=np.float64),
                "var": np.asarray(rms.var, dtype=np.float64),
                "count": float(rms.count),
            },
        }
        for cfg, rms in zip(cfg_list, obs_rms_list)
    ]


def restore_obs_rms_state(
    state: List[Dict[str, Any]],
    cfgs: Optional[List[SubAgentConfig]] = None,
) -> List[RunningMeanStd]:
    restored: List[RunningMeanStd] = []
    if cfgs is None:
        for payload in state:
            state_payload = dict(payload.get("state", {}))
            mean = np.asarray(state_payload.get("mean"), dtype=np.float64)
            var = np.asarray(state_payload.get("var"), dtype=np.float64)
            count = float(state_payload.get("count", 1.0))
            if mean.shape != var.shape:
                raise ValueError(f"ObsRMS state shape mismatch: mean={mean.shape}, var={var.shape}")
            rms = RunningMeanStd(shape=mean.shape)
            rms.mean = mean
            rms.var = var
            rms.count = count
            restored.append(rms)
        return restored

    cfg_list = list(cfgs)
    if len(state) != len(cfg_list):
        raise ValueError(f"ObsRMS state count mismatch: expected {len(cfg_list)}, got {len(state)}")
    for cfg, payload in zip(cfg_list, state):
        state_payload = dict(payload.get("state", {}))
        mean = np.asarray(state_payload.get("mean"), dtype=np.float64)
        var = np.asarray(state_payload.get("var"), dtype=np.float64)
        count = float(state_payload.get("count", 1.0))
        expected_shape = (len(cfg.obs_indices),)
        if mean.shape != expected_shape or var.shape != expected_shape:
            raise ValueError(
                f"ObsRMS state shape mismatch for {cfg.name}: mean={mean.shape}, var={var.shape}, expected={expected_shape}"
            )
        rms = RunningMeanStd(shape=expected_shape)
        rms.mean = mean
        rms.var = var
        rms.count = count
        restored.append(rms)
    return restored


def _experiment_cfg_from_run_config(config: Dict[str, Any]) -> ExperimentConfig:
    exp_payload = dict(config.get("experiment", {}))
    model_payload = exp_payload.pop("model_cfg", config.get("model_cfg", {}))
    model_cfg = SplitFLModelConfig(
        **{
            key: tuple(value) if isinstance(value, list) and key in {"actor_local_hidden_dims", "actor_shared_hidden_dims", "actor_action_hidden_dims", "critic_local_pre_hidden_dims", "critic_readout_hidden_dims", "critic_value_hidden_dims"} else value
            for key, value in model_payload.items()
            if key in SplitFLModelConfig.__dataclass_fields__
        }
    )
    exp_payload["model_cfg"] = model_cfg
    default_exp = make_default_experiment()
    allowed = set(ExperimentConfig.__dataclass_fields__)
    updates = {}
    for key, value in exp_payload.items():
        if key in allowed:
            if key == "checkpoint_steps" and isinstance(value, list):
                value = tuple(int(item) for item in value)
            updates[key] = value
    if "name" not in updates:
        updates["name"] = str(config.get("experiment_name", default_exp.name))
    return replace(default_exp, **updates)


def capture_checkpoint_state(
    *,
    checkpoint_step: int,
    bundle_root: Path,
    seed: int,
    device: str,
    exp_cfg: ExperimentConfig,
    trainer: SplitFedPPOTrainer,
    joint_mv_indices: Tuple[int, ...],
    obs_rms_list: List[RunningMeanStd],
    env: Any,
    loop_state: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": int(checkpoint_step),
        "bundle_root": str(bundle_root),
        "seed": int(seed),
        "device": str(device),
        "experiment": json_safe(asdict(exp_cfg)),
        "joint_mv_indices": tuple(int(item) for item in joint_mv_indices),
        "trainer_state": trainer.checkpoint_state(),
        "obs_rms_state": capture_obs_rms_state(obs_rms_list),
        "rng_state": rng_state_payload(),
        "env_snapshot_bytes": pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL),
        "loop_state": loop_state,
    }


def checkpoint_bundle_root(run_root: str | Path, checkpoint_step: int, total_steps: int) -> Path:
    root = Path(run_root)
    if int(checkpoint_step) >= int(total_steps):
        return root
    return root / "milestones" / f"step_{int(checkpoint_step)}"


def save_milestone_checkpoint(
    *,
    bundle_root: Path,
    checkpoint_step: int,
    seed: int,
    device: str,
    exp_cfg: ExperimentConfig,
    trainer: SplitFedPPOTrainer,
    joint_mv_indices: Tuple[int, ...],
    obs_rms_list: List[RunningMeanStd],
    env: Any,
    loop_state: Dict[str, Any],
    parameter_counts: Optional[Dict[str, Any]] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    result_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bundle_root.mkdir(parents=True, exist_ok=True)
    write_config(
        str(bundle_root),
        exp_cfg,
        trainer.hp,
        joint_mv_indices,
        device,
        seed,
        parameter_counts=parameter_counts or trainer_parameter_counts(trainer),
    )
    trainer.save(str(bundle_root))
    save_obs_rms(str(bundle_root), AGENT_CFGS, obs_rms_list)
    if result_summary is not None:
        write_json_file(bundle_root / "result_summary.json", result_summary)
    payload = checkpoint_state or capture_checkpoint_state(
        checkpoint_step=checkpoint_step,
        bundle_root=bundle_root,
        seed=seed,
        device=device,
        exp_cfg=exp_cfg,
        trainer=trainer,
        joint_mv_indices=joint_mv_indices,
        obs_rms_list=obs_rms_list,
        env=env,
        loop_state=loop_state,
    )
    save_checkpoint_state(bundle_root, payload)

    file_names = [
        "config.json",
        *(["result_summary.json"] if result_summary is not None else []),
        *[f"split_actor_{i}.pt" for i in range(len(AGENT_CFGS))],
        *[f"split_bi_encoder_{i}.pt" for i in range(len(AGENT_CFGS))],
        "split_global_bi_server_critic.pt",
        *[f"{cfg.name}_obs_rms.npz" for cfg in AGENT_CFGS],
        "checkpoint_state.pt",
        "milestone_manifest.json",
    ]
    manifest = {
        "schema_version": 1,
        "checkpoint_step": int(checkpoint_step),
        "bundle_root": str(bundle_root),
        "checkpoint_state_path": str(bundle_root / "checkpoint_state.pt"),
        "manifest_path": str(bundle_root / "milestone_manifest.json"),
        "files": file_names,
        "file_set_hash": file_set_hash([bundle_root / name for name in file_names if name != "milestone_manifest.json"]),
        "checkpoint_state_summary": {
            "trainer_version": str(payload.get("trainer_state", {}).get("trainer_version", TRAINER_VERSION)),
            "round_idx": int(payload.get("trainer_state", {}).get("round_idx", 0)),
            "seed": int(payload.get("seed", seed)),
            "device": str(payload.get("device", device)),
        },
    }
    write_json_file(bundle_root / "milestone_manifest.json", manifest)
    return manifest


def load_milestone_checkpoint(
    bundle_root: Path,
    device: str = "cpu",
    exp_cfg: Optional[ExperimentConfig] = None,
) -> Dict[str, Any]:
    bundle_root = Path(bundle_root)
    config = load_json(bundle_root / "config.json")
    resolved_exp_cfg = exp_cfg or _experiment_cfg_from_run_config(config)
    trainer = build_trainer(resolved_exp_cfg, device=device, writer=None)
    checkpoint_state = load_checkpoint_state(bundle_root, map_location=device)
    trainer_summary = trainer.load_checkpoint_state(dict(checkpoint_state.get("trainer_state", {})))
    obs_rms_state = restore_obs_rms_state(list(checkpoint_state.get("obs_rms_state", [])))
    restore_rng_state(dict(checkpoint_state.get("rng_state", {})))
    env_snapshot_bytes = checkpoint_state.get("env_snapshot_bytes")
    env = pickle.loads(env_snapshot_bytes) if env_snapshot_bytes is not None else None
    manifest_path = bundle_root / "milestone_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    return {
        "bundle_root": bundle_root,
        "config": config,
        "exp_cfg": resolved_exp_cfg,
        "trainer": trainer,
        "trainer_summary": trainer_summary,
        "obs_rms_list": obs_rms_state,
        "checkpoint_state": checkpoint_state,
        "env": env,
        "manifest": manifest,
    }


def _single_step_checkpoint_self_test_rollout(
    *,
    trainer: SplitFedPPOTrainer,
    env: Any,
    obs_rms_list: List[RunningMeanStd],
    full_obs: np.ndarray,
    prev_actions: List[np.ndarray],
    critic_hidden_list: List[torch.Tensor],
    step_count: int,
    episode_seed: int,
    exp_cfg: ExperimentConfig,
) -> Dict[str, Any]:
    local_obs_list = make_local_obs(full_obs, obs_rms_list, update_rms=True)
    actions_list, logps_list, values_list, critic_hidden_list = trainer.act(
        obs_list=local_obs_list,
        prev_act_list=prev_actions,
        critic_hidden_list=critic_hidden_list,
        deterministic=False,
    )
    joint_action = concat_joint_action(actions_list)
    next_full_obs, reward, terminated, truncated, info = env.step(joint_action)
    done = bool(terminated or truncated)
    buffers = [{key: [] for key in ("obs", "act", "prev_act", "logp", "val", "rew", "done")} for _ in AGENT_CFGS]
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
    next_full_obs = np.asarray(next_full_obs, dtype=np.float32)
    if done:
        full_obs, _ = env.reset(seed=episode_seed + 1)
        full_obs = np.asarray(full_obs, dtype=np.float32)
        critic_hidden_list = trainer.init_hidden(batch_size=1)
        prev_actions = [np.zeros(len(cfg.mv_indices), dtype=np.float32) for cfg in AGENT_CFGS]
    else:
        full_obs = next_full_obs
    agent_rollouts = buffers_to_rollouts(buffers)
    last_values = [0.0 for _ in AGENT_CFGS] if done else trainer.value(
        obs_list=make_local_obs(full_obs, obs_rms_list, update_rms=False),
        prev_act_list=prev_actions,
        critic_hidden_list=critic_hidden_list,
    )[0]
    logs = trainer.update(agent_rollouts=agent_rollouts, last_values=last_values, global_step=step_count + 1)
    return {
        "obs": full_obs,
        "reward": float(reward),
        "done": done,
        "info": json_safe(info),
        "critic_hidden_list": critic_hidden_list,
        "prev_actions": prev_actions,
        "logs": logs,
        "step_count": step_count + 1,
    }


def checkpoint_self_test() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir) / "checkpoint_self_test"
        exp_cfg = ExperimentConfig(
            name="checkpoint_self_test",
            total_steps=1,
            rollout_horizon=1,
            min_rollout_steps=1,
            log_every=1,
            n_epochs=1,
            batch_size=1,
            smoke=False,
            checkpoint_steps=(1,),
            run_root=str(run_root),
            device="cpu",
        )
        validate_experiment(exp_cfg)
        device = resolve_device(exp_cfg.device)
        seed = 123
        set_global_seed(seed)
        run_root.mkdir(parents=True, exist_ok=True)
        joint_mv_indices = build_joint_mv_indices(AGENT_CFGS)
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
        trainer = build_trainer(exp_cfg, device=device, writer=None)
        parameter_counts = trainer_parameter_counts(trainer)
        obs_rms_list = [RunningMeanStd(shape=(len(cfg.obs_indices),)) for cfg in AGENT_CFGS]
        prev_actions = [np.zeros(len(cfg.mv_indices), dtype=np.float32) for cfg in AGENT_CFGS]
        critic_hidden_list = trainer.init_hidden(batch_size=1)
        full_obs = np.asarray(full_obs, dtype=np.float32)
        step_state = _single_step_checkpoint_self_test_rollout(
            trainer=trainer,
            env=env,
            obs_rms_list=obs_rms_list,
            full_obs=full_obs,
            prev_actions=prev_actions,
            critic_hidden_list=critic_hidden_list,
            step_count=0,
            episode_seed=seed,
            exp_cfg=exp_cfg,
        )
        loop_state = {
            "step_count": step_state["step_count"],
            "update_count": 1,
            "skipped_short_count": 0,
            "episode_seed": seed,
            "episode_return": 0.0,
            "episode_len": 0,
            "prod_buf": [],
            "cost_buf": [],
            "profit_buf": [],
            "shutdown_buf": [],
            "profit_values": [],
            "shutdown_values": [],
            "episode_returns": [],
            "prev_actions": step_state["prev_actions"],
            "critic_hidden_list": step_state["critic_hidden_list"],
            "current_obs": step_state["obs"],
            "buffers": [{key: [] for key in ("obs", "act", "prev_act", "logp", "val", "rew", "done")} for _ in AGENT_CFGS],
            "final_logs": step_state["logs"],
        }
        payload = capture_checkpoint_state(
            checkpoint_step=1,
            bundle_root=run_root,
            seed=seed,
            device=device,
            exp_cfg=exp_cfg,
            trainer=trainer,
            joint_mv_indices=joint_mv_indices,
            obs_rms_list=obs_rms_list,
            env=env,
            loop_state=loop_state,
        )
        manifest = save_milestone_checkpoint(
            bundle_root=run_root,
            checkpoint_step=1,
            seed=seed,
            device=device,
            exp_cfg=exp_cfg,
            trainer=trainer,
            joint_mv_indices=joint_mv_indices,
            obs_rms_list=obs_rms_list,
            env=env,
            loop_state=loop_state,
            parameter_counts=parameter_counts,
            checkpoint_state=payload,
        )
        loaded_a = load_milestone_checkpoint(run_root, device=device, exp_cfg=exp_cfg)
        loaded_b = load_milestone_checkpoint(run_root, device=device, exp_cfg=exp_cfg)

        state_a = loaded_a["checkpoint_state"]
        state_b = loaded_b["checkpoint_state"]
        trainer_state_diff = nested_max_abs_diff(state_a["trainer_state"], state_b["trainer_state"])
        obs_rms_diff = nested_max_abs_diff(state_a["obs_rms_state"], state_b["obs_rms_state"])
        rng_diff = nested_max_abs_diff(state_a["rng_state"], state_b["rng_state"])
        loop_state_diff = nested_max_abs_diff(state_a["loop_state"], state_b["loop_state"])

        def _advance(bundle: Dict[str, Any]) -> Dict[str, Any]:
            trainer_obj: SplitFedPPOTrainer = bundle["trainer"]
            env_obj = bundle["env"]
            obs_rms_obj = bundle["obs_rms_list"]
            loop = dict(bundle["checkpoint_state"]["loop_state"])
            return _single_step_checkpoint_self_test_rollout(
                trainer=trainer_obj,
                env=env_obj,
                obs_rms_list=obs_rms_obj,
                full_obs=np.asarray(loop["current_obs"], dtype=np.float32),
                prev_actions=[np.asarray(x, dtype=np.float32) for x in loop["prev_actions"]],
                critic_hidden_list=loop["critic_hidden_list"],
                step_count=int(loop["step_count"]),
                episode_seed=int(loop["episode_seed"]),
                exp_cfg=exp_cfg,
            )

        advance_a = _advance(loaded_a)
        advance_b = _advance(loaded_b)
        continuation_diff = {
            "obs": nested_max_abs_diff(advance_a["obs"], advance_b["obs"]),
            "reward": nested_max_abs_diff(advance_a["reward"], advance_b["reward"]),
            "done": nested_max_abs_diff(advance_a["done"], advance_b["done"]),
            "info": nested_max_abs_diff(advance_a["info"], advance_b["info"]),
        }
        continuation_pass = all(v == 0.0 for v in continuation_diff.values())
        return {
            "schema_version": 1,
            "bundle_root": str(run_root),
            "manifest": manifest,
            "trainer_state_diff": trainer_state_diff,
            "obs_rms_diff": obs_rms_diff,
            "rng_diff": rng_diff,
            "loop_state_diff": loop_state_diff,
            "continuation_diff": continuation_diff,
            "continuation_pass": continuation_pass,
            "pass": trainer_state_diff == 0.0 and obs_rms_diff == 0.0 and rng_diff == 0.0 and loop_state_diff == 0.0,
        }


def make_env(
    mv_indices,
    seed: int,
    rl_dt_seconds: int,
    max_steps: int,
    delta_mv_max: float,
    production_index: int,
    baseline: float,
    scale: float,
    reset_reward: bool,
):
    try:
        from envs.cost import Cost
        from envs.env import Env
        from envs.reward import Reward
        from envs.wrappers import ActionWrapper
    except Exception as exc:
        raise TEPUnavailableError(f"TEP simulator unavailable during import: {type(exc).__name__}: {exc}") from exc

    try:
        base_env = Env(
            mv_indices=tuple(mv_indices),
            rl_dt_seconds=rl_dt_seconds,
            max_steps=max_steps,
        )
        cost_fn = Cost(product_flow_index=production_index)
        reward_fn = Reward(
            cost_fn=cost_fn,
            production_index=production_index,
            baseline=baseline,
            scale=scale,
        )
        env = ActionWrapper(
            base_env,
            delta_mv_max=delta_mv_max,
            reward_fn=reward_fn,
            reset_reward=reset_reward,
        )
        obs, info = env.reset(seed=seed)
    except Exception as exc:
        raise TEPUnavailableError(f"TEP simulator unavailable during reset: {type(exc).__name__}: {exc}") from exc
    return env, np.asarray(obs, dtype=np.float32), info


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device cuda but CUDA is not available")
    return requested


def resolve_run_root(path_arg: str) -> str:
    if os.path.isabs(path_arg):
        return path_arg
    return os.path.join(PROJECT_ROOT, path_arg)


def append_transition(
    buffers: List[Dict[str, list]],
    local_obs_list: List[np.ndarray],
    actions_list: List[np.ndarray],
    prev_actions: List[np.ndarray],
    logps_list: List[float],
    values_list: List[float],
    reward: float,
    done: bool,
):
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


def make_local_obs(full_obs: np.ndarray, obs_rms_list: List[RunningMeanStd], update_rms: bool) -> List[np.ndarray]:
    local_obs_list = []
    for i, cfg in enumerate(AGENT_CFGS):
        raw_obs_i = full_obs[np.asarray(cfg.obs_indices, dtype=np.int64)].astype(np.float32)
        if update_rms:
            obs_rms_list[i].update(raw_obs_i[None, :])
        local_obs_list.append(obs_rms_list[i].normalize(raw_obs_i))
    return local_obs_list


def build_hparams(exp_cfg: ExperimentConfig) -> SplitFedPPOHyperParams:
    return SplitFedPPOHyperParams(
        actor_lr=exp_cfg.actor_lr,
        critic_lr=exp_cfg.critic_lr,
        server_lr=exp_cfg.server_lr,
        gamma=exp_cfg.gamma,
        gae_lambda=exp_cfg.gae_lambda,
        clip_range=exp_cfg.clip_range,
        ent_coef=exp_cfg.ent_coef,
        vf_coef=exp_cfg.vf_coef,
        max_grad_norm=exp_cfg.max_grad_norm,
        n_epochs=exp_cfg.n_epochs,
        batch_size=exp_cfg.batch_size,
        rollout_horizon=exp_cfg.rollout_horizon,
        total_steps=exp_cfg.total_steps,
        log_every=exp_cfg.log_every,
        target_kl=exp_cfg.target_kl,
        fedavg_interval=exp_cfg.fedavg_interval,
        fedavg_scope=exp_cfg.fedavg_scope,
        bi_type_grl=exp_cfg.bi_type_grl,
        bi_type_loss_coef=exp_cfg.bi_type_loss_coef,
        bi_grl_alpha=exp_cfg.bi_grl_alpha,
    )


def build_trainer(exp_cfg: ExperimentConfig, device: str, writer=None) -> SplitFedPPOTrainer:
    obs_dims = [len(cfg.obs_indices) for cfg in AGENT_CFGS]
    act_dims = [len(cfg.mv_indices) for cfg in AGENT_CFGS]
    model_cfg = exp_cfg.model_cfg
    trainer = SplitFedContinueRolloutPPOTrainer.build_default(
        obs_dims=obs_dims,
        act_dims=act_dims,
        device=device,
        hp=build_hparams(exp_cfg),
        actor_local_out=model_cfg.actor_local_out,
        actor_shared_out=model_cfg.actor_shared_out,
        critic_local_pre_out=model_cfg.critic_local_pre_out,
        critic_bi_hidden=model_cfg.critic_bi_hidden,
        critic_gru_layers=model_cfg.critic_gru_layers,
        critic_gat_hidden=model_cfg.critic_gat_hidden,
        critic_readout_dim=model_cfg.critic_readout_dim,
        actor_local_hidden_dims=model_cfg.actor_local_hidden_dims,
        actor_shared_hidden_dims=model_cfg.actor_shared_hidden_dims,
        actor_action_hidden_dims=model_cfg.actor_action_hidden_dims,
        critic_local_pre_hidden_dims=model_cfg.critic_local_pre_hidden_dims,
        critic_readout_hidden_dims=model_cfg.critic_readout_hidden_dims,
        critic_value_hidden_dims=model_cfg.critic_value_hidden_dims,
        activate_feature_outputs=model_cfg.activate_feature_outputs,
        ablate_enhancer=model_cfg.ablate_enhancer,
        bi_extractor=model_cfg.bi_extractor,
        bi_post_gru_mlp=model_cfg.bi_post_gru_mlp,
        aggregation=model_cfg.aggregation,
        action_pathway=model_cfg.action_pathway,
        fuse=model_cfg.fuse,
    )
    trainer.writer = writer
    return trainer


def trainer_fedavg_groups_from_config(exp_cfg: ExperimentConfig) -> List[str]:
    if exp_cfg.fedavg_scope == "both":
        groups = ["enhancer", "gru"]
    elif exp_cfg.fedavg_scope == "enhancer":
        groups = ["enhancer"]
    elif exp_cfg.fedavg_scope == "gru":
        groups = ["gru"]
    elif exp_cfg.fedavg_scope == "none":
        groups = []
    else:
        raise ValueError(f"Unsupported fedavg_scope={exp_cfg.fedavg_scope!r}")
    if exp_cfg.model_cfg.ablate_enhancer and "enhancer" in groups:
        groups = [g for g in groups if g != "enhancer"]
    if exp_cfg.model_cfg.bi_extractor == "none" and "gru" in groups:
        groups = [g for g in groups if g != "gru"]
    return groups


def write_config(
    run_root: str,
    exp_cfg: ExperimentConfig,
    hp: SplitFedPPOHyperParams,
    joint_mv_indices: Tuple[int, ...],
    device: str,
    seed: int,
    parameter_counts: Optional[Dict[str, Any]] = None,
):
    config = {
        "trainer_version": getattr(hp, "trainer_version", TRAINER_VERSION),
        "seed": seed,
        "device": device,
        "experiment": json_safe(asdict(exp_cfg)),
        "ppo_hp": hp.__dict__,
        "model_cfg": json_safe(asdict(exp_cfg.model_cfg)),
        "agents": [json_safe(asdict(cfg)) for cfg in AGENT_CFGS],
        "joint_mv_indices": list(joint_mv_indices),
        "ablation": "fl_split_shallow_hybrid_tepenv",
        "split_boundary": "bi_only",
        "rollout_collection": {
            "mode": "fixed_horizon_continue_across_done",
            "horizon": int(exp_cfg.rollout_horizon),
            "done_behavior": (
                "Do not end the PPO update on episode done. Log the completed episode, "
                "reset env/prev_actions/critic hidden, and continue filling the same rollout."
            ),
            "critic_update_done_resets": (
                "SplitFedContinueRolloutPPOTrainer re-encodes local BI segments with hidden resets "
                "at internal done boundaries before server/local critic updates."
            ),
        },
        "feature_output_activation": "ReLU" if exp_cfg.model_cfg.activate_feature_outputs else None,
        "feature_output_activation_scope": "existing shallow feature projections only; no action/value body depth added",
        "shallow_hidden_dim_preset": json_safe(SHALLOW_HIDDEN_DIM_PRESET),
        "model_structure_deltas": list(MODEL_STRUCTURE_DELTAS),
        "structural_ablation": {
            "ablate_enhancer": bool(exp_cfg.model_cfg.ablate_enhancer),
            "bi_extractor": exp_cfg.model_cfg.bi_extractor,
            "aggregation": exp_cfg.model_cfg.aggregation,
            "action_pathway": exp_cfg.model_cfg.action_pathway,
            "fuse": exp_cfg.model_cfg.fuse,
            "fedavg_scope": exp_cfg.fedavg_scope,
            "fedavg_interval": exp_cfg.fedavg_interval,
            "scope_note": (
                "FL-only structural ablation flags. Defaults preserve the full shallow split-FL path; "
                "non-default flags are interpreted as structural/federation-surface ablations."
            ),
        },
        "fedavg_groups": trainer_fedavg_groups_from_config(exp_cfg),
        "parameter_counts": json_safe(parameter_counts or {}),
        "logging_contract": {
            "baseline_tags": list(BASELINE_TB_TAGS),
            "split_fl_tags": list(SPLIT_FL_TB_TAGS),
        },
        "non_goals": [
            "non_fl_hhac_shallow_hpo_runner",
            "real_network_client_server",
            "encryption",
            "differential_privacy",
            "secure_aggregation",
        ],
        "obs_index_semantics": "direct_numpy_indices_matching_existing_baseline",
        "mv_index_semantics": "one_based_env_mv_indices",
        "batch_size_note": "metadata only; current SplitFedPPOTrainer.update uses full-rollout tensors",
    }
    with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def validate_experiment(exp_cfg: ExperimentConfig):
    if exp_cfg.total_steps <= 0 or exp_cfg.rollout_horizon <= 0:
        raise ValueError("total_steps and rollout_horizon must be positive")
    if exp_cfg.log_every <= 0:
        raise ValueError("log_every must be positive")
    if exp_cfg.min_rollout_steps <= 0:
        raise ValueError("min_rollout_steps must be positive")
    checkpoint_steps = tuple(int(step) for step in exp_cfg.checkpoint_steps)
    if checkpoint_steps:
        if any(step <= 0 for step in checkpoint_steps):
            raise ValueError("checkpoint_steps must contain positive integers")
        if any(left >= right for left, right in zip(checkpoint_steps, checkpoint_steps[1:])):
            raise ValueError("checkpoint_steps must be strictly increasing")
        if any(step > exp_cfg.total_steps for step in checkpoint_steps):
            raise ValueError("checkpoint_steps cannot exceed total_steps")
    if exp_cfg.fedavg_interval <= 0:
        raise ValueError("shallow FL/FedAvg experiments require fedavg_interval > 0")
    if exp_cfg.fedavg_scope not in {"both", "gru", "enhancer", "none"}:
        raise ValueError(f"Unsupported fedavg_scope={exp_cfg.fedavg_scope!r}")
    if exp_cfg.model_cfg.bi_extractor not in {"gru", "mlp", "none"}:
        raise ValueError(f"Unsupported bi_extractor={exp_cfg.model_cfg.bi_extractor!r}")
    if exp_cfg.model_cfg.aggregation not in {"gat", "mean", "none"}:
        raise ValueError(f"Unsupported aggregation={exp_cfg.model_cfg.aggregation!r}")
    if exp_cfg.model_cfg.action_pathway not in {"on", "off"}:
        raise ValueError(f"Unsupported action_pathway={exp_cfg.model_cfg.action_pathway!r}")
    if exp_cfg.model_cfg.fuse not in {"learned", "concat"}:
        raise ValueError(f"Unsupported fuse={exp_cfg.model_cfg.fuse!r}")
    if (exp_cfg.bi_type_grl or exp_cfg.bi_type_loss_coef != 0.0) and not exp_cfg.model_cfg.bi_post_gru_mlp:
        raise ValueError("bi_type_grl or bi_type_loss_coef requires model_cfg.bi_post_gru_mlp=True")
    if exp_cfg.bi_type_grl and exp_cfg.bi_type_loss_coef <= 0.0:
        raise ValueError("bi_type_grl requires bi_type_loss_coef > 0.0")
    if exp_cfg.bi_type_loss_coef < 0.0:
        raise ValueError("bi_type_loss_coef must be non-negative")
    if exp_cfg.bi_grl_alpha <= 0.0:
        raise ValueError("bi_grl_alpha must be positive")
    if exp_cfg.smoke:
        if exp_cfg.total_steps != exp_cfg.rollout_horizon:
            raise ValueError("smoke requires total_steps to equal rollout_horizon")
        if exp_cfg.min_rollout_steps > exp_cfg.rollout_horizon:
            raise ValueError("smoke requires min_rollout_steps <= rollout_horizon")
        if exp_cfg.n_epochs < 1:
            raise ValueError("smoke requires n_epochs >= 1")


def resolve_experiment_run_root(exp_cfg: ExperimentConfig, run_root_override: Optional[str]) -> str:
    if run_root_override:
        return resolve_run_root(run_root_override)
    run_root = resolve_run_root(exp_cfg.run_root)
    if exp_cfg.timestamp_run_dir:
        run_root = os.path.join(run_root, f"fl_split_shallow_hybrid_tuning_{exp_cfg.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    return run_root


def run_one_experiment(
    exp_cfg: ExperimentConfig,
    seed: int = 42,
    run_root_override: Optional[str] = None,
    runtime_hooks: Optional[Mapping[str, Callable[[Dict[str, Any]], None]]] = None,
) -> int:
    """Run one experiment with optional post-update runtime hooks.

    The canonical path supplies no hooks.  Consequently its pre-existing
    rollout, PPO update, logging, and normal save behavior is unchanged.
    """
    validate_experiment(exp_cfg)
    device = resolve_device(exp_cfg.device)
    set_global_seed(seed)

    from torch.utils.tensorboard import SummaryWriter

    run_root = resolve_experiment_run_root(exp_cfg, run_root_override)
    os.makedirs(run_root, exist_ok=True)
    writer = SummaryWriter(run_root)

    joint_mv_indices = build_joint_mv_indices(AGENT_CFGS)
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
    # Canonical A0 enables every fixed logging group.  The trainer receives the
    # same instance so update captures and transition state share one owner.
    metrics_logger = TrainingMetricsLogger(writer=writer, log_every=exp_cfg.log_every, marl=True, fedavg=True, privacy=True)
    trainer.metrics_logger = metrics_logger
    performance = metrics_logger.performance
    final_logs: Dict[str, Any] = {}
    checkpoint_steps = tuple(sorted({int(step) for step in exp_cfg.checkpoint_steps}))
    checkpoint_step_set = set(checkpoint_steps)
    saved_checkpoint_steps: set[int] = set()
    milestone_entries: List[Dict[str, Any]] = []

    def emit_runtime_hook(event: str) -> None:
        if runtime_hooks is None:
            return
        callback = runtime_hooks.get(event)
        if callback is None:
            return
        callback(
            {
                "event": event,
                "exp_cfg": exp_cfg,
                "seed": seed,
                "run_root": run_root,
                "device": device,
                "trainer": trainer,
                "env": env,
                "joint_mv_indices": joint_mv_indices,
                "obs_rms_list": obs_rms_list,
                "step_count": step_count,
                "update_count": update_count,
                "skipped_short_count": skipped_short_count,
                "episode_seed": episode_seed,
                "episode_return": performance.episode_return,
                "episode_len": performance.episode_length,
                "prod_buf": performance.prod_buffer,
                "cost_buf": performance.cost_buffer,
                "profit_buf": performance.profit_buffer,
                "shutdown_buf": performance.shutdown_buffer,
                "profit_values": performance.profit_history,
                "shutdown_values": performance.shutdown_history,
                "episode_returns": performance.episode_return_history,
                "prev_actions": prev_actions,
                "critic_hidden_list": critic_hidden_list,
                "current_obs": full_obs,
                "buffers": buffers,
                "last_transition_done": last_transition_done,
                "episodes_in_rollout": episodes_in_rollout,
                "final_logs": final_logs,
            }
        )

    def save_checkpoint_entry(
        current_step: int,
        *,
        final: bool = False,
        result_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not checkpoint_steps:
            return None
        if not final:
            if current_step not in checkpoint_step_set or current_step in saved_checkpoint_steps or current_step >= exp_cfg.total_steps:
                return None
        bundle_root = checkpoint_bundle_root(run_root, current_step, exp_cfg.total_steps)
        loop_state = {
            "step_count": step_count,
            "update_count": update_count,
            "skipped_short_count": skipped_short_count,
            "episode_seed": episode_seed,
            "episode_return": performance.episode_return,
            "episode_len": performance.episode_length,
            "prod_buf": list(performance.prod_buffer),
            "cost_buf": list(performance.cost_buffer),
            "profit_buf": list(performance.profit_buffer),
            "shutdown_buf": list(performance.shutdown_buffer),
            "profit_values": list(performance.profit_history),
            "shutdown_values": list(performance.shutdown_history),
            "episode_returns": list(performance.episode_return_history),
            "prev_actions": prev_actions,
            "critic_hidden_list": critic_hidden_list,
            "current_obs": full_obs,
            "buffers": buffers,
            "last_transition_done": last_transition_done,
            "episodes_in_rollout": episodes_in_rollout,
            "final_logs": final_logs,
        }
        checkpoint_state = capture_checkpoint_state(
            checkpoint_step=current_step,
            bundle_root=bundle_root,
            seed=seed,
            device=device,
            exp_cfg=exp_cfg,
            trainer=trainer,
            joint_mv_indices=joint_mv_indices,
            obs_rms_list=obs_rms_list,
            env=env,
            loop_state=loop_state,
        )
        summary_payload = result_summary or {
            "variant": "fl_split_shallow_fixed_horizon",
            "trainer_version": getattr(trainer.hp, "trainer_version", TRAINER_VERSION),
            "run_root": run_root,
            "seed": seed,
            "total_steps": exp_cfg.total_steps,
            "checkpoint_step": int(current_step),
            "updates": update_count,
            "skipped_short": skipped_short_count,
            "recent_profit_mean": mean_or_none(performance.profit_history[-100:]),
            "episode_return_mean": mean_or_none(performance.episode_return_history[-20:]),
            "shutdown_rate": mean_or_none(performance.shutdown_history),
            "checkpoint_steps_requested": list(checkpoint_steps),
            "milestone_checkpoint_manifest": {
                "path": os.path.join(run_root, "milestone_checkpoint_manifest.json"),
                "saved_checkpoint_steps": sorted(set(saved_checkpoint_steps) | {int(current_step)}),
            },
            "checkpoint_files_exist": {
                "split_actor_0": True,
                "split_actor_1": True,
                "split_actor_2": True,
                "split_bi_encoder_0": True,
                "split_bi_encoder_1": True,
                "split_bi_encoder_2": True,
                "split_global_bi_server_critic": True,
                "checkpoint_state": True,
                "milestone_manifest": True,
                "result_summary": True,
            },
        }
        manifest = save_milestone_checkpoint(
            bundle_root=bundle_root,
            checkpoint_step=current_step,
            seed=seed,
            device=device,
            exp_cfg=exp_cfg,
            trainer=trainer,
            joint_mv_indices=joint_mv_indices,
            obs_rms_list=obs_rms_list,
            env=env,
            loop_state=loop_state,
            parameter_counts=parameter_counts,
            checkpoint_state=checkpoint_state,
            result_summary=json_safe(summary_payload),
        )
        entry = {
            "checkpoint_step": int(current_step),
            "bundle_root": str(bundle_root),
            "manifest_path": str(bundle_root / "milestone_manifest.json"),
            "checkpoint_state_path": str(bundle_root / "checkpoint_state.pt"),
            "file_set_hash": str(manifest.get("file_set_hash", "")),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "final": bool(final),
        }
        milestone_entries.append(entry)
        saved_checkpoint_steps.add(int(current_step))
        return entry

    while step_count < exp_cfg.total_steps:
        remaining = exp_cfg.total_steps - step_count
        target_horizon = min(exp_cfg.rollout_horizon, remaining)
        buffers = [{key: [] for key in ("obs", "act", "prev_act", "logp", "val", "rew", "done")} for _ in AGENT_CFGS]
        last_transition_done = False
        episodes_in_rollout = 0

        for _ in range(target_horizon):
            local_obs_list = make_local_obs(full_obs, obs_rms_list, update_rms=True)
            actions_list, logps_list, values_list, critic_hidden_list = trainer.act(
                obs_list=local_obs_list,
                prev_act_list=prev_actions,
                critic_hidden_list=critic_hidden_list,
                deterministic=False,
            )
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
            metrics_logger.record_action(
                step=step_count,
                action_parts=actions_list,
                mv_indices=[cfg.mv_indices for cfg in AGENT_CFGS],
                info=info,
                delta_mv_max=exp_cfg.delta_mv_max,
            )
            last_transition_done = done
            raw_prod = info.get("raw_prod", None)
            raw_cost = info.get("raw_cost", None)
            completed_episode = metrics_logger.record_transition(
                step=step_count,
                reward=float(reward),
                raw_prod=raw_prod,
                raw_cost=raw_cost,
                shutdown=info.get("shutdown", False),
                done=done,
            )

            if completed_episode is not None:
                episodes_in_rollout += 1
                episode_seed += 1
                if step_count >= exp_cfg.total_steps:
                    break
                full_obs, _ = env.reset(seed=episode_seed)
                full_obs = np.asarray(full_obs, dtype=np.float32)
                critic_hidden_list = trainer.init_hidden(batch_size=1)
                prev_actions = [np.zeros(len(cfg.mv_indices), dtype=np.float32) for cfg in AGENT_CFGS]
                if step_count in checkpoint_step_set:
                    save_checkpoint_entry(step_count)
                continue
            if step_count >= exp_cfg.total_steps:
                break
            if step_count in checkpoint_step_set:
                save_checkpoint_entry(step_count)

        collected = len(buffers[0]["obs"])
        if collected == 0:
            raise RuntimeError("No rollout steps were collected")

        metrics_logger.record_rollout(step=step_count, length=collected, episodes=episodes_in_rollout)
        too_short = collected < exp_cfg.min_rollout_steps
        final_partial_allowed = exp_cfg.allow_short_final_rollout and step_count >= exp_cfg.total_steps
        if too_short and not final_partial_allowed:
            skipped_short_count += 1
            metrics_logger.record_update({"rollout/skipped_short": 1.0}, step=step_count)
            writer.flush()
            if exp_cfg.smoke:
                raise RuntimeError(
                    f"Smoke collected rollout_T={collected}, below min_rollout_steps={exp_cfg.min_rollout_steps}"
                )
            continue

        metrics_logger.record_update({"rollout/skipped_short": 0.0}, step=step_count)
        if last_transition_done:
            last_values = [0.0 for _ in AGENT_CFGS]
        else:
            last_local_obs_list = make_local_obs(full_obs, obs_rms_list, update_rms=False)
            last_values, _ = trainer.value(
                obs_list=last_local_obs_list,
                prev_act_list=prev_actions,
                critic_hidden_list=critic_hidden_list,
            )

        agent_rollouts = buffers_to_rollouts(buffers)
        logs = trainer.update(agent_rollouts=agent_rollouts, last_values=last_values, global_step=step_count)
        logs["rollout/episodes_per_update"] = float(episodes_in_rollout)
        final_logs = dict(logs)
        update_count += 1
        metrics_logger.record_update({
            "train/update_count": float(update_count),
            "train/updates_per_env_step": float(update_count) / max(float(step_count), 1.0),
        }, step=step_count)

        # Requested only by external runtime infrastructure; this is an
        # accepted PPO-update boundary and is a no-op on the canonical path.
        emit_runtime_hook("after_update")

        print(
            f"[{exp_cfg.name}] update={update_count} steps={step_count}/{exp_cfg.total_steps} "
            f"rollout_T={collected} episodes_per_update={episodes_in_rollout} "
            f"actor={logs['loss/actor']:.6f} "
            f"critic={logs['loss/critic']:.6f} kl={logs['train/approx_kl']:.6f} "
            f"kl_post={logs['train/approx_kl_post']:.6f} "
            f"actor_delta={logs['debug/actor_param_delta']:.6e} "
            f"fedavg_round={logs['fedavg/round']:.0f}"
        )

        if exp_cfg.smoke:
            break

    metrics_logger.flush(step=max(step_count, 1))
    emit_runtime_hook("finalizing")
    if checkpoint_steps:
        save_checkpoint_entry(step_count, final=True)
    else:
        trainer.save(run_root)
        save_obs_rms(run_root, AGENT_CFGS, obs_rms_list)
    writer.flush()
    writer.close()
    recent_profit_mean = mean_or_none(performance.profit_history[-100:])
    episode_return_mean = mean_or_none(performance.episode_return_history[-20:])
    shutdown_rate = mean_or_none(performance.shutdown_history)
    result = {
        "variant": "fl_split_shallow_fixed_horizon",
        "trainer_version": getattr(trainer.hp, "trainer_version", TRAINER_VERSION),
        "run_root": run_root,
        "seed": seed,
        "total_steps": step_count,
        "updates": update_count,
        "skipped_short": skipped_short_count,
        "final_logs": json_safe(final_logs),
        "recent_profit_mean": recent_profit_mean,
        "episode_return_mean": episode_return_mean,
        "shutdown_rate": shutdown_rate,
        "selection_metrics": {
            "primary": "recent_profit_mean",
            "secondary": ["episode_return_mean", "shutdown_rate"],
            "note": "Weighted composite score removed; compare metrics directly instead of using arbitrary fixed weights.",
        },
        "rollout_collection": {
            "mode": "fixed_horizon_continue_across_done",
            "horizon": int(exp_cfg.rollout_horizon),
            "last_update_rollout_T": int(final_logs.get("rollout/length", 0) or 0),
            "last_update_episodes_per_update": float(final_logs.get("rollout/episodes_per_update", 0.0) or 0.0),
        },
        "training_hps": json_safe(trainer.hp.__dict__),
        "model_cfg": json_safe(asdict(exp_cfg.model_cfg)),
        "structural_ablation": {
            "ablate_enhancer": bool(exp_cfg.model_cfg.ablate_enhancer),
            "bi_extractor": exp_cfg.model_cfg.bi_extractor,
            "aggregation": exp_cfg.model_cfg.aggregation,
            "action_pathway": exp_cfg.model_cfg.action_pathway,
            "fuse": exp_cfg.model_cfg.fuse,
            "fedavg_scope": exp_cfg.fedavg_scope,
            "fedavg_groups": trainer_fedavg_groups_from_config(exp_cfg),
        },
        "parameter_counts": json_safe(parameter_counts),
        "metric_windows": {
            "recent_profit_mean": "last 100 raw profit samples",
            "episode_return_mean": "last 20 completed episodes",
            "shutdown_rate": "all collected rollout steps",
        },
        "checkpoint_steps_requested": list(checkpoint_steps),
        "milestone_checkpoint_manifest": {
            "path": os.path.join(run_root, "milestone_checkpoint_manifest.json"),
            "entries": json_safe(milestone_entries),
            "saved_checkpoint_steps": sorted(saved_checkpoint_steps),
        },
        "checkpoint_files_exist": {
            "split_actor_0": os.path.exists(os.path.join(run_root, "split_actor_0.pt")),
            "split_actor_1": os.path.exists(os.path.join(run_root, "split_actor_1.pt")),
            "split_actor_2": os.path.exists(os.path.join(run_root, "split_actor_2.pt")),
            "split_bi_encoder_0": os.path.exists(os.path.join(run_root, "split_bi_encoder_0.pt")),
            "split_bi_encoder_1": os.path.exists(os.path.join(run_root, "split_bi_encoder_1.pt")),
            "split_bi_encoder_2": os.path.exists(os.path.join(run_root, "split_bi_encoder_2.pt")),
            "split_global_bi_server_critic": os.path.exists(
                os.path.join(run_root, "split_global_bi_server_critic.pt")
            ),
            "checkpoint_state": os.path.exists(os.path.join(run_root, "checkpoint_state.pt")),
            "milestone_manifest": os.path.exists(os.path.join(run_root, "milestone_manifest.json")),
            "result_summary": os.path.exists(os.path.join(run_root, "result_summary.json")),
        },
    }
    if checkpoint_steps:
        manifest_path = os.path.join(run_root, "milestone_checkpoint_manifest.json")
        write_json_file(Path(manifest_path), json_safe(result["milestone_checkpoint_manifest"]))
        result["milestone_checkpoint_manifest"]["path"] = manifest_path
    else:
        result["milestone_checkpoint_manifest"]["path"] = None
    with open(os.path.join(run_root, "result_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2, ensure_ascii=False)
    print(f"[done] {exp_cfg.name}: updates={update_count}, skipped_short={skipped_short_count}, steps={step_count}, run_root={run_root}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Shallow split-federated hybrid TEP experiment-style training entrypoint.")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--experiment-index",
        type=int,
        default=None,
        help="Override PYCHARM_EXPERIMENT_INDICES and run one experiment index.",
    )
    parser.add_argument("--smoke", action="store_true", default=None, help="Run exactly one short rollout/update and exit.")
    parser.add_argument("--no-smoke", action="store_true", help="Disable smoke mode for explicit short/long experiments.")
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-horizon", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--min-rollout-steps", type=int, default=None)
    parser.add_argument("--allow-short-final-rollout", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=PYCHARM_SEED)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "auto"))
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--timestamp-run-dir", action="store_true", default=None)
    parser.add_argument("--fedavg-interval", type=int, default=None)
    parser.add_argument("--fedavg-scope", choices=("both", "gru", "enhancer", "none"), default=None)
    parser.add_argument("--bi-type-grl", action="store_true", default=None)
    parser.add_argument("--no-bi-type-grl", action="store_false", dest="bi_type_grl", default=None)
    parser.add_argument("--bi-type-loss-coef", type=float, default=None)
    parser.add_argument("--bi-grl-alpha", type=float, default=None)
    parser.add_argument("--ablate-enhancer", action="store_true", default=None)
    parser.add_argument("--bi-extractor", choices=("gru", "mlp", "none"), default=None)
    parser.add_argument("--bi-post-gru-mlp", action="store_true", default=None)
    parser.add_argument("--no-bi-post-gru-mlp", action="store_false", dest="bi_post_gru_mlp", default=None)
    parser.add_argument("--aggregation", choices=("gat", "mean", "none"), default=None)
    parser.add_argument("--action-pathway", choices=("on", "off"), default=None)
    parser.add_argument("--fuse", choices=("learned", "concat"), default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Metadata only; current trainer uses full rollout updates.")
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--server-lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--rl-dt-seconds", type=int, default=None)
    parser.add_argument("--delta-mv-max", type=float, default=None)
    parser.add_argument("--production-index", type=int, default=None)
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--reset-reward", action="store_true", default=None)
    return parser.parse_args(argv)


def experiment_from_args(args, experiment_index: int) -> ExperimentConfig:
    if experiment_index < 0 or experiment_index >= len(EXPERIMENTS):
        raise ValueError(f"experiment-index {experiment_index} out of range")
    exp_cfg = EXPERIMENTS[experiment_index]
    overrides = {}
    for arg_name, field_name in [
        ("total_steps", "total_steps"),
        ("rollout_horizon", "rollout_horizon"),
        ("max_steps", "max_steps"),
        ("min_rollout_steps", "min_rollout_steps"),
        ("fedavg_interval", "fedavg_interval"),
        ("n_epochs", "n_epochs"),
        ("batch_size", "batch_size"),
        ("actor_lr", "actor_lr"),
        ("critic_lr", "critic_lr"),
        ("server_lr", "server_lr"),
        ("gamma", "gamma"),
        ("gae_lambda", "gae_lambda"),
        ("clip_range", "clip_range"),
        ("ent_coef", "ent_coef"),
        ("vf_coef", "vf_coef"),
        ("max_grad_norm", "max_grad_norm"),
        ("target_kl", "target_kl"),
        ("log_every", "log_every"),
        ("rl_dt_seconds", "rl_dt_seconds"),
        ("delta_mv_max", "delta_mv_max"),
        ("production_index", "production_index"),
        ("baseline", "baseline"),
        ("scale", "scale"),
        ("bi_type_grl", "bi_type_grl"),
        ("bi_type_loss_coef", "bi_type_loss_coef"),
        ("bi_grl_alpha", "bi_grl_alpha"),
        ("device", "device"),
        ("reset_reward", "reset_reward"),
        ("timestamp_run_dir", "timestamp_run_dir"),
        ("allow_short_final_rollout", "allow_short_final_rollout"),
        ("fedavg_scope", "fedavg_scope"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    if args.run_root is not None:
        overrides["run_root"] = args.run_root
    if args.smoke is True:
        overrides["smoke"] = True
    if args.no_smoke:
        overrides["smoke"] = False
    model_cfg = exp_cfg.model_cfg
    model_overrides = {}
    for arg_name, field_name in [
        ("ablate_enhancer", "ablate_enhancer"),
        ("bi_extractor", "bi_extractor"),
        ("bi_post_gru_mlp", "bi_post_gru_mlp"),
        ("aggregation", "aggregation"),
        ("action_pathway", "action_pathway"),
        ("fuse", "fuse"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            model_overrides[field_name] = value
    if model_overrides:
        overrides["model_cfg"] = replace(model_cfg, **model_overrides)
    return replace(exp_cfg, **overrides)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        custom_overrides = [
            "name",
            "checkpoint_steps",
            "total_steps",
            "rollout_horizon",
            "max_steps",
            "min_rollout_steps",
            "fedavg_interval",
            "n_epochs",
            "batch_size",
            "actor_lr",
            "critic_lr",
            "server_lr",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
            "log_every",
            "rl_dt_seconds",
            "delta_mv_max",
            "production_index",
            "baseline",
            "scale",
            "bi_type_grl",
            "bi_type_loss_coef",
            "bi_grl_alpha",
            "device",
            "reset_reward",
            "timestamp_run_dir",
            "allow_short_final_rollout",
            "fedavg_scope",
            "ablate_enhancer",
            "bi_extractor",
            "bi_post_gru_mlp",
            "aggregation",
            "action_pathway",
            "fuse",
        ]
        has_custom_overrides = any(getattr(args, name, None) is not None for name in custom_overrides)
        if args.experiment_index is not None:
            experiment_indices = (args.experiment_index,)
        elif has_custom_overrides:
            experiment_indices = (None,)
        elif args.smoke is True:
            experiment_indices = (0,)
        else:
            experiment_indices = PYCHARM_EXPERIMENT_INDICES
        for experiment_index in experiment_indices:
            if experiment_index is None:
                exp_cfg = apply_common_overrides(make_default_experiment(), args)
            else:
                exp_cfg = apply_common_overrides(experiment_from_args(args, experiment_index), args)
            run_one_experiment(exp_cfg, seed=args.seed)
        return 0
    except TEPUnavailableError as exc:
        print(f"TEP simulator unavailable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
