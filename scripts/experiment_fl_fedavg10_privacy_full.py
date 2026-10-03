"""FedAvg10 full-privacy experiment driver.

This entrypoint keeps the full FedAvg10 rollout/update contract from the
privacy-module-only v3 path, but switches the privacy contract so the BI
encoder receives task plus privacy gradients while the reconstructor is first
trained on detached BI features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.experiment_fl_fedavg10_privacy_module_only as base_mod  # noqa: E402
import scripts.experiment_fl_fedavg10_privacy_module_only_v3 as base_exp  # noqa: E402
from trainers.fl_split_ppo_trainer_privacy_full import (  # noqa: E402
    TRAINER_VERSION,
    DRAVLSplitFedPPOHyperParamsFullV3,
    PrivacyFullSplitFedPPOTrainerV3,
)

_TASK_NETWORK_TRAINER_PARAMETER_COUNTS = base_mod.base.trainer_parameter_counts
_BASE_WRITE_CONFIG = base_exp.write_config
_BASE_VALIDATE_EXPERIMENT = base_exp.validate_experiment

BASELINE_ANCHOR = base_exp.BASELINE_ANCHOR
EXPERIMENT_SET = "fl_fedavg10_privacy_full"
SPEC_VERSION = "fl_fedavg10_privacy_full_v1"


@dataclass
class ExperimentConfig(base_exp.ExperimentConfig):
    name: str = "fl_fedavg10_privacy_full"
    actor_lr: float = 1.2e-4
    critic_lr: float = 1e-4
    server_lr: float = 1e-4
    clip_range: float = 0.2
    ent_coef: float = 0.05
    n_epochs: int = 4
    critic_epochs: int = 16
    target_kl: Optional[float] = 0.03
    delta_mv_max: float = 0.3
    rollout_horizon: int = 1024
    total_steps: int = 200_000
    log_every: int = 1000
    fedavg_interval: int = 10
    bi_extractor_sharing_mode: str = "independent"
    bi_extractor_fedavg_interval: int = 10
    max_steps: int = 300
    min_rollout_steps: int = 512
    smoke: bool = False
    run_root: str = os.path.join("runs", EXPERIMENT_SET)
    model_cfg: base_mod.base.SplitFLModelConfig = field(default_factory=base_mod.base.SplitFLModelConfig)

    privacy_enabled: bool = True
    priv_ar_coef: float = 0.0
    priv_nr_coef: float = 0.0
    priv_dcor_coef: float = 0.0
    grl_lambda: float = 1.0
    priv_noise_std: float = 1.0
    priv_lr: float = 1e-4
    priv_dcor_max_samples: int = 2048
    priv_recon_hidden_dims: Tuple[int, ...] = (128, 128)
    privacy_mode: str = "full"
    detached_privacy_sidecar: bool = False
    privacy_reconstructor_steps: int = 1
    privacy_encoder_steps: int = 1
    privacy_log_grad_components: bool = True
    privacy_posthoc_eval: bool = True

    experiment_name: str = "fl_fedavg10_privacy_full"
    sweep_name: Optional[str] = None
    sweep_index: Optional[int] = None
    spec_version: str = SPEC_VERSION


def json_safe(value: Any) -> Any:
    return base_exp.json_safe(value)


def privacy_hps(exp_cfg: ExperimentConfig) -> Dict[str, Any]:
    lambda_eff = float(exp_cfg.grl_lambda) * float(exp_cfg.priv_ar_coef)
    return {
        "privacy_enabled": bool(exp_cfg.privacy_enabled),
        "privacy_mode": exp_cfg.privacy_mode,
        "detached_privacy_sidecar": bool(exp_cfg.detached_privacy_sidecar),
        "priv_ar_coef": float(exp_cfg.priv_ar_coef),
        "priv_nr_coef": float(exp_cfg.priv_nr_coef),
        "priv_dcor_coef": float(exp_cfg.priv_dcor_coef),
        "grl_lambda": float(exp_cfg.grl_lambda),
        "lambda_eff": float(lambda_eff),
        "priv_noise_std": float(exp_cfg.priv_noise_std),
        "priv_lr": float(exp_cfg.priv_lr),
        "priv_dcor_max_samples": int(exp_cfg.priv_dcor_max_samples),
        "priv_recon_hidden_dims": list(exp_cfg.priv_recon_hidden_dims),
        "privacy_reconstructor_steps": int(exp_cfg.privacy_reconstructor_steps),
        "privacy_encoder_steps": int(exp_cfg.privacy_encoder_steps),
        "privacy_log_grad_components": bool(exp_cfg.privacy_log_grad_components),
        "privacy_posthoc_eval": bool(exp_cfg.privacy_posthoc_eval),
    }


def build_hparams(exp_cfg: ExperimentConfig) -> DRAVLSplitFedPPOHyperParamsFullV3:
    return DRAVLSplitFedPPOHyperParamsFullV3(
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
        critic_epochs=exp_cfg.critic_epochs,
        batch_size=exp_cfg.batch_size,
        rollout_horizon=exp_cfg.rollout_horizon,
        total_steps=exp_cfg.total_steps,
        log_every=exp_cfg.log_every,
        target_kl=exp_cfg.target_kl,
        fedavg_interval=exp_cfg.fedavg_interval,
        bi_extractor_sharing_mode=exp_cfg.bi_extractor_sharing_mode,
        bi_extractor_fedavg_interval=exp_cfg.bi_extractor_fedavg_interval,
        fedavg_scope=exp_cfg.fedavg_scope,
        bi_type_grl=exp_cfg.bi_type_grl,
        bi_type_loss_coef=exp_cfg.bi_type_loss_coef,
        bi_grl_alpha=exp_cfg.bi_grl_alpha,
        privacy_enabled=exp_cfg.privacy_enabled,
        priv_ar_coef=exp_cfg.priv_ar_coef,
        priv_nr_coef=exp_cfg.priv_nr_coef,
        priv_dcor_coef=exp_cfg.priv_dcor_coef,
        grl_lambda=exp_cfg.grl_lambda,
        priv_noise_std=exp_cfg.priv_noise_std,
        priv_lr=exp_cfg.priv_lr,
        priv_dcor_max_samples=exp_cfg.priv_dcor_max_samples,
        priv_recon_hidden_dims=tuple(exp_cfg.priv_recon_hidden_dims),
        privacy_reconstructor_steps=exp_cfg.privacy_reconstructor_steps,
        privacy_encoder_steps=exp_cfg.privacy_encoder_steps,
        privacy_log_grad_components=exp_cfg.privacy_log_grad_components,
    )


def build_trainer(exp_cfg: ExperimentConfig, device: str, writer=None) -> PrivacyFullSplitFedPPOTrainerV3:
    obs_dims = [len(cfg.obs_indices) for cfg in base_mod.base.AGENT_CFGS]
    act_dims = [len(cfg.mv_indices) for cfg in base_mod.base.AGENT_CFGS]
    model_cfg = exp_cfg.model_cfg
    trainer = PrivacyFullSplitFedPPOTrainerV3.build_default(
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


def trainer_parameter_counts(trainer: PrivacyFullSplitFedPPOTrainerV3) -> Dict[str, Any]:
    # Capture the original task-network counter before the base runner is
    # monkey-patched. The privacy-module-only counter already adds the privacy
    # defenses, so calling it here would double-count them.
    counts = _TASK_NETWORK_TRAINER_PARAMETER_COUNTS(trainer)
    privacy_counts = [base_mod.base.count_parameters(defense) for defense in trainer.priv_defenses]
    counts["privacy_defenses"] = privacy_counts
    counts["active_task_total"] = int(counts.get("total", 0))
    counts["privacy_sidecar_total"] = int(sum(privacy_counts))
    counts["total"] = int(counts["active_task_total"] + counts["privacy_sidecar_total"])
    return counts


def write_config(
    run_root: str,
    exp_cfg: ExperimentConfig,
    hp: DRAVLSplitFedPPOHyperParamsFullV3,
    joint_mv_indices: Tuple[int, ...],
    device: str,
    seed: int,
    parameter_counts: Optional[Dict[str, Any]] = None,
):
    _BASE_WRITE_CONFIG(run_root, exp_cfg, hp, joint_mv_indices, device, seed, parameter_counts=parameter_counts)
    config_path = os.path.join(run_root, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(
        {
            "ablation": "fl_split_full_privacy",
            "canonical_variant": "fl_fedavg10_privacy_full",
            "trainer_version": getattr(hp, "trainer_version", TRAINER_VERSION),
            "spec_version": SPEC_VERSION,
            "experiment_name": exp_cfg.experiment_name,
            "sweep_name": exp_cfg.sweep_name,
            "sweep_index": exp_cfg.sweep_index,
            "baseline_anchor": BASELINE_ANCHOR,
            "privacy_mode": exp_cfg.privacy_mode,
            "detached_privacy_sidecar": False,
            "main_model_contract": "baseline fedavg10 plus BI privacy gradients",
            "training_schedule": {
                **config.get("training_schedule", {}),
                "actor_n_epochs": int(exp_cfg.n_epochs),
                "critic_epochs": int(exp_cfg.critic_epochs),
                "target_kl": exp_cfg.target_kl,
            },
            "fl": {
                "fedavg_interval": int(exp_cfg.fedavg_interval),
                "bi_extractor_sharing_mode": exp_cfg.bi_extractor_sharing_mode,
                "bi_extractor_fedavg_interval": int(exp_cfg.bi_extractor_fedavg_interval),
                "fedavg_scope": exp_cfg.fedavg_scope,
                "split_boundary": "bi_only",
                "main_model_contract": "baseline fedavg10 plus BI privacy gradients",
            },
            "privacy_hps": privacy_hps(exp_cfg),
            "logging_contract": {
                **config.get("logging_contract", {}),
                "privacy_full_tags": [
                    "loss/privacy_recon_attacker_total",
                    "loss/recon_attacker",
                    "privacy/attacker_applied",
                    "privacy/full_privacy_applied",
                    "privacy/bi_grad_norm_td_{i}",
                    "privacy/bi_grad_norm_priv_{i}",
                    "privacy/bi_grad_ratio_{i}",
                    "grad_norm/priv_defense_{i}",
                    "grad_norm/bi_encoder_{i}",
                    "loss/privacy_total",
                    "ratio/privacy_backward_to_critic",
                    "debug/privacy_reconstructor_detached",
                ],
            },
            "privacy_gradient_contract": {
                "main_model": "baseline critic/actor/server update plus live BI privacy gradients",
                "reconstructor": "AR raw trains on detached BI features before the main privacy update",
                "bi_update": "task + AR + NR + dCor gradients update the live BI encoder",
                "fedavg": "privacy defense parameters excluded from run_fedavg",
            },
            "non_goals": sorted(
                set(config.get("non_goals", []))
                | {"main_model_privacy_backprop_disabled", "privacy_sidecar_detach"}
            ),
        }
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2, ensure_ascii=False)


def validate_experiment(exp_cfg: ExperimentConfig):
    _BASE_VALIDATE_EXPERIMENT(exp_cfg)
    if exp_cfg.critic_epochs <= 0:
        raise ValueError("critic_epochs must be positive")
    if (exp_cfg.bi_type_grl or exp_cfg.bi_type_loss_coef != 0.0) and not exp_cfg.model_cfg.bi_post_gru_mlp:
        raise ValueError("bi_type_grl or bi_type_loss_coef requires model_cfg.bi_post_gru_mlp=True")
    if exp_cfg.bi_type_grl and exp_cfg.bi_type_loss_coef <= 0.0:
        raise ValueError("bi_type_grl requires bi_type_loss_coef > 0.0")
    if exp_cfg.bi_type_loss_coef < 0.0:
        raise ValueError("bi_type_loss_coef must be non-negative")
    if exp_cfg.bi_grl_alpha <= 0.0:
        raise ValueError("bi_grl_alpha must be positive")
    if exp_cfg.priv_dcor_max_samples <= 0:
        raise ValueError("priv_dcor_max_samples must be positive")
    if exp_cfg.priv_lr <= 0:
        raise ValueError("priv_lr must be positive")
    if exp_cfg.privacy_reconstructor_steps <= 0:
        raise ValueError("privacy_reconstructor_steps must be positive")
    if exp_cfg.privacy_encoder_steps <= 0:
        raise ValueError("privacy_encoder_steps must be positive")
    if any(int(dim) <= 0 for dim in exp_cfg.priv_recon_hidden_dims):
        raise ValueError(f"priv_recon_hidden_dims must be positive, got {exp_cfg.priv_recon_hidden_dims}")
    if exp_cfg.bi_extractor_sharing_mode not in {"independent", "true_shared", "fedavg10"}:
        raise ValueError(
            "bi_extractor_sharing_mode must be one of independent/true_shared/fedavg10"
        )
    if exp_cfg.bi_extractor_sharing_mode == "fedavg10" and exp_cfg.bi_extractor_fedavg_interval <= 0:
        raise ValueError("bi_extractor_fedavg_interval must be positive when sharing mode is fedavg10")
    if exp_cfg.bi_extractor_sharing_mode != "fedavg10" and exp_cfg.bi_extractor_fedavg_interval < 0:
        raise ValueError("bi_extractor_fedavg_interval must be non-negative")


@contextmanager
def patched_base_runner() -> Iterator[None]:
    base_runner = base_mod.base
    originals = {
        "build_hparams": base_runner.build_hparams,
        "build_trainer": base_runner.build_trainer,
        "trainer_parameter_counts": base_runner.trainer_parameter_counts,
        "write_config": base_runner.write_config,
        "validate_experiment": base_runner.validate_experiment,
    }
    try:
        base_runner.build_hparams = build_hparams  # type: ignore[assignment]
        base_runner.build_trainer = build_trainer  # type: ignore[assignment]
        base_runner.trainer_parameter_counts = trainer_parameter_counts  # type: ignore[assignment]
        base_runner.write_config = write_config  # type: ignore[assignment]
        base_runner.validate_experiment = validate_experiment  # type: ignore[assignment]
        yield
    finally:
        for name, func in originals.items():
            setattr(base_runner, name, func)


def default_run_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(PROJECT_ROOT) / "runs" / f"{EXPERIMENT_SET}_{stamp}"


def resolve_run_root(exp_cfg: ExperimentConfig, run_root_override: Optional[str]) -> Path:
    if run_root_override is not None:
        run_root = Path(run_root_override)
        if exp_cfg.timestamp_run_dir:
            run_root = run_root / f"{EXPERIMENT_SET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return run_root
    return default_run_root()


def enrich_result_summary(run_root: str, exp_cfg: ExperimentConfig, seed: int) -> Dict[str, Any]:
    path = os.path.join(run_root, "result_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    privacy_files = {
        f"fl_bi_privacy_defense_{i}": os.path.exists(os.path.join(run_root, f"fl_bi_privacy_defense_{i}.pt"))
        for i in range(len(base_mod.base.AGENT_CFGS))
    }
    final_logs = summary.get("final_logs", {}) or {}
    privacy_logs = {
        k: v
        for k, v in final_logs.items()
        if k.startswith("privacy/") or "priv" in k or k.startswith("loss/privacy") or k.startswith("ratio/privacy")
    }
    summary.update(
        {
            "variant": "fl_fedavg10_privacy_full",
            "canonical_variant": "fl_fedavg10_privacy_full",
            "trainer_version": TRAINER_VERSION,
            "spec_version": SPEC_VERSION,
            "experiment_name": exp_cfg.experiment_name,
            "sweep_name": exp_cfg.sweep_name,
            "sweep_index": exp_cfg.sweep_index,
            "seed": seed,
            "canonical_run_root": run_root,
            "baseline_anchor": BASELINE_ANCHOR,
            "privacy_mode": exp_cfg.privacy_mode,
            "detached_privacy_sidecar": False,
            "fl": {
                "fedavg_interval": int(exp_cfg.fedavg_interval),
                "fedavg_scope": exp_cfg.fedavg_scope,
                "split_boundary": "bi_only",
                "main_model_contract": "baseline fedavg10 plus BI privacy gradients",
            },
            "privacy_hps": privacy_hps(exp_cfg),
            "privacy_logs": privacy_logs,
            "tail20": {
                "critic": summary.get("loss/critic", final_logs.get("loss/critic")),
                "privacy_total": final_logs.get("loss/privacy_total"),
                "privacy_recon_attacker_total": final_logs.get("loss/privacy_recon_attacker_total"),
                "approx_kl": final_logs.get("train/approx_kl"),
                "profit": summary.get("recent_profit_mean"),
            },
            "checkpoint_files_exist": {
                **(summary.get("checkpoint_files_exist", {}) or {}),
                **privacy_files,
            },
            "cli_echo": {
                "privacy_enabled": bool(exp_cfg.privacy_enabled),
                "privacy_mode": exp_cfg.privacy_mode,
                "detached_privacy_sidecar": bool(exp_cfg.detached_privacy_sidecar),
                "priv_ar_coef": float(exp_cfg.priv_ar_coef),
                "priv_nr_coef": float(exp_cfg.priv_nr_coef),
                "priv_dcor_coef": float(exp_cfg.priv_dcor_coef),
                "grl_lambda": float(exp_cfg.grl_lambda),
                "fedavg_interval": int(exp_cfg.fedavg_interval),
                "total_steps": int(exp_cfg.total_steps),
                "rollout_horizon": int(exp_cfg.rollout_horizon),
                "critic_epochs": int(exp_cfg.critic_epochs),
                "seed": int(seed),
            },
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    return summary


def run_one_experiment(exp_cfg: ExperimentConfig, seed: int = 42, run_root_override: Optional[str] = None) -> Dict[str, Any]:
    run_root = str(resolve_run_root(exp_cfg, run_root_override))
    with patched_base_runner():
        base_mod.base.run_one_experiment(exp_cfg, seed=seed, run_root_override=run_root)
    return enrich_result_summary(run_root, exp_cfg, seed=seed)


def make_default_experiment() -> ExperimentConfig:
    return ExperimentConfig()


def apply_common_overrides(exp_cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updates: Dict[str, Any] = {}
    for field_name in (
        "total_steps",
        "rollout_horizon",
        "max_steps",
        "min_rollout_steps",
        "n_epochs",
        "critic_epochs",
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
        "fedavg_interval",
        "bi_extractor_fedavg_interval",
        "priv_ar_coef",
        "priv_nr_coef",
        "priv_dcor_coef",
        "grl_lambda",
        "bi_type_grl",
        "bi_type_loss_coef",
        "bi_grl_alpha",
        "priv_noise_std",
        "priv_lr",
        "priv_dcor_max_samples",
        "privacy_reconstructor_steps",
        "privacy_encoder_steps",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            updates[field_name] = value
    if args.device is not None:
        updates["device"] = args.device
    if args.fedavg_scope is not None:
        updates["fedavg_scope"] = args.fedavg_scope
    if args.bi_extractor_sharing_mode is not None:
        updates["bi_extractor_sharing_mode"] = args.bi_extractor_sharing_mode
    if args.bi_extractor_fedavg_interval is not None:
        updates["bi_extractor_fedavg_interval"] = int(args.bi_extractor_fedavg_interval)
    if args.timestamp_run_dir is not None:
        updates["timestamp_run_dir"] = bool(args.timestamp_run_dir)
    if args.allow_short_final_rollout is not None:
        updates["allow_short_final_rollout"] = bool(args.allow_short_final_rollout)
    if args.privacy_enabled is not None:
        updates["privacy_enabled"] = bool(args.privacy_enabled)
    if args.privacy_mode is not None:
        updates["privacy_mode"] = args.privacy_mode
    if args.detached_privacy_sidecar is not None:
        updates["detached_privacy_sidecar"] = bool(args.detached_privacy_sidecar)
    if args.privacy_log_grad_components is not None:
        updates["privacy_log_grad_components"] = bool(args.privacy_log_grad_components)
    if args.privacy_posthoc_eval is not None:
        updates["privacy_posthoc_eval"] = bool(args.privacy_posthoc_eval)
    if args.bi_type_grl is not None:
        updates["bi_type_grl"] = bool(args.bi_type_grl)
    if args.bi_type_loss_coef is not None:
        updates["bi_type_loss_coef"] = float(args.bi_type_loss_coef)
    if args.bi_grl_alpha is not None:
        updates["bi_grl_alpha"] = float(args.bi_grl_alpha)
    if args.priv_recon_hidden_dims:
        updates["priv_recon_hidden_dims"] = tuple(int(x) for x in args.priv_recon_hidden_dims.split(",") if x.strip())
    if args.bi_post_gru_mlp is not None:
        updates["model_cfg"] = replace(exp_cfg.model_cfg, bi_post_gru_mlp=bool(args.bi_post_gru_mlp))
    if args.smoke is True:
        updates["smoke"] = True
        updates.setdefault("total_steps", 16)
        updates.setdefault("rollout_horizon", 16)
        updates.setdefault("n_epochs", 1)
        updates.setdefault("max_steps", 16)
        updates.setdefault("min_rollout_steps", 16)
        updates.setdefault("log_every", 16)
    if args.no_smoke:
        updates["smoke"] = False
        updates.setdefault("n_epochs", 4)
        updates.setdefault("critic_epochs", 16)
        updates.setdefault("rollout_horizon", 1024)
        updates.setdefault("total_steps", 200_000)
        updates.setdefault("max_steps", 300)
        updates.setdefault("min_rollout_steps", 512)
        updates.setdefault("log_every", 1000)
    if args.name:
        updates["name"] = args.name
        updates["experiment_name"] = args.name
    return replace(exp_cfg, **updates)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fedavg10 full privacy with live BI gradients.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-horizon", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--min-rollout-steps", type=int, default=None)
    parser.add_argument("--allow-short-final-rollout", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "auto"))
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--timestamp-run-dir", action="store_true", default=None)
    parser.add_argument("--fedavg-interval", type=int, default=None)
    parser.add_argument(
        "--bi-extractor-sharing-mode",
        choices=("independent", "true_shared", "fedavg10"),
        default=None,
    )
    parser.add_argument("--bi-extractor-fedavg-interval", type=int, default=None)
    parser.add_argument("--fedavg-scope", choices=("both", "gru", "enhancer", "none"), default=None)
    parser.add_argument("--bi-post-gru-mlp", action="store_true", default=None)
    parser.add_argument("--no-bi-post-gru-mlp", action="store_false", dest="bi_post_gru_mlp", default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--critic-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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
    parser.add_argument("--bi-type-grl", action="store_true", default=None)
    parser.add_argument("--no-bi-type-grl", action="store_false", dest="bi_type_grl", default=None)
    parser.add_argument("--bi-type-loss-coef", type=float, default=None)
    parser.add_argument("--bi-grl-alpha", type=float, default=None)

    privacy_group = parser.add_mutually_exclusive_group()
    privacy_group.add_argument("--privacy-enabled", dest="privacy_enabled", action="store_true", default=None)
    privacy_group.add_argument("--privacy-disabled", dest="privacy_enabled", action="store_false", default=None)
    parser.add_argument("--privacy-mode", default=None)
    detached_group = parser.add_mutually_exclusive_group()
    detached_group.add_argument("--detached-privacy-sidecar", dest="detached_privacy_sidecar", action="store_true", default=None)
    detached_group.add_argument("--attached-privacy-sidecar", dest="detached_privacy_sidecar", action="store_false", default=None)
    parser.add_argument("--priv-ar-coef", dest="priv_ar_coef", type=float, default=None)
    parser.add_argument("--priv-nr-coef", dest="priv_nr_coef", type=float, default=None)
    parser.add_argument("--priv-dcor-coef", dest="priv_dcor_coef", type=float, default=None)
    parser.add_argument("--grl-lambda", type=float, default=None)
    parser.add_argument("--priv-noise-std", type=float, default=None)
    parser.add_argument("--priv-lr", type=float, default=None)
    parser.add_argument("--priv-dcor-max-samples", type=int, default=None)
    parser.add_argument("--priv-recon-hidden-dims", default=None, help="Comma-separated hidden sizes, e.g. 128,128")
    parser.add_argument("--privacy-reconstructor-steps", type=int, default=None)
    parser.add_argument("--privacy-encoder-steps", type=int, default=None)
    privacy_log_group = parser.add_mutually_exclusive_group()
    privacy_log_group.add_argument("--privacy-log-grad-components", dest="privacy_log_grad_components", action="store_true", default=None)
    privacy_log_group.add_argument("--no-privacy-log-grad-components", dest="privacy_log_grad_components", action="store_false", default=None)
    privacy_posthoc_group = parser.add_mutually_exclusive_group()
    privacy_posthoc_group.add_argument("--privacy-posthoc-eval", dest="privacy_posthoc_eval", action="store_true", default=None)
    privacy_posthoc_group.add_argument("--no-privacy-posthoc-eval", dest="privacy_posthoc_eval", action="store_false", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    exp_cfg = apply_common_overrides(make_default_experiment(), args)
    summary = run_one_experiment(exp_cfg, seed=args.seed, run_root_override=args.run_root)
    print(json.dumps(json_safe({
        "run_root": summary.get("run_root"),
        "recent_profit_mean": summary.get("recent_profit_mean"),
        "episode_return_mean": summary.get("episode_return_mean"),
        "shutdown_rate": summary.get("shutdown_rate"),
        "bi_extractor_sharing_mode": summary.get("config", {}).get("fl", {}).get("bi_extractor_sharing_mode"),
        "privacy_hps": summary.get("privacy_hps"),
        "checkpoint_files_exist": summary.get("checkpoint_files_exist"),
    }), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
