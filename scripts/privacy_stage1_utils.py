"""Stage 1 fixed-eval and post-hoc BI probe utilities for full-privacy runs."""

from __future__ import annotations

import csv
import inspect
import hashlib
import json
import os
import shutil
import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experiment_fl_fedavg10_privacy_full as full  # noqa: E402
from scripts.train_fl_split_shallow_hybrid import (  # noqa: E402
    AGENT_CFGS,
    RunningMeanStd,
    build_joint_mv_indices,
    make_env,
    make_local_obs,
    resolve_device,
    set_global_seed,
    SplitFLModelConfig,
)

SPEC_VERSION = "privacy_stage1_eval_v1"
PROBE_SEEDS = (0, 1, 2)
TARGET_NAMES = ("obs", "act", "obs_action")
PROBE_TYPES = ("constant", "linear", "small_mlp", "strong_mlp", "shuffled")
COMMON_INPUT_PROBE_ARCH = {
    "constant": None,
    "linear": "linear",
    "small_mlp": "small_mlp",
    "strong_mlp": "strong_mlp",
    "shuffled": "strong_mlp",
}


@dataclass
class FrozenRun:
    label: str
    run_dir: Path
    exp_cfg: full.ExperimentConfig
    config: Dict[str, Any]
    trainer: Any
    obs_rms_list: List[RunningMeanStd]
    result_summary: Dict[str, Any]
    resolved_checkpoints: Dict[str, str]
    device: str
    checkpoint_hash: str = ""


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
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def parse_labels(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        labels = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    else:
        labels = [str(item).strip() for item in value if str(item).strip()]
    if not labels:
        raise argparse.ArgumentTypeError("labels must contain at least one item")
    return labels


def _tuple_fields(dataclass_type) -> set[str]:
    return {f.name for f in fields(dataclass_type) if "Tuple" in str(f.type) or "tuple" in str(f.type)}


def split_model_cfg_from_dict(payload: Mapping[str, Any]) -> SplitFLModelConfig:
    allowed = {f.name for f in fields(SplitFLModelConfig)}
    tuple_names = _tuple_fields(SplitFLModelConfig)
    kwargs: Dict[str, Any] = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        if key in tuple_names and isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return SplitFLModelConfig(**kwargs)


def experiment_cfg_from_run_config(config: Mapping[str, Any]) -> full.ExperimentConfig:
    exp_payload = dict(config.get("experiment", {}))
    model_payload = exp_payload.pop("model_cfg", config.get("model_cfg", {}))
    exp_payload["model_cfg"] = split_model_cfg_from_dict(model_payload)

    default_exp = full.make_default_experiment()
    allowed = {f.name for f in fields(full.ExperimentConfig)}
    tuple_names = _tuple_fields(full.ExperimentConfig)
    updates: Dict[str, Any] = {}
    for key, value in exp_payload.items():
        if key not in allowed:
            continue
        if key in tuple_names and isinstance(value, list):
            value = tuple(value)
        updates[key] = value
    if "name" not in updates:
        updates["name"] = str(config.get("experiment_name", default_exp.name))
    return replace(default_exp, **updates)


def load_obs_rms(run_root: Path, allow_missing: bool = False) -> List[RunningMeanStd]:
    rms_list: List[RunningMeanStd] = []
    for cfg in AGENT_CFGS:
        path = run_root / f"{cfg.name}_obs_rms.npz"
        rms = RunningMeanStd(shape=(len(cfg.obs_indices),))
        if path.exists():
            data = np.load(path)
            missing = {"mean", "var", "count"} - set(data.files)
            if missing:
                raise KeyError(f"{path} is missing obs RMS keys: {sorted(missing)}")
            rms.mean = np.asarray(data["mean"], dtype=np.float64)
            rms.var = np.asarray(data["var"], dtype=np.float64)
            rms.count = float(data["count"])
        elif not allow_missing:
            raise FileNotFoundError(f"Missing obs normalization artifact {path}")
        rms_list.append(rms)
    return rms_list


def load_result_summary(run_root: Path) -> Dict[str, Any]:
    path = run_root / "result_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_resolved_checkpoints(resolved_checkpoints: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(resolved_checkpoints):
        path = Path(resolved_checkpoints[key])
        digest.update(str(key).encode("utf-8"))
        digest.update(str(path).encode("utf-8"))
        if path.exists():
            digest.update(_sha256_file(path).encode("utf-8"))
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    data = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(data.tobytes())
    digest.update(str(tuple(data.shape)).encode("utf-8"))
    digest.update(str(data.dtype).encode("utf-8"))
    return digest.hexdigest()


def _stable_hash(parts: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(parts):
        value = parts[key]
        digest.update(str(key).encode("utf-8"))
        if isinstance(value, Mapping):
            digest.update(_stable_hash(value).encode("utf-8"))
        elif isinstance(value, (list, tuple)):
            digest.update(json.dumps(json_safe(value), sort_keys=True).encode("utf-8"))
        elif torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous().to(dtype=torch.float32)
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            arr = np.asarray(value)
            digest.update(str(tuple(arr.shape)).encode("utf-8"))
            digest.update(str(arr.dtype).encode("utf-8"))
            digest.update(np.ascontiguousarray(arr).tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _build_cache_key(
    *,
    mode: str,
    label: str,
    checkpoint_hash: str,
    target: str,
    probe: str,
    seed: int,
    bi_feature_mode: str,
    input_hash: str,
    target_hash: str,
) -> str:
    return "|".join(
        [
            f"mode={mode}",
            f"branch={label}",
            f"checkpoint_hash={checkpoint_hash}",
            f"target={target}",
            f"probe={probe}",
            f"seed={int(seed)}",
            f"bi_feature_mode={bi_feature_mode}",
            f"input_hash={input_hash}",
            f"target_hash={target_hash}",
        ]
    )


def load_frozen_run(run_dir: Path, device: str) -> FrozenRun:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {run_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    exp_cfg = experiment_cfg_from_run_config(config)
    trainer = full.build_trainer(exp_cfg, device=device)

    resolved_checkpoints: Dict[str, str] = {}
    for i, client in enumerate(trainer.clients):
        actor_path = run_dir / f"split_actor_{i}.pt"
        bi_path = run_dir / f"split_bi_encoder_{i}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Missing actor checkpoint {actor_path}")
        if not bi_path.exists():
            raise FileNotFoundError(f"Missing BI checkpoint {bi_path}")
        client.actor.load_state_dict(torch.load(actor_path, map_location=device))
        client.bi_encoder.load_state_dict(torch.load(bi_path, map_location=device))
        resolved_checkpoints[f"split_actor_{i}"] = str(actor_path)
        resolved_checkpoints[f"split_bi_encoder_{i}"] = str(bi_path)

    server_path = run_dir / "split_global_bi_server_critic.pt"
    if not server_path.exists():
        raise FileNotFoundError(f"Missing server checkpoint {server_path}")
    trainer.server.load_state_dict(torch.load(server_path, map_location=device))
    resolved_checkpoints["split_global_bi_server_critic"] = str(server_path)

    if hasattr(trainer, "priv_defenses"):
        for i, defense in enumerate(trainer.priv_defenses):
            defense_path = run_dir / f"fl_bi_privacy_defense_{i}.pt"
            if defense_path.exists():
                defense.load_state_dict(torch.load(defense_path, map_location=device))
                resolved_checkpoints[f"fl_bi_privacy_defense_{i}"] = str(defense_path)

    trainer.clients.eval()
    trainer.server.eval()
    if hasattr(trainer, "priv_defenses"):
        for defense in trainer.priv_defenses:
            defense.eval()
    for module in [trainer.clients, trainer.server]:
        for param in module.parameters():
            param.requires_grad_(False)
    if hasattr(trainer, "priv_defenses"):
        for defense in trainer.priv_defenses:
            for param in defense.parameters():
                param.requires_grad_(False)

    obs_rms_list = load_obs_rms(run_dir)
    result_summary = load_result_summary(run_dir)
    checkpoint_hash = _hash_resolved_checkpoints(resolved_checkpoints)
    return FrozenRun(
        label=str(config.get("experiment_name", run_dir.name)),
        run_dir=run_dir,
        exp_cfg=exp_cfg,
        config=config,
        trainer=trainer,
        obs_rms_list=obs_rms_list,
        result_summary=result_summary,
        resolved_checkpoints=resolved_checkpoints,
        device=device,
        checkpoint_hash=checkpoint_hash,
    )


def resolve_run_dirs(sweep_root: Path, labels: Sequence[str], seed: int) -> Dict[str, Path]:
    manifest_path = sweep_root / "launcher_manifest.json"
    resolved: Dict[str, Path] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_runs: Dict[str, Path] = {}
        for batch in manifest.get("batches", []):
            for run in batch.get("runs", []):
                label = str(run.get("label", ""))
                run_root = Path(run.get("run_root", ""))
                if label and run_root:
                    manifest_runs[label] = run_root
        for label in labels:
            path = manifest_runs.get(label)
            if path is not None and path.exists():
                resolved[label] = path
    for label in labels:
        if label in resolved:
            continue
        matches = sorted(sweep_root.glob(f"**/coef_{label}/critic*/seed{int(seed)}"))
        if not matches:
            raise FileNotFoundError(f"Could not resolve run dir for label={label!r} seed={seed} under {sweep_root}")
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous run dirs for label={label!r} seed={seed}: {[str(p) for p in matches]}")
        resolved[label] = matches[0]
    return resolved


def _ensure_dir(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Run root already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _series_from_unique_count(n: int, train_frac: float = 0.6, val_frac: float = 0.2) -> Tuple[int, int, int]:
    if n < 3:
        raise ValueError("Episode split requires at least 3 episodes")
    train_n = max(1, int(round(train_frac * n)))
    val_n = max(1, int(round(val_frac * n)))
    if train_n + val_n >= n:
        overflow = train_n + val_n - (n - 1)
        while overflow > 0 and train_n > 1:
            train_n -= 1
            overflow -= 1
        while overflow > 0 and val_n > 1:
            val_n -= 1
            overflow -= 1
    test_n = n - train_n - val_n
    if test_n < 1:
        if train_n > val_n and train_n > 1:
            train_n -= 1
        elif val_n > 1:
            val_n -= 1
        else:
            train_n = max(1, train_n - 1)
        test_n = n - train_n - val_n
    if min(train_n, val_n, test_n) < 1:
        raise ValueError(f"Could not split {n} episodes into non-empty train/val/test sets")
    return train_n, val_n, test_n


def split_episode_ids(episode_ids: np.ndarray) -> Dict[str, Any]:
    unique = np.sort(np.unique(np.asarray(episode_ids, dtype=np.int64)))
    train_n, val_n, test_n = _series_from_unique_count(unique.size)
    train_eps = unique[:train_n]
    val_eps = unique[train_n : train_n + val_n]
    test_eps = unique[train_n + val_n : train_n + val_n + test_n]
    return {
        "unit": "episode",
        "strategy": "ordered_60_20_20",
        "train_episodes": train_eps.tolist(),
        "val_episodes": val_eps.tolist(),
        "test_episodes": test_eps.tolist(),
    }


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fv):
        return None
    return fv


def _series_stats(values: Sequence[float]) -> Dict[str, Any]:
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if vals.size == 0:
        return {"mean": None, "std": None, "median": None}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "median": float(np.median(vals)),
    }


def bootstrap_ci(values: Sequence[float], *, samples: int, seed: int, alpha: float = 0.05) -> Dict[str, Any]:
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if vals.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "samples": int(samples)}
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        sample = rng.choice(vals, size=vals.size, replace=True)
        means[i] = float(np.mean(sample))
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(means, 100.0 * (alpha / 2.0))),
        "ci_high": float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))),
        "samples": int(samples),
    }


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    a = np.asarray([float(v) for v in left], dtype=np.float64)
    b = np.asarray([float(v) for v in right], dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired bootstrap requires equal-length series")
    if a.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "samples": int(samples)}
    rng = np.random.default_rng(int(seed))
    diffs = a - b
    draws = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        sample = rng.choice(diffs, size=diffs.size, replace=True)
        draws[i] = float(np.mean(sample))
    return {
        "mean": float(np.mean(diffs)),
        "ci_low": float(np.percentile(draws, 100.0 * (alpha / 2.0))),
        "ci_high": float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0))),
        "samples": int(samples),
    }


def _last_step_vector(x: torch.Tensor) -> np.ndarray:
    arr = x.reshape(-1, x.shape[-1])[-1].detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float32)


def _target_dim_labels(agent_idx: int, target_name: str) -> List[str]:
    obs_dim = len(AGENT_CFGS[agent_idx].obs_indices)
    act_dim = len(AGENT_CFGS[agent_idx].mv_indices)
    if target_name == "obs":
        return [f"obs_{i}" for i in range(obs_dim)]
    if target_name == "act":
        return [f"act_{i}" for i in range(act_dim)]
    if target_name == "obs_action":
        return [f"obs_{i}" for i in range(obs_dim)] + [f"act_{i}" for i in range(act_dim)]
    raise ValueError(f"Unknown target name {target_name!r}")


def _make_target_matrix(data: Mapping[str, np.ndarray], target_name: str) -> np.ndarray:
    if target_name == "obs":
        return np.asarray(data["target_obs_normalized"], dtype=np.float32)
    if target_name == "act":
        return np.asarray(data["target_action"], dtype=np.float32)
    if target_name == "obs_action":
        return np.asarray(data["target_obs_action"], dtype=np.float32)
    raise ValueError(f"Unknown target name {target_name!r}")


def _target_mean_std(train_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_y.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_y.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _compute_metrics(
    pred_norm: np.ndarray,
    y_norm: np.ndarray,
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Dict[str, Any]:
    pred_norm = np.asarray(pred_norm, dtype=np.float32)
    y_norm = np.asarray(y_norm, dtype=np.float32)
    target_mean = np.asarray(target_mean, dtype=np.float32)
    target_std = np.asarray(target_std, dtype=np.float32)
    pred_raw = pred_norm * target_std + target_mean
    y_raw = y_norm * target_std + target_mean
    err = pred_norm - y_norm
    mse_norm = np.mean(err**2, axis=0)
    mse_norm_scalar = float(np.mean(err**2))
    baseline_mse_norm = float(np.mean(y_norm**2))
    relative_mse = float(mse_norm_scalar / baseline_mse_norm) if baseline_mse_norm > 0 else None
    sse = np.sum((pred_norm - y_norm) ** 2, axis=0)
    centered = y_norm - y_norm.mean(axis=0, keepdims=True)
    sst = np.sum(centered**2, axis=0)
    per_dim_r2 = np.where(sst > 1e-12, 1.0 - sse / sst, 0.0)
    macro_r2 = float(np.mean(per_dim_r2))
    weighted_r2 = float(1.0 - float(np.sum(sse)) / max(float(np.sum(sst)), 1e-12))
    per_dim_rmse_norm = np.sqrt(mse_norm)
    per_dim_mse_raw = np.mean((pred_raw - y_raw) ** 2, axis=0)
    per_dim_rmse_raw = np.sqrt(per_dim_mse_raw)
    return {
        "mse_norm": mse_norm_scalar,
        "baseline_mse_norm": baseline_mse_norm,
        "relative_mse": relative_mse,
        "r2_macro": macro_r2,
        "r2_weighted": weighted_r2,
        "per_dim_mse_norm": mse_norm.astype(float).tolist(),
        "per_dim_rmse_norm": per_dim_rmse_norm.astype(float).tolist(),
        "per_dim_r2": per_dim_r2.astype(float).tolist(),
        "per_dim_mse_raw": per_dim_mse_raw.astype(float).tolist(),
        "per_dim_rmse_raw": per_dim_rmse_raw.astype(float).tolist(),
        "raw_rmse_mean": float(np.mean(per_dim_rmse_raw)),
        "raw_rmse_median": float(np.median(per_dim_rmse_raw)),
    }


def _distance_correlation_np(x: np.ndarray, y: np.ndarray, max_samples: int = 8192) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return None
    n = min(int(max_samples), int(x.shape[0]), int(y.shape[0]))
    x = x[:n]
    y = y[:n]
    a = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    b = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    a = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    b = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = np.mean(a * b)
    dvar_x = np.mean(a * a)
    dvar_y = np.mean(b * b)
    denom = np.sqrt(max(dvar_x, 0.0) * max(dvar_y, 0.0)) + 1e-12
    value = np.sqrt(max(dcov2, 0.0) / denom + 1e-12)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _build_probe(input_dim: int, target_dim: int, arch: str) -> nn.Module:
    if arch == "linear":
        return nn.Linear(input_dim, target_dim)
    if arch == "small_mlp":
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, target_dim),
        )
    if arch == "strong_mlp":
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, target_dim),
        )
    raise ValueError(f"Unknown probe arch {arch!r}")


def _train_probe_seed(
    *,
    x_train_np: np.ndarray,
    y_train_norm_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_norm_np: np.ndarray,
    x_test_np: np.ndarray,
    y_test_norm_np: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    arch: str,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: str,
    shuffle_train: bool,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = _build_probe(x_train_np.shape[-1], y_train_norm_np.shape[-1], arch=arch).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-5)

    x_train = torch.as_tensor(x_train_np, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train_norm_np, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_val_norm_np, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(x_test_np, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(y_test_norm_np, dtype=torch.float32, device=device)

    if shuffle_train:
        perm = rng.permutation(y_train.shape[0])
        y_train = y_train.index_select(0, torch.as_tensor(perm, dtype=torch.long, device=device))

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    train_loss_initial: Optional[float] = None

    for epoch in range(int(max_epochs)):
        perm = rng.permutation(x_train.shape[0])
        epoch_losses: List[float] = []
        for start in range(0, x_train.shape[0], int(batch_size)):
            idx = torch.as_tensor(perm[start : start + int(batch_size)], dtype=torch.long, device=device)
            pred = model(x_train.index_select(0, idx))
            loss = ((pred - y_train.index_select(0, idx)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        if train_loss_initial is None and epoch_losses:
            train_loss_initial = float(np.mean(epoch_losses))
        with torch.no_grad():
            pred_train = model(x_train).detach().cpu().numpy()
            pred_val = model(x_val).detach().cpu().numpy()
            pred_test = model(x_test).detach().cpu().numpy()
        train_metrics = _compute_metrics(pred_train, y_train_norm_np, target_mean=target_mean, target_std=target_std)
        val_metrics = _compute_metrics(pred_val, y_val_norm_np, target_mean=target_mean, target_std=target_std)
        test_metrics = _compute_metrics(pred_test, y_test_norm_np, target_mean=target_mean, target_std=target_std)
        history.append(
            {
                "epoch": int(epoch),
                "train_mse_norm": train_metrics["mse_norm"],
                "val_mse_norm": val_metrics["mse_norm"],
                "test_mse_norm": test_metrics["mse_norm"],
                "train_r2_macro": train_metrics["r2_macro"],
                "val_r2_macro": val_metrics["r2_macro"],
                "test_r2_macro": test_metrics["r2_macro"],
            }
        )
        if val_metrics["mse_norm"] < best_val - 1e-9:
            best_val = float(val_metrics["mse_norm"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        pred_train = model(x_train).detach().cpu().numpy()
        pred_val = model(x_val).detach().cpu().numpy()
        pred_test = model(x_test).detach().cpu().numpy()
    train_metrics = _compute_metrics(pred_train, y_train_norm_np, target_mean=target_mean, target_std=target_std)
    val_metrics = _compute_metrics(pred_val, y_val_norm_np, target_mean=target_mean, target_std=target_std)
    test_metrics = _compute_metrics(pred_test, y_test_norm_np, target_mean=target_mean, target_std=target_std)
    return {
        "seed": int(seed),
        "arch": arch,
        "shuffle_train": bool(shuffle_train),
        "best_val_loss": float(best_val),
        "epochs_ran": int(len(history)),
        "train_loss_initial": train_loss_initial,
        "history": history,
        "final": {
            "train_mse_norm": train_metrics["mse_norm"],
            "train_r2_macro": train_metrics["r2_macro"],
            "train_r2_weighted": train_metrics["r2_weighted"],
            "val_mse_norm": val_metrics["mse_norm"],
            "val_r2_macro": val_metrics["r2_macro"],
            "val_r2_weighted": val_metrics["r2_weighted"],
            "test_mse_norm": test_metrics["mse_norm"],
            "test_r2_macro": test_metrics["r2_macro"],
            "test_r2_weighted": test_metrics["r2_weighted"],
            "test_relative_mse": test_metrics["relative_mse"],
            "test_raw_rmse_mean": test_metrics["raw_rmse_mean"],
            "test_raw_rmse_median": test_metrics["raw_rmse_median"],
            "test_per_dim_r2": test_metrics["per_dim_r2"],
            "test_per_dim_mse_norm": test_metrics["per_dim_mse_norm"],
            "test_per_dim_rmse_norm": test_metrics["per_dim_rmse_norm"],
            "test_per_dim_mse_raw": test_metrics["per_dim_mse_raw"],
            "test_per_dim_rmse_raw": test_metrics["per_dim_rmse_raw"],
            "test_baseline_mse_norm": test_metrics["baseline_mse_norm"],
            "epochs_ran": int(len(history)),
        },
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


def _train_probe_constant(
    *,
    y_train_norm_np: np.ndarray,
    y_val_norm_np: np.ndarray,
    y_test_norm_np: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Dict[str, Any]:
    zeros_train = np.zeros_like(y_train_norm_np, dtype=np.float32)
    zeros_val = np.zeros_like(y_val_norm_np, dtype=np.float32)
    zeros_test = np.zeros_like(y_test_norm_np, dtype=np.float32)
    train_metrics = _compute_metrics(zeros_train, y_train_norm_np, target_mean=target_mean, target_std=target_std)
    val_metrics = _compute_metrics(zeros_val, y_val_norm_np, target_mean=target_mean, target_std=target_std)
    test_metrics = _compute_metrics(zeros_test, y_test_norm_np, target_mean=target_mean, target_std=target_std)
    return {
        "seed": -1,
        "arch": "constant",
        "shuffle_train": False,
        "best_val_loss": float(val_metrics["mse_norm"]),
        "epochs_ran": 0,
        "train_loss_initial": float(train_metrics["mse_norm"]),
        "history": [],
        "final": {
            "train_mse_norm": train_metrics["mse_norm"],
            "train_r2_macro": train_metrics["r2_macro"],
            "train_r2_weighted": train_metrics["r2_weighted"],
            "val_mse_norm": val_metrics["mse_norm"],
            "val_r2_macro": val_metrics["r2_macro"],
            "val_r2_weighted": val_metrics["r2_weighted"],
            "test_mse_norm": test_metrics["mse_norm"],
            "test_r2_macro": test_metrics["r2_macro"],
            "test_r2_weighted": test_metrics["r2_weighted"],
            "test_relative_mse": test_metrics["relative_mse"],
            "test_raw_rmse_mean": test_metrics["raw_rmse_mean"],
            "test_raw_rmse_median": test_metrics["raw_rmse_median"],
            "test_per_dim_r2": test_metrics["per_dim_r2"],
            "test_per_dim_mse_norm": test_metrics["per_dim_mse_norm"],
            "test_per_dim_rmse_norm": test_metrics["per_dim_rmse_norm"],
            "test_per_dim_mse_raw": test_metrics["per_dim_mse_raw"],
            "test_per_dim_rmse_raw": test_metrics["per_dim_rmse_raw"],
            "test_baseline_mse_norm": test_metrics["baseline_mse_norm"],
            "epochs_ran": 0,
        },
        "state_dict": {
            "target_mean": torch.as_tensor(target_mean, dtype=torch.float32),
            "target_std": torch.as_tensor(target_std, dtype=torch.float32),
        },
    }


def _aggregate_seed_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for run in runs
            for key, value in (run.get("final") or {}).items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    vector_keys = sorted(
        {
            key
            for run in runs
            for key, value in (run.get("final") or {}).items()
            if isinstance(value, list)
        }
    )
    out: Dict[str, Any] = {
        "runs": runs,
        "final_mean": {},
        "final_std": {},
    }
    for key in numeric_keys:
        vals = [float((run.get("final") or {}).get(key)) for run in runs if (run.get("final") or {}).get(key) is not None]
        if vals:
            out["final_mean"][key] = float(np.mean(vals))
            out["final_std"][key] = float(np.std(vals, ddof=0))
        else:
            out["final_mean"][key] = None
            out["final_std"][key] = None
    for key in vector_keys:
        arrays = [np.asarray((run.get("final") or {}).get(key), dtype=np.float64) for run in runs if (run.get("final") or {}).get(key) is not None]
        arrays = [arr for arr in arrays if arr.ndim == 1]
        if arrays:
            width = arrays[0].shape[0]
            arrays = [arr for arr in arrays if arr.shape[0] == width]
            stacked = np.stack(arrays, axis=0)
            out["final_mean"][key] = stacked.mean(axis=0).astype(float).tolist()
            out["final_std"][key] = stacked.std(axis=0, ddof=0).astype(float).tolist()
        else:
            out["final_mean"][key] = None
            out["final_std"][key] = None
    return out


def train_probe_suite(
    data: Mapping[str, np.ndarray],
    *,
    target_name: str,
    split: Mapping[str, Any],
    probe_seeds: Sequence[int],
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: str,
    shuffle_control: bool = True,
) -> Dict[str, Any]:
    episode_ids = np.asarray(data.get("episode_id", data.get("episode_index")), dtype=np.int64)
    train_mask = np.isin(episode_ids, np.asarray(split["train_episodes"], dtype=np.int64))
    val_mask = np.isin(episode_ids, np.asarray(split["val_episodes"], dtype=np.int64))
    test_mask = np.isin(episode_ids, np.asarray(split["test_episodes"], dtype=np.int64))
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Probe split produced an empty partition")

    x = np.asarray(data["z"], dtype=np.float32)
    y = _make_target_matrix(data, target_name)
    target_mean, target_std = _target_mean_std(y[train_mask])
    y_train_norm = (y[train_mask] - target_mean) / target_std
    y_val_norm = (y[val_mask] - target_mean) / target_std
    y_test_norm = (y[test_mask] - target_mean) / target_std
    baseline_metrics = _train_probe_constant(
        y_train_norm_np=y_train_norm,
        y_val_norm_np=y_val_norm,
        y_test_norm_np=y_test_norm,
        target_mean=target_mean,
        target_std=target_std,
    )
    target_input = np.concatenate([np.asarray(data["target_obs_normalized"], dtype=np.float32), np.asarray(data["target_action"], dtype=np.float32)], axis=-1)
    dcor = _distance_correlation_np(x[test_mask], _make_target_matrix(data, target_name)[test_mask])

    probe_payload: Dict[str, Any] = {
        "target": target_name,
        "target_dim": int(y.shape[-1]),
        "obs_dim": int(np.asarray(data["target_obs_normalized"], dtype=np.float32).shape[-1]),
        "act_dim": int(np.asarray(data["target_action"], dtype=np.float32).shape[-1]),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "target_mean": target_mean.reshape(-1).astype(float).tolist(),
        "target_std": target_std.reshape(-1).astype(float).tolist(),
        "dcor_test": dcor,
        "baseline": baseline_metrics,
        "targets": {
            "obs": np.asarray(data["target_obs_normalized"], dtype=np.float32),
            "act": np.asarray(data["target_action"], dtype=np.float32),
            "obs_action": target_input.astype(np.float32),
        },
    }

    probe_specs = [
        ("constant", None, False),
        ("linear", COMMON_INPUT_PROBE_ARCH["linear"], False),
        ("small_mlp", COMMON_INPUT_PROBE_ARCH["small_mlp"], False),
        ("strong_mlp", COMMON_INPUT_PROBE_ARCH["strong_mlp"], False),
        ("shuffled", COMMON_INPUT_PROBE_ARCH["shuffled"], True),
    ]
    probe_results: Dict[str, Any] = {}
    for probe_name, arch, shuffle_train in probe_specs:
        if probe_name == "constant":
            runs = []
            for seed in probe_seeds:
                run = dict(baseline_metrics)
                run.update({"seed": int(seed), "arch": "constant", "shuffle_train": False})
                runs.append(run)
        else:
            runs = [
                _train_probe_seed(
                    x_train_np=x[train_mask],
                    y_train_norm_np=y_train_norm,
                    x_val_np=x[val_mask],
                    y_val_norm_np=y_val_norm,
                    x_test_np=x[test_mask],
                    y_test_norm_np=y_test_norm,
                    target_mean=target_mean,
                    target_std=target_std,
                    arch=str(arch),
                    seed=int(seed),
                    max_epochs=int(max_epochs),
                    patience=int(patience),
                    batch_size=int(batch_size),
                    lr=float(lr),
                    device=device,
                    shuffle_train=bool(shuffle_train and shuffle_control),
                )
                for seed in probe_seeds
            ]
        probe_results[probe_name] = _aggregate_seed_runs(runs)
        probe_results[probe_name]["final_mean"].setdefault("test_baseline_mse_norm", baseline_metrics["final"]["test_baseline_mse_norm"])
        probe_results[probe_name]["final_std"].setdefault("test_baseline_mse_norm", 0.0)
        probe_results[probe_name]["baseline"] = baseline_metrics
        probe_results[probe_name]["dcor_test"] = dcor

    probe_payload["probes"] = probe_results
    return probe_payload


def _episode_group_by_seed(rows: Sequence[Mapping[str, Any]], eval_seed_key: str = "eval_seed") -> Dict[int, List[Mapping[str, Any]]]:
    groups: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row[eval_seed_key])].append(row)
    return groups


def _collect_episode_step_arrays(
    *,
    trainer: Any,
    exp_cfg: full.ExperimentConfig,
    run_dir: Path,
    eval_seeds: Sequence[int],
    deterministic: bool,
    max_steps: int,
    label: str,
    training_seed: int,
    bi_feature_mode: str = "decoded",
    checkpoint_hash: str = "",
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if bi_feature_mode not in {"decoded", "transmitted"}:
        raise ValueError(f"Unsupported bi_feature_mode={bi_feature_mode!r}; expected decoded/transmitted")
    obs_rms_list = load_obs_rms(run_dir)
    joint_mv_indices = build_joint_mv_indices(AGENT_CFGS)
    checkpoint_hash = str(checkpoint_hash)
    per_agent: List[Dict[str, List[Any]]] = [
        {
            "z": [],
            "obs_raw": [],
            "obs_normalized": [],
            "action_actor_mean": [],
            "action_bi_input": [],
            "action_env": [],
            "reward": [],
            "raw_profit": [],
            "shutdown": [],
            "terminated": [],
            "truncated": [],
            "eval_seed": [],
            "episode_id": [],
            "episode_index": [],
            "timestep": [],
        }
        for _ in AGENT_CFGS
    ]
    episode_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for episode_index, eval_seed in enumerate(eval_seeds):
            env, full_obs, _ = make_env(
                mv_indices=joint_mv_indices,
                seed=int(eval_seed),
                rl_dt_seconds=exp_cfg.rl_dt_seconds,
                max_steps=max_steps,
                delta_mv_max=exp_cfg.delta_mv_max,
                production_index=exp_cfg.production_index,
                baseline=exp_cfg.baseline,
                scale=exp_cfg.scale,
                reset_reward=exp_cfg.reset_reward,
            )
            prev_actions = [np.zeros(len(cfg.mv_indices), dtype=np.float32) for cfg in AGENT_CFGS]
            hidden_list = trainer.init_hidden(batch_size=1)
            total_reward = 0.0
            raw_profit_sum = 0.0
            raw_profit_steps = 0
            shutdown = False
            terminated = False
            truncated = False
            t = 0
            while True:
                local_obs_list = make_local_obs(np.asarray(full_obs, dtype=np.float32), obs_rms_list, update_rms=False)
                obs_t_list = trainer._tensor_list(local_obs_list)
                prev_act_t_list = trainer._tensor_list(prev_actions)
                action_t_list: List[torch.Tensor] = []
                actor_mean_list: List[np.ndarray] = []
                bi_list: List[torch.Tensor] = []
                z_list: List[torch.Tensor] = []
                new_hidden_list: List[torch.Tensor] = []
                for i, client in enumerate(trainer.clients):
                    mu, _ = client.actor(obs_t_list[i])
                    action_i, _ = client.act(obs_t_list[i], deterministic=deterministic)
                    if bi_feature_mode == "transmitted":
                        bi_i, h_i, z_i = client.encode_bi(
                            obs_t_list[i],
                            prev_act_t_list[i],
                            hidden_list[i],
                            return_transmitted=True,
                        )
                        z_list.append(z_i)
                    else:
                        bi_i, h_i = client.encode_bi(obs_t_list[i], prev_act_t_list[i], hidden_list[i])
                    action_t_list.append(action_i)
                    actor_mean_list.append(np.asarray(mu.detach().cpu().numpy(), dtype=np.float32))
                    bi_list.append(bi_i)
                    new_hidden_list.append(h_i.detach())

                actions_list = [a.detach().cpu().numpy().astype(np.float32) for a in action_t_list]
                next_full_obs, reward, terminated, truncated, info = env.step(np.concatenate(actions_list, axis=0))
                reward_scalar = float(np.asarray(reward, dtype=np.float32).reshape(-1)[0])
                raw_prod = _as_float(info.get("raw_prod"))
                raw_cost = _as_float(info.get("raw_cost"))
                if raw_prod is not None and raw_cost is not None:
                    raw_profit = float(raw_prod - raw_cost)
                else:
                    raw_profit = float(reward_scalar * float(exp_cfg.scale) + float(exp_cfg.baseline))
                shutdown = bool(info.get("shutdown", False))
                total_reward += reward_scalar
                raw_profit_sum += raw_profit
                raw_profit_steps += 1

                for i in range(len(AGENT_CFGS)):
                    feature_seq = z_list[i] if bi_feature_mode == "transmitted" else bi_list[i]
                    z_vec = _last_step_vector(feature_seq)
                    obs_raw_i = np.asarray(full_obs, dtype=np.float32)[np.asarray(AGENT_CFGS[i].obs_indices, dtype=np.int64)]
                    obs_norm_i = np.asarray(local_obs_list[i], dtype=np.float32)
                    prev_act_i = np.asarray(prev_actions[i], dtype=np.float32)
                    action_env_i = np.asarray(actions_list[i], dtype=np.float32)
                    step_rows.append(
                        {
                            "label": label,
                            "bi_feature_mode": bi_feature_mode,
                            "checkpoint_hash": checkpoint_hash,
                            "eval_seed": int(eval_seed),
                            "episode_index": int(episode_index),
                            "timestep": int(t),
                            "agent_id": int(i),
                            "obs_raw": obs_raw_i,
                            "obs_normalized": obs_norm_i,
                            "action_actor_mean": actor_mean_list[i],
                            "action_bi_input": prev_act_i,
                            "action_env": action_env_i,
                            "bi_z": z_vec,
                            "reward": reward_scalar,
                            "raw_profit": raw_profit,
                            "shutdown": float(shutdown),
                            "terminated": float(bool(terminated)),
                            "truncated": float(bool(truncated)),
                        }
                    )
                    agent_rows = per_agent[i]
                    agent_rows["z"].append(z_vec)
                    agent_rows["obs_raw"].append(obs_raw_i)
                    agent_rows["obs_normalized"].append(obs_norm_i)
                    agent_rows["action_actor_mean"].append(actor_mean_list[i])
                    agent_rows["action_bi_input"].append(prev_act_i)
                    agent_rows["action_env"].append(action_env_i)
                    agent_rows["reward"].append(np.float32(reward_scalar))
                    agent_rows["raw_profit"].append(np.float32(raw_profit))
                    agent_rows["shutdown"].append(np.float32(shutdown))
                    agent_rows["terminated"].append(np.float32(bool(terminated)))
                    agent_rows["truncated"].append(np.float32(bool(truncated)))
                    agent_rows["eval_seed"].append(np.int64(eval_seed))
                    agent_rows["episode_id"].append(np.int64(episode_index))
                    agent_rows["episode_index"].append(np.int64(episode_index))
                    agent_rows["timestep"].append(np.int64(t))

                full_obs = np.asarray(next_full_obs, dtype=np.float32)
                prev_actions = actions_list
                hidden_list = new_hidden_list
                t += 1
                if terminated or truncated or t >= max_steps:
                    break

            episode_rows.append(
                    {
                        "label": label,
                        "bi_feature_mode": bi_feature_mode,
                        "checkpoint_hash": checkpoint_hash,
                        "training_seed": int(training_seed),
                        "eval_seed": int(eval_seed),
                        "episode_index": int(episode_index),
                        "steps": int(t),
                    "raw_profit_sum": float(raw_profit_sum),
                    "raw_profit_mean": float(raw_profit_sum / max(raw_profit_steps, 1)),
                    "episode_return": float(total_reward),
                    "shutdown": bool(shutdown),
                    "termination_reason": "shutdown" if shutdown else ("terminated" if terminated else ("truncated" if truncated else "completed")),
                }
            )

    agent_arrays: List[Dict[str, np.ndarray]] = []
    for rows in per_agent:
        arrays = {
            key: np.asarray(values, dtype=(np.int64 if key in {"eval_seed", "episode_id", "episode_index", "timestep"} else np.float32))
            for key, values in rows.items()
        }
        agent_arrays.append(arrays)
    return agent_arrays, episode_rows, step_rows


def collect_policy_run(
    *,
    frozen_run: FrozenRun,
    eval_seeds: Sequence[int],
    deterministic: bool,
    max_steps: int,
    label: str,
    bi_feature_mode: str = "decoded",
) -> Dict[str, Any]:
    agent_arrays, episode_rows, step_rows = _collect_episode_step_arrays(
        trainer=frozen_run.trainer,
        exp_cfg=frozen_run.exp_cfg,
        run_dir=frozen_run.run_dir,
        eval_seeds=eval_seeds,
        deterministic=deterministic,
        max_steps=max_steps,
        label=label,
        training_seed=int(frozen_run.config.get("seed", 0)),
        bi_feature_mode=bi_feature_mode,
        checkpoint_hash=frozen_run.checkpoint_hash,
    )
    return {
        "agent_arrays": agent_arrays,
        "episode_rows": episode_rows,
        "step_rows": step_rows,
        "bi_feature_mode": bi_feature_mode,
        "checkpoint_hash": frozen_run.checkpoint_hash,
    }


def _save_agent_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def _save_reference_dataset(run_root: Path, reference_rows: Sequence[Mapping[str, Any]], agent_arrays: Sequence[Mapping[str, np.ndarray]]) -> None:
    ref_dir = run_root / "datasets" / "reference_zero"
    ref_dir.mkdir(parents=True, exist_ok=True)
    write_json(ref_dir / "episodes.json", list(reference_rows))
    for i, arrays in enumerate(agent_arrays):
        _save_agent_npz(ref_dir / f"agent_{i}.npz", arrays)


def _save_common_or_onpolicy_dataset(
    run_root: Path,
    *,
    group_name: str,
    label: str,
    agent_arrays: Sequence[Mapping[str, np.ndarray]],
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> None:
    label_dir = run_root / "datasets" / group_name / label
    label_dir.mkdir(parents=True, exist_ok=True)
    for i, arrays in enumerate(agent_arrays):
        _save_agent_npz(label_dir / f"agent_{i}.npz", arrays)
    if extra_meta is not None:
        write_json(label_dir / "meta.json", dict(extra_meta))


def _agent_target_hash(arrays: Mapping[str, np.ndarray]) -> str:
    data = np.ascontiguousarray(np.asarray(arrays["target_obs_action"], dtype=np.float32))
    h = hashlib.sha256()
    h.update(data.tobytes())
    h.update(str(data.shape).encode("utf-8"))
    h.update(str(data.dtype).encode("utf-8"))
    return h.hexdigest()


def _agent_input_hash(arrays: Mapping[str, np.ndarray]) -> str:
    data = np.ascontiguousarray(np.asarray(arrays["z"], dtype=np.float32))
    h = hashlib.sha256()
    h.update(data.tobytes())
    h.update(str(data.shape).encode("utf-8"))
    h.update(str(data.dtype).encode("utf-8"))
    return h.hexdigest()


def _prepare_target_arrays_from_policy_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, np.ndarray]]:
    grouped = _group_rows_by_agent(rows)
    agent_arrays: List[Dict[str, np.ndarray]] = []
    for agent_id in sorted(grouped):
        agent_rows = grouped[agent_id]
        obs_norm = np.stack([np.asarray(row["obs_normalized"], dtype=np.float32) for row in agent_rows], axis=0)
        action_bi = np.stack([np.asarray(row["action_bi_input"], dtype=np.float32) for row in agent_rows], axis=0)
        z = np.stack([np.asarray(row["bi_z"], dtype=np.float32) for row in agent_rows], axis=0)
        eval_seed = np.asarray([int(row["eval_seed"]) for row in agent_rows], dtype=np.int64)
        episode_index = np.asarray([int(row["episode_index"]) for row in agent_rows], dtype=np.int64)
        timestep = np.asarray([int(row["timestep"]) for row in agent_rows], dtype=np.int64)
        target_obs_action = np.concatenate([obs_norm, action_bi], axis=-1)
        agent_arrays.append(
            {
                "z": z,
                "target_obs_normalized": obs_norm,
                "target_action": action_bi,
                "target_obs_action": target_obs_action,
                "eval_seed": eval_seed,
                "episode_id": episode_index,
                "episode_index": episode_index,
                "timestep": timestep,
            }
        )
    return agent_arrays


def _prepare_reference_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "label": row["label"],
                "eval_seed": int(row["eval_seed"]),
                "episode_index": int(row["episode_index"]),
                "steps": int(row["steps"]),
                "raw_profit_sum": float(row["raw_profit_sum"]),
                "raw_profit_mean": float(row["raw_profit_mean"]),
                "episode_return": float(row["episode_return"]),
                "shutdown": bool(row["shutdown"]),
                "termination_reason": str(row["termination_reason"]),
            }
        )
    return out


def _group_rows_by_agent(rows: Sequence[Mapping[str, Any]]) -> Dict[int, List[Mapping[str, Any]]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["agent_id"])].append(row)
    return grouped


def _replay_reference_through_encoder(
    *,
    frozen_run: FrozenRun,
    reference_rows: Sequence[Mapping[str, Any]],
    bi_feature_mode: str = "decoded",
) -> List[Dict[str, np.ndarray]]:
    if bi_feature_mode not in {"decoded", "transmitted"}:
        raise ValueError(f"Unsupported bi_feature_mode={bi_feature_mode!r}; expected decoded/transmitted")
    grouped = _group_rows_by_agent(reference_rows)
    output_agent_arrays: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for agent_id, client in enumerate(frozen_run.trainer.clients):
            agent_rows = grouped[agent_id]
            z_list: List[np.ndarray] = []
            obs_norm_list: List[np.ndarray] = []
            action_bi_list: List[np.ndarray] = []
            eval_seed_list: List[int] = []
            episode_index_list: List[int] = []
            timestep_list: List[int] = []
            current_episode = None
            hidden = client.init_hidden(batch_size=1, device=frozen_run.device)
            for row in agent_rows:
                if current_episode is None or int(row["episode_index"]) != current_episode:
                    current_episode = int(row["episode_index"])
                    hidden = client.init_hidden(batch_size=1, device=frozen_run.device)
                obs_norm = torch.as_tensor(row["obs_normalized"], dtype=torch.float32, device=frozen_run.device)
                prev_act = torch.as_tensor(row["action_bi_input"], dtype=torch.float32, device=frozen_run.device)
                encoded = client.encode_bi(obs_norm, prev_act, hidden, return_transmitted=bi_feature_mode == "transmitted")
                if bi_feature_mode == "transmitted":
                    bi_seq, hidden, z_seq = encoded
                    z_list.append(_last_step_vector(z_seq))
                else:
                    bi_seq, hidden = encoded
                    z_list.append(_last_step_vector(bi_seq))
                obs_norm_list.append(np.asarray(row["obs_normalized"], dtype=np.float32))
                action_bi_list.append(np.asarray(row["action_bi_input"], dtype=np.float32))
                eval_seed_list.append(int(row["eval_seed"]))
                episode_index_list.append(int(row["episode_index"]))
                timestep_list.append(int(row["timestep"]))
            obs_norm_arr = np.stack(obs_norm_list, axis=0)
            action_arr = np.stack(action_bi_list, axis=0)
            z_arr = np.stack(z_list, axis=0)
            output_agent_arrays.append(
                {
                    "z": z_arr,
                    "target_obs_normalized": obs_norm_arr,
                    "target_action": action_arr,
                    "target_obs_action": np.concatenate([obs_norm_arr, action_arr], axis=-1),
                    "eval_seed": np.asarray(eval_seed_list, dtype=np.int64),
                    "episode_id": np.asarray(episode_index_list, dtype=np.int64),
                    "episode_index": np.asarray(episode_index_list, dtype=np.int64),
                    "timestep": np.asarray(timestep_list, dtype=np.int64),
                }
            )
    return output_agent_arrays


def _build_policy_dataset_from_rows(step_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, np.ndarray]]:
    return _prepare_target_arrays_from_policy_rows(step_rows)


def _evaluate_fixed_task_stats(episode_rows: Sequence[Mapping[str, Any]], bootstrap_samples: int, seed: int) -> Dict[str, Any]:
    raw_profit = [float(row["raw_profit_sum"]) for row in episode_rows]
    episode_return = [float(row["episode_return"]) for row in episode_rows]
    shutdown = [float(bool(row["shutdown"])) for row in episode_rows]
    length = [float(row["steps"]) for row in episode_rows]
    return {
        "raw_profit": {
            **bootstrap_ci(raw_profit, samples=bootstrap_samples, seed=seed),
            "std": _series_stats(raw_profit)["std"],
            "median": _series_stats(raw_profit)["median"],
        },
        "episode_return": {
            **bootstrap_ci(episode_return, samples=bootstrap_samples, seed=seed + 1),
            "std": _series_stats(episode_return)["std"],
            "median": _series_stats(episode_return)["median"],
        },
        "shutdown_rate": {
            **bootstrap_ci(shutdown, samples=bootstrap_samples, seed=seed + 2),
            "std": _series_stats(shutdown)["std"],
            "median": _series_stats(shutdown)["median"],
        },
        "episode_length": {
            **bootstrap_ci(length, samples=bootstrap_samples, seed=seed + 3),
            "std": _series_stats(length)["std"],
            "median": _series_stats(length)["median"],
        },
    }


def _paired_vs_zero(
    labels: Sequence[str],
    episode_rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference_label: str,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    reference_rows = episode_rows_by_label[reference_label]
    reference_by_seed = {int(row["eval_seed"]): row for row in reference_rows}
    out: Dict[str, Any] = {}
    for label in labels:
        if label == reference_label:
            continue
        rows = episode_rows_by_label[label]
        by_seed = {int(row["eval_seed"]): row for row in rows}
        common_seeds = sorted(set(reference_by_seed).intersection(by_seed))
        if not common_seeds:
            raise ValueError(f"No paired eval seeds shared between {reference_label} and {label}")
        left_profit = [float(by_seed[s]["raw_profit_sum"]) for s in common_seeds]
        right_profit = [float(reference_by_seed[s]["raw_profit_sum"]) for s in common_seeds]
        left_return = [float(by_seed[s]["episode_return"]) for s in common_seeds]
        right_return = [float(reference_by_seed[s]["episode_return"]) for s in common_seeds]
        left_shutdown = [float(bool(by_seed[s]["shutdown"])) for s in common_seeds]
        right_shutdown = [float(bool(reference_by_seed[s]["shutdown"])) for s in common_seeds]
        left_length = [float(by_seed[s]["steps"]) for s in common_seeds]
        right_length = [float(reference_by_seed[s]["steps"]) for s in common_seeds]
        out[label] = {
            "common_seeds": common_seeds,
            "raw_profit_sum": paired_bootstrap_ci(left_profit, right_profit, samples=bootstrap_samples, seed=seed),
            "episode_return": paired_bootstrap_ci(left_return, right_return, samples=bootstrap_samples, seed=seed + 1),
            "shutdown_rate": paired_bootstrap_ci(left_shutdown, right_shutdown, samples=bootstrap_samples, seed=seed + 2),
            "episode_length": paired_bootstrap_ci(left_length, right_length, samples=bootstrap_samples, seed=seed + 3),
        }
    return out


def _aggregate_probe_target_table(
    *,
    labels: Sequence[str],
    probe_results_by_label: Mapping[str, Mapping[str, Any]],
    probe_name: str = "strong_mlp",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label in labels:
        probe_results = probe_results_by_label[label]
        for agent_idx in range(len(AGENT_CFGS)):
            final_mean = probe_results["agent_%d" % agent_idx]["probes"][probe_name]["final_mean"] if "probes" in probe_results["agent_%d" % agent_idx] else None
        # This helper is intentionally not used directly; the caller constructs rows with richer context.
    return rows


def _probe_summary_rows(
    *,
    label: str,
    agent_id: int,
    target_name: str,
    probe_name: str,
    probe_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    probe = probe_payload["probes"][probe_name]
    final_mean = probe["final_mean"]
    final_std = probe["final_std"]
    row = {
        "label": label,
        "agent_id": int(agent_id),
        "target": target_name,
        "probe": probe_name,
        "mode": probe_payload.get("mode"),
        "checkpoint_hash": probe_payload.get("checkpoint_hash"),
        "bi_feature_mode": probe_payload.get("bi_feature_mode"),
        "dataset_cache_key": probe_payload.get("dataset_cache_key"),
        "input_hash": probe_payload.get("input_hash"),
        "target_hash": probe_payload.get("target_hash"),
        "split_hash": probe_payload.get("split_hash"),
        "dcor_test": probe_payload.get("dcor_test"),
        "train_samples": probe_payload["train_samples"],
        "val_samples": probe_payload["val_samples"],
        "test_samples": probe_payload["test_samples"],
        "target_dim": probe_payload["target_dim"],
        "obs_dim": probe_payload["obs_dim"],
        "act_dim": probe_payload["act_dim"],
        "baseline_test_mse_norm": probe["baseline"]["final"]["test_mse_norm"],
        "baseline_test_r2_macro": probe["baseline"]["final"]["test_r2_macro"],
        "baseline_test_relative_mse": probe["baseline"]["final"]["test_relative_mse"],
        "test_mse_norm_mean": final_mean.get("test_mse_norm"),
        "test_mse_norm_std": final_std.get("test_mse_norm"),
        "test_r2_macro_mean": final_mean.get("test_r2_macro"),
        "test_r2_macro_std": final_std.get("test_r2_macro"),
        "test_r2_weighted_mean": final_mean.get("test_r2_weighted"),
        "test_r2_weighted_std": final_std.get("test_r2_weighted"),
        "test_relative_mse_mean": final_mean.get("test_relative_mse"),
        "test_relative_mse_std": final_std.get("test_relative_mse"),
        "test_raw_rmse_mean_mean": final_mean.get("test_raw_rmse_mean"),
        "test_raw_rmse_mean_std": final_std.get("test_raw_rmse_mean"),
        "test_raw_rmse_median_mean": final_mean.get("test_raw_rmse_median"),
        "test_raw_rmse_median_std": final_std.get("test_raw_rmse_median"),
    }
    return row


def _dimension_rows(
    *,
    label: str,
    agent_id: int,
    target_name: str,
    probe_payload: Mapping[str, Any],
    probe_name: str = "strong_mlp",
) -> List[Dict[str, Any]]:
    probe = probe_payload["probes"][probe_name]
    final_mean = probe["final_mean"]
    final_std = probe["final_std"]
    dim_labels = _target_dim_labels(agent_id, target_name)
    rows: List[Dict[str, Any]] = []
    per_dim_r2 = final_mean.get("test_per_dim_r2") or []
    per_dim_mse_norm = final_mean.get("test_per_dim_mse_norm") or []
    per_dim_rmse_norm = final_mean.get("test_per_dim_rmse_norm") or []
    per_dim_mse_raw = final_mean.get("test_per_dim_mse_raw") or []
    per_dim_rmse_raw = final_mean.get("test_per_dim_rmse_raw") or []
    for idx, dim_label in enumerate(dim_labels):
        rows.append(
            {
                "label": label,
                "agent_id": int(agent_id),
                "target": target_name,
                "probe": probe_name,
                "dimension": dim_label,
                "dim_index": int(idx),
                "r2": per_dim_r2[idx] if idx < len(per_dim_r2) else None,
                "mse_norm": per_dim_mse_norm[idx] if idx < len(per_dim_mse_norm) else None,
                "rmse_norm": per_dim_rmse_norm[idx] if idx < len(per_dim_rmse_norm) else None,
                "mse_raw": per_dim_mse_raw[idx] if idx < len(per_dim_mse_raw) else None,
                "rmse_raw": per_dim_rmse_raw[idx] if idx < len(per_dim_rmse_raw) else None,
                "dcor_test": probe_payload.get("dcor_test"),
                "baseline_test_mse_norm": probe["baseline"]["final"]["test_mse_norm"],
            }
        )
    return rows


def _probe_rows_for_dataset(
    *,
    label: str,
    agent_arrays: Sequence[Mapping[str, np.ndarray]],
    split: Mapping[str, Any],
    probe_seeds: Sequence[int],
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: str,
    dataset_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    per_agent: Dict[int, Dict[str, Any]] = {}
    shared_meta = dict(dataset_meta or {})
    for agent_id, arrays in enumerate(agent_arrays):
        per_agent[agent_id] = {}
        for target_name in TARGET_NAMES:
            payload = train_probe_suite(
                arrays,
                target_name=target_name,
                split=split,
                probe_seeds=probe_seeds,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                device=device,
            )
            payload.update(shared_meta)
            payload["label"] = label
            payload["agent_key"] = f"agent_{agent_id}"
            payload["input_hash"] = shared_meta.get("input_hashes", {}).get(f"agent_{agent_id}")
            payload["target_hash"] = shared_meta.get("target_hashes", {}).get(f"agent_{agent_id}", {}).get(target_name)
            per_agent[agent_id][target_name] = payload
    return per_agent


def _save_probe_artifacts(
    *,
    run_root: Path,
    group_name: str,
    label: str,
    agent_id: int,
    target_name: str,
    probe_payload: Mapping[str, Any],
) -> None:
    probe_dir = run_root / "probes" / group_name / label / f"agent_{agent_id}" / target_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    for probe_name, probe in probe_payload["probes"].items():
        probe_name_dir = probe_dir / probe_name
        probe_name_dir.mkdir(parents=True, exist_ok=True)
        runs = probe["runs"]
        for run in runs:
            seed = int(run["seed"])
            run_cache_key = _build_cache_key(
                mode=str(probe_payload.get("mode", group_name)),
                label=str(label),
                checkpoint_hash=str(probe_payload.get("checkpoint_hash", "")),
                target=str(target_name),
                probe=str(probe_name),
                seed=seed,
                bi_feature_mode=str(probe_payload.get("bi_feature_mode", "decoded")),
                input_hash=str(probe_payload.get("input_hashes", {}).get(f"agent_{agent_id}", "")),
                target_hash=str(probe_payload.get("target_hashes", {}).get(f"agent_{agent_id}", {}).get(target_name, "")),
            )
            run["cache_key"] = run_cache_key
            run["cache_key_components"] = {
                "mode": probe_payload.get("mode", group_name),
                "branch": label,
                "checkpoint_hash": probe_payload.get("checkpoint_hash"),
                "target": target_name,
                "probe": probe_name,
                "seed": seed,
                "bi_feature_mode": probe_payload.get("bi_feature_mode"),
                "input_hash": probe_payload.get("input_hashes", {}).get(f"agent_{agent_id}"),
                "target_hash": probe_payload.get("target_hashes", {}).get(f"agent_{agent_id}", {}).get(target_name),
            }
            state_path = probe_name_dir / f"seed_{seed}.pt"
            if probe_name == "constant":
                torch.save(
                    {
                        "probe_type": "constant",
                        "seed": seed,
                        "target_mean": probe_payload["target_mean"],
                        "target_std": probe_payload["target_std"],
                    },
                    state_path,
                )
            else:
                torch.save(run["state_dict"], state_path)
        write_json(
            probe_name_dir / "metrics.json",
            {
                "probe": probe_name,
                "target": target_name,
                "label": label,
                "agent_id": int(agent_id),
                "mode": probe_payload.get("mode", group_name),
                "checkpoint_hash": probe_payload.get("checkpoint_hash"),
                "bi_feature_mode": probe_payload.get("bi_feature_mode"),
                "dataset_cache_key": probe_payload.get("dataset_cache_key"),
                "cache_key": runs[0]["cache_key"] if runs else None,
                "input_hash": probe_payload.get("input_hash"),
                "target_hash": probe_payload.get("target_hash"),
                "split_hash": probe_payload.get("split_hash"),
                "summary": {
                    "dcor_test": probe["dcor_test"],
                    "baseline": probe["baseline"],
                    "final_mean": probe["final_mean"],
                    "final_std": probe["final_std"],
                },
                "runs": runs,
                "cache_key_components": {
                    "mode": probe_payload.get("mode", group_name),
                    "branch": label,
                    "checkpoint_hash": probe_payload.get("checkpoint_hash"),
                    "target": target_name,
                    "probe": probe_name,
                    "seed": int(runs[0]["seed"]) if runs else 0,
                    "bi_feature_mode": probe_payload.get("bi_feature_mode"),
                },
            },
        )


def _write_summary_tables(
    *,
    run_root: Path,
    labels: Sequence[str],
    reference_label: str,
    task_eval_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    probe_by_group: Mapping[str, Mapping[str, Dict[int, Dict[str, Any]]]],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    summaries_dir = run_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    task_table_rows: List[Dict[str, Any]] = []
    label_summary: Dict[str, Any] = {}
    episode_rows_by_label = task_eval_by_label
    for label in labels:
        rows = episode_rows_by_label[label]
        stats = _evaluate_fixed_task_stats(rows, bootstrap_samples=bootstrap_samples, seed=seed)
        label_summary[label] = stats
        task_table_rows.append(
            {
                "label": label,
                "profit_mean": stats["raw_profit"]["mean"],
                "profit_ci_low": stats["raw_profit"]["ci_low"],
                "profit_ci_high": stats["raw_profit"]["ci_high"],
                "return_mean": stats["episode_return"]["mean"],
                "return_ci_low": stats["episode_return"]["ci_low"],
                "return_ci_high": stats["episode_return"]["ci_high"],
                "shutdown_rate": stats["shutdown_rate"]["mean"],
                "shutdown_ci_low": stats["shutdown_rate"]["ci_low"],
                "shutdown_ci_high": stats["shutdown_rate"]["ci_high"],
                "episode_length_mean": stats["episode_length"]["mean"],
                "episode_length_ci_low": stats["episode_length"]["ci_low"],
                "episode_length_ci_high": stats["episode_length"]["ci_high"],
            }
        )

    paired_vs_zero = _paired_vs_zero(
        labels,
        episode_rows_by_label,
        reference_label=reference_label,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    common_rows: List[Dict[str, Any]] = []
    on_policy_rows: List[Dict[str, Any]] = []
    dimension_rows: List[Dict[str, Any]] = []
    for group_name, group_payload in probe_by_group.items():
        for label in labels:
            if label not in group_payload:
                continue
            for agent_id in range(len(AGENT_CFGS)):
                for target_name in TARGET_NAMES:
                    probe_payload = group_payload[label][agent_id][target_name]
                    for probe_name in PROBE_TYPES:
                        row = _probe_summary_rows(
                            label=label,
                            agent_id=agent_id,
                            target_name=target_name,
                            probe_name=probe_name,
                            probe_payload=probe_payload,
                        )
                        if group_name == "common_input":
                            common_rows.append(row)
                            if probe_name == "strong_mlp":
                                dimension_rows.extend(
                                    _dimension_rows(
                                        label=label,
                                        agent_id=agent_id,
                                        target_name=target_name,
                                        probe_name=probe_name,
                                        probe_payload=probe_payload,
                                    )
                                )
                        else:
                            on_policy_rows.append(row)
    if not common_rows:
        raise ValueError("No common-input probe rows were generated")

    common_input_target_hashes: Dict[str, Dict[str, str]] = {}
    for label in labels:
        common_input_target_hashes[label] = {}
        label_payload = probe_by_group["common_input"][label]
        for agent_id in range(len(AGENT_CFGS)):
            common_input_target_hashes[label][f"agent_{agent_id}"] = _agent_target_hash(
                {
                    "target_obs_action": label_payload[agent_id]["obs_action"]["targets"]["obs_action"],
                }
            )
    reference_hashes = common_input_target_hashes[reference_label]
    for label in labels:
        if label == reference_label:
            continue
        for agent_key, reference_hash in reference_hashes.items():
            if common_input_target_hashes[label][agent_key] != reference_hash:
                raise AssertionError(f"Common-input target hash mismatch for label={label} {agent_key}")

    write_csv(summaries_dir / "task_table.csv", task_table_rows, task_table_rows[0].keys())
    write_csv(summaries_dir / "common_input_probe_table.csv", common_rows, common_rows[0].keys())
    on_policy_headers = on_policy_rows[0].keys() if on_policy_rows else common_rows[0].keys()
    write_csv(summaries_dir / "on_policy_probe_table.csv", on_policy_rows, on_policy_headers)
    dim_headers = dimension_rows[0].keys() if dimension_rows else ["label"]
    write_csv(summaries_dir / "per_dimension_probe_table.csv", dimension_rows, dim_headers)
    coefficient_comparison = {
        "task": label_summary,
        "paired_vs_zero": paired_vs_zero,
        "common_input_target_hashes": common_input_target_hashes,
    }
    write_json(summaries_dir / "coefficient_comparison.json", coefficient_comparison)
    write_json(summaries_dir / "label_summary.json", label_summary)
    write_json(summaries_dir / "paired_vs_zero.json", paired_vs_zero)
    return {
        "task_table": task_table_rows,
        "common_input_rows": common_rows,
        "on_policy_rows": on_policy_rows,
        "dimension_rows": dimension_rows,
        "label_summary": label_summary,
        "paired_vs_zero": paired_vs_zero,
        "common_input_target_hashes": common_input_target_hashes,
        "coefficient_comparison": coefficient_comparison,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No rows._"

    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (list, tuple)):
            return json.dumps(json_safe(value), ensure_ascii=False)
        return str(value)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    return "\n".join(lines)


def _build_report(
    *,
    run_root: Path,
    labels: Sequence[str],
    sweep_root: Path,
    reference_label: str,
    bi_feature_mode: str,
    eval_seeds: Sequence[int],
    resolved_dirs: Mapping[str, Path],
    frozen_runs: Mapping[str, FrozenRun],
    label_summary: Mapping[str, Any],
    task_table: Sequence[Mapping[str, Any]],
    common_input_rows: Sequence[Mapping[str, Any]],
    on_policy_rows: Sequence[Mapping[str, Any]],
    dimension_rows: Sequence[Mapping[str, Any]],
    paired_vs_zero: Mapping[str, Any],
    common_input_target_hashes: Mapping[str, Mapping[str, str]],
) -> str:
    lines: List[str] = []
    lines.append("# Stage 1 Fixed Evaluation and Post-hoc BI Probe Report")
    lines.append("")
    lines.append("## 1. Environment and checkpoints")
    lines.append("")
    lines.append(f"- Sweep root: `{sweep_root}`")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Labels: `{', '.join(labels)}`")
    lines.append(f"- Reference label: `{reference_label}`")
    lines.append(f"- BI feature mode: `{bi_feature_mode}`")
    lines.append(f"- Evaluation seeds: `{len(eval_seeds)}`")
    lines.append("")
    env_rows = []
    for label in labels:
        fr = frozen_runs[label]
        env_rows.append(
            {
                "label": label,
                "run_dir": fr.run_dir,
                "trainer_version": fr.config.get("trainer_version"),
                "privacy_mode": fr.config.get("privacy_mode"),
                "checkpoint_hash": fr.checkpoint_hash,
                "critic_epochs": fr.exp_cfg.critic_epochs,
                "seed": fr.config.get("seed"),
                "recent_profit_mean": fr.result_summary.get("recent_profit_mean"),
                "episode_return_mean": fr.result_summary.get("episode_return_mean"),
                "shutdown_rate": fr.result_summary.get("shutdown_rate"),
            }
        )
    lines.append(_markdown_table(["label", "run_dir", "trainer_version", "privacy_mode", "checkpoint_hash", "critic_epochs", "seed", "recent_profit_mean", "episode_return_mean", "shutdown_rate"], env_rows))
    lines.append("")
    lines.append("## 2. Fixed task evaluation")
    lines.append("")
    lines.append(_markdown_table(["label", "profit_mean", "profit_ci_low", "profit_ci_high", "return_mean", "return_ci_low", "return_ci_high", "shutdown_rate", "shutdown_ci_low", "shutdown_ci_high", "episode_length_mean"], task_table))
    lines.append("")
    lines.append("### Paired difference vs zero")
    lines.append("")
    paired_rows = []
    for label, payload in paired_vs_zero.items():
        paired_rows.append(
            {
                "label": label,
                "profit_delta": payload["raw_profit_sum"]["mean"],
                "profit_ci_low": payload["raw_profit_sum"]["ci_low"],
                "profit_ci_high": payload["raw_profit_sum"]["ci_high"],
                "return_delta": payload["episode_return"]["mean"],
                "shutdown_delta": payload["shutdown_rate"]["mean"],
                "length_delta": payload["episode_length"]["mean"],
            }
        )
    lines.append(_markdown_table(["label", "profit_delta", "profit_ci_low", "profit_ci_high", "return_delta", "shutdown_delta", "length_delta"], paired_rows))
    lines.append("")
    lines.append("## 3. Training summary vs fixed evaluation")
    lines.append("")
    compare_rows = []
    for label in labels:
        fr = frozen_runs[label]
        compare_rows.append(
            {
                "label": label,
                "recent_profit_mean": fr.result_summary.get("recent_profit_mean"),
                "fixed_profit_mean": task_table[[row["label"] for row in task_table].index(label)]["profit_mean"],
                "recent_episode_return_mean": fr.result_summary.get("episode_return_mean"),
                "fixed_episode_return_mean": task_table[[row["label"] for row in task_table].index(label)]["return_mean"],
            }
        )
    lines.append(
        _markdown_table(
            ["label", "recent_profit_mean", "fixed_profit_mean", "recent_episode_return_mean", "fixed_episode_return_mean"],
            compare_rows,
        )
    )
    lines.append("")
    lines.append("> `recent_profit_mean` is the training summary's last-100-step raw-profit average; it is not directly the same statistic as the fixed-eval per-episode profit mean.")
    lines.append("")
    lines.append("## 4. Common-input probe results")
    lines.append("")
    strong_common_rows = [row for row in common_input_rows if row["probe"] == "strong_mlp"]
    lines.append(_markdown_table(["label", "agent_id", "target", "probe", "test_r2_macro_mean", "test_relative_mse_mean", "dcor_test"], strong_common_rows))
    lines.append("")
    lines.append("### Common-input target hashes")
    lines.append("")
    hash_rows = []
    for label in labels:
        for agent_key, target_hash in common_input_target_hashes[label].items():
            hash_rows.append({"label": label, "agent": agent_key, "target_hash": target_hash})
    lines.append(_markdown_table(["label", "agent", "target_hash"], hash_rows))
    lines.append("")
    lines.append("## 5. On-policy probe results")
    lines.append("")
    strong_on_rows = [row for row in on_policy_rows if row["probe"] == "strong_mlp"]
    lines.append(_markdown_table(["label", "agent_id", "target", "probe", "test_r2_macro_mean", "test_relative_mse_mean", "dcor_test"], strong_on_rows))
    lines.append("")
    lines.append("## 6. Agent-wise results")
    lines.append("")
    agent_rows = [row for row in common_input_rows if row["probe"] == "strong_mlp" and row["target"] == "obs_action"]
    lines.append(_markdown_table(["label", "agent_id", "target", "test_r2_macro_mean", "test_relative_mse_mean"], agent_rows))
    lines.append("")
    lines.append("## 7. Dimension-wise results")
    lines.append("")
    dim_rows = [row for row in dimension_rows if row.get("probe") == "strong_mlp"]
    lines.append(_markdown_table(["label", "agent_id", "target", "dimension", "r2", "rmse_norm", "rmse_raw"], dim_rows[: min(len(dim_rows), 40)]))
    if len(dim_rows) > 40:
        lines.append("")
        lines.append(f"_Truncated to first 40 of {len(dim_rows)} dimension rows._")
    lines.append("")
    lines.append("## 8. Shuffled and constant control")
    lines.append("")
    control_rows = []
    for row in common_input_rows:
        if row["probe"] in {"constant", "shuffled", "linear", "small_mlp", "strong_mlp"}:
            control_rows.append(row)
    lines.append(_markdown_table(["label", "agent_id", "target", "probe", "test_r2_macro_mean", "test_relative_mse_mean"], control_rows[: min(len(control_rows), 40)]))
    if len(control_rows) > 40:
        lines.append("")
        lines.append(f"_Truncated to first 40 of {len(control_rows)} control rows._")
    lines.append("")
    lines.append("## 9. Conclusion and limitations")
    lines.append("")
    lines.append("- Common-input targets are hash-checked across labels and must match for a valid replay audit.")
    lines.append("- Strong MLP is the primary leakage probe; constant and shuffled probes are controls.")
    lines.append("- On-policy probes are auxiliary because their input distribution differs from the common-input audit.")
    lines.append("- Task profit comparisons are based on fixed-eval episodes and paired eval seeds.")
    lines.append("- If `fixed_profit_mean` diverges from `recent_profit_mean`, the training summary and fixed evaluation are measuring different windows.")
    lines.append("")
    return "\n".join(lines)


def run_stage1_eval(
    *,
    sweep_root: Path,
    labels: Sequence[str],
    seed: int,
    eval_seed_start: int,
    eval_seed_count: int,
    max_steps: int,
    reference_label: str,
    bi_feature_mode: str = "decoded",
    deterministic_policy: bool,
    include_on_policy_probes: bool,
    probe_epochs: int,
    probe_patience: int,
    probe_batch_size: int,
    probe_learning_rate: float,
    bootstrap_samples: int,
    device: str,
    run_root: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    resolved_device = resolve_device(device)
    labels = [str(label) for label in labels]
    if reference_label not in labels:
        raise ValueError(f"reference_label {reference_label!r} must be present in labels")
    if bi_feature_mode not in {"decoded", "transmitted"}:
        raise ValueError(f"Unsupported bi_feature_mode={bi_feature_mode!r}; expected decoded/transmitted")
    eval_seeds = [int(eval_seed_start) + i for i in range(int(eval_seed_count))]
    if len(eval_seeds) < 3:
        raise ValueError("Stage 1 evaluation requires at least 3 eval seeds")
    resolved_dirs = resolve_run_dirs(sweep_root, labels, seed=seed)
    root = run_root or (PROJECT_ROOT / "runs" / f"privacy_stage1_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    root = _ensure_dir(root, overwrite=overwrite)
    (root / "task_eval").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "probes").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    frozen_runs: Dict[str, FrozenRun] = {}
    for label in labels:
        frozen_runs[label] = load_frozen_run(resolved_dirs[label], device=resolved_device)

    task_eval_by_label: Dict[str, List[Dict[str, Any]]] = {}
    step_rows_by_label: Dict[str, List[Dict[str, Any]]] = {}
    policy_dataset_by_label: Dict[str, List[Dict[str, np.ndarray]]] = {}
    checkpoint_hash_by_label: Dict[str, str] = {}
    reference_rows: List[Dict[str, Any]] = []
    reference_agent_arrays: List[Dict[str, np.ndarray]] = []

    collect_policy_target = getattr(collect_policy_run, "side_effect", None)
    if not callable(collect_policy_target):
        collect_policy_target = collect_policy_run
    try:
        collect_policy_signature = inspect.signature(collect_policy_target)
    except (TypeError, ValueError):
        collect_policy_signature = None
    collect_policy_accepts_bi_feature_mode = False
    if collect_policy_signature is not None:
        collect_policy_accepts_bi_feature_mode = (
            "bi_feature_mode" in collect_policy_signature.parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in collect_policy_signature.parameters.values())
        )
    replay_target = getattr(_replay_reference_through_encoder, "side_effect", None)
    if not callable(replay_target):
        replay_target = _replay_reference_through_encoder
    try:
        replay_signature = inspect.signature(replay_target)
    except (TypeError, ValueError):
        replay_signature = None
    replay_accepts_bi_feature_mode = False
    if replay_signature is not None:
        replay_accepts_bi_feature_mode = (
            "bi_feature_mode" in replay_signature.parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in replay_signature.parameters.values())
        )
    probe_target = getattr(_probe_rows_for_dataset, "side_effect", None)
    if not callable(probe_target):
        probe_target = _probe_rows_for_dataset
    try:
        probe_signature = inspect.signature(probe_target)
    except (TypeError, ValueError):
        probe_signature = None
    probe_accepts_dataset_meta = False
    if probe_signature is not None:
        probe_accepts_dataset_meta = (
            "dataset_meta" in probe_signature.parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in probe_signature.parameters.values())
        )

    # Fixed evaluation and on-policy dataset collection.
    for label in labels:
        collect_kwargs = {
            "frozen_run": frozen_runs[label],
            "eval_seeds": eval_seeds,
            "deterministic": bool(deterministic_policy),
            "max_steps": int(max_steps),
            "label": label,
        }
        if collect_policy_accepts_bi_feature_mode:
            collect_kwargs["bi_feature_mode"] = bi_feature_mode
        result = collect_policy_run(**collect_kwargs)
        task_eval_by_label[label] = result["episode_rows"]
        step_rows_by_label[label] = result["step_rows"]
        policy_dataset_by_label[label] = result["agent_arrays"]
        checkpoint_hash_by_label[label] = str(result.get("checkpoint_hash", frozen_runs[label].checkpoint_hash))
        if label == reference_label:
            reference_rows = _prepare_reference_rows(result["episode_rows"])
            reference_agent_arrays = result["agent_arrays"]

    if not reference_rows:
        raise RuntimeError("Reference trajectory collection failed")

    _save_reference_dataset(root, reference_rows, reference_agent_arrays)
    reference_episode_ids = np.asarray(reference_agent_arrays[0]["episode_id"], dtype=np.int64)
    split = split_episode_ids(reference_episode_ids)
    split_hash = _stable_hash(split)
    common_input_agent_arrays_by_label: Dict[str, List[Dict[str, np.ndarray]]] = {}
    on_policy_agent_arrays_by_label: Dict[str, List[Dict[str, np.ndarray]]] = {}
    common_input_dataset_meta_by_label: Dict[str, Dict[str, Any]] = {}
    on_policy_dataset_meta_by_label: Dict[str, Dict[str, Any]] = {}

    def _dataset_meta_for_group(
        *,
        group_name: str,
        label: str,
        agent_arrays: Sequence[Mapping[str, np.ndarray]],
    ) -> Dict[str, Any]:
        input_hashes = {f"agent_{agent_id}": _hash_array(arrays["z"]) for agent_id, arrays in enumerate(agent_arrays)}
        target_hashes = {
            f"agent_{agent_id}": {
                "obs": _hash_array(arrays["target_obs_normalized"]),
                "act": _hash_array(arrays["target_action"]),
                "obs_action": _hash_array(arrays["target_obs_action"]),
            }
            for agent_id, arrays in enumerate(agent_arrays)
        }
        dataset_input_hash = _stable_hash(input_hashes)
        dataset_target_hash = _stable_hash(target_hashes)
        checkpoint_hash = checkpoint_hash_by_label[label]
        dataset_cache_key = _build_cache_key(
            mode=group_name,
            label=label,
            checkpoint_hash=checkpoint_hash,
            target="dataset",
            probe="dataset",
            seed=int(seed),
            bi_feature_mode=bi_feature_mode,
            input_hash=dataset_input_hash,
            target_hash=dataset_target_hash,
        )
        return {
            "mode": group_name,
            "label": label,
            "checkpoint_hash": checkpoint_hash,
            "bi_feature_mode": bi_feature_mode,
            "split_hash": split_hash,
            "dataset_cache_key": dataset_cache_key,
            "dataset_input_hash": dataset_input_hash,
            "dataset_target_hash": dataset_target_hash,
            "input_hashes": input_hashes,
            "target_hashes": target_hashes,
            "reference_label": reference_label,
            "source_eval_seeds": list(eval_seeds),
            "deterministic_policy": bool(deterministic_policy),
        }

    # Common-input replay: replay reference zero inputs through each label's encoder.
    for label in labels:
        replay_rows = []
        grouped = _group_rows_by_agent(step_rows_by_label[reference_label])
        for agent_id in range(len(AGENT_CFGS)):
            replay_rows.extend(grouped[agent_id])
        replay_kwargs = {
            "frozen_run": frozen_runs[label],
            "reference_rows": replay_rows,
        }
        if replay_accepts_bi_feature_mode:
            replay_kwargs["bi_feature_mode"] = bi_feature_mode
        common_input_agent_arrays_by_label[label] = _replay_reference_through_encoder(**replay_kwargs)
        common_input_dataset_meta_by_label[label] = _dataset_meta_for_group(
            group_name="common_input",
            label=label,
            agent_arrays=common_input_agent_arrays_by_label[label],
        )
        _save_common_or_onpolicy_dataset(
            root,
            group_name="common_input",
            label=label,
            agent_arrays=common_input_agent_arrays_by_label[label],
            extra_meta=common_input_dataset_meta_by_label[label],
        )

    if include_on_policy_probes:
        for label in labels:
            on_policy_agent_arrays_by_label[label] = _build_policy_dataset_from_rows(step_rows_by_label[label])
            on_policy_dataset_meta_by_label[label] = _dataset_meta_for_group(
                group_name="on_policy",
                label=label,
                agent_arrays=on_policy_agent_arrays_by_label[label],
            )
            _save_common_or_onpolicy_dataset(
                root,
                group_name="on_policy",
                label=label,
                agent_arrays=on_policy_agent_arrays_by_label[label],
                extra_meta=on_policy_dataset_meta_by_label[label],
            )

    # Probes and summaries.
    probe_by_group: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {"common_input": {}, "on_policy": {}}
    common_input_probe_payloads: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for label in labels:
        common_input_probe_kwargs = {
            "label": label,
            "agent_arrays": common_input_agent_arrays_by_label[label],
            "split": split,
            "probe_seeds": PROBE_SEEDS,
            "max_epochs": int(probe_epochs),
            "patience": int(probe_patience),
            "batch_size": int(probe_batch_size),
            "lr": float(probe_learning_rate),
            "device": resolved_device,
        }
        if probe_accepts_dataset_meta:
            common_input_probe_kwargs["dataset_meta"] = common_input_dataset_meta_by_label[label]
        common_input_probe_payloads[label] = _probe_rows_for_dataset(**common_input_probe_kwargs)
        for agent_id in range(len(AGENT_CFGS)):
            for target_name in TARGET_NAMES:
                _save_probe_artifacts(
                    run_root=root,
                    group_name="common_input",
                    label=label,
                    agent_id=agent_id,
                    target_name=target_name,
                    probe_payload=common_input_probe_payloads[label][agent_id][target_name],
                )
    probe_by_group["common_input"] = common_input_probe_payloads

    if include_on_policy_probes:
        on_policy_probe_payloads: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for label in labels:
            on_policy_probe_kwargs = {
                "label": label,
                "agent_arrays": on_policy_agent_arrays_by_label[label],
                "split": split,
                "probe_seeds": PROBE_SEEDS,
                "max_epochs": int(probe_epochs),
                "patience": int(probe_patience),
                "batch_size": int(probe_batch_size),
                "lr": float(probe_learning_rate),
                "device": resolved_device,
            }
            if probe_accepts_dataset_meta:
                on_policy_probe_kwargs["dataset_meta"] = on_policy_dataset_meta_by_label[label]
            on_policy_probe_payloads[label] = _probe_rows_for_dataset(**on_policy_probe_kwargs)
            for agent_id in range(len(AGENT_CFGS)):
                for target_name in TARGET_NAMES:
                    _save_probe_artifacts(
                        run_root=root,
                        group_name="on_policy",
                        label=label,
                        agent_id=agent_id,
                        target_name=target_name,
                        probe_payload=on_policy_probe_payloads[label][agent_id][target_name],
                )
        probe_by_group["on_policy"] = on_policy_probe_payloads
    else:
        probe_by_group["on_policy"] = {}

    summary_tables = _write_summary_tables(
        run_root=root,
        labels=labels,
        reference_label=reference_label,
        task_eval_by_label=task_eval_by_label,
        probe_by_group=probe_by_group,
        bootstrap_samples=int(bootstrap_samples),
        seed=int(seed),
    )

    report = _build_report(
        run_root=root,
        labels=labels,
        sweep_root=sweep_root,
        reference_label=reference_label,
        bi_feature_mode=bi_feature_mode,
        eval_seeds=eval_seeds,
        resolved_dirs=resolved_dirs,
        frozen_runs=frozen_runs,
        label_summary=summary_tables["label_summary"],
        task_table=summary_tables["task_table"],
        common_input_rows=summary_tables["common_input_rows"],
        on_policy_rows=summary_tables["on_policy_rows"],
        dimension_rows=summary_tables["dimension_rows"],
        paired_vs_zero=summary_tables["paired_vs_zero"],
        common_input_target_hashes=summary_tables["common_input_target_hashes"],
    )
    (root / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "spec_version": SPEC_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_root": str(sweep_root),
        "run_root": str(root),
        "labels": list(labels),
        "reference_label": reference_label,
        "bi_feature_mode": bi_feature_mode,
        "seed": int(seed),
        "eval_seed_start": int(eval_seed_start),
        "eval_seed_count": int(eval_seed_count),
        "eval_seeds": eval_seeds,
        "max_steps": int(max_steps),
        "deterministic_policy": bool(deterministic_policy),
        "include_on_policy_probes": bool(include_on_policy_probes),
        "checkpoint_hashes": checkpoint_hash_by_label,
        "common_input_dataset_cache_keys": {label: meta["dataset_cache_key"] for label, meta in common_input_dataset_meta_by_label.items()},
        "on_policy_dataset_cache_keys": {label: meta["dataset_cache_key"] for label, meta in on_policy_dataset_meta_by_label.items()},
        "common_input_dataset_hashes": {
            label: {"input_hashes": meta["input_hashes"], "target_hashes": meta["target_hashes"], "split_hash": meta["split_hash"]}
            for label, meta in common_input_dataset_meta_by_label.items()
        },
        "on_policy_dataset_hashes": {
            label: {"input_hashes": meta["input_hashes"], "target_hashes": meta["target_hashes"], "split_hash": meta["split_hash"]}
            for label, meta in on_policy_dataset_meta_by_label.items()
        },
        "probe": {
            "seeds": list(PROBE_SEEDS),
            "epochs": int(probe_epochs),
            "patience": int(probe_patience),
            "batch_size": int(probe_batch_size),
            "learning_rate": float(probe_learning_rate),
            "bootstrap_samples": int(bootstrap_samples),
        },
        "device": resolved_device,
        "resolved_dirs": {label: str(path) for label, path in resolved_dirs.items()},
    }
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "resolved_checkpoints.json",
        {
            "labels": list(labels),
            "reference_label": reference_label,
            "resolved_dirs": {label: str(path) for label, path in resolved_dirs.items()},
            "checkpoints": {label: frozen_runs[label].resolved_checkpoints for label in labels},
            "config": {label: frozen_runs[label].config for label in labels},
        },
    )
    return {
        "run_root": str(root),
        "manifest": manifest,
        "bi_feature_mode": bi_feature_mode,
        "resolved_checkpoints": {label: frozen_runs[label].resolved_checkpoints for label in labels},
        "task_table": summary_tables["task_table"],
        "common_input_probe_rows": summary_tables["common_input_rows"],
        "on_policy_probe_rows": summary_tables["on_policy_rows"],
        "dimension_rows": summary_tables["dimension_rows"],
        "label_summary": summary_tables["label_summary"],
        "paired_vs_zero": summary_tables["paired_vs_zero"],
        "common_input_target_hashes": summary_tables["common_input_target_hashes"],
        "report_path": str(root / "report.md"),
    }
