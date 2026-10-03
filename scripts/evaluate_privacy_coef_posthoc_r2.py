"""Independent common-input post-hoc observation reconstruction audit.

The evaluator combines the completed low- and high-coefficient FedAvg10
privacy sweeps, freezes every source checkpoint, builds one deterministic
reference trajectory cache, and attacks only the transmitted BI surface.
Training-time reconstructors are never loaded as probes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import privacy_stage1_utils as stage1  # noqa: E402
from scripts.train_fl_split_shallow_hybrid import AGENT_CFGS  # noqa: E402

SPEC_VERSION = "privacy_coef_posthoc_r2_v1"
BRANCH_ORDER = ("zero", "current", "x10", "x100", "x1000", "x10000", "x100000", "x1000000")
PROBE_ORDER = ("constant", "linear", "strong_mlp", "shuffled")
PROBE_SEEDS = (0, 1, 2)
R2_EPSILON = 1e-12
DEFAULT_LOW_ROOT = PROJECT_ROOT / "runs" / "fl_fedavg10_privacy_full_coef_sweep_full_20260715T183541Z"
DEFAULT_HIGH_ROOT = PROJECT_ROOT / "runs" / "fl_fedavg10_privacy_full_high_coef_sweep_20260723T022427Z"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json_safe(row.get(name)) for name in fieldnames})


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(valid)) if valid else None


def std_or_none(values: Iterable[Optional[float]], *, require_multiple: bool = False) -> Optional[float]:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not valid or (require_multiple and len(valid) < 2):
        return None
    return float(np.std(valid, ddof=0))


def discover_runs(source_roots: Sequence[Path]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for source_root in source_roots:
        manifest_path = source_root / "launcher_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing launcher manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for batch in manifest.get("batches", []):
            for run in batch.get("runs", []):
                label = str(run.get("label", "")).strip()
                run_root = Path(str(run.get("run_root", "")))
                if label and run_root:
                    if label in resolved and resolved[label].resolve() != run_root.resolve():
                        raise RuntimeError(f"Duplicate branch {label!r} resolves to different paths")
                    resolved[label] = run_root
    missing = [label for label in BRANCH_ORDER if label not in resolved]
    if missing:
        raise FileNotFoundError(f"Missing coefficient branches: {missing}")
    return {label: resolved[label] for label in BRANCH_ORDER}


def audit_run(label: str, run_dir: Path) -> Dict[str, Any]:
    required = [
        "config.json",
        "result_summary.json",
        *(f"agent_{i}_obs_rms.npz" for i in range(len(AGENT_CFGS))),
        *(f"split_actor_{i}.pt" for i in range(len(AGENT_CFGS))),
        *(f"split_bi_encoder_{i}.pt" for i in range(len(AGENT_CFGS))),
        "split_global_bi_server_critic.pt",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label}: missing final artifacts: {missing}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((run_dir / "result_summary.json").read_text(encoding="utf-8-sig"))
    experiment = config.get("experiment", {})
    model_cfg = config.get("model_cfg", experiment.get("model_cfg", {}))
    config_step = int(experiment.get("total_steps", config.get("ppo_hp", {}).get("total_steps", -1)))
    summary_step = int(summary.get("total_steps", -1))
    if config_step != summary_step or summary_step <= 0:
        raise RuntimeError(f"{label}: config/result global-step mismatch: {config_step} vs {summary_step}")
    state = torch.load(run_dir / "split_bi_encoder_0.pt", map_location="cpu")
    state_keys = list(state.keys()) if isinstance(state, Mapping) else []
    bottleneck_keys = [key for key in state_keys if "bottleneck" in str(key).lower() or "compress" in str(key).lower()]
    width = int(model_cfg.get("critic_bi_hidden", -1))
    if width <= 0:
        raise RuntimeError(f"{label}: invalid BI width in config: {width}")
    rms = {}
    for agent_id in range(len(AGENT_CFGS)):
        rms_path = run_dir / f"agent_{agent_id}_obs_rms.npz"
        data = np.load(rms_path)
        rms[f"agent_{agent_id}"] = {
            "path": str(rms_path),
            "sha256": file_hash(rms_path),
            "count": float(data["count"]),
            "dimension": int(np.asarray(data["mean"]).size),
        }
    coefficients = {
        "ar": float(experiment.get("priv_ar_coef", 0.0)),
        "nr": float(experiment.get("priv_nr_coef", 0.0)),
        "dcor": float(experiment.get("priv_dcor_coef", 0.0)),
    }
    checkpoints = {
        name[:-3]: {"path": str(run_dir / name), "sha256": file_hash(run_dir / name)}
        for name in required
        if name.endswith(".pt")
    }
    return {
        "branch": label,
        "run_dir": str(run_dir),
        "training_seed": int(config.get("seed", summary.get("seed", -1))),
        "global_step": summary_step,
        "global_step_source": "result_summary.total_steps corroborated by config.experiment.total_steps; raw state_dict has no embedded step",
        "bi_width": width,
        "coefficients": coefficients,
        "bottleneck_enabled": bool(bottleneck_keys),
        "bottleneck_state_keys": bottleneck_keys,
        "attack_surface_name": "transmitted_bi",
        "attack_surface_dimension": width,
        "target_name": "agent_full_observation",
        "target_dimensions": [len(cfg.obs_indices) for cfg in AGENT_CFGS],
        "target_space": "normalized",
        "normalization": rms,
        "checkpoints": checkpoints,
        "utility_summary": {
            "source": "training result_summary (not fixed evaluation)",
            "recent_profit_mean": summary.get("recent_profit_mean"),
            "episode_return_mean": summary.get("episode_return_mean"),
            "shutdown_rate": summary.get("shutdown_rate"),
        },
    }


def audit_all_runs(run_dirs: Mapping[str, Path]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    audits = {label: audit_run(label, run_dirs[label]) for label in BRANCH_ORDER}
    steps = sorted({int(item["global_step"]) for item in audits.values()})
    seeds = sorted({int(item["training_seed"]) for item in audits.values()})
    widths = sorted({int(item["bi_width"]) for item in audits.values()})
    if len(steps) != 1:
        raise RuntimeError(f"Final continuation-step mismatch; refusing comparison: {steps}")
    if len(seeds) != 1:
        raise RuntimeError(f"This coefficient audit requires a common training seed; found {seeds}")
    if len(widths) != 1:
        raise RuntimeError(f"Width mismatch requires width-matched zero controls; found {widths}")
    return audits, steps[0]


def make_episode_split(
    episode_ids: Sequence[int],
    *,
    seed: int = 2026,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> Dict[str, Any]:
    unique = np.asarray(sorted(set(int(v) for v in episode_ids)), dtype=np.int64)
    if unique.size < 10:
        raise ValueError(f"At least 10 episodes are required for a stable 70/15/15 split; found {unique.size}")
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    train_n = max(1, int(math.floor(train_fraction * unique.size)))
    val_n = max(1, int(math.floor(validation_fraction * unique.size)))
    test_n = int(unique.size - train_n - val_n)
    if test_n < 1:
        raise ValueError("Episode split produced an empty test partition")
    split = {
        "unit": "episode",
        "strategy": "deterministic_shuffled_70_15_15",
        "split_seed": int(seed),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(1.0 - train_fraction - validation_fraction),
        "train_episodes": sorted(int(v) for v in shuffled[:train_n]),
        "validation_episodes": sorted(int(v) for v in shuffled[train_n : train_n + val_n]),
        "test_episodes": sorted(int(v) for v in shuffled[train_n + val_n :]),
    }
    assert_episode_split_disjoint(split)
    split["split_manifest_hash"] = canonical_hash(split)
    return split


def assert_episode_split_disjoint(split: Mapping[str, Any]) -> None:
    train = set(int(v) for v in split["train_episodes"])
    val = set(int(v) for v in split["validation_episodes"])
    test = set(int(v) for v in split["test_episodes"])
    if train & val or train & test or val & test:
        raise ValueError("Episode-level split overlap detected")
    if not train or not val or not test:
        raise ValueError("Episode-level split contains an empty partition")


def transmitted_attack_surface(client: Any, obs: torch.Tensor, prev_act: torch.Tensor, hidden: torch.Tensor):
    decoded, hidden_out, transmitted = client.encode_bi(
        obs,
        prev_act,
        hidden,
        return_transmitted=True,
    )
    del decoded
    return transmitted, hidden_out


def group_reference_rows(step_rows: Sequence[Mapping[str, Any]]) -> Dict[int, List[Mapping[str, Any]]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        grouped[int(row["agent_id"])].append(row)
    if sorted(grouped) != list(range(len(AGENT_CFGS))):
        raise ValueError("Reference trajectory is missing an agent")
    return grouped


def reference_arrays(grouped_rows: Mapping[int, Sequence[Mapping[str, Any]]]) -> List[Dict[str, np.ndarray]]:
    arrays: List[Dict[str, np.ndarray]] = []
    for agent_id in range(len(AGENT_CFGS)):
        rows = grouped_rows[agent_id]
        arrays.append(
            {
                "target_obs_raw": np.stack([np.asarray(row["obs_raw"], dtype=np.float32) for row in rows]),
                "target_obs_normalized": np.stack([np.asarray(row["obs_normalized"], dtype=np.float32) for row in rows]),
                "previous_action": np.stack([np.asarray(row["action_bi_input"], dtype=np.float32) for row in rows]),
                "eval_seed": np.asarray([int(row["eval_seed"]) for row in rows], dtype=np.int64),
                "episode_id": np.asarray([int(row["episode_index"]) for row in rows], dtype=np.int64),
                "timestep": np.asarray([int(row["timestep"]) for row in rows], dtype=np.int64),
            }
        )
    return arrays


def sample_order_hash(arrays: Mapping[str, np.ndarray]) -> str:
    order = np.stack(
        [
            np.asarray(arrays["eval_seed"], dtype=np.int64),
            np.asarray(arrays["episode_id"], dtype=np.int64),
            np.asarray(arrays["timestep"], dtype=np.int64),
        ],
        axis=1,
    )
    return array_hash(order)


def target_hashes(agent_arrays: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, Any]:
    per_agent = {
        f"agent_{agent_id}": {
            "normalized": array_hash(np.asarray(arrays["target_obs_normalized"], dtype=np.float32)),
            "raw": array_hash(np.asarray(arrays["target_obs_raw"], dtype=np.float32)),
        }
        for agent_id, arrays in enumerate(agent_arrays)
    }
    return {"per_agent": per_agent, "combined": canonical_hash(per_agent)}


def assert_common_integrity(datasets: Mapping[str, Sequence[Mapping[str, np.ndarray]]]) -> Dict[str, Any]:
    expected_target = None
    expected_order = None
    branch_details = {}
    for branch, agents in datasets.items():
        hashes = target_hashes(agents)
        orders = [sample_order_hash(agent) for agent in agents]
        if len(set(orders)) != 1:
            raise ValueError(f"{branch}: agents do not share sample order")
        if expected_target is None:
            expected_target = hashes["combined"]
            expected_order = orders[0]
        if hashes["combined"] != expected_target or orders[0] != expected_order:
            raise ValueError(f"{branch}: common-input target/order hash mismatch")
        branch_details[branch] = {"target_hash": hashes["combined"], "sample_order_hash": orders[0]}
    return {
        "valid": True,
        "common_target_hash": expected_target,
        "common_sample_order_hash": expected_order,
        "branches": branch_details,
    }


def replay_transmitted_representations(
    frozen_run: stage1.FrozenRun,
    reference: Sequence[Mapping[str, np.ndarray]],
) -> List[Dict[str, np.ndarray]]:
    outputs: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for agent_id, client in enumerate(frozen_run.trainer.clients):
            source = reference[agent_id]
            z_values: List[np.ndarray] = []
            hidden = client.init_hidden(batch_size=1, device=torch.device(frozen_run.device))
            previous_episode: Optional[int] = None
            for index in range(len(source["episode_id"])):
                episode = int(source["episode_id"][index])
                if previous_episode is None or episode != previous_episode:
                    hidden = client.init_hidden(batch_size=1, device=torch.device(frozen_run.device))
                    previous_episode = episode
                raw_obs = np.asarray(source["target_obs_raw"][index], dtype=np.float32)
                branch_obs = frozen_run.obs_rms_list[agent_id].normalize(raw_obs)
                obs_t = torch.as_tensor(branch_obs, dtype=torch.float32, device=frozen_run.device)
                act_t = torch.as_tensor(source["previous_action"][index], dtype=torch.float32, device=frozen_run.device)
                transmitted, hidden = transmitted_attack_surface(client, obs_t, act_t, hidden)
                z_values.append(stage1._last_step_vector(transmitted))
                hidden = hidden.detach()
            outputs.append(
                {
                    "z": np.stack(z_values).astype(np.float32),
                    "target_obs_raw": np.asarray(source["target_obs_raw"], dtype=np.float32),
                    "target_obs_normalized": np.asarray(source["target_obs_normalized"], dtype=np.float32),
                    "eval_seed": np.asarray(source["eval_seed"], dtype=np.int64),
                    "episode_id": np.asarray(source["episode_id"], dtype=np.int64),
                    "timestep": np.asarray(source["timestep"], dtype=np.int64),
                }
            )
    return outputs


def compute_dimension_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_true_raw: Optional[np.ndarray] = None,
    y_pred_raw: Optional[np.ndarray] = None,
    epsilon: float = R2_EPSILON,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(f"Expected matching [samples, dimensions] arrays, got {y_true.shape} and {y_pred.shape}")
    error = y_pred - y_true
    sse = np.sum(error**2, axis=0)
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    sst = np.sum(centered**2, axis=0)
    mse = np.mean(error**2, axis=0)
    variance = np.var(y_true, axis=0)
    valid = sst > float(epsilon)
    r2: List[Optional[float]] = []
    relative: List[Optional[float]] = []
    for dim in range(y_true.shape[1]):
        if valid[dim]:
            ratio = float(sse[dim] / sst[dim])
            r2.append(float(1.0 - ratio))
            relative.append(ratio)
        else:
            r2.append(None)
            relative.append(None)
    raw_mse = raw_rmse = None
    if y_true_raw is not None and y_pred_raw is not None:
        raw_error = np.asarray(y_pred_raw, dtype=np.float64) - np.asarray(y_true_raw, dtype=np.float64)
        raw_mse_arr = np.mean(raw_error**2, axis=0)
        raw_mse = raw_mse_arr.astype(float).tolist()
        raw_rmse = np.sqrt(raw_mse_arr).astype(float).tolist()
    return {
        "epsilon": float(epsilon),
        "r2": r2,
        "mse": mse.astype(float).tolist(),
        "rmse": np.sqrt(mse).astype(float).tolist(),
        "relative_mse": relative,
        "target_std": np.sqrt(variance).astype(float).tolist(),
        "target_variance": variance.astype(float).tolist(),
        "sst": sst.astype(float).tolist(),
        "valid_r2": valid.astype(bool).tolist(),
        "constant_or_near_constant_count": int((~valid).sum()),
        "r2_macro": mean_or_none(r2),
        "relative_mse_valid_dimension_macro": mean_or_none(relative),
        "raw_mse": raw_mse,
        "raw_rmse": raw_rmse,
    }


def aggregate_agent_metrics(agent_metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    r2_values = [item.get("r2_macro") for item in agent_metrics]
    relative_values = [item.get("relative_mse_valid_dimension_macro") for item in agent_metrics]
    all_dim_r2 = [value for item in agent_metrics for value in item.get("r2", []) if value is not None]
    valid_agents = [float(value) for value in r2_values if value is not None]
    return {
        "branch_r2_agent_macro": mean_or_none(r2_values),
        "branch_r2_dimension_macro": mean_or_none(all_dim_r2),
        "branch_r2_worst_agent": max(valid_agents) if valid_agents else None,
        "branch_relative_mse": mean_or_none(relative_values),
    }


def shuffle_control_targets(
    y_train: np.ndarray,
    y_validation: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Break input/target alignment in both model-fit partitions."""
    return (
        y_train[rng.permutation(y_train.shape[0])],
        y_validation[rng.permutation(y_validation.shape[0])],
    )


def _fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    split: Mapping[str, Any],
    arch: str,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    shuffle_targets: bool,
) -> Dict[str, Any]:
    episode_ids = np.asarray(split["episode_ids"], dtype=np.int64)
    train_mask = np.isin(episode_ids, np.asarray(split["train_episodes"], dtype=np.int64))
    val_mask = np.isin(episode_ids, np.asarray(split["validation_episodes"], dtype=np.int64))
    test_mask = np.isin(episode_ids, np.asarray(split["test_episodes"], dtype=np.int64))
    if np.any(train_mask & test_mask) or not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Invalid episode masks for probe fitting")
    x_train = np.asarray(x[train_mask], dtype=np.float32)
    x_val = np.asarray(x[val_mask], dtype=np.float32)
    x_test = np.asarray(x[test_mask], dtype=np.float32)
    y_train = np.asarray(y[train_mask], dtype=np.float32)
    y_val = np.asarray(y[val_mask], dtype=np.float32)
    y_test = np.asarray(y[test_mask], dtype=np.float32)
    target_mean = y_train.mean(axis=0, keepdims=True)
    target_std = y_train.std(axis=0, keepdims=True)
    target_std = np.where(target_std < 1e-6, 1.0, target_std).astype(np.float32)
    y_train_scaled = ((y_train - target_mean) / target_std).astype(np.float32)
    y_val_scaled = ((y_val - target_mean) / target_std).astype(np.float32)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))
    if shuffle_targets:
        # Validation labels must be independently shuffled as well.  Selecting
        # an epoch against the true validation alignment would otherwise let
        # the control cherry-pick a chance-correlated state.
        y_train_scaled, y_val_scaled = shuffle_control_targets(y_train_scaled, y_val_scaled, rng)
    model = stage1._build_probe(x.shape[1], y.shape[1], arch=arch).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    tx = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y_train_scaled, dtype=torch.float32, device=device)
    vx = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    vy = torch.as_tensor(y_val_scaled, dtype=torch.float32, device=device)
    test_x = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    epochs_ran = 0
    for epoch in range(int(max_epochs)):
        model.train()
        permutation = rng.permutation(tx.shape[0])
        for start in range(0, tx.shape[0], int(batch_size)):
            idx = torch.as_tensor(permutation[start : start + int(batch_size)], dtype=torch.long, device=device)
            loss = ((model(tx.index_select(0, idx)) - ty.index_select(0, idx)) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(((model(vx) - vy) ** 2).mean().cpu().item())
        epochs_ran = epoch + 1
        if val_loss < best_loss - 1e-9:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("Probe failed to produce a validation-selected state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(test_x).detach().cpu().numpy()
    prediction = pred_scaled * target_std + target_mean
    return {
        "prediction": prediction.astype(np.float32),
        "test_indices": np.flatnonzero(test_mask).astype(np.int64),
        "best_validation_loss": best_loss,
        "epochs_ran": epochs_ran,
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }


def train_and_score_probe(
    dataset: Mapping[str, np.ndarray],
    *,
    split_manifest: Mapping[str, Any],
    probe: str,
    seed: int,
    rms_mean: np.ndarray,
    rms_var: np.ndarray,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> Dict[str, Any]:
    x = np.asarray(dataset["z"], dtype=np.float32)
    y = np.asarray(dataset["target_obs_normalized"], dtype=np.float32)
    episode_ids = np.asarray(dataset["episode_id"], dtype=np.int64)
    split = dict(split_manifest)
    split["episode_ids"] = episode_ids
    test_mask = np.isin(episode_ids, np.asarray(split_manifest["test_episodes"], dtype=np.int64))
    y_test = y[test_mask]
    raw_test = np.asarray(dataset["target_obs_raw"], dtype=np.float32)[test_mask]
    if probe == "constant":
        prediction = np.broadcast_to(y_test.mean(axis=0, keepdims=True), y_test.shape).copy()
        fit = {
            "prediction": prediction,
            "test_indices": np.flatnonzero(test_mask),
            "best_validation_loss": None,
            "epochs_ran": 0,
            "train_samples": int(np.isin(episode_ids, split_manifest["train_episodes"]).sum()),
            "validation_samples": int(np.isin(episode_ids, split_manifest["validation_episodes"]).sum()),
            "test_samples": int(test_mask.sum()),
        }
    else:
        arch = "linear" if probe == "linear" else "strong_mlp"
        fit = _fit_probe(
            x,
            y,
            split=split,
            arch=arch,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            shuffle_targets=probe == "shuffled",
        )
        prediction = np.asarray(fit["prediction"], dtype=np.float32)
    rms_scale = np.sqrt(np.asarray(rms_var, dtype=np.float64)) + 1e-8
    prediction_raw = prediction * rms_scale.reshape(1, -1) + np.asarray(rms_mean, dtype=np.float64).reshape(1, -1)
    metrics = compute_dimension_metrics(
        y_test,
        prediction,
        y_true_raw=raw_test,
        y_pred_raw=prediction_raw,
        epsilon=R2_EPSILON,
    )
    return {**fit, "metrics": metrics}


def hierarchical_seed_summary(seed_rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    by_training_seed: Dict[int, List[float]] = defaultdict(list)
    for row in seed_rows:
        value = row.get(metric)
        if value is not None:
            by_training_seed[int(row["training_seed"])].append(float(value))
    training_seed_values = {seed: float(np.mean(values)) for seed, values in by_training_seed.items() if values}
    all_probe_values = [float(row[metric]) for row in seed_rows if row.get(metric) is not None]
    return {
        "probe_seed_mean": mean_or_none(all_probe_values),
        "probe_seed_std": std_or_none(all_probe_values),
        "training_seed_values": training_seed_values,
        "training_seed_mean": mean_or_none(training_seed_values.values()),
        "training_seed_std": std_or_none(training_seed_values.values(), require_multiple=True),
        "training_seed_count": len(training_seed_values),
        "probe_seed_count": len({int(row["probe_seed"]) for row in seed_rows}),
    }


def control_sanity_checks(seed_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_probe: Dict[str, List[float]] = defaultdict(list)
    for row in seed_rows:
        value = row.get("branch_r2_agent_macro")
        if value is not None:
            by_probe[str(row["probe"])].append(float(value))
    means = {probe: float(np.mean(values)) for probe, values in by_probe.items() if values}
    issues = []
    if abs(means.get("constant", 0.0)) > 1e-8:
        issues.append("constant_r2_not_zero")
    if means.get("shuffled", 0.0) > 0.05:
        issues.append("shuffled_r2_significantly_positive")
    if "strong_mlp" in means and "linear" in means and means["strong_mlp"] + 0.02 < means["linear"]:
        issues.append("strong_mlp_materially_below_linear")
    return {"probe_means": means, "issues": issues, "passed": not issues}


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    *,
    logical_output_root: Path,
    source_roots: Sequence[Path],
    global_step: int,
    audits: Mapping[str, Mapping[str, Any]],
    branch_summaries: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
) -> str:
    summary_by_branch = {str(row["branch"]): row for row in branch_summaries}
    lines = [
        "# Independent Common-input Post-hoc Observation Reconstruction R²",
        "",
        f"- Output: `{logical_output_root}`",
        f"- Source roots: {', '.join(f'`{root}`' for root in source_roots)}",
        f"- Final checkpoint step: **{global_step}**",
        "- Attack surface: **transmitted BI64** (no bottleneck; transmitted tensor equals BI representation)",
        "- Target: full per-agent observation in the zero-control training-normalized convention",
        "- Official probe: independent strong MLP `[512, 512, 256]`, seeds `[0,1,2]`",
        "- Training-time online reconstructor metrics were not used.",
        "",
        "## Verdict",
        "",
    ]
    best_privacy = min(branch_summaries, key=lambda row: float(row["obs_r2_agent_macro"]))
    lines.append(
        f"Lowest reconstruction leakage was `{best_privacy['branch']}` "
        f"(agent-macro R²={float(best_privacy['obs_r2_agent_macro']):.6f}); paired deltas and controls below determine whether this is credible."
    )
    lines.extend(["", "## Branch summary", ""])
    lines.append(
        _markdown_table(
            ["Branch", "Width", "Training seeds", "Obs R² agent-macro ↓", "Training-seed std", "Worst-agent R² ↓", "Dimension-macro R² ↓", "Relative MSE ↑", "ΔR² vs zero ↓"],
            [
                [
                    row["branch"], row["width"], row["training_seed_count"], row["obs_r2_agent_macro"],
                    row["training_seed_std"], row["worst_agent_r2"], row["dimension_macro_r2"],
                    row["relative_mse"], row["delta_r2_vs_zero"],
                ]
                for row in branch_summaries
            ],
        )
    )
    lines.extend(["", "## Agent-level strong MLP", ""])
    agent_table = []
    for branch in BRANCH_ORDER:
        values = []
        for agent_id in range(len(AGENT_CFGS)):
            selected = [row["agent_r2_macro"] for row in agent_rows if row["branch"] == branch and row["probe"] == "strong_mlp" and row["agent_id"] == agent_id]
            values.append(mean_or_none(selected))
        agent_table.append([branch, 42, *values, summary_by_branch[branch]["obs_r2_agent_macro"]])
    lines.append(_markdown_table(["Branch", "Seed", "Agent 0 R²", "Agent 1 R²", "Agent 2 R²", "Branch macro R²"], agent_table))
    lines.extend(["", "## Controls", ""])
    control_table = []
    for branch in BRANCH_ORDER:
        means = controls[branch]["probe_means"]
        control_table.append([branch, means.get("constant"), means.get("linear"), means.get("strong_mlp"), means.get("shuffled")])
    lines.append(_markdown_table(["Branch", "Constant R²", "Linear R²", "Strong MLP R²", "Shuffled R²"], control_table))
    lines.extend(["", "## Existing utility summary (training window; not fixed evaluation)", ""])
    utility_rows = []
    for row in branch_summaries:
        utility = audits[row["branch"]]["utility_summary"]
        utility_rows.append([row["branch"], row["obs_r2_agent_macro"], row["relative_mse"], utility.get("recent_profit_mean"), utility.get("episode_return_mean")])
    lines.append(_markdown_table(["Branch", "Obs R² ↓", "Relative MSE ↑", "Recent profit ↑", "Episode return ↑"], utility_rows))
    lines.extend(
        [
            "",
            "## Integrity and limitations",
            "",
            f"- Common-input target/order integrity: **{'PASS' if integrity.get('valid') else 'FAIL'}**.",
            "- Episode split is deterministic 70/15/15 with split seed 2026 and zero sample overlap.",
            "- Only one training seed (42) exists, so training-seed standard deviation is unavailable (`NA`).",
            "- Checkpoints are raw state dictionaries without embedded global-step metadata; co-located config and final result summary both report 200000.",
            "- The common trajectory follows the existing official convention: deterministic zero-control policy over eval seeds 1000–1029, then exact replay through every frozen encoder.",
            "- Utility values above are training-summary windows, not newly executed fixed evaluation.",
            "- Raw RMSE is auxiliary; the official target and R² remain in the saved normalized-observation convention.",
            "",
            "## Control verdicts",
            "",
        ]
    )
    for branch in BRANCH_ORDER:
        issue_text = ", ".join(controls[branch]["issues"]) if controls[branch]["issues"] else "PASS"
        lines.append(f"- `{branch}`: {issue_text}")
    lines.extend(["", "## Artifact index", ""])
    for name in (
        "evaluation_manifest.json", "split_manifest.json", "common_input_manifest.json", "probe_config.json",
        "dimension_level_results.csv", "agent_level_results.csv", "seed_level_results.csv", "branch_summary.csv",
        "branch_summary.json", "tensorboard/",
    ):
        lines.append(f"- `{logical_output_root / name}`")
    return "\n".join(lines) + "\n"


def run_evaluation(
    *,
    source_roots: Sequence[Path],
    output_root: Path,
    common_cache_root: Optional[Path],
    eval_seed_start: int,
    eval_seed_count: int,
    max_steps: int,
    split_seed: int,
    encoder_device: str,
    probe_device: str,
    probe_epochs: int,
    probe_patience: int,
    probe_batch_size: int,
    probe_learning_rate: float,
) -> Dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation directory: {output_root}")
    work_root = output_root.with_name(output_root.name + "_in_progress_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    work_root.mkdir(parents=True, exist_ok=False)
    run_dirs = discover_runs(source_roots)
    audits, global_step = audit_all_runs(run_dirs)
    reference_label = "zero"
    training_seed = int(audits[reference_label]["training_seed"])
    eval_seeds = [int(eval_seed_start) + index for index in range(int(eval_seed_count))]

    print(f"[audit] branches={list(BRANCH_ORDER)} common_step={global_step}", flush=True)
    cache_dir = work_root / "common_input_cache"
    if common_cache_root is None:
        reference_run = stage1.load_frozen_run(run_dirs[reference_label], device=encoder_device)
        collected = stage1.collect_policy_run(
            frozen_run=reference_run,
            eval_seeds=eval_seeds,
            deterministic=True,
            max_steps=int(max_steps),
            label=reference_label,
            bi_feature_mode="transmitted",
        )
        grouped = group_reference_rows(collected["step_rows"])
        reference = reference_arrays(grouped)
        cache_dir.mkdir(parents=True)
        for agent_id, arrays in enumerate(reference):
            np.savez_compressed(cache_dir / f"agent_{agent_id}.npz", **arrays)
        write_json(cache_dir / "episodes.json", collected["episode_rows"])
        cache_source = cache_dir
        cache_rule = "existing official deterministic zero-control policy convention; cache reused unchanged for all branch encoders"
        del reference_run
    else:
        cache_source = Path(common_cache_root).resolve()
        source_manifest_path = cache_source.parent / "common_input_manifest.json"
        if not source_manifest_path.exists():
            raise FileNotFoundError(f"Missing source common-input manifest: {source_manifest_path}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8-sig"))
        if list(source_manifest.get("evaluation_seeds", [])) != eval_seeds:
            raise ValueError("Requested evaluation seeds do not match the reusable common-input cache")
        reference = []
        for agent_id in range(len(AGENT_CFGS)):
            source_npz = cache_source / f"agent_{agent_id}.npz"
            if not source_npz.exists():
                raise FileNotFoundError(f"Missing cached agent dataset: {source_npz}")
            with np.load(source_npz, allow_pickle=False) as cached:
                reference.append({key: np.asarray(cached[key]) for key in cached.files})
        episodes_path = cache_source / "episodes.json"
        collected = {"episode_rows": json.loads(episodes_path.read_text(encoding="utf-8-sig"))}
        computed_targets = target_hashes(reference)
        computed_order = sample_order_hash(reference[0])
        if computed_targets["combined"] != source_manifest.get("observation_target_hash"):
            raise ValueError("Reusable cache observation target hash mismatch")
        if computed_order != source_manifest.get("sample_order_hash"):
            raise ValueError("Reusable cache sample order hash mismatch")
        if len({sample_order_hash(agent) for agent in reference}) != 1:
            raise ValueError("Reusable cache agent sample orders do not match")
        shutil.copytree(cache_source, cache_dir)
        cache_rule = "integrity-verified reuse of the existing deterministic common-input cache; unchanged for all branch encoders"

    episode_ids = sorted({int(row["episode_index"]) for row in collected["episode_rows"]})
    split_manifest = make_episode_split(episode_ids, seed=split_seed)
    split_manifest["sample_counts"] = {
        partition: int(sum(np.isin(reference[0]["episode_id"], split_manifest[f"{partition}_episodes"])))
        for partition in ("train", "validation", "test")
    }
    split_manifest["train_test_sample_overlap"] = 0
    split_manifest.pop("split_manifest_hash", None)
    split_manifest["split_manifest_hash"] = canonical_hash(split_manifest)
    write_json(work_root / "split_manifest.json", split_manifest)

    reference_run_config = json.loads((run_dirs[reference_label] / "config.json").read_text(encoding="utf-8-sig"))
    reference_env_config = stage1.experiment_cfg_from_run_config(reference_run_config)
    env_config = {
        key: getattr(reference_env_config, key)
        for key in ("rl_dt_seconds", "max_steps", "delta_mv_max", "production_index", "baseline", "scale", "reset_reward")
    }
    env_config["agent_obs_indices"] = [list(cfg.obs_indices) for cfg in AGENT_CFGS]
    env_config["agent_mv_indices"] = [list(cfg.mv_indices) for cfg in AGENT_CFGS]
    base_hashes = target_hashes(reference)
    order_hash = sample_order_hash(reference[0])
    common_manifest = {
        "environment_config": env_config,
        "environment_config_hash": canonical_hash(env_config),
        "source_checkpoint_independent_trajectory_cache_path": str(cache_source),
        "local_cache_copy_path": str(output_root / "common_input_cache"),
        "cache_generation_rule": cache_rule,
        "cache_is_branch_independent_after_generation": True,
        "cache_generation_reference_branch": reference_label,
        "evaluation_seeds": eval_seeds,
        "episode_count": len(episode_ids),
        "sample_count_per_agent": int(len(reference[0]["episode_id"])),
        "episode_ids": episode_ids,
        "episode_ids_hash": canonical_hash(episode_ids),
        "sample_order_hash": order_hash,
        "observation_target_hash": base_hashes["combined"],
        "observation_target_hashes_per_agent": base_hashes["per_agent"],
        "split_manifest_hash": split_manifest["split_manifest_hash"],
        "target_space": "normalized",
        "normalization_source": str(run_dirs[reference_label]) + "/agent_{i}_obs_rms.npz",
        "raw_target_saved": True,
    }
    write_json(work_root / "common_input_manifest.json", common_manifest)

    datasets: Dict[str, List[Dict[str, np.ndarray]]] = {}
    representation_hashes: Dict[str, Any] = {}
    for branch in BRANCH_ORDER:
        print(f"[encode] branch={branch}", flush=True)
        frozen = stage1.load_frozen_run(run_dirs[branch], device=encoder_device)
        datasets[branch] = replay_transmitted_representations(frozen, reference)
        branch_dir = work_root / "representations" / branch
        branch_dir.mkdir(parents=True, exist_ok=True)
        for agent_id, arrays in enumerate(datasets[branch]):
            np.savez_compressed(branch_dir / f"agent_{agent_id}.npz", **arrays)
        representation_hashes[branch] = {
            f"agent_{agent_id}": array_hash(arrays["z"]) for agent_id, arrays in enumerate(datasets[branch])
        }
        observed_widths = {int(arrays["z"].shape[1]) for arrays in datasets[branch]}
        if observed_widths != {int(audits[branch]["bi_width"])}:
            raise RuntimeError(f"{branch}: transmitted attack width mismatch: {observed_widths}")
        del frozen

    integrity = assert_common_integrity(datasets)
    if not integrity["valid"]:
        raise RuntimeError("Common-input integrity validation failed")

    probe_config = {
        "official_probe": "strong_mlp",
        "architecture": {"strong_mlp": [512, 512, 256], "linear": [], "shuffled": [512, 512, 256]},
        "source_implementation": "scripts/privacy_stage1_utils.py::_build_probe",
        "optimizer": "AdamW",
        "learning_rate": float(probe_learning_rate),
        "weight_decay": 1e-5,
        "max_epochs": int(probe_epochs),
        "early_stopping_patience": int(probe_patience),
        "selection_metric": "validation normalized MSE",
        "batch_size": int(probe_batch_size),
        "probe_seeds": list(PROBE_SEEDS),
        "probes": list(PROBE_ORDER),
        "shuffled_control": "train and validation targets independently permuted; true held-out test alignment used only for final scoring",
        "target": "full per-agent observation",
        "target_space": "normalized",
        "r2_baseline": "held-out test target mean per dimension",
        "r2_epsilon": R2_EPSILON,
        "constant_dimensions": "excluded from R2 macro; RMSE retained",
        "probe_device": probe_device,
        "encoder_device": encoder_device,
    }
    write_json(work_root / "probe_config.json", probe_config)

    dimension_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    for branch in BRANCH_ORDER:
        agent_probe_seed_metrics: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for agent_id, dataset in enumerate(datasets[branch]):
            rms_path = run_dirs[reference_label] / f"agent_{agent_id}_obs_rms.npz"
            rms_data = np.load(rms_path)
            for probe in PROBE_ORDER:
                for probe_seed in PROBE_SEEDS:
                    print(f"[probe] branch={branch} agent={agent_id} probe={probe} seed={probe_seed}", flush=True)
                    result = train_and_score_probe(
                        dataset,
                        split_manifest=split_manifest,
                        probe=probe,
                        seed=probe_seed,
                        rms_mean=np.asarray(rms_data["mean"], dtype=np.float64),
                        rms_var=np.asarray(rms_data["var"], dtype=np.float64),
                        max_epochs=probe_epochs,
                        patience=probe_patience,
                        batch_size=probe_batch_size,
                        learning_rate=probe_learning_rate,
                        device=probe_device,
                    )
                    metrics = result["metrics"]
                    agent_row = {
                        "branch": branch,
                        "width": audits[branch]["bi_width"],
                        "training_seed": training_seed,
                        "probe_seed": probe_seed,
                        "probe": probe,
                        "agent_id": agent_id,
                        "agent_r2_macro": metrics["r2_macro"],
                        "agent_relative_mse": metrics["relative_mse_valid_dimension_macro"],
                        "valid_dimension_count": int(sum(metrics["valid_r2"])),
                        "excluded_dimension_count": metrics["constant_or_near_constant_count"],
                        "best_validation_loss": result["best_validation_loss"],
                        "epochs_ran": result["epochs_ran"],
                        "train_samples": result["train_samples"],
                        "validation_samples": result["validation_samples"],
                        "test_samples": result["test_samples"],
                    }
                    agent_rows.append(agent_row)
                    agent_probe_seed_metrics[(probe, probe_seed)].append(metrics)
                    for dim, obs_index in enumerate(AGENT_CFGS[agent_id].obs_indices):
                        dimension_rows.append(
                            {
                                "branch": branch,
                                "width": audits[branch]["bi_width"],
                                "training_seed": training_seed,
                                "probe_seed": probe_seed,
                                "probe": probe,
                                "agent_id": agent_id,
                                "dimension_index": dim,
                                "dimension_name": f"obs_index_{obs_index}",
                                "r2": metrics["r2"][dim],
                                "mse": metrics["mse"][dim],
                                "rmse": metrics["rmse"][dim],
                                "normalized_rmse": metrics["rmse"][dim],
                                "relative_mse": metrics["relative_mse"][dim],
                                "raw_mse": metrics["raw_mse"][dim] if metrics["raw_mse"] is not None else None,
                                "raw_rmse": metrics["raw_rmse"][dim] if metrics["raw_rmse"] is not None else None,
                                "target_std": metrics["target_std"][dim],
                                "target_variance": metrics["target_variance"][dim],
                                "valid_r2": metrics["valid_r2"][dim],
                                "constant_or_near_constant": not metrics["valid_r2"][dim],
                                "r2_epsilon": R2_EPSILON,
                            }
                        )
        for (probe, probe_seed), metrics_by_agent in agent_probe_seed_metrics.items():
            aggregate = aggregate_agent_metrics(metrics_by_agent)
            seed_rows.append(
                {
                    "branch": branch,
                    "width": audits[branch]["bi_width"],
                    "training_seed": training_seed,
                    "probe_seed": probe_seed,
                    "probe": probe,
                    **aggregate,
                }
            )
        if probe_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    controls = {
        branch: control_sanity_checks([row for row in seed_rows if row["branch"] == branch]) for branch in BRANCH_ORDER
    }
    summaries: List[Dict[str, Any]] = []
    for branch in BRANCH_ORDER:
        strong_rows = [row for row in seed_rows if row["branch"] == branch and row["probe"] == "strong_mlp"]
        r2 = hierarchical_seed_summary(strong_rows, "branch_r2_agent_macro")
        worst = hierarchical_seed_summary(strong_rows, "branch_r2_worst_agent")
        dim = hierarchical_seed_summary(strong_rows, "branch_r2_dimension_macro")
        relative = hierarchical_seed_summary(strong_rows, "branch_relative_mse")
        summaries.append(
            {
                "branch": branch,
                "width": audits[branch]["bi_width"],
                "training_seed_count": r2["training_seed_count"],
                "probe_seed_count": r2["probe_seed_count"],
                "probe_seed_mean": r2["probe_seed_mean"],
                "probe_seed_std": r2["probe_seed_std"],
                "obs_r2_agent_macro": r2["training_seed_mean"],
                "training_seed_std": r2["training_seed_std"],
                "worst_agent_r2": worst["training_seed_mean"],
                "dimension_macro_r2": dim["training_seed_mean"],
                "relative_mse": relative["training_seed_mean"],
                "control_passed": controls[branch]["passed"],
                "control_issues": controls[branch]["issues"],
                "coefficients": audits[branch]["coefficients"],
            }
        )
    zero = next(row for row in summaries if row["branch"] == reference_label)
    for row in summaries:
        row["delta_r2_vs_zero"] = float(row["obs_r2_agent_macro"] - zero["obs_r2_agent_macro"])
        zero_relative = float(zero["relative_mse"])
        row["relative_mse_ratio_vs_zero"] = float(row["relative_mse"] / zero_relative) if zero_relative != 0 else None

    dimension_fields = list(dimension_rows[0].keys())
    agent_fields = list(agent_rows[0].keys())
    seed_fields = list(seed_rows[0].keys())
    summary_fields = [
        "branch", "width", "training_seed_count", "probe_seed_count", "probe_seed_mean", "probe_seed_std",
        "obs_r2_agent_macro", "training_seed_std", "worst_agent_r2", "dimension_macro_r2", "relative_mse",
        "delta_r2_vs_zero", "relative_mse_ratio_vs_zero", "control_passed", "control_issues", "coefficients",
    ]
    write_csv(work_root / "dimension_level_results.csv", dimension_rows, dimension_fields)
    write_csv(work_root / "agent_level_results.csv", agent_rows, agent_fields)
    write_csv(work_root / "seed_level_results.csv", seed_rows, seed_fields)
    write_csv(work_root / "branch_summary.csv", summaries, summary_fields)
    write_json(work_root / "branch_summary.json", {"branches": summaries, "controls": controls})

    for branch in BRANCH_ORDER:
        writer = SummaryWriter(log_dir=str(work_root / "tensorboard" / branch))
        summary = next(row for row in summaries if row["branch"] == branch)
        writer.add_scalar("privacy/posthoc/common/strong_mlp/r2_agent_macro", summary["obs_r2_agent_macro"], global_step)
        writer.add_scalar("privacy/posthoc/common/strong_mlp/r2_dimension_macro", summary["dimension_macro_r2"], global_step)
        writer.add_scalar("privacy/posthoc/common/strong_mlp/r2_worst_agent", summary["worst_agent_r2"], global_step)
        writer.add_scalar("privacy/posthoc/common/strong_mlp/relative_mse", summary["relative_mse"], global_step)
        for agent_id in range(len(AGENT_CFGS)):
            values = [row["agent_r2_macro"] for row in agent_rows if row["branch"] == branch and row["probe"] == "strong_mlp" and row["agent_id"] == agent_id]
            writer.add_scalar(f"privacy/posthoc/common/strong_mlp/agent_{agent_id}/r2_macro", float(np.mean(values)), global_step)
        for probe in ("constant", "linear", "shuffled"):
            writer.add_scalar(f"privacy/posthoc/common/{probe}/r2_agent_macro", controls[branch]["probe_means"].get(probe, 0.0), global_step)
        writer.close()

    evaluation_manifest = {
        "spec_version": SPEC_VERSION,
        "created_at_utc": utc_now(),
        "status": "complete",
        "source_experiment_roots": [str(root) for root in source_roots],
        "output_root": str(output_root),
        "branches": list(BRANCH_ORDER),
        "training_seeds": [training_seed],
        "probe_seeds": list(PROBE_SEEDS),
        "common_final_global_step": global_step,
        "branch_audit": audits,
        "attack_surface_name": "transmitted_bi",
        "attack_surface_dimension": 64,
        "target_name": "full_agent_observation",
        "target_dimensions": [len(cfg.obs_indices) for cfg in AGENT_CFGS],
        "target_space": "normalized",
        "normalization_source": str(run_dirs[reference_label]) + "/agent_{i}_obs_rms.npz",
        "common_input_integrity": integrity,
        "representation_hashes": representation_hashes,
        "split_manifest_hash": split_manifest["split_manifest_hash"],
        "probe_config_hash": canonical_hash(probe_config),
        "existing_posthoc_evaluator": "scripts/privacy_stage1_utils.py and Stage 4A wrappers; strong MLP reused, strict metric/output layer added here",
        "fixed_utility_evaluation": "not available; not newly executed",
    }
    write_json(work_root / "evaluation_manifest.json", evaluation_manifest)
    report = build_report(
        logical_output_root=output_root,
        source_roots=source_roots,
        global_step=global_step,
        audits=audits,
        branch_summaries=summaries,
        seed_rows=seed_rows,
        agent_rows=agent_rows,
        integrity=integrity,
        controls=controls,
    )
    (work_root / "POSTHOC_R2_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    work_root.rename(output_root)
    return {
        "output_root": str(output_root),
        "report": str(output_root / "POSTHOC_R2_EVALUATION_REPORT.md"),
        "global_step": global_step,
        "summaries": summaries,
        "controls": controls,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-root", type=Path, default=DEFAULT_LOW_ROOT)
    parser.add_argument("--high-root", type=Path, default=DEFAULT_HIGH_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--common-cache-root", type=Path, default=None)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--eval-seed-count", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--probe-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--probe-patience", type=int, default=30)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.probe_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe device requested but CUDA is unavailable")
    output_root = args.output_root or (args.high_root / "posthoc_r2_eval")
    result = run_evaluation(
        source_roots=[args.low_root, args.high_root],
        output_root=output_root,
        common_cache_root=args.common_cache_root,
        eval_seed_start=args.eval_seed_start,
        eval_seed_count=args.eval_seed_count,
        max_steps=args.max_steps,
        split_seed=args.split_seed,
        encoder_device=args.encoder_device,
        probe_device=args.probe_device,
        probe_epochs=args.probe_epochs,
        probe_patience=args.probe_patience,
        probe_batch_size=args.probe_batch_size,
        probe_learning_rate=args.probe_learning_rate,
    )
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
