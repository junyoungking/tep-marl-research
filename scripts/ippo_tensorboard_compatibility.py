from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


SPEC_VERSION = "ippo_tensorboard_compatibility_v2"
RAW_REPLAY_ABS_TOL = 1e-5
RAW_REPLAY_REL_TOL = 1e-6

RAW_TAGS: Tuple[str, ...] = (
    "raw/prod",
    "raw/cost",
    "raw/profit",
    "raw/shutdown_rate",
)

REQUIRED_COMMON_TAGS: Tuple[str, ...] = (
    *RAW_TAGS,
    "episode/return",
    "episode/length",
    "loss/actor",
    "loss/critic",
    "loss/entropy",
    "train/approx_kl",
    "train/clipfrac",
    "train/actor_epochs_used",
    "train/critic_epochs_used",
    "rollout/length",
    "rollout/episodes_per_update",
    "train/update_count",
    "train/updates_per_env_step",
)

END_COVERAGE_TAGS: Tuple[str, ...] = (
    *RAW_TAGS,
    "loss/actor",
    "loss/critic",
    "loss/entropy",
    "train/approx_kl",
    "train/clipfrac",
    "train/actor_epochs_used",
    "train/critic_epochs_used",
    "rollout/length",
    "rollout/episodes_per_update",
    "train/update_count",
    "train/updates_per_env_step",
)

FORBIDDEN_TAG_PREFIXES: Tuple[str, ...] = ("fedavg/", "privacy/", "server/")
EXPECTED_ACTOR_COUNTS = [4738, 4930, 4802]
EXPECTED_CRITIC_COUNTS = [25601, 25793, 25665]
EXPECTED_TOTAL_UNIQUE = 91529


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fields = list(fieldnames or [])
    if not resolved_fields:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(str(key))
        resolved_fields = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not resolved_fields:
            return
        writer = csv.DictWriter(handle, fieldnames=resolved_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in resolved_fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.10g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def assert_finite_values(label: str, values: Iterable[Any]) -> None:
    for index, value in enumerate(values):
        array = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{label}[{index}] contains NaN/Inf")


@dataclass
class RawMetricWindow:
    start_step: Optional[int] = None
    prod: List[float] = field(default_factory=list)
    cost: List[float] = field(default_factory=list)
    profit: List[float] = field(default_factory=list)
    shutdown: List[float] = field(default_factory=list)

    def append(self, *, step: int, raw_prod: Any, raw_cost: Any, shutdown: Any) -> None:
        if self.start_step is None:
            self.start_step = int(step)
        prod_value = None if raw_prod is None else float(raw_prod)
        cost_value = None if raw_cost is None else float(raw_cost)
        if prod_value is not None:
            self.prod.append(prod_value)
        if cost_value is not None:
            self.cost.append(cost_value)
        if prod_value is not None and cost_value is not None:
            self.profit.append(prod_value - cost_value)
        self.shutdown.append(float(bool(shutdown)))

    def flush(self, *, end_step: int) -> Optional[Dict[str, Any]]:
        if self.start_step is None:
            return None
        row = {
            "window_start_step": int(self.start_step),
            "window_end_step": int(end_step),
            "raw_prod_count": len(self.prod),
            "raw_cost_count": len(self.cost),
            "raw_profit_count": len(self.profit),
            "raw_shutdown_rate_count": len(self.shutdown),
            "raw_prod_mean": float(np.mean(self.prod)) if self.prod else None,
            "raw_cost_mean": float(np.mean(self.cost)) if self.cost else None,
            "raw_profit_mean": float(np.mean(self.profit)) if self.profit else None,
            "raw_shutdown_rate_mean": float(np.mean(self.shutdown)) if self.shutdown else None,
        }
        self.start_step = None
        self.prod.clear()
        self.cost.clear()
        self.profit.clear()
        self.shutdown.clear()
        return row


RAW_WINDOW_FIELDS: Tuple[str, ...] = (
    "window_start_step",
    "window_end_step",
    "raw_prod_count",
    "raw_cost_count",
    "raw_profit_count",
    "raw_shutdown_rate_count",
    "raw_prod_mean",
    "raw_cost_mean",
    "raw_profit_mean",
    "raw_shutdown_rate_mean",
)


def append_raw_window_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RAW_WINDOW_FIELDS), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: json_safe(row.get(key)) for key in RAW_WINDOW_FIELDS})


def write_raw_window_to_tensorboard(writer: Any, row: Mapping[str, Any]) -> None:
    step = int(row["window_end_step"])
    mapping = {
        "raw/prod": row.get("raw_prod_mean"),
        "raw/cost": row.get("raw_cost_mean"),
        "raw/profit": row.get("raw_profit_mean"),
        "raw/shutdown_rate": row.get("raw_shutdown_rate_mean"),
    }
    for tag, value in mapping.items():
        if value is not None:
            writer.add_scalar(tag, float(value), step)


def load_event_data(
    log_dir: Path,
    *,
    required_tags: Sequence[str] = REQUIRED_COMMON_TAGS,
    include_all_scalar_samples: bool = True,
    run_id: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # pragma: no cover - environment failure
        raise RuntimeError("TensorBoard EventAccumulator is required for compatibility validation") from exc

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = sorted(accumulator.Tags().get("scalars", []))
    inventory: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    tags = list(dict.fromkeys([*required_tags, *available]))
    for tag in tags:
        values = accumulator.Scalars(tag) if tag in available else []
        finite = all(finite_scalar(item.value) for item in values)
        inventory.append(
            {
                "run_id": run_id,
                "tag": tag,
                "exists": bool(values),
                "point_count": len(values),
                "first_step": values[0].step if values else None,
                "last_step": values[-1].step if values else None,
                "first_value": values[0].value if values else None,
                "last_value": values[-1].value if values else None,
                "all_values_finite": finite,
            }
        )
        if include_all_scalar_samples:
            for index, item in enumerate(values):
                samples.append(
                    {
                        "run_id": run_id,
                        "tag": tag,
                        "sample_index": index,
                        "wall_time": item.wall_time,
                        "step": item.step,
                        "value": item.value,
                        "finite": finite_scalar(item.value),
                    }
                )
    return inventory, samples, available


def validate_event_directory(
    log_dir: Path,
    *,
    total_steps: int,
    required_tags: Sequence[str] = REQUIRED_COMMON_TAGS,
    run_id: str | None = None,
) -> Dict[str, Any]:
    inventory, samples, available = load_event_data(log_dir, required_tags=required_tags, run_id=run_id)
    by_tag = {row["tag"]: row for row in inventory}
    missing = [tag for tag in required_tags if not bool(by_tag[tag]["exists"])]
    non_finite = [tag for tag in required_tags if by_tag[tag]["exists"] and not bool(by_tag[tag]["all_values_finite"])]
    forbidden = [tag for tag in available if any(tag.startswith(prefix) for prefix in FORBIDDEN_TAG_PREFIXES)]
    end_coverage_failures = [
        tag
        for tag in END_COVERAGE_TAGS
        if tag in by_tag and (by_tag[tag]["last_step"] is None or int(by_tag[tag]["last_step"]) < int(total_steps))
    ]
    last_step = max((int(row["last_step"]) for row in inventory if row["last_step"] is not None), default=0)
    event_files = sorted(str(path) for path in log_dir.glob("events.out.tfevents*")) if log_dir.exists() else []
    passed = bool(event_files) and not missing and not non_finite and not forbidden and not end_coverage_failures and last_step >= int(total_steps)
    return {
        "spec_version": SPEC_VERSION,
        "run_id": run_id,
        "log_dir": str(log_dir),
        "event_files": event_files,
        "required_tags": list(required_tags),
        "available_tags": available,
        "missing_tags": missing,
        "non_finite_tags": non_finite,
        "forbidden_tags": forbidden,
        "end_coverage_failures": end_coverage_failures,
        "last_logged_global_step": last_step,
        "expected_total_steps": int(total_steps),
        "inventory": inventory,
        "samples": samples,
        "passed": passed,
    }


def raw_replay_rows(
    window_rows: Sequence[Mapping[str, Any]],
    scalar_samples: Sequence[Mapping[str, Any]],
    *,
    abs_tolerance: float = RAW_REPLAY_ABS_TOL,
    rel_tolerance: float = RAW_REPLAY_REL_TOL,
) -> List[Dict[str, Any]]:
    actual: Dict[Tuple[str, int], List[float]] = {}
    for row in scalar_samples:
        tag = str(row.get("tag"))
        if tag not in RAW_TAGS:
            continue
        key = (tag, int(row["step"]))
        actual.setdefault(key, []).append(float(row["value"]))

    mapping = {
        "raw/prod": ("raw_prod_mean", "raw_prod_count"),
        "raw/cost": ("raw_cost_mean", "raw_cost_count"),
        "raw/profit": ("raw_profit_mean", "raw_profit_count"),
        "raw/shutdown_rate": ("raw_shutdown_rate_mean", "raw_shutdown_rate_count"),
    }
    rows: List[Dict[str, Any]] = []
    for window in window_rows:
        step = int(window["window_end_step"])
        for tag, (mean_key, count_key) in mapping.items():
            raw_expected = window.get(mean_key)
            if raw_expected in (None, ""):
                continue
            expected = float(raw_expected)
            values = actual.get((tag, step), [])
            observed = values[-1] if values else None
            delta = abs(observed - expected) if observed is not None else None
            tolerance = max(float(abs_tolerance), float(rel_tolerance) * abs(expected))
            passed = observed is not None and delta is not None and delta <= tolerance
            rows.append(
                {
                    "tag": tag,
                    "window_start_step": int(window["window_start_step"]),
                    "window_end_step": step,
                    "source_count": int(window[count_key]),
                    "expected_value": expected,
                    "tensorboard_value": observed,
                    "absolute_delta": delta,
                    "absolute_tolerance": abs_tolerance,
                    "relative_tolerance": rel_tolerance,
                    "effective_tolerance": tolerance,
                    "tensorboard_match_count": len(values),
                    "passed": passed,
                }
            )
    return rows


def replay_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(str(row.get("passed")).lower() in ("true", "1") or row.get("passed") is True for row in rows)


def safe_reset_exact(path: Path, *, expected: Path) -> None:
    resolved = path.resolve()
    expected_resolved = expected.resolve()
    if resolved != expected_resolved:
        raise RuntimeError(f"Refusing to reset unexpected path: {resolved} != {expected_resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def snapshot_tree(root: Path) -> Dict[str, Any]:
    resolved = root.resolve()
    files: List[Dict[str, Any]] = []
    if resolved.exists():
        for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
            stat = path.stat()
            files.append(
                {
                    "relative_path": str(path.relative_to(resolved)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"root": str(resolved), "captured_at_utc": utc_now(), "file_count": len(files), "files": files}


def inventory_unchanged(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> bool:
    def normalized(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: normalized(item) for key, item in value.items() if key != "captured_at_utc"}
        if isinstance(value, list):
            return [normalized(item) for item in value]
        return value

    return before is not None and after is not None and normalized(before) == normalized(after)


def validate_pre_run_contract(
    *,
    candidate_id: str,
    candidate_spec: Mapping[str, Any],
    experiment: Any,
    requested_seed: int,
    child_process_seed: int,
    parameter_counts: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = {
        "candidate_id_match": str(experiment.name) == str(candidate_id),
        "actor_lr_match": math.isclose(float(experiment.actor_lr), float(candidate_spec["actor_lr"]), rel_tol=0.0, abs_tol=1e-12),
        "critic_lr_match": math.isclose(float(experiment.critic_lr), float(candidate_spec["critic_lr"]), rel_tol=0.0, abs_tol=1e-12),
        "ent_coef_match": math.isclose(float(experiment.ent_coef), float(candidate_spec["ent_coef"]), rel_tol=0.0, abs_tol=1e-12),
        "requested_seed_match": int(requested_seed) == 42,
        "child_process_seed_match": int(child_process_seed) == 42,
        "total_steps_match": int(experiment.total_steps) == 200000,
        "rollout_horizon_match": int(experiment.rollout_horizon) == 1024,
        "gamma_match": math.isclose(float(experiment.gamma), 0.99),
        "gae_lambda_match": math.isclose(float(experiment.gae_lambda), 0.95),
        "clip_range_match": math.isclose(float(experiment.clip_range), 0.2),
        "actor_epochs_match": int(experiment.actor_epochs) == 4,
        "critic_epochs_match": int(experiment.critic_epochs) == 16,
        "target_kl_match": math.isclose(float(experiment.target_kl), 0.03),
        "max_grad_norm_match": math.isclose(float(experiment.max_grad_norm), 0.5),
        "actor_counts_match": list(parameter_counts.get("actors", [])) == EXPECTED_ACTOR_COUNTS,
        "critic_counts_match": list(parameter_counts.get("critics", [])) == EXPECTED_CRITIC_COUNTS,
        "total_unique_match": int(parameter_counts.get("total_unique", -1)) == EXPECTED_TOTAL_UNIQUE,
    }
    return {
        "spec_version": SPEC_VERSION,
        "candidate_id": candidate_id,
        "requested_seed": int(requested_seed),
        "child_process_seed": int(child_process_seed),
        "checks": checks,
        "passed": all(checks.values()),
        "validated_at_utc": utc_now(),
    }


def validate_candidate_completion(
    *,
    candidate_id: str,
    run_root: Path,
    requested_seed: int,
    child_process_seed: int,
    total_steps: int,
    required_tags: Sequence[str] = REQUIRED_COMMON_TAGS,
) -> Dict[str, Any]:
    config_path = run_root / "config.json"
    result_path = run_root / "result_summary.json"
    pre_run_path = run_root / "stage_a_v2_pre_run_validation.json"
    config = load_json(config_path) if config_path.exists() else {}
    result = load_json(result_path) if result_path.exists() else {}
    pre_run = load_json(pre_run_path) if pre_run_path.exists() else {}
    tensorboard = validate_event_directory(
        run_root / "tensorboard",
        total_steps=int(total_steps),
        required_tags=required_tags,
        run_id=candidate_id,
    )
    parameter_counts = result.get("parameter_counts", config.get("parameter_counts", {}))
    checkpoint_names = [
        "config.json",
        "result_summary.json",
        *[f"agent_{index}_actor.pt" for index in range(3)],
        *[f"agent_{index}_critic.pt" for index in range(3)],
        *[f"agent_{index}_obs_rms.npz" for index in range(3)],
    ]
    final_logs = result.get("final_logs", {})
    finite_log_keys = (
        "loss/actor",
        "loss/critic",
        "loss/entropy",
        "train/approx_kl",
        "train/clipfrac",
    )
    checks = {
        "pre_run_validation_pass": bool(pre_run.get("passed")),
        "requested_seed_match": int(requested_seed) == 42,
        "child_process_seed_match": int(child_process_seed) == 42,
        "config_seed_match": config.get("seed") == 42,
        "result_seed_match": result.get("seed") == 42,
        "total_steps_match": int(result.get("total_steps", -1)) == int(total_steps),
        "actor_counts_match": list(parameter_counts.get("actors", [])) == EXPECTED_ACTOR_COUNTS,
        "critic_counts_match": list(parameter_counts.get("critics", [])) == EXPECTED_CRITIC_COUNTS,
        "total_unique_match": int(parameter_counts.get("total_unique", -1)) == EXPECTED_TOTAL_UNIQUE,
        "checkpoints_complete": all((run_root / name).exists() for name in checkpoint_names),
        "finite_training_logs": all(finite_scalar(final_logs.get(key)) for key in finite_log_keys),
        "tensorboard_complete": bool(tensorboard.get("passed")),
    }
    payload = {
        "spec_version": SPEC_VERSION,
        "candidate_id": candidate_id,
        "requested_seed": int(requested_seed),
        "child_process_seed": int(child_process_seed),
        "config_seed": config.get("seed"),
        "result_seed": result.get("seed"),
        "total_steps": result.get("total_steps"),
        "checks": checks,
        "tensorboard": tensorboard,
        "passed": all(checks.values()),
        "validated_at_utc": utc_now(),
    }
    write_json(run_root / "stage_a_v2_tensorboard_validation.json", payload)
    return payload


def interval_overlap(left_low: Any, left_high: Any, right_low: Any, right_high: Any) -> bool:
    try:
        values = [float(left_low), float(left_high), float(right_low), float(right_high)]
    except Exception:
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    if values[0] > values[1] or values[2] > values[3]:
        return False
    return max(values[0], values[2]) <= min(values[1], values[3])


def ci_overlap_matrix(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for left in records:
        left_eval = left.get("evaluation_result", {})
        for right in records:
            right_eval = right.get("evaluation_result", {})
            rows.append(
                {
                    "left_candidate": left.get("candidate_id"),
                    "right_candidate": right.get("candidate_id"),
                    "left_ci_low": left_eval.get("fixed_profit_ci95_low"),
                    "left_ci_high": left_eval.get("fixed_profit_ci95_high"),
                    "right_ci_low": right_eval.get("fixed_profit_ci95_low"),
                    "right_ci_high": right_eval.get("fixed_profit_ci95_high"),
                    "profit_ci_overlap": interval_overlap(
                        left_eval.get("fixed_profit_ci95_low"),
                        left_eval.get("fixed_profit_ci95_high"),
                        right_eval.get("fixed_profit_ci95_low"),
                        right_eval.get("fixed_profit_ci95_high"),
                    ),
                }
            )
    return rows


def _candidate_tb_validation(record: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(record["run_root"])) / "stage_a_v2_tensorboard_validation.json"
    return load_json(path) if path.exists() else {}


def rank_v2_records(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eligible = []
    for record in records:
        evaluation = record.get("evaluation_result", {})
        validation = _candidate_tb_validation(record)
        if (
            record.get("status") == "COMPLETE"
            and record.get("config_parity", {}).get("overall_pass")
            and validation.get("passed")
            and evaluation.get("fixed_profit_mean") is not None
        ):
            eligible.append(record)
    profit_sorted = sorted(
        eligible,
        key=lambda record: (-float(record["evaluation_result"]["fixed_profit_mean"]), str(record["candidate_id"])),
    )
    near_tie: List[Mapping[str, Any]] = []
    remaining: List[Mapping[str, Any]] = []
    if profit_sorted:
        leader = profit_sorted[0]
        leader_eval = leader["evaluation_result"]
        for record in profit_sorted:
            evaluation = record["evaluation_result"]
            if interval_overlap(
                leader_eval.get("fixed_profit_ci95_low"),
                leader_eval.get("fixed_profit_ci95_high"),
                evaluation.get("fixed_profit_ci95_low"),
                evaluation.get("fixed_profit_ci95_high"),
            ):
                near_tie.append(record)
            else:
                remaining.append(record)
    near_tie = sorted(
        near_tie,
        key=lambda record: (
            -float(record["evaluation_result"].get("fixed_return_mean", float("-inf"))),
            float(record["evaluation_result"].get("shutdown_rate", float("inf"))),
            float(record.get("anchor_distance", float("inf"))),
            str(record["candidate_id"]),
        ),
    )
    ordered = [*near_tie, *remaining]
    ranking: List[Dict[str, Any]] = []
    for index, record in enumerate(ordered, start=1):
        evaluation = record["evaluation_result"]
        ranking.append(
            {
                "rank": index,
                "candidate_id": record["candidate_id"],
                "status": record["status"],
                "eligible_for_ranking": True,
                "selected": index == 1,
                "in_leading_ci_overlap_set": record in near_tie,
                "fixed_profit_mean": evaluation.get("fixed_profit_mean"),
                "fixed_profit_ci95_low": evaluation.get("fixed_profit_ci95_low"),
                "fixed_profit_ci95_high": evaluation.get("fixed_profit_ci95_high"),
                "fixed_return_mean": evaluation.get("fixed_return_mean"),
                "shutdown_rate": evaluation.get("shutdown_rate"),
                "anchor_distance": record.get("anchor_distance"),
            }
        )
    for record in records:
        if any(row["candidate_id"] == record["candidate_id"] for row in ranking):
            continue
        evaluation = record.get("evaluation_result", {})
        ranking.append(
            {
                "rank": None,
                "candidate_id": record["candidate_id"],
                "status": record.get("status"),
                "eligible_for_ranking": False,
                "selected": False,
                "in_leading_ci_overlap_set": False,
                "fixed_profit_mean": evaluation.get("fixed_profit_mean"),
                "fixed_profit_ci95_low": evaluation.get("fixed_profit_ci95_low"),
                "fixed_profit_ci95_high": evaluation.get("fixed_profit_ci95_high"),
                "fixed_return_mean": evaluation.get("fixed_return_mean"),
                "shutdown_rate": evaluation.get("shutdown_rate"),
                "anchor_distance": record.get("anchor_distance"),
            }
        )
    selected = ordered[0] if ordered else None
    selection = {
        "selected_candidate": selected["candidate_id"] if selected else None,
        "selection_scope": "seed42_stage_a_tb_v2_screen_only",
        "primary_metric": "fixed_profit_mean",
        "supporting_metric": "fixed_return_mean",
        "near_tie_rule_applied": len(near_tie) > 1,
        "near_tie_method": "actual_profit_ci95_overlap_with_highest_profit_candidate; no practical threshold was defined",
        "leading_ci_overlap_set": [record["candidate_id"] for record in near_tie],
        "reason": (
            "No eligible candidate completed Stage A TB v2."
            if selected is None
            else (
                f"Selected {selected['candidate_id']} from the leading Profit CI-overlap set using Fixed Return, "
                "shutdown rate, and anchor distance tie-breakers."
                if len(near_tie) > 1
                else f"Selected {selected['candidate_id']} because it has the highest Fixed Profit mean without a leading CI-overlap tie."
            )
        ),
        "not_final_benchmark": True,
    }
    return ranking, selection


def legacy_tag_contract_rows() -> List[Dict[str, Any]]:
    rows = []
    for tag in REQUIRED_COMMON_TAGS:
        if tag == "raw/prod":
            source, aggregation, cadence = 'info["raw_prod"]', "mean over log_every window", "log_every + final partial"
        elif tag == "raw/cost":
            source, aggregation, cadence = 'info["raw_cost"]', "mean over log_every window", "log_every + final partial"
        elif tag == "raw/profit":
            source, aggregation, cadence = 'info["raw_prod"] - info["raw_cost"]', "mean over log_every window", "log_every + final partial"
        elif tag == "raw/shutdown_rate":
            source, aggregation, cadence = 'bool(info["shutdown"])', "mean over log_every window", "log_every + final partial"
        elif tag.startswith("episode/"):
            source, aggregation, cadence = "completed episode accumulator", "one scalar per completed episode", "on done before reset"
        elif tag.startswith("loss/") or tag.startswith("train/"):
            source, aggregation, cadence = "IPPO per-agent update diagnostics", "mean across three agents unless counter", "once per PPO update"
        else:
            source, aggregation, cadence = "rollout/update diagnostics", "per update", "once per PPO update"
        rows.append(
            {
                "tag": tag,
                "source_variable": source,
                "aggregation_rule": aggregation,
                "aggregation_window": cadence,
                "tensorboard_global_step": "environment step_count",
            }
        )
    return rows


def mapping_rows() -> List[Dict[str, Any]]:
    return [
        {
            **row,
            "ippo_status": "restored common tag" if row["tag"] in (*RAW_TAGS, "episode/return", "episode/length", "train/update_count", "train/updates_per_env_step") else "existing aggregate retained",
            "agent_specific_additional": row["tag"].startswith(("loss/", "train/")),
        }
        for row in legacy_tag_contract_rows()
    ]


def build_logging_restore_artifacts(*, smoke_root: Path, report_root: Path, command: Sequence[str]) -> Dict[str, Any]:
    result = load_json(smoke_root / "result_summary.json")
    config = load_json(smoke_root / "config.json")
    validation = validate_event_directory(
        smoke_root / "tensorboard",
        total_steps=2048,
        required_tags=REQUIRED_COMMON_TAGS,
        run_id="logging_smoke",
    )
    windows = read_csv(smoke_root / "tensorboard_raw_metric_windows.csv")
    replay = raw_replay_rows(windows, validation["samples"])
    counts = result.get("parameter_counts", {})
    architecture_rows = [
        {"check": "actor_counts", "expected": EXPECTED_ACTOR_COUNTS, "actual": counts.get("actors"), "passed": counts.get("actors") == EXPECTED_ACTOR_COUNTS},
        {"check": "critic_counts", "expected": EXPECTED_CRITIC_COUNTS, "actual": counts.get("critics"), "passed": counts.get("critics") == EXPECTED_CRITIC_COUNTS},
        {"check": "total_unique", "expected": EXPECTED_TOTAL_UNIQUE, "actual": counts.get("total_unique"), "passed": counts.get("total_unique") == EXPECTED_TOTAL_UNIQUE},
        {"check": "no_server", "expected": True, "actual": result.get("final_logs", {}).get("debug/no_server_critic") == 1.0, "passed": result.get("final_logs", {}).get("debug/no_server_critic") == 1.0},
        {"check": "no_gat", "expected": True, "actual": result.get("final_logs", {}).get("debug/no_server_gat") == 1.0, "passed": result.get("final_logs", {}).get("debug/no_server_gat") == 1.0},
        {"check": "no_fedavg", "expected": True, "actual": result.get("final_logs", {}).get("debug/no_fedavg") == 1.0, "passed": result.get("final_logs", {}).get("debug/no_fedavg") == 1.0},
        {"check": "no_privacy", "expected": True, "actual": result.get("final_logs", {}).get("debug/no_privacy") == 1.0, "passed": result.get("final_logs", {}).get("debug/no_privacy") == 1.0},
    ]
    rollout = result.get("rollout_collection", {})
    training_hps = result.get("training_hps", {})
    rollout_rows = [
        {"check": "total_steps", "expected": 2048, "actual": result.get("total_steps"), "passed": result.get("total_steps") == 2048},
        {"check": "minimum_two_updates", "expected": ">=2", "actual": result.get("updates"), "passed": int(result.get("updates", 0)) >= 2},
        {"check": "rollout_horizon", "expected": 1024, "actual": rollout.get("horizon"), "passed": rollout.get("horizon") == 1024},
        {"check": "continue_across_done", "expected": "fixed_horizon_continue_across_done", "actual": rollout.get("mode"), "passed": rollout.get("mode") == "fixed_horizon_continue_across_done"},
        {"check": "critic_epochs", "expected": 16, "actual": training_hps.get("critic_epochs"), "passed": training_hps.get("critic_epochs") == 16},
        {"check": "actor_epochs_max", "expected": 4, "actual": training_hps.get("actor_epochs"), "passed": training_hps.get("actor_epochs") == 4},
        {"check": "target_kl", "expected": 0.03, "actual": training_hps.get("target_kl"), "passed": training_hps.get("target_kl") == 0.03},
    ]
    source_diff_rows = [
        {"file": "scripts/train_ippo_architecture_matched.py", "scope": "logging only", "change": "common raw/episode/update-counter TensorBoard tags and replay source windows"},
        {"file": "scripts/ippo_stage_a_tuning_common.py", "scope": "v2 validation only", "change": "pre-run and post-training gates before fixed evaluation"},
        {"file": "scripts/launch_ippo_stage_a_tuning.py", "scope": "opt-in v2 orchestration", "change": "smoke-first gate, v2 roots, v2 reports/artifacts"},
        {"file": "scripts/ippo_tensorboard_compatibility.py", "scope": "new evidence helper", "change": "event/replay/seed/CI/report validation"},
    ]
    report_root.mkdir(parents=True, exist_ok=True)
    write_csv(report_root / "legacy_tensorboard_tag_contract.csv", legacy_tag_contract_rows())
    write_csv(report_root / "ippo_tensorboard_tag_mapping.csv", mapping_rows())
    write_csv(report_root / "logging_smoke_tag_inventory.csv", validation["inventory"])
    write_csv(report_root / "logging_smoke_scalar_samples.csv", validation["samples"])
    write_csv(report_root / "tensorboard_raw_metric_replay.csv", replay)
    write_csv(report_root / "logging_smoke_architecture_parity.csv", architecture_rows)
    write_csv(report_root / "logging_smoke_rollout_validation.csv", rollout_rows)
    write_csv(report_root / "logging_restore_source_diff.csv", source_diff_rows)
    gate_checks = {
        "tensorboard_tags_pass": bool(validation["passed"]),
        "raw_replay_pass": replay_passed(replay),
        "architecture_non_regression_pass": all(bool(row["passed"]) for row in architecture_rows),
        "rollout_non_regression_pass": all(bool(row["passed"]) for row in rollout_rows),
    }
    gate = {
        "spec_version": SPEC_VERSION,
        "smoke_root": str(smoke_root),
        "report_root": str(report_root),
        "command": list(command),
        "checks": gate_checks,
        "passed": all(gate_checks.values()),
        "validated_at_utc": utc_now(),
    }
    write_json(report_root / "logging_smoke_gate.json", gate)
    inventory_table = markdown_table(
        ["tag", "exists", "points", "first_step", "last_step", "first_value", "last_value"],
        [[row["tag"], row["exists"], row["point_count"], row["first_step"], row["last_step"], row["first_value"], row["last_value"]] for row in validation["inventory"] if row["tag"] in REQUIRED_COMMON_TAGS],
    )
    report = f"""# IPPO TensorBoard Logging Restore Report

## 1. Scope

Restore legacy-compatible common TensorBoard tags for the verified architecture-matched recurrent IPPO only. Training and evaluation semantics are unchanged.

## 2. Files modified

{markdown_table(["file", "scope", "change"], [[row["file"], row["scope"], row["change"]] for row in source_diff_rows])}

## 3. Canonical Ours TensorBoard tag contract

Canonical raw metrics use environment `info` values, window means at `log_every`, and a final partial flush. Episode scalars are emitted on done before reset.

## 4. IPPO common-tag mapping

See `legacy_tensorboard_tag_contract.csv` and `ippo_tensorboard_tag_mapping.csv`.

## 5. Agent-specific tags

Existing `agent_N/*` IPPO diagnostics remain additional signals and do not replace common tags.

## 6. Raw metric source and aggregation

- `raw/prod = info[\"raw_prod\"]`
- `raw/cost = info[\"raw_cost\"]`
- `raw/profit = raw_prod - raw_cost`
- `raw/shutdown_rate = mean(bool(info[\"shutdown\"]))`

## 7. Smoke configuration

- seed: 42
- total steps: 2048
- rollout horizon: 1024
- updates: {result.get("updates")}
- command: `{command}`

## 8. Smoke actual event tag inventory

{inventory_table}

## 9. Raw metric replay validation

- absolute tolerance: {RAW_REPLAY_ABS_TOL}
- relative tolerance: {RAW_REPLAY_REL_TOL}
- rows: {len(replay)}
- verdict: {"PASS" if gate_checks["raw_replay_pass"] else "FAIL"}

## 10. Architecture non-regression

{markdown_table(["check", "expected", "actual", "passed"], [[row["check"], row["expected"], row["actual"], row["passed"]] for row in architecture_rows])}

## 11. Rollout non-regression

{markdown_table(["check", "expected", "actual", "passed"], [[row["check"], row["expected"], row["actual"], row["passed"]] for row in rollout_rows])}

## 12. Smoke PASS gate

{markdown_table(["check", "passed"], [[key, value] for key, value in gate_checks.items()])}

## 13. Exact commands

`{' '.join(command)}`

## 14. Raw artifact index

- `legacy_tensorboard_tag_contract.csv`
- `ippo_tensorboard_tag_mapping.csv`
- `logging_smoke_tag_inventory.csv`
- `logging_smoke_scalar_samples.csv`
- `tensorboard_raw_metric_replay.csv`
- `logging_smoke_architecture_parity.csv`
- `logging_smoke_rollout_validation.csv`
- `logging_restore_source_diff.csv`
- `logging_smoke_gate.json`

## 15. Final logging compatibility verdict

**{"PASS" if gate["passed"] else "FAIL"}**
"""
    (report_root / "IPPO_TENSORBOARD_LOGGING_RESTORE_REPORT.md").write_text(report, encoding="utf-8")
    return gate


def run_logging_smoke(*, smoke_root: Path, report_root: Path, device: str, reset: bool, command: Sequence[str]) -> Dict[str, Any]:
    if reset:
        safe_reset_exact(smoke_root, expected=smoke_root)
    else:
        smoke_root.mkdir(parents=True, exist_ok=True)
    from dataclasses import replace
    from torch.utils.tensorboard import SummaryWriter
    from scripts.train_ippo_architecture_matched import make_default_experiment, run_one_experiment

    writer = SummaryWriter(log_dir=str(smoke_root / "tensorboard"))
    try:
        experiment = replace(
            make_default_experiment(),
            name="ippo_tensorboard_logging_smoke",
            total_steps=2048,
            rollout_horizon=1024,
            log_every=1000,
            smoke=False,
            stage_a_tuning=False,
            device=device,
            run_root=str(smoke_root),
        )
        result = run_one_experiment(experiment, seed=42, run_root_override=str(smoke_root), writer=writer)
        if int(result.get("updates", 0)) < 2:
            raise RuntimeError(f"Logging smoke requires at least two updates, observed {result.get('updates')}")
        writer.flush()
    finally:
        writer.close()
    gate = build_logging_restore_artifacts(smoke_root=smoke_root, report_root=report_root, command=command)
    if not gate["passed"]:
        raise RuntimeError(f"TensorBoard logging smoke gate failed: {gate['checks']}")
    return gate


def _flatten_config_parity(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {"candidate_id": record["candidate_id"], "status": record.get("status"), **record.get("config_parity", {})}


def _training_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    result = record.get("training_result", {})
    logs = result.get("final_logs", {})
    return {
        "candidate_id": record["candidate_id"],
        "status": record.get("status"),
        "total_steps": result.get("total_steps"),
        "updates": result.get("updates"),
        "recent_profit_mean": result.get("recent_profit_mean"),
        "episode_return_mean": result.get("episode_return_mean"),
        "shutdown_rate": result.get("shutdown_rate"),
        "final_actor_loss": logs.get("loss/actor"),
        "final_critic_loss": logs.get("loss/critic"),
        "final_entropy": logs.get("loss/entropy"),
        "final_approx_kl": logs.get("train/approx_kl"),
        "final_clipfrac": logs.get("train/clipfrac"),
        "final_actor_epochs_used": logs.get("train/actor_epochs_used"),
        "final_critic_epochs_used": logs.get("train/critic_epochs_used"),
        "final_rollout_length": logs.get("rollout/length"),
        "final_episodes_per_update": logs.get("rollout/episodes_per_update"),
    }


def _fixed_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    evaluation = record.get("evaluation_result", {})
    fields = (
        "evaluation_seed_count",
        "valid_episodes",
        "failed_episodes",
        "truncated_episodes",
        "mean_episode_shutdown_rate",
        "shutdown_rate",
        "mean_episode_length",
        "fixed_profit_mean",
        "fixed_profit_std",
        "fixed_profit_median",
        "fixed_profit_minimum",
        "fixed_profit_maximum",
        "fixed_profit_q25",
        "fixed_profit_q75",
        "fixed_profit_ci95_low",
        "fixed_profit_ci95_high",
        "fixed_return_mean",
        "fixed_return_std",
        "fixed_return_median",
        "fixed_return_minimum",
        "fixed_return_maximum",
        "fixed_return_q25",
        "fixed_return_q75",
        "fixed_return_ci95_low",
        "fixed_return_ci95_high",
    )
    return {"candidate_id": record["candidate_id"], "status": record.get("status"), **{field: evaluation.get(field) for field in fields}}


def build_stage_a_v2_artifacts(
    *,
    stage_a_root: Path,
    report_root: Path,
    manifest: MutableMapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    launcher_command: Sequence[str],
    legacy_inventory_before: Mapping[str, Any] | None,
    legacy_inventory_after: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    ranking, selection = rank_v2_records(records)
    overlap = ci_overlap_matrix(records)
    grid_rows = []
    for spec in manifest.get("candidate_grid", []):
        grid_rows.append(
            {
                "candidate_id": spec.get("candidate_id"),
                "actor_lr": spec.get("actor_lr"),
                "critic_lr": spec.get("critic_lr"),
                "ent_coef": spec.get("ent_coef"),
                "seed": manifest.get("seed"),
                "total_steps": manifest.get("training_contract", {}).get("total_steps"),
                "rollout_horizon": manifest.get("training_contract", {}).get("rollout_horizon"),
                "run_root": spec.get("run_root"),
                "eval_root": spec.get("eval_root"),
            }
        )
    execution_rows = [
        {"execution_index": index, "candidate_id": candidate_id}
        for index, candidate_id in enumerate(manifest.get("execution_order", []), start=1)
    ]
    seed_rows = []
    run_rows = []
    tb_rows = []
    tag_rows: List[Dict[str, Any]] = []
    kl_rows = []
    failure_rows = []
    candidate_runtime = []
    for record in records:
        validation = _candidate_tb_validation(record)
        tensorboard = validation.get("tensorboard", {})
        config = record.get("config", {})
        result = record.get("training_result", {})
        seed_row = {
            "candidate_id": record["candidate_id"],
            "requested_seed": validation.get("requested_seed"),
            "child_process_seed": validation.get("child_process_seed"),
            "config_seed": config.get("seed"),
            "result_seed": result.get("seed"),
        }
        seed_row["passed"] = all(seed_row[key] == 42 for key in ("requested_seed", "child_process_seed", "config_seed", "result_seed"))
        seed_rows.append(seed_row)
        run_rows.append(
            {
                "candidate_id": record["candidate_id"],
                "status": record.get("status"),
                "phase": record.get("phase"),
                "complete": record.get("complete"),
                "run_root": record.get("run_root"),
                "eval_root": record.get("eval_root"),
                "training_started_at": record.get("training_started_at"),
                "training_completed_at": record.get("training_completed_at"),
                "evaluation_started_at": record.get("evaluation_started_at"),
                "evaluation_completed_at": record.get("evaluation_completed_at"),
                "error_type": record.get("error_type"),
                "error_message": record.get("error_message"),
            }
        )
        tb_rows.append(
            {
                "candidate_id": record["candidate_id"],
                "passed": validation.get("passed"),
                "tensorboard_passed": tensorboard.get("passed"),
                "event_file_count": len(tensorboard.get("event_files", [])),
                "missing_tags": ";".join(tensorboard.get("missing_tags", [])),
                "forbidden_tags": ";".join(tensorboard.get("forbidden_tags", [])),
                "non_finite_tags": ";".join(tensorboard.get("non_finite_tags", [])),
                "end_coverage_failures": ";".join(tensorboard.get("end_coverage_failures", [])),
                "last_logged_global_step": tensorboard.get("last_logged_global_step"),
                "expected_total_steps": tensorboard.get("expected_total_steps"),
            }
        )
        tag_rows.extend(tensorboard.get("inventory", []))
        logs = result.get("final_logs", {})
        kl_row: Dict[str, Any] = {"candidate_id": record["candidate_id"], "status": record.get("status")}
        for index in range(3):
            for suffix in ("approx_kl", "approx_kl_post", "actor_epochs_used", "critic_epochs_used", "entropy"):
                kl_row[f"agent_{index}_{suffix}"] = logs.get(f"agent_{index}/{suffix}")
        kl_rows.append(kl_row)
        if record.get("status") != "COMPLETE" or record.get("failure_text"):
            failure_rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "status": record.get("status"),
                    "phase": record.get("phase"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "failure_path": record.get("failure_path"),
                    "failure_exists": bool(record.get("failure_text")),
                }
            )
        candidate_runtime.append(
            {
                "candidate_id": record["candidate_id"],
                "actor_lr": record.get("spec", {}).get("actor_lr"),
                "critic_lr": record.get("spec", {}).get("critic_lr"),
                "ent_coef": record.get("spec", {}).get("ent_coef"),
                "requested_seed": validation.get("requested_seed"),
                "child_process_seed": validation.get("child_process_seed"),
                "config_seed": config.get("seed"),
                "result_seed": result.get("seed"),
                "total_steps": result.get("total_steps"),
                "rollout_horizon": config.get("experiment", {}).get("rollout_horizon"),
                "run_status": record.get("status"),
                "tensorboard_status": "PASS" if validation.get("passed") else "FAIL",
                "fixed_eval_status": "PASS" if record.get("evaluation_result_exists") else "FAIL",
                "run_root": record.get("run_root"),
                "eval_root": record.get("eval_root"),
            }
        )
    write_csv(report_root / "stage_a_v2_grid.csv", grid_rows)
    write_csv(report_root / "stage_a_v2_execution_order.csv", execution_rows)
    write_csv(report_root / "stage_a_v2_seed_validation.csv", seed_rows)
    write_csv(report_root / "stage_a_v2_config_parity.csv", [_flatten_config_parity(record) for record in records])
    write_csv(report_root / "stage_a_v2_run_status.csv", run_rows)
    write_csv(report_root / "stage_a_v2_tensorboard_completion.csv", tb_rows)
    write_csv(report_root / "stage_a_v2_tag_inventory.csv", tag_rows)
    write_csv(report_root / "stage_a_v2_training_diagnostics.csv", [_training_row(record) for record in records])
    write_csv(report_root / "stage_a_v2_kl_actor_epochs.csv", kl_rows)
    write_csv(report_root / "stage_a_v2_fixed_utility_summary.csv", [_fixed_row(record) for record in records])
    write_csv(report_root / "stage_a_v2_ci_overlap_matrix.csv", overlap)
    write_csv(report_root / "stage_a_v2_ranking.csv", ranking)
    write_csv(report_root / "stage_a_v2_failures.csv", failure_rows, ["candidate_id", "status", "phase", "error_type", "error_message", "failure_path", "failure_exists"])
    write_json(report_root / "stage_a_v2_selection.json", selection)
    manifest["spec_version"] = "ippo_stage_a_tuning_tb_v2"
    manifest["candidate_runtime"] = candidate_runtime
    manifest["logging_restore_report"] = str(report_root / "IPPO_TENSORBOARD_LOGGING_RESTORE_REPORT.md")
    manifest["stage_a_v2_report"] = str(report_root / "IPPO_STAGE_A_TUNING_TB_V2_REPORT.md")
    manifest["selection"] = selection
    manifest["reported_at_utc"] = utc_now()
    manifest["legacy_stage_a_status"] = "LEGACY_STAGE_A_LOGGING_INCOMPLETE"
    manifest["legacy_inventory_before"] = legacy_inventory_before
    manifest["legacy_inventory_after"] = legacy_inventory_after
    manifest["legacy_inventory_unchanged"] = inventory_unchanged(legacy_inventory_before, legacy_inventory_after)
    write_json(stage_a_root / "stage_a_manifest.json", manifest)
    write_json(report_root / "stage_a_v2_manifest_snapshot.json", manifest)
    report = _stage_a_v2_report(
        stage_a_root=stage_a_root,
        report_root=report_root,
        manifest=manifest,
        records=records,
        ranking=ranking,
        selection=selection,
        overlap=overlap,
        launcher_command=launcher_command,
    )
    (report_root / "IPPO_STAGE_A_TUNING_TB_V2_REPORT.md").write_text(report, encoding="utf-8")
    return {"ranking": ranking, "selection": selection, "manifest": manifest, "report_path": str(report_root / "IPPO_STAGE_A_TUNING_TB_V2_REPORT.md")}


def _stage_a_v2_report(
    *,
    stage_a_root: Path,
    report_root: Path,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    overlap: Sequence[Mapping[str, Any]],
    launcher_command: Sequence[str],
) -> str:
    completed = sum(record.get("status") == "COMPLETE" for record in records)
    tb_passed = sum(bool(_candidate_tb_validation(record).get("passed")) for record in records)
    fixed_table = markdown_table(
        ["candidate", "profit_mean", "profit_ci_low", "profit_ci_high", "return_mean", "shutdown", "status"],
        [[record["candidate_id"], record.get("evaluation_result", {}).get("fixed_profit_mean"), record.get("evaluation_result", {}).get("fixed_profit_ci95_low"), record.get("evaluation_result", {}).get("fixed_profit_ci95_high"), record.get("evaluation_result", {}).get("fixed_return_mean"), record.get("evaluation_result", {}).get("shutdown_rate"), record.get("status")] for record in records],
    )
    ranking_table = markdown_table(
        ["rank", "candidate", "eligible", "selected", "ci_tie_set", "profit", "return", "shutdown"],
        [[row.get("rank"), row.get("candidate_id"), row.get("eligible_for_ranking"), row.get("selected"), row.get("in_leading_ci_overlap_set"), row.get("fixed_profit_mean"), row.get("fixed_return_mean"), row.get("shutdown_rate")] for row in ranking],
    )
    matrix_rows = [
        ["logging compatibility", "PASS" if (report_root / "logging_smoke_gate.json").exists() and load_json(report_root / "logging_smoke_gate.json").get("passed") else "FAIL"],
        ["candidate completion", "PASS" if completed == 9 else "PARTIAL"],
        ["TensorBoard completion", "PASS" if tb_passed == 9 else "PARTIAL"],
        ["legacy root unchanged", "PASS" if manifest.get("legacy_inventory_unchanged") else "FAIL"],
        ["selection", "PASS" if selection.get("selected_candidate") else "PARTIAL"],
    ]
    return f"""# IPPO Stage A TensorBoard v2 Tuning Report

## 1. Executive summary

Completed candidates: **{completed}/9**. TensorBoard-complete candidates: **{tb_passed}/9**. Selected Stage B candidate for screening only: **{selection.get('selected_candidate')}**.

## 2. Why previous Stage A was superseded

The previous root remains preserved and is classified `LEGACY_STAGE_A_LOGGING_INCOMPLETE` because it lacks the required legacy-compatible raw and episode curves.

## 3. TensorBoard logging compatibility status

See `IPPO_TENSORBOARD_LOGGING_RESTORE_REPORT.md` and `logging_smoke_gate.json`.

## 4. Fixed architecture contract

Actor counts `4738/4930/4802`, critic counts `25601/25793/25665`, total unique `91529`.

## 5. Fixed training contract

Seed 42, 200,000 steps, horizon 1,024, fixed-horizon continue-across-done, PPO/Adam contract unchanged.

## 6. Predeclared grid

See `stage_a_v2_grid.csv`.

## 7. Actual execution order

See `stage_a_v2_execution_order.csv`.

## 8. Seed propagation validation

See `stage_a_v2_seed_validation.csv` for requested/child/config/result values.

## 9. Candidate config parity

See `stage_a_v2_config_parity.csv`.

## 10. Run completion matrix

See `stage_a_v2_run_status.csv`.

## 11. TensorBoard completion matrix

See `stage_a_v2_tensorboard_completion.csv`.

## 12. Required tag inventory per candidate

See `stage_a_v2_tag_inventory.csv`.

## 13. Training stability

See `stage_a_v2_training_diagnostics.csv` and `stage_a_v2_kl_actor_epochs.csv`.

## 14. raw/profit diagnostics

Training raw profit is diagnostic only and is not used for ranking.

## 15. episode/return diagnostics

Episode return is a training diagnostic only and is not used for ranking.

## 16. PPO loss/KL diagnostics

Common and per-agent loss/KL/epoch metrics are retained in TensorBoard and tabulated in the diagnostics artifacts.

## 17. Fixed evaluation contract

Frozen deterministic actor/critic/obs RMS; evaluation seeds 1000-1099; zero optimizer or gradient steps.

## 18. Fixed Profit comparison

{fixed_table}

## 19. Fixed Return comparison

See the comparison table and `stage_a_v2_fixed_utility_summary.csv`.

## 20. Shutdown comparison

Shutdown rates and failed episode counts are reported in `stage_a_v2_fixed_utility_summary.csv`.

## 21. Confidence intervals

Profit and Return bootstrap confidence intervals use the fixed evaluation summaries.

## 22. CI overlap / near-tie analysis

Full pairwise evidence is in `stage_a_v2_ci_overlap_matrix.csv` ({len(overlap)} rows). Method: {selection.get('near_tie_method')}.

## 23. Stage A ranking

{ranking_table}

## 24. Selected Stage B candidate

`{selection.get('selected_candidate')}`. Reason: {selection.get('reason')} This is not a final benchmark conclusion.

## 25. Failure log

See `stage_a_v2_failures.csv`.

## 26. Reproducibility limits

This is a single training seed Stage A screen. It does not establish multi-seed or final general superiority.

## 27. TensorBoard launch command

`tensorboard --logdir {stage_a_root}`

## 28. Raw artifact index

All `stage_a_v2_*` CSV/JSON files plus both Markdown reports under `{report_root}`.

## 29. Exact commands

`{' '.join(str(item) for item in launcher_command)}`

## 30. PASS/PARTIAL/FAIL matrix

{markdown_table(["criterion", "status"], matrix_rows)}
"""
