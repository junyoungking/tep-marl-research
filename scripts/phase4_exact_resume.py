"""Scoped exact-state checkpoint support for the Phase 4 Python TEP backend.

Checkpoints contain only primitive containers and tensors and are loaded with
torch.load(weights_only=True). No historical checkpoint is accepted here.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import tempfile
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from trainers.fl_split_ppo_trainer_globalbi_phase4 import CONTRACT_SHA256


FORMAT = "phase4_full_privacy_exact_v1"
VERIFIED_FORMAT = "phase6_verified_checkpoint_v1"


class UnsupportedExactResume(RuntimeError):
    pass


class CheckpointTrustError(ValueError):
    """A checkpoint has no valid committed verification record."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_json_atomic(path: Path, value: dict) -> None:
    """Commit a small trust record with the same durability order as a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".verified_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def verified_manifest_path(run_root: Path, *, final: bool = False) -> Path:
    return Path(run_root) / ("final/verified_final.json" if final else "checkpoints/latest_verified.json")


def verified_record(run_root: Path, path: Path, info: dict, *, step: int,
                    update_count: int, collector_cursor: int, variant: str, seed: int,
                    source_sha256: str, final: bool, previous: dict | None = None) -> dict:
    run_root = Path(run_root).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(run_root):
        raise CheckpointTrustError("Checkpoint is outside its experiment namespace")
    return {"schema": VERIFIED_FORMAT, "kind": "final" if final else "interim",
            "run_root": str(run_root), "checkpoint_relpath": str(path.relative_to(run_root)),
            "step": int(step), "update_count": int(update_count),
            "collector_cursor": int(collector_cursor), "variant": variant,
            "seed": int(seed), "source_sha256": source_sha256,
            "contract_sha256": CONTRACT_SHA256, "backend": "python",
            "sha256": info["sha256"], "size_bytes": int(info["size_bytes"]),
            "verification": {"reload_exact": True, "fresh_process_one_step_exact": True},
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "previous": previous}


def read_verified_record(run_root: Path, *, final: bool = False) -> dict | None:
    path = verified_manifest_path(run_root, final=final)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CheckpointTrustError(f"Verified manifest unreadable: {path}") from exc
    if not isinstance(record, dict) or record.get("schema") != VERIFIED_FORMAT:
        raise CheckpointTrustError(f"Verified manifest schema invalid: {path}")
    return record


def verify_published_checkpoint(path: Path, *, variant: str, seed: int,
                                source_sha256: str) -> tuple[dict, dict]:
    """Accept an explicit path only when its run's manifest commits it."""
    path = Path(path).resolve()
    kind = path.parent.name
    if kind not in ("checkpoints", "final"):
        raise CheckpointTrustError("Checkpoint is not in a published namespace")
    run_root = path.parent.parent.resolve()
    final = kind == "final"
    record = read_verified_record(run_root, final=final)
    if record is None:
        raise CheckpointTrustError("No verified manifest; legacy files need read-only migration verification")
    expected = {
        "schema": VERIFIED_FORMAT, "kind": "final" if final else "interim",
        "run_root": str(run_root), "checkpoint_relpath": str(path.relative_to(run_root)),
        "variant": variant, "seed": int(seed), "source_sha256": source_sha256,
        "contract_sha256": CONTRACT_SHA256, "backend": "python",
        "verification": {"reload_exact": True, "fresh_process_one_step_exact": True},
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise CheckpointTrustError(f"Verified manifest {key} mismatch")
    if not path.is_file() or path.stat().st_size != record.get("size_bytes"):
        raise CheckpointTrustError("Verified checkpoint missing or size mismatch")
    if file_sha256(path) != record.get("sha256"):
        raise CheckpointTrustError("Verified checkpoint SHA256 mismatch")
    payload = load_payload(path, variant=variant, source_sha256=source_sha256)
    collector = unpack(payload["collector"])
    if (payload.get("seed") != int(seed) or collector.get("step_count") != record.get("step")
            or collector.get("update_count") != record.get("update_count")
            or len(collector["buffers"][0]["obs"]) != record.get("collector_cursor")):
        raise CheckpointTrustError("Verified checkpoint counters or collector mismatch")
    return record, payload


def discover_latest_verified(run_root: Path, *, variant: str, seed: int,
                             source_sha256: str) -> Path:
    record = read_verified_record(run_root)
    if record is None:
        raise CheckpointTrustError("No verified interim manifest for automatic resume")
    path = Path(run_root).resolve() / record["checkpoint_relpath"]
    if path.resolve().parent != (Path(run_root).resolve() / "checkpoints"):
        raise CheckpointTrustError("Verified manifest checkpoint path escapes interim namespace")
    verify_published_checkpoint(path, variant=variant, seed=seed,
                                source_sha256=source_sha256)
    return path


def quarantine_file(path: Path, run_root: Path, *, reason: str) -> dict | None:
    """Move only a caller-owned candidate into a named quarantine with an audit record."""
    path = Path(path)
    if not path.exists():
        return None
    run_root = Path(run_root).resolve()
    if not path.resolve().is_relative_to(run_root):
        raise CheckpointTrustError("Refusing to quarantine a file outside the experiment")
    folder = run_root / "checkpoints/quarantine"
    folder.mkdir(parents=True, exist_ok=True)
    digest, size = file_sha256(path), path.stat().st_size
    destination = folder / f"{path.stem}_{digest[:16]}.pt"
    if destination.exists():
        destination = folder / f"{path.stem}_{digest[:16]}_{os.urandom(4).hex()}.pt"
    os.replace(path, destination)
    fsync_directory(path.parent)
    fsync_directory(folder)
    info = {"source_path": str(path), "quarantine_path": str(destination),
            "sha256": digest, "size_bytes": size, "reason": reason,
            "quarantined_utc": datetime.now(timezone.utc).isoformat()}
    save_json_atomic(destination.with_suffix(".json"), info)
    return info


def reconcile_interim_orphans(run_root: Path, *, variant: str, seed: int,
                              source_sha256: str) -> list[dict]:
    """Quarantine files owned by this protocol that are not manifest-selected."""
    run_root = Path(run_root).resolve()
    folder = run_root / "checkpoints"
    record = read_verified_record(run_root)
    if record is None:
        owned = list(folder.glob("interim_step_*.pt")) + list((folder / "staging").glob("candidate_*.pt"))
        if owned and (run_root / "checkpoint_events.jsonl").exists():
            for path in owned:
                quarantine_file(path, run_root, reason="uncommitted_run_without_manifest")
        if owned:
            raise CheckpointTrustError("Interim files exist without a verified manifest")
        return []
    chosen = discover_latest_verified(run_root, variant=variant, seed=seed,
                                      source_sha256=source_sha256)
    moved = []
    for path in sorted(folder.glob("interim_step_*.pt")):
        if path.resolve() != chosen.resolve():
            moved.append(quarantine_file(path, run_root, reason="orphan_after_verified_commit"))
    for path in sorted((folder / "staging").glob("candidate_*.pt")):
        moved.append(quarantine_file(path, run_root, reason="uncommitted_staging_candidate"))
    return moved


def pack(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": torch.from_numpy(np.ascontiguousarray(value.copy())),
                "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return {f.name: pack(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return {"__tuple__": [pack(x) for x in value]}
    if isinstance(value, list):
        return [pack(x) for x in value]
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise UnsupportedExactResume(f"Unserializable exact-state field: {type(value).__name__}")


def unpack(value: Any) -> Any:
    if isinstance(value, dict) and "__ndarray__" in value:
        return value["__ndarray__"].cpu().numpy().astype(np.dtype(value["dtype"])).reshape(value["shape"]).copy()
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(unpack(x) for x in value["__tuple__"])
    if isinstance(value, list):
        return [unpack(x) for x in value]
    if isinstance(value, dict):
        return {k: unpack(v) for k, v in value.items()}
    return value


def rng_state() -> dict:
    return pack({"python": random.getstate(), "numpy_global": np.random.get_state(),
                 "torch_cpu": torch.get_rng_state(),
                 "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})


def restore_rng(payload: dict) -> None:
    state = unpack(payload)
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _restore_dataclass(obj: Any, state: dict) -> None:
    names = {f.name for f in fields(obj)}
    if set(state) != names:
        raise UnsupportedExactResume(f"Dataclass state mismatch for {type(obj).__name__}")
    for name in names:
        setattr(obj, name, state[name])


def snapshot_env(env) -> dict:
    base = env.unwrapped
    sim = base.sim
    if sim._backend_type != "python" or type(sim.process).__name__ != "PythonTEProcess":
        raise UnsupportedExactResume(f"Only current Python TEP backend has a field-complete snapshot: {sim._backend_type}")
    if sim._detectors or sim._detection_results:
        raise UnsupportedExactResume("TEP detector state is not part of this exact snapshot contract")
    process = sim.process
    process_state = {k: v for k, v in vars(process).items() if k not in ("state", "disturbances")}
    sim_state = {k: v for k, v in vars(sim).items()
                 if k not in ("process", "controller", "control_mode", "_detectors", "_detection_results")}
    wrapper_state = {"_last_mv": env._last_mv, "_prev_mv": env._prev_mv,
                     "delta_mv_max": env.delta_mv_max, "reset_reward": env.reset_reward}
    base_state = {"_step": base._step, "_mv": base._mv,
                  "np_random_state": base._np_random.bit_generator.state,
                  "np_random_seed": base._np_random_seed}
    return pack({"backend": "python", "process": process_state, "sim": sim_state,
                 "controller_mv_values": sim.controller.mv_values,
                 "base": base_state, "wrapper": wrapper_state})


def restore_env(env, payload: dict) -> None:
    state = unpack(payload)
    if state["backend"] != "python":
        raise UnsupportedExactResume("Wrong TEP backend in checkpoint")
    base = env.unwrapped
    sim = base.sim
    if sim._backend_type != "python" or type(sim.process).__name__ != "PythonTEProcess":
        raise UnsupportedExactResume("Current environment backend differs from checkpoint")
    process = sim.process
    expected = set(vars(process)) - {"state", "disturbances"}
    if set(state["process"]) != expected:
        raise UnsupportedExactResume("TEP process field inventory changed")
    for k, v in state["process"].items():
        if is_dataclass(getattr(process, k)):
            _restore_dataclass(getattr(process, k), v)
        else:
            setattr(process, k, v)
    expected_sim = set(vars(sim)) - {"process", "controller", "control_mode", "_detectors", "_detection_results"}
    if set(state["sim"]) != expected_sim:
        raise UnsupportedExactResume("TEP simulator field inventory changed")
    for k, v in state["sim"].items():
        setattr(sim, k, v)
    sim.controller.mv_values = state["controller_mv_values"]
    base._step = state["base"]["_step"]
    base._mv = state["base"]["_mv"]
    base._np_random.bit_generator.state = state["base"]["np_random_state"]
    base._np_random_seed = state["base"]["np_random_seed"]
    env._last_mv = state["wrapper"]["_last_mv"]
    env._prev_mv = state["wrapper"]["_prev_mv"]
    if env.delta_mv_max != state["wrapper"]["delta_mv_max"] or env.reset_reward != state["wrapper"]["reset_reward"]:
        raise UnsupportedExactResume("TEP action/reward wrapper configuration changed")


def checkpoint_payload(trainer, env, obs_rms_list, collector: dict, *, seed: int, source_sha256: str,
                       runtime_config: dict) -> dict:
    return {"format": FORMAT, "contract_sha256": CONTRACT_SHA256, "variant": trainer.variant,
            "source_sha256": source_sha256, "seed": int(seed),
            "runtime_config": pack(runtime_config),
            "packages": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                         "tep": importlib.metadata.version("tep")},
            "trainer": pack(trainer.checkpoint_state()),
            "obs_rms": pack([vars(rms) for rms in obs_rms_list]),
            "collector": pack(collector), "environment": snapshot_env(env), "rng": rng_state()}


def save_atomic(path: Path, payload: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".train_resume_", suffix=".pt", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_payload(path: Path, *, variant: str, source_sha256: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT or payload.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("Wrong Phase 4 checkpoint format or contract")
    if payload.get("variant") != variant or payload.get("source_sha256") != source_sha256:
        raise ValueError("Wrong variant or source hash")
    if payload.get("packages") != {"torch": str(torch.__version__), "numpy": str(np.__version__),
                                   "tep": importlib.metadata.version("tep")}:
        raise ValueError("Runtime package version mismatch")
    for key in ("trainer", "obs_rms", "collector", "environment", "rng", "runtime_config"):
        if key not in payload:
            raise ValueError(f"Missing exact-resume field: {key}")
    return payload


def restore_checkpoint(payload: dict, trainer, env, obs_rms_list) -> dict:
    trainer.load_checkpoint_state(unpack(payload["trainer"]))
    rms_state = unpack(payload["obs_rms"])
    if len(rms_state) != len(obs_rms_list):
        raise ValueError("ObsRMS state count mismatch")
    for rms, state in zip(obs_rms_list, rms_state):
        if set(vars(rms)) != set(state):
            raise ValueError("ObsRMS field inventory changed")
        if state["mean"].shape != rms.mean.shape or state["var"].shape != rms.var.shape:
            raise ValueError("ObsRMS agent ordering or shape mismatch")
        vars(rms).update(state)
    restore_env(env, payload["environment"])
    restore_rng(payload["rng"])
    return unpack(payload["collector"])
