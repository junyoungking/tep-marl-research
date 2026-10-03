from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_set_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(item) for item in paths}, key=lambda item: str(item)):
        digest.update(path.name.encode("utf-8"))
        digest.update(file_hash(path).encode("ascii"))
    return digest.hexdigest()


def rng_state_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        payload["torch_cuda"] = None
    return payload


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    python_state = payload.get("python")
    numpy_state = payload.get("numpy")
    torch_state = payload.get("torch")
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and payload.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def safe_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def safe_torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_state_path(bundle_root: Path) -> Path:
    return bundle_root / "checkpoint_state.pt"


def save_checkpoint_state(bundle_root: Path, payload: Mapping[str, Any]) -> Path:
    bundle_root.mkdir(parents=True, exist_ok=True)
    path = checkpoint_state_path(bundle_root)
    safe_torch_save(dict(payload), path)
    return path


def load_checkpoint_state(bundle_root: Path, map_location: Any = "cpu") -> Dict[str, Any]:
    path = checkpoint_state_path(bundle_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint state file: {path}")
    payload = safe_torch_load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint state must be a dict payload, got {type(payload).__name__}")
    return payload


def tensor_max_abs_diff(left: Any, right: Any) -> float:
    if torch.is_tensor(left) or torch.is_tensor(right):
        left_t = torch.as_tensor(left, dtype=torch.float64)
        right_t = torch.as_tensor(right, dtype=torch.float64)
        return float(torch.max(torch.abs(left_t - right_t)).item())
    left_a = np.asarray(left, dtype=np.float64)
    right_a = np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left_a - right_a)))


def nested_max_abs_diff(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        if keys != set(left) or keys != set(right):
            raise KeyError(f"Mapping key mismatch: left={sorted(left)} right={sorted(right)}")
        diffs = [nested_max_abs_diff(left[key], right[key]) for key in keys]
        return float(max(diffs) if diffs else 0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"Sequence length mismatch: {len(left)} != {len(right)}")
        diffs = [nested_max_abs_diff(l_item, r_item) for l_item, r_item in zip(left, right)]
        return float(max(diffs) if diffs else 0.0)
    if torch.is_tensor(left) or torch.is_tensor(right) or isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return tensor_max_abs_diff(left, right)
    if isinstance(left, (float, int, np.number)) and isinstance(right, (float, int, np.number)):
        return float(abs(float(left) - float(right)))
    return 0.0 if left == right else float("inf")


