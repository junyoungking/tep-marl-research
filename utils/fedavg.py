from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


def _as_float_weights(num_items: int, weights: Optional[Sequence[float]], device: torch.device) -> torch.Tensor:
    if weights is None:
        return torch.full((num_items,), 1.0 / float(num_items), dtype=torch.float32, device=device)
    if len(weights) != num_items:
        raise ValueError(f"Expected {num_items} FedAvg weights, got {len(weights)}")
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if torch.any(w < 0):
        raise ValueError("FedAvg weights must be non-negative")
    total = w.sum()
    if float(total.item()) <= 0.0:
        raise ValueError("FedAvg weights must sum to a positive value")
    return w / total


def average_state_dicts(state_dicts: Sequence[OrderedDict], weights: Optional[Sequence[float]] = None) -> OrderedDict:
    """Return a weighted average state_dict with strict key/shape validation."""
    if len(state_dicts) == 0:
        raise ValueError("Cannot FedAvg an empty state_dict list")
    ref_keys = list(state_dicts[0].keys())
    for idx, sd in enumerate(state_dicts):
        if list(sd.keys()) != ref_keys:
            raise ValueError(f"State dict {idx} has different keys")
    first_tensor = next((v for v in state_dicts[0].values() if torch.is_tensor(v)), None)
    if first_tensor is None:
        raise ValueError("State dict has no tensors to average")
    w = _as_float_weights(len(state_dicts), weights, first_tensor.device)
    averaged = OrderedDict()
    for key in ref_keys:
        ref = state_dicts[0][key]
        if not torch.is_tensor(ref):
            averaged[key] = ref
            continue
        for idx, sd in enumerate(state_dicts[1:], start=1):
            val = sd[key]
            if not torch.is_tensor(val):
                raise ValueError(f"State dict {idx} key {key} is not a tensor")
            if tuple(val.shape) != tuple(ref.shape):
                raise ValueError(f"Shape mismatch for key {key}: {tuple(ref.shape)} vs {tuple(val.shape)}")
            if val.dtype != ref.dtype:
                raise ValueError(f"Dtype mismatch for key {key}: {ref.dtype} vs {val.dtype}")
        if torch.is_floating_point(ref) or torch.is_complex(ref):
            acc = torch.zeros_like(ref)
            for coeff, sd in zip(w.to(ref.device, dtype=ref.dtype), state_dicts):
                acc = acc + sd[key].to(device=ref.device, dtype=ref.dtype) * coeff
            averaged[key] = acc
        else:
            # Non-floating buffers are required to be identical. Averaging counters would be misleading.
            for idx, sd in enumerate(state_dicts[1:], start=1):
                if not torch.equal(ref, sd[key].to(device=ref.device)):
                    raise ValueError(f"Non-floating state key {key} differs for state dict {idx}")
            averaged[key] = ref.clone()
    return averaged


def fedavg_modules(modules: Sequence[nn.Module], weights: Optional[Sequence[float]] = None) -> OrderedDict:
    """Average module state_dicts in-place. Optimizer state is intentionally untouched."""
    if len(modules) == 0:
        raise ValueError("Cannot FedAvg an empty module list")
    state_dicts = [m.state_dict() for m in modules]
    averaged = average_state_dicts(state_dicts, weights=weights)
    for module in modules:
        module.load_state_dict(averaged)
    return averaged


def parameters_share_storage(left: nn.Module, right: nn.Module) -> bool:
    """True if any parameters from two modules share underlying storage."""
    for p_left in left.parameters():
        for p_right in right.parameters():
            if p_left.data_ptr() == p_right.data_ptr():
                return True
    return False


def clone_state_dict(module: nn.Module) -> OrderedDict:
    return OrderedDict((k, v.detach().clone() if torch.is_tensor(v) else v) for k, v in module.state_dict().items())


def state_dict_allclose(left: OrderedDict, right: OrderedDict, atol: float = 1e-6) -> bool:
    if list(left.keys()) != list(right.keys()):
        return False
    for key in left:
        lv, rv = left[key], right[key]
        if torch.is_tensor(lv):
            if not torch.allclose(lv, rv.to(device=lv.device, dtype=lv.dtype), atol=atol):
                return False
        elif lv != rv:
            return False
    return True


def state_dict_any_changed(left: OrderedDict, right: OrderedDict, atol: float = 1e-8) -> bool:
    return not state_dict_allclose(left, right, atol=atol)


def parameter_ids(modules: Iterable[nn.Module]) -> List[int]:
    ids: List[int] = []
    for module in modules:
        ids.extend(id(p) for p in module.parameters())
    return ids
