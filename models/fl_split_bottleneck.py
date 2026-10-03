from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _hadamard(order: int) -> torch.Tensor:
    """Return the normalized Walsh-Hadamard matrix of ``order``.

    The Stage 4A bottleneck needs a nested orthogonal basis so that widths 64,
    32, and 16 can share a single deterministic construction. Hadamard order is
    fixed to powers of two and is therefore a good fit for the 64-d BI path.
    """

    if order <= 0 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {order}")
    basis = torch.tensor([[1.0]], dtype=torch.float32)
    size = 1
    while size < order:
        top = torch.cat([basis, basis], dim=1)
        bottom = torch.cat([basis, -basis], dim=1)
        basis = torch.cat([top, bottom], dim=0)
        size *= 2
    return basis / math.sqrt(float(order))


def _basis_hash(basis: torch.Tensor, *, seed: int, bottleneck_dim: int) -> str:
    digest = hashlib.sha256()
    tensor = basis.detach().cpu().contiguous().to(dtype=torch.float32)
    digest.update(str(int(seed)).encode("utf-8"))
    digest.update(str(int(bottleneck_dim)).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class NestedOrthogonalBottleneck(nn.Module):
    """Deterministic 64->k->64 bottleneck with a nested orthogonal basis.

    The module is intentionally interface-preserving: the compressor emits the
    transmitted representation ``z_k`` and the decompressor reconstructs the
    64-d feature that the unchanged downstream critic expects.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        bottleneck_dim: int = 64,
        *,
        seed: int = 0,
    ):
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if int(bottleneck_dim) <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        if int(bottleneck_dim) > int(feature_dim):
            raise ValueError(
                f"bottleneck_dim must be <= feature_dim, got bottleneck_dim={bottleneck_dim} feature_dim={feature_dim}"
            )
        if int(feature_dim) != 64:
            raise ValueError(
                f"Stage 4A bottleneck is defined for the 64-d BI path, got feature_dim={feature_dim}"
            )

        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.initialization_seed = int(seed)

        basis = _hadamard(self.feature_dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.initialization_seed)
        # Deterministic column-wise sign flips preserve orthogonality and the
        # nested-prefix property while still giving the seed a real effect.
        column_signs = torch.where(
            torch.rand(self.feature_dim, generator=generator) >= 0.5,
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(-1.0, dtype=torch.float32),
        )
        basis = basis * column_signs.unsqueeze(0)
        self.register_buffer("projection_basis", basis, persistent=True)
        self.projection_basis_hash = _basis_hash(
            basis,
            seed=self.initialization_seed,
            bottleneck_dim=self.bottleneck_dim,
        )

        self.compressor = nn.Linear(self.feature_dim, self.bottleneck_dim, bias=False)
        self.decompressor = nn.Linear(self.bottleneck_dim, self.feature_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            basis_rows = self.projection_basis[: self.bottleneck_dim, :]
            self.compressor.weight.copy_(basis_rows)
            self.decompressor.weight.copy_(basis_rows.t())

    @property
    def transmitted_dim(self) -> int:
        return self.bottleneck_dim

    @property
    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstructed_64d, transmitted_z)``.

        ``features`` may be 2D or 3D; the linear layers operate on the last
        dimension and preserve the leading batch/sequence dimensions.
        """

        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected last dim {self.feature_dim}, got {features.shape[-1]}")
        transmitted = self.compressor(features)
        reconstructed = self.decompressor(transmitted)
        return reconstructed, transmitted

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Return the transmitted representation ``z_k``."""

        return self.compressor(features)

    def decode(self, transmitted: torch.Tensor) -> torch.Tensor:
        """Reconstruct the 64-d feature from ``z_k``."""

        if transmitted.shape[-1] != self.bottleneck_dim:
            raise ValueError(f"Expected last dim {self.bottleneck_dim}, got {transmitted.shape[-1]}")
        return self.decompressor(transmitted)

    def state_summary(self) -> Dict[str, object]:
        return {
            "feature_dim": self.feature_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "initialization_seed": self.initialization_seed,
            "projection_basis_hash": self.projection_basis_hash,
            "compressor_hash": _basis_hash(self.compressor.weight.data, seed=self.initialization_seed, bottleneck_dim=self.bottleneck_dim),
            "decompressor_hash": _basis_hash(self.decompressor.weight.data, seed=self.initialization_seed, bottleneck_dim=self.bottleneck_dim),
        }


class BottleneckFedAvgView(nn.Module):
    """Ephemeral wrapper used for FedAvg over the BI extractor + bottleneck."""

    def __init__(self, bi_extractor: nn.Module, bottleneck: NestedOrthogonalBottleneck):
        super().__init__()
        self.bi_extractor = bi_extractor
        self.bottleneck = bottleneck

    def forward(self, *args, **kwargs):  # pragma: no cover - FedAvg wrapper is state-only.
        raise NotImplementedError("BottleneckFedAvgView is a state-dict wrapper only")

