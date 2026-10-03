"""BI cut-layer privacy defense modules inspired by DRAVL.

This module is intentionally training-only. It protects BI/cut-layer tensors with
adversarial reconstruction (AR), noise regularization (NR), and distance
correlation (dCor) losses. The independent evaluation attacker/decoder described
in the DRAVL paper is not implemented here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    """Gradient reversal layer primitive.

    Forward is identity. Backward multiplies the upstream gradient by ``-lambd``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):  # type: ignore[override]
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


class GradReverseLambda(_GradReverse):
    """Public GRL primitive for client-side FL-DRAVL.

    This is intentionally an alias of the existing implementation: forward is
    identity and backward multiplies the upstream gradient by ``-lambd``.
    """


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal while leaving the forward value unchanged."""

    return _GradReverse.apply(x, lambd)


def grad_reverse_lambda(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply the public FL-DRAVL gradient reversal primitive."""

    return GradReverseLambda.apply(x, lambd)


def distance_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Biased distance-correlation estimator for two sample matrices.

    Args:
        x: Tensor shaped ``(N, Dx)``.
        y: Tensor shaped ``(N, Dy)``. ``Dx`` and ``Dy`` may differ.
        eps: Numerical stabilizer.

    Returns:
        Scalar tensor in a finite range close to ``[0, 1]`` for ordinary inputs.
    """

    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(
            f"distance_correlation expects 2D tensors, got x={tuple(x.shape)}, y={tuple(y.shape)}"
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"distance_correlation expects matching sample counts, got {x.shape[0]} and {y.shape[0]}"
        )
    if x.shape[0] < 2:
        # Degenerate but finite; keeps validation/smoke runs robust for tiny inputs.
        return x.new_tensor(0.0)

    a = torch.cdist(x, x)
    b = torch.cdist(y, y)
    A = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
    B = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    denom = torch.sqrt(dvar_x.clamp_min(0) * dvar_y.clamp_min(0)) + eps
    return torch.sqrt(dcov2.clamp_min(0) / denom + eps)


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _fork_rng_from_tensor(tensor: torch.Tensor):
    """Fork RNG state for the device that owns ``tensor``.

    The privacy module samples logging-time noise and dCor subsamples. Those
    samples should not perturb the caller's global RNG stream, otherwise a
    privacy-attached run can diverge from the fedavg10 baseline even when the
    privacy coefficients are zero.
    """

    devices: List[int] = []
    if tensor.is_cuda:
        device_index = tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        devices = [int(device_index)]
    return torch.random.fork_rng(devices=devices)


class BIPrivacyDefense(nn.Module):
    """DRAVL-style training-time privacy defense for BI tensors.

    ``reconstructors`` are training-time adversaries ``R``. They are updated by
    AR only. NR and dCor must route gradients to BI/F while leaving ``R`` frozen.
    """

    def __init__(
        self,
        bi_dim: int,
        target_dims: Sequence[int],
        recon_hidden: Sequence[int] = (128, 128),
        grl_lambda: float = 1.0,
        noise_std: float = 1.0,
        dcor_max_samples: int = 2048,
        eps: float = 1e-8,
    ):
        super().__init__()
        if bi_dim <= 0:
            raise ValueError(f"bi_dim must be positive, got {bi_dim}")
        if len(target_dims) == 0:
            raise ValueError("target_dims must contain at least one agent target dimension")
        if any(int(dim) <= 0 for dim in target_dims):
            raise ValueError(f"all target_dims must be positive, got {target_dims}")

        self.bi_dim = int(bi_dim)
        self.target_dims = [int(dim) for dim in target_dims]
        self.grl_lambda = float(grl_lambda)
        self.noise_std = float(noise_std)
        self.dcor_max_samples = int(dcor_max_samples)
        self.eps = float(eps)
        with torch.random.fork_rng(devices=[]):
            self.reconstructors = nn.ModuleList(
                [_build_mlp(self.bi_dim, recon_hidden, target_dim) for target_dim in self.target_dims]
            )

    @staticmethod
    def _flat(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        if x.dim() == 2:
            return x
        raise ValueError(f"Expected 2D or 3D tensor, got shape={tuple(x.shape)}")

    def _validate_inputs(
        self,
        bi_list: Sequence[torch.Tensor],
        target_list: Sequence[torch.Tensor],
    ) -> None:
        if len(bi_list) != len(self.reconstructors):
            raise ValueError(
                f"Expected {len(self.reconstructors)} BI tensors, got {len(bi_list)}"
            )
        if len(target_list) != len(self.reconstructors):
            raise ValueError(
                f"Expected {len(self.reconstructors)} target tensors, got {len(target_list)}"
            )

    def _frozen_recon(self, i: int, bi_flat: torch.Tensor) -> torch.Tensor:
        """Run reconstructor ``R[i]`` with detached parameters but live BI grads."""

        out = bi_flat
        for layer in self.reconstructors[i]:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach()
                bias = layer.bias.detach() if layer.bias is not None else None
                out = F.linear(out, weight, bias)
            else:
                out = layer(out)
        return out

    def ar_loss(
        self,
        bi_list: Sequence[torch.Tensor],
        target_list: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, float]:
        """Adversarial reconstruction loss with GRL.

        Returns ``(loss, recon_mse_float)``. ``R`` receives normal MSE gradients;
        BI/F receives reversed gradients through GRL.
        """

        self._validate_inputs(bi_list, target_list)
        losses: List[torch.Tensor] = []
        mse_values: List[float] = []
        for i, (bi, target) in enumerate(zip(bi_list, target_list)):
            bi_flat = self._flat(bi)
            target_flat = self._flat(target).detach()
            if bi_flat.shape[-1] != self.bi_dim:
                raise ValueError(f"BI dim mismatch for agent {i}: expected {self.bi_dim}, got {bi_flat.shape[-1]}")
            if target_flat.shape[-1] != self.target_dims[i]:
                raise ValueError(
                    f"Target dim mismatch for agent {i}: expected {self.target_dims[i]}, got {target_flat.shape[-1]}"
                )
            recon = self.reconstructors[i](grad_reverse(bi_flat, self.grl_lambda))
            loss_i = F.mse_loss(recon, target_flat)
            losses.append(loss_i)
            mse_values.append(float(loss_i.detach().item()))
        loss = torch.stack(losses).mean()
        return loss, float(sum(mse_values) / max(len(mse_values), 1))

    def nr_loss(
        self,
        bi_list: Sequence[torch.Tensor],
        target_list: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Noise regularization loss that updates BI/F but not R parameters."""

        self._validate_inputs(bi_list, target_list)
        losses: List[torch.Tensor] = []
        reference = self._flat(bi_list[0])
        with _fork_rng_from_tensor(reference):
            for i, (bi, target) in enumerate(zip(bi_list, target_list)):
                bi_flat = self._flat(bi)
                target_flat = self._flat(target)
                if target_flat.shape[-1] != self.target_dims[i]:
                    raise ValueError(
                        f"Target dim mismatch for agent {i}: expected {self.target_dims[i]}, got {target_flat.shape[-1]}"
                    )
                noise = torch.randn_like(target_flat) * self.noise_std
                recon = self._frozen_recon(i, bi_flat)
                losses.append(F.mse_loss(recon, noise))
        return torch.stack(losses).mean()

    def dcor_loss(
        self,
        bi_list: Sequence[torch.Tensor],
        target_list: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Log distance-correlation loss between target X and BI F(X)."""

        self._validate_inputs(bi_list, target_list)
        losses: List[torch.Tensor] = []
        reference = self._flat(bi_list[0])
        with _fork_rng_from_tensor(reference):
            for bi, target in zip(bi_list, target_list):
                bi_flat = self._flat(bi)
                target_flat = self._flat(target).detach()
                n_samples = bi_flat.shape[0]
                if target_flat.shape[0] != n_samples:
                    raise ValueError(
                        f"BI/target sample count mismatch: {n_samples} vs {target_flat.shape[0]}"
                    )
                if self.dcor_max_samples > 0 and n_samples > self.dcor_max_samples:
                    idx = torch.randperm(n_samples, device=bi_flat.device)[: self.dcor_max_samples]
                    bi_flat = bi_flat.index_select(0, idx)
                    target_flat = target_flat.index_select(0, idx)
                dcor = distance_correlation(target_flat, bi_flat, eps=self.eps)
                losses.append(torch.log(dcor + self.eps))
        return torch.stack(losses).mean()


class ClientBIPrivacyDefense(nn.Module):
    """Single-client BI cut-layer defense for split-FL DRAVL.

    Unlike ``BIPrivacyDefense``'s monolithic weighted objective, this class
    decouples reconstructor learning from defense strength:

    * AR trains ``R`` at full strength every call.
    * The BI/encoder-side AR defense pressure is controlled only by
      ``lambda_eff = grl_lambda * ar_coef`` through GRL.
    * NR and dCor remain weighted by their coefficients.
    """

    def __init__(
        self,
        bi_dim: int,
        target_dim: int,
        recon_hidden_dims: Sequence[int] = (128, 128),
        nr_reconstructor: Optional[nn.Module] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        if int(bi_dim) <= 0:
            raise ValueError(f"bi_dim must be positive, got {bi_dim}")
        if int(target_dim) <= 0:
            raise ValueError(f"target_dim must be positive, got {target_dim}")
        self.bi_dim = int(bi_dim)
        self.target_dim = int(target_dim)
        self.eps = float(eps)
        with torch.random.fork_rng(devices=[]):
            self.reconstructor = _build_mlp(self.bi_dim, recon_hidden_dims, self.target_dim)
        self.nr_reconstructor = nr_reconstructor

    @staticmethod
    def _flat(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        if x.dim() == 2:
            return x
        raise ValueError(f"Expected 2D or 3D tensor, got shape={tuple(x.shape)}")

    def _validate_flat(self, bi_flat: torch.Tensor, target_flat: torch.Tensor) -> None:
        if bi_flat.dim() != 2 or target_flat.dim() != 2:
            raise ValueError(
                f"ClientBIPrivacyDefense expects flat 2D tensors, got bi={tuple(bi_flat.shape)}, "
                f"target={tuple(target_flat.shape)}"
            )
        if bi_flat.shape[0] != target_flat.shape[0]:
            raise ValueError(f"Sample count mismatch: bi={bi_flat.shape[0]} vs target={target_flat.shape[0]}")
        if bi_flat.shape[-1] != self.bi_dim:
            raise ValueError(f"BI dim mismatch: expected {self.bi_dim}, got {bi_flat.shape[-1]}")
        if target_flat.shape[-1] != self.target_dim:
            raise ValueError(f"Target dim mismatch: expected {self.target_dim}, got {target_flat.shape[-1]}")

    @staticmethod
    def _run_with_frozen_params(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        params = list(module.parameters())
        old_requires_grad = [p.requires_grad for p in params]
        try:
            for p in params:
                p.requires_grad_(False)
            return module(x)
        finally:
            for p, flag in zip(params, old_requires_grad):
                p.requires_grad_(flag)

    def _nr_module(self) -> nn.Module:
        return self.nr_reconstructor if self.nr_reconstructor is not None else self.reconstructor

    def _dcor_values(
        self,
        bi_flat: torch.Tensor,
        target_flat: torch.Tensor,
        dcor_max_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(dcor_max_samples) > 0 and bi_flat.shape[0] > int(dcor_max_samples):
            idx = torch.randperm(bi_flat.shape[0], device=bi_flat.device)[: int(dcor_max_samples)]
            bi_flat = bi_flat.index_select(0, idx)
            target_flat = target_flat.index_select(0, idx)
        dcor_raw = distance_correlation(target_flat.detach(), bi_flat, eps=self.eps)
        dcor_raw = dcor_raw.clamp(min=self.eps, max=1.0)
        dcor_log = torch.log(dcor_raw + self.eps)
        return dcor_raw, dcor_log

    def compute(
        self,
        bi_flat: torch.Tensor,
        target_flat: torch.Tensor,
        *,
        ar_coef: float,
        nr_coef: float,
        dcor_coef: float,
        grl_lambda: float,
        noise_std: float,
        dcor_max_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute client-side FL-DRAVL loss terms and logging tensors.

        ``backward_loss`` is the scalar that should be backpropagated on the
        client side. AR is deliberately unweighted for reconstructor learning;
        the BI-side defense pressure is controlled by ``lambda_eff`` in GRL.
        """

        bi_flat = self._flat(bi_flat)
        target_flat = self._flat(target_flat).detach()
        self._validate_flat(bi_flat, target_flat)
        with _fork_rng_from_tensor(bi_flat):
            ar_coef_t = float(ar_coef)
            nr_coef_t = float(nr_coef)
            dcor_coef_t = float(dcor_coef)
            lambda_eff = float(grl_lambda) * ar_coef_t

            recon = self.reconstructor(grad_reverse_lambda(bi_flat, lambda_eff))
            ar_raw = F.mse_loss(recon, target_flat)

            nr_module = self._nr_module()
            if nr_coef_t != 0.0:
                noise = torch.randn_like(target_flat) * float(noise_std)
                nr_raw = F.mse_loss(self._run_with_frozen_params(nr_module, bi_flat), noise)
                nr_weighted = nr_raw * nr_coef_t
            else:
                with torch.no_grad():
                    noise = torch.randn_like(target_flat) * float(noise_std)
                    nr_raw = F.mse_loss(nr_module(bi_flat.detach()), noise)
                nr_weighted = ar_raw.new_zeros(())

            if dcor_coef_t != 0.0:
                dcor_raw, dcor_log = self._dcor_values(bi_flat, target_flat, dcor_max_samples)
                dcor_weighted = dcor_log * dcor_coef_t
            else:
                with torch.no_grad():
                    dcor_raw, dcor_log = self._dcor_values(bi_flat.detach(), target_flat, dcor_max_samples)
                dcor_weighted = ar_raw.new_zeros(())

            backward_loss = ar_raw + nr_weighted + dcor_weighted
            return {
                "backward_loss": backward_loss,
                "loss/priv_ar_raw": ar_raw,
                "loss/priv_nr_raw": nr_raw,
                "privacy/dcor_raw": dcor_raw,
                "privacy/dcor_log": dcor_log,
                "loss/priv_nr_weighted": nr_weighted,
                "loss/priv_dcor_weighted": dcor_weighted,
                "privacy/ar_lambda_eff": ar_raw.new_tensor(lambda_eff),
                "loss/privacy_backward_total": backward_loss,
                "privacy/recon_mse": ar_raw.detach(),
            }
