from __future__ import annotations

import math
from typing import Any, Dict, List


FIXED = "fixed"
HOLD_THEN_LINEAR = "hold_then_linear"
SUPPORTED_REGIMES = (FIXED, HOLD_THEN_LINEAR)


def actor_lr_at_step(
    *,
    regime: str,
    start_lr: float,
    final_lr: float | None,
    hold_steps: int | None,
    total_steps: int,
    step: int,
) -> float:
    """Return the predeclared actor LR for a global environment step."""
    if regime not in SUPPORTED_REGIMES:
        raise ValueError(f"Unsupported actor LR regime: {regime!r}")
    if not math.isfinite(float(start_lr)) or float(start_lr) <= 0:
        raise ValueError(f"start_lr must be positive and finite, got {start_lr}")
    if int(total_steps) <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if int(step) < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if regime == FIXED:
        resolved_final = float(start_lr if final_lr is None else final_lr)
        if not math.isclose(resolved_final, float(start_lr), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("fixed actor LR requires final_lr == start_lr")
        if hold_steps not in (None, 0):
            raise ValueError("fixed actor LR does not accept hold_steps")
        return float(start_lr)

    if final_lr is None or not math.isfinite(float(final_lr)) or float(final_lr) <= 0:
        raise ValueError(f"hold_then_linear requires a positive finite final_lr, got {final_lr}")
    if hold_steps is None or int(hold_steps) < 0 or int(hold_steps) >= int(total_steps):
        raise ValueError(f"hold_steps must be within [0, {total_steps}), got {hold_steps}")
    if int(step) <= int(hold_steps):
        return float(start_lr)
    if int(step) >= int(total_steps):
        return float(final_lr)
    progress = (float(step) - float(hold_steps)) / (float(total_steps) - float(hold_steps))
    return float(start_lr) + progress * (float(final_lr) - float(start_lr))


def optimizer_group_lrs(optimizers: Any) -> List[List[float]]:
    return [[float(group["lr"]) for group in optimizer.param_groups] for optimizer in optimizers]


def apply_actor_lr(
    trainer: Any,
    *,
    actor_lr: float,
    expected_critic_lr: float,
) -> Dict[str, List[List[float]]]:
    """Apply LR only to actor optimizers and prove critic LR was not changed."""
    if not math.isfinite(float(actor_lr)) or float(actor_lr) <= 0:
        raise ValueError(f"actor_lr must be positive and finite, got {actor_lr}")
    critic_before = optimizer_group_lrs(trainer.critic_opts)
    if any(
        not math.isclose(lr, float(expected_critic_lr), rel_tol=0.0, abs_tol=1e-15)
        for groups in critic_before
        for lr in groups
    ):
        raise RuntimeError(f"critic optimizer LR mismatch before actor LR update: {critic_before}")

    for optimizer in trainer.actor_opts:
        for group in optimizer.param_groups:
            group["lr"] = float(actor_lr)

    actor_after = optimizer_group_lrs(trainer.actor_opts)
    critic_after = optimizer_group_lrs(trainer.critic_opts)
    if any(
        not math.isclose(lr, float(expected_critic_lr), rel_tol=0.0, abs_tol=1e-15)
        for groups in critic_after
        for lr in groups
    ):
        raise RuntimeError(f"actor LR update mutated critic optimizer LR: {critic_after}")
    return {
        "actor_optimizer_lrs": actor_after,
        "critic_optimizer_lrs_before": critic_before,
        "critic_optimizer_lrs_after": critic_after,
    }
