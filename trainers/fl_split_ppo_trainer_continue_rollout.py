"""Compatibility alias for fixed-horizon split-FL PPO.

`SplitFedPPOTrainer` is the canonical fl_v2 implementation.  It now owns
done-boundary BI segmentation, hidden resets, actor epoch looping, and KL
early-stop semantics directly, so historical imports of the continue-rollout
variant can inherit the same behavior without duplicating update logic.
"""

from trainers.fl_split_ppo_trainer import SplitFedPPOTrainer


class SplitFedContinueRolloutPPOTrainer(SplitFedPPOTrainer):
    """Backward-compatible name for the fl_v2 base split-FL trainer."""

    pass
