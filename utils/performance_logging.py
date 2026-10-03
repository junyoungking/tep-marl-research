"""Algorithm-neutral online performance logging.

``PerformanceTracker`` deliberately owns all aggregation, reset, tag, and
flush timing for its six performance scalars.  Callers only supply one raw
environment transition at a time; they must not pre-aggregate these values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass(frozen=True)
class CompletedEpisode:
    """Episode accounting emitted when a terminal transition is recorded."""

    return_value: float
    length: int


@dataclass
class PerformanceTracker:
    """Emit transition-window and completed-episode performance scalars.

    Required transition inputs are the environment reward, raw production and
    cost (each may be ``None``), shutdown flag, done flag, and the already
    incremented environment/global step.  ``done`` is the caller's combined
    terminated-or-truncated condition.

    At every ``log_every`` step the tracker writes the means of values observed
    since the preceding flush, clears those windows, and flushes the writer.
    A completed episode is written after that boundary flush, matching the
    historical write order when both occur on the same transition.  ``flush``
    writes any residual raw window but intentionally does not flush the writer;
    that lifecycle action remains with the caller.
    """

    writer: Any
    log_every: int
    episode_return: float = 0.0
    episode_length: int = 0
    prod_buffer: List[float] = field(default_factory=list)
    cost_buffer: List[float] = field(default_factory=list)
    profit_buffer: List[float] = field(default_factory=list)
    shutdown_buffer: List[float] = field(default_factory=list)
    profit_history: List[float] = field(default_factory=list)
    shutdown_history: List[float] = field(default_factory=list)
    episode_return_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.log_every) <= 0:
            raise ValueError("log_every must be positive")
        self.log_every = int(self.log_every)

    def state_dict(self) -> dict:
        """Primitive observer state; the writer is deliberately not serialized."""
        from dataclasses import fields
        return {f.name: list(getattr(self, f.name)) if isinstance(getattr(self, f.name), list)
                else getattr(self, f.name) for f in fields(self) if f.name != "writer"}

    def load_state_dict(self, state: dict) -> None:
        expected = set(self.state_dict())
        if set(state) != expected or state["log_every"] != self.log_every:
            raise ValueError("PerformanceTracker state schema/window mismatch")
        if not isinstance(state["episode_length"], int) or state["episode_length"] < 0:
            raise ValueError("Invalid PerformanceTracker episode length")
        for name, value in state.items():
            current = getattr(self, name)
            if isinstance(current, list):
                if not isinstance(value, list):
                    raise ValueError(f"PerformanceTracker list missing: {name}")
                value = list(value)
            setattr(self, name, value)

    def record_transition(
        self,
        *,
        step: int,
        reward: float,
        raw_prod: Any,
        raw_cost: Any,
        shutdown: Any,
        done: bool,
    ) -> Optional[CompletedEpisode]:
        """Record one transition and return episode accounting at ``done``."""

        self.episode_return += float(reward)
        self.episode_length += 1

        if raw_prod is not None:
            self.prod_buffer.append(float(raw_prod))
        if raw_cost is not None:
            self.cost_buffer.append(float(raw_cost))
        if raw_prod is not None and raw_cost is not None:
            # Preserve the historical operation order exactly.
            profit_value = float(raw_prod) - float(raw_cost)
            self.profit_buffer.append(profit_value)
            self.profit_history.append(profit_value)
        shutdown_value = float(bool(shutdown))
        self.shutdown_buffer.append(shutdown_value)
        self.shutdown_history.append(shutdown_value)

        if int(step) % self.log_every == 0:
            self.flush(step=int(step))
            self.writer.flush()

        if not done:
            return None
        completed = CompletedEpisode(float(self.episode_return), int(self.episode_length))
        self.writer.add_scalar("episode/return", self.episode_return, int(step))
        self.writer.add_scalar("episode/length", self.episode_length, int(step))
        self.episode_return_history.append(float(self.episode_return))
        self.episode_return = 0.0
        self.episode_length = 0
        return completed

    def flush(self, *, step: int) -> None:
        """Write and reset the current raw metric buffers, if non-empty."""

        if self.prod_buffer:
            self.writer.add_scalar("raw/prod", float(np.mean(self.prod_buffer)), int(step))
        if self.cost_buffer:
            self.writer.add_scalar("raw/cost", float(np.mean(self.cost_buffer)), int(step))
        if self.profit_buffer:
            self.writer.add_scalar("raw/profit", float(np.mean(self.profit_buffer)), int(step))
        if self.shutdown_buffer:
            self.writer.add_scalar("raw/shutdown_rate", float(np.mean(self.shutdown_buffer)), int(step))
        self.prod_buffer.clear()
        self.cost_buffer.clear()
        self.profit_buffer.clear()
        self.shutdown_buffer.clear()
