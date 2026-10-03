"""Neutral, mode-aware TensorBoard logging composed with PerformanceTracker.

Trainers capture tensors/values only at their scientifically necessary timing;
this module owns diagnostic reductions, group ownership, and all writes.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from utils.performance_logging import PerformanceTracker


class TrainingMetricsLogger:
    """The shared BASE/MARL/FEDAVG/PRIVACY metric implementation."""

    def __init__(self, writer: Any, log_every: int, *, marl: bool, fedavg: bool, privacy: bool):
        self.writer, self.marl, self.fedavg, self.privacy = writer, bool(marl), bool(fedavg), bool(privacy)
        # This is deliberately composition, not a reimplementation of the six
        # validated performance metric calculations.
        self.performance = PerformanceTracker(writer, log_every)

    @staticmethod
    def mode_for_tag(tag: str) -> str:
        if tag.startswith(("fedavg/", "bi_fedavg/")):
            return "FEDAVG"
        if tag.startswith(("privacy/", "grad_norm/priv_defense_", "grad_norm/bi_encoder_", "loss/privacy", "loss/recon", "ratio/privacy", "debug/privacy_")):
            return "PRIVACY"
        if tag.startswith("agent_") and any(x in tag for x in ("/privacy/", "/loss/priv", "/loss/recon")):
            return "PRIVACY"
        if tag.startswith(("action/", "agent_", "critic/value_loss_agent_")):
            return "MARL"
        # These use split clients, the local BI/recurrent path, or a server
        # handoff.  A joint C0 actor/critic has no corresponding object.
        if tag in {
            "rollout/num_episodes", "rollout/mean_episode_len",
            "debug/actor_grad_norm", "debug/actor_param_delta",
            "debug/actor_param_delta_min_epoch", "debug/actor_logp_delta_post",
            "debug/actor_post_kl_available", "debug/server_advantage_owner",
            "debug/local_actor_reward_value_free", "debug/critic_gru_reset_enabled",
            "debug/critic_gru_done_resets", "debug/hidden_resets",
            "debug/critic_dL_dBI_requires_grad_rate",
            "debug/critic_local_changed_on_handoff",
            "debug/critic_server_changed_during_local_step",
            "debug/critic_server_changed_on_server_step",
            "debug/critic_server_grads_created_during_local",
        }:
            return "MARL"
        return "BASE"

    def enabled(self, mode: str) -> bool:
        return mode == "BASE" or (mode == "MARL" and self.marl) or (mode == "FEDAVG" and self.fedavg) or (mode == "PRIVACY" and self.privacy)

    def _write(self, tag: str, value: Any, step: int) -> None:
        if self.writer is not None and self.enabled(self.mode_for_tag(tag)):
            self.writer.add_scalar(tag, float(value), int(step))

    def record_transition(self, **kwargs: Any):
        return self.performance.record_transition(**kwargs)

    def flush(self, *, step: int) -> None:
        self.performance.flush(step=step)

    def record_action(self, *, step: int, action_parts: Any, mv_indices: Sequence[Sequence[int]], info: Any = None, delta_mv_max: Optional[float] = None, prefix: str = "action", log_every: int = 1) -> None:
        """Exact A0 action-policy/delta/absolute action diagnostic logic."""
        if not self.marl or int(step) <= 0 or int(log_every) <= 0 or int(step) % int(log_every):
            return
        groups = [np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=np.float32).reshape(-1) for x in action_parts]
        for agent, (actions, indices) in enumerate(zip(groups, mv_indices)):
            if len(actions) != len(indices):
                raise ValueError("action and MV group dimensions differ")
            for value, mv in zip(actions, indices):
                self._write(f"{prefix}/policy/agent_{agent}/mv_{int(mv)}/value", value, step)
        flat = np.concatenate(groups) if groups else np.empty(0, dtype=np.float32)
        if flat.size:
            self._write(f"{prefix}/policy_abs_mean", np.mean(np.abs(flat)), step)
            self._write(f"{prefix}/policy_max_abs", np.max(np.abs(flat)), step)
        if delta_mv_max is not None:
            deltas = [np.clip(group, -1.0, 1.0) * float(delta_mv_max) for group in groups]
            for agent, (values, indices) in enumerate(zip(deltas, mv_indices)):
                for value, mv in zip(values, indices):
                    self._write(f"{prefix}/delta/agent_{agent}/mv_{int(mv)}/value", value, step)
            if flat.size:
                self._write(f"{prefix}/delta_abs_mean", np.mean(np.abs(np.concatenate(deltas))), step)
        if isinstance(info, dict) and "mv" in info:
            try:
                full = np.asarray(info["mv"], dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                full = None
            if full is not None:
                values = []
                for mv in dict.fromkeys(int(x) for group in mv_indices for x in group):
                    if 0 <= mv - 1 < len(full):
                        value = float(full[mv - 1]); values.append(value)
                        self._write(f"{prefix}/absolute/mv_{mv}/value", value, step)
                if values:
                    self._write(f"{prefix}/absolute_mean", np.mean(values), step)

    def record_rollout(self, *, step: int, length: int, episodes: int,
                       skipped_short: Optional[bool] = None,
                       emit_length: bool = True) -> None:
        """Record runner-owned rollout facts.

        ``emit_length`` is an opt-in routing control for runners whose
        validated update producer owns ``rollout/length``.  The default keeps
        every existing caller and its event ordering unchanged.
        """
        if emit_length:
            self._write("rollout/length", length, step)
        self._write("rollout/episodes_per_update", episodes, step)
        if skipped_short is not None:
            self._write("rollout/skipped_short", float(bool(skipped_short)), step)

    @staticmethod
    def _scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def aggregate_actor_epochs(cls, rows: Sequence[Mapping[str, Any]], *, include_marl: bool = True) -> Dict[str, Any]:
        """A0's exact actor diagnostic reduction: last KL, otherwise mean."""
        if not rows:
            raise ValueError("Cannot aggregate empty actor epoch captures")
        last, out = rows[-1], {}
        last_keys = {"approx_kl", "approx_kl_post"}
        keys = ["actor_loss", "entropy", "approx_kl", "clipfrac", "approx_kl_post"]
        if include_marl:
            keys += ["actor_grad_norm", "actor_param_delta", "actor_logp_delta_post", "actor_post_kl_available"]
        for key in keys:
            values = [cls._scalar(row.get(key)) for row in rows]
            values = [x for x in values if x is not None and np.isfinite(x)]
            if values:
                out[key] = float(last[key]) if key in last_keys else float(np.mean(values))
        last_lists = {"per_agent_approx_kl", "per_agent_approx_kl_post"}
        if include_marl:
            for key in ("per_agent_actor_loss", "per_agent_entropy", "per_agent_approx_kl", "per_agent_approx_kl_post", "per_agent_clipfrac", "per_agent_ratio_std", "per_agent_actor_grad_norm", "per_agent_actor_param_delta"):
                if key in last:
                    out[key] = [float(x) for x in last[key]] if key in last_lists else [float(x) for x in np.asarray([r[key] for r in rows if key in r], dtype=np.float64).mean(axis=0)]
        out["actor_epochs_used"] = len(rows)
        out["approx_kl_epoch0"], out["approx_kl_last"] = float(rows[0].get("approx_kl", 0.0)), float(last.get("approx_kl", 0.0))
        if include_marl:
            deltas = [cls._scalar(row.get("actor_param_delta")) for row in rows]
            deltas = [x for x in deltas if x is not None and np.isfinite(x)]
            out["actor_param_delta_min_epoch"] = float(min(deltas)) if deltas else 0.0
        return out

    @classmethod
    def _critic_scalars(cls, diag: Mapping[str, Any]) -> Dict[str, float]:
        names = {"server_changed_on_server_step": "debug/critic_server_changed_on_server_step", "server_changed_during_local_step": "debug/critic_server_changed_during_local_step", "server_grads_created_during_local": "debug/critic_server_grads_created_during_local", "local_changed_on_handoff": "debug/critic_local_changed_on_handoff"}
        out = {dst: value for src, dst in names.items() if (value := cls._scalar(diag.get(src))) is not None}
        for i, value in enumerate(diag.get("value_losses", [])):
            if (scalar := cls._scalar(value)) is not None:
                out[f"critic/value_loss_agent_{i}"] = scalar
        flags = [cls._scalar(x) for x in diag.get("dL_dBI_requires_grad", [])]
        flags = [x for x in flags if x is not None]
        if flags:
            out["debug/critic_dL_dBI_requires_grad_rate"] = float(np.mean(flags))
        return out

    @staticmethod
    def _mean_logs(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        return {key: float(np.mean([float(row[key]) for row in rows if key in row])) for key in sorted({key for row in rows for key in row})}

    def compose_generic_update(self, *, actor_epochs: Sequence[Mapping[str, Any]],
                               critic_epochs: Sequence[Mapping[str, Any]],
                               rollout_length: int) -> Dict[str, Any]:
        """Compose the validated BASE PPO surface from raw epoch captures.

        This is the BASE portion of :meth:`compose_split_update`, factored so
        a non-split C0 runner can feed the same shared reducer without
        reconstructing any scalar formula in its trainer or runner.
        """
        if not critic_epochs:
            raise ValueError("Cannot compose empty critic epoch captures")
        actor = self.aggregate_actor_epochs(actor_epochs, include_marl=self.marl)
        critic, first = critic_epochs[-1], critic_epochs[0]
        return {
            "loss/actor": actor["actor_loss"],
            "loss/critic": float(critic["loss"]),
            "loss/critic_first_epoch": float(first["loss"]),
            "loss/critic_last_epoch": float(critic["loss"]),
            "loss/entropy": actor["entropy"],
            "train/approx_kl": actor["approx_kl"],
            "train/approx_kl_post": actor["approx_kl_post"],
            "train/actor_epochs_used": float(actor["actor_epochs_used"]),
            "train/critic_epochs_used": float(len(critic_epochs)),
            "train/approx_kl_epoch0": actor["approx_kl_epoch0"],
            "train/approx_kl_last": actor["approx_kl_last"],
            "train/clipfrac": actor["clipfrac"],
            "rollout/length": float(rollout_length),
        }

    def compose_split_update(self, *, actor_epochs: Sequence[Mapping[str, Any]], critic_epochs: Sequence[Mapping[str, Any]], server_advantage: Any, normalized_adv_stds: Sequence[float], round_idx: int, fedavg_groups: Sequence[str], privacy_epochs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Compose the canonical A0 update surface from timing-captured raw state."""
        actor = self.aggregate_actor_epochs(actor_epochs, include_marl=self.marl)
        critic = critic_epochs[-1]
        logs: Dict[str, Any] = self.compose_generic_update(
            actor_epochs=actor_epochs,
            critic_epochs=critic_epochs,
            rollout_length=int(server_advantage.advantages[0].shape[0]),
        )
        if self.marl:
            logs.update({"rollout/num_episodes": float(critic.get("num_segments", 0.0)), "rollout/mean_episode_len": float(critic.get("mean_segment_len", 0.0)), "debug/actor_grad_norm": actor["actor_grad_norm"], "debug/actor_param_delta": actor["actor_param_delta"], "debug/actor_param_delta_min_epoch": actor["actor_param_delta_min_epoch"], "debug/actor_logp_delta_post": actor["actor_logp_delta_post"], "debug/actor_post_kl_available": actor["actor_post_kl_available"], "debug/server_advantage_owner": 1.0, "debug/local_actor_reward_value_free": 1.0, "debug/critic_gru_reset_enabled": float(critic.get("critic_gru_reset_enabled", False)), "debug/critic_gru_done_resets": float(critic.get("critic_gru_done_resets", 0.0)), "debug/hidden_resets": float(critic.get("hidden_resets", 0.0))})
            logs.update(self._critic_scalars(critic))
            for i in range(len(normalized_adv_stds)):
                logs.update({f"agent_{i}/adv_mean_raw": server_advantage.adv_mean_raw[i], f"agent_{i}/adv_std_raw": server_advantage.adv_std_raw[i], f"agent_{i}/adv_std_normed": normalized_adv_stds[i], f"agent_{i}/loss_actor": actor["per_agent_actor_loss"][i], f"agent_{i}/loss_critic": float(critic["value_losses"][i]), f"agent_{i}/entropy": actor["per_agent_entropy"][i], f"agent_{i}/approx_kl": actor["per_agent_approx_kl"][i], f"agent_{i}/clipfrac": actor["per_agent_clipfrac"][i], f"agent_{i}/ratio_std": actor["per_agent_ratio_std"][i], f"agent_{i}/approx_kl_post": actor["per_agent_approx_kl_post"][i], f"agent_{i}/debug/actor_grad_norm": actor["per_agent_actor_grad_norm"][i], f"agent_{i}/debug/actor_param_delta": actor["per_agent_actor_param_delta"][i]})
        if self.fedavg:
            logs.update({"fedavg/round": float(round_idx), "fedavg/applied": float(bool(fedavg_groups)), "fedavg/scope_enhancer": float("enhancer" in fedavg_groups), "fedavg/scope_gru": float("gru" in fedavg_groups)})
        if self.privacy:
            logs.update(self._mean_logs(privacy_epochs))
            if "loss/privacy_backward_total" in logs:
                logs["ratio/privacy_backward_to_critic"] = float(logs["loss/privacy_backward_total"] / (logs["loss/critic"] + 1e-12))
                logs["loss/privacy_total"] = float(logs["loss/privacy_backward_total"])
        # The production writer's historical order was base -> FedAvg ->
        # rollout/debug -> critic -> privacy -> per-agent.  Dict insertion is
        # observable in event-file order, so preserve it explicitly.
        def write_rank(tag: str) -> int:
            if tag == "fedavg/round":
                return 1
            if tag in {
                "debug/critic_server_changed_on_server_step",
                "debug/critic_server_changed_during_local_step",
                "debug/critic_server_grads_created_during_local",
                "debug/critic_local_changed_on_handoff",
                "debug/critic_dL_dBI_requires_grad_rate",
            }:
                return 4
            # Event ordering is producer chronology, not mode ownership.
            if tag.startswith(("rollout/", "debug/")) and not tag.startswith("debug/privacy_"):
                return 2
            if tag.startswith("fedavg/") or tag.startswith("bi_fedavg/"):
                return 3
            if tag.startswith("critic/"):
                return 4
            if self.mode_for_tag(tag) == "PRIVACY":
                return 5
            if self.mode_for_tag(tag) == "MARL":
                return 6
            return 0
        return dict(sorted(logs.items(), key=lambda item: write_rank(item[0])))

    def record_update(self, raw: Mapping[str, Any], *, step: int) -> Dict[str, Any]:
        emitted = {}
        for tag, value in raw.items():
            if tag != "diagnostics" and self.enabled(self.mode_for_tag(tag)):
                self._write(tag, value, step); emitted[tag] = value
        return emitted
