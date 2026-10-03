"""Fresh CPU F/M/G generation with the existing shared 156-tag logger."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from scripts import train_a0_globalbi_extendable as base
from scripts.experiment_fl_fedavg10_privacy_full import ExperimentConfig
from scripts.train_fl_split_shallow_hybrid import AGENT_CFGS, SplitFLModelConfig
from scripts.train_a0_globalbi_phase4 import ACCOUNTING, ROOT, _equal, source_sha256
from scripts.phase4_exact_resume import (
    checkpoint_payload, discover_latest_verified, file_sha256, load_payload,
    restore_rng, save_json_atomic, unpack,
)
from trainers.fl_split_ppo_trainer_globalbi_phase4 import CONTRACT_SHA256
from utils.training_metrics_logger import TrainingMetricsLogger
from utils.globalbi_metric_journal import (
    SCHEMA, MetricJournalWriter, content_digest, prefix_bytes, scalar_timeline,
)


def tag_definition_sha256():
    return hashlib.sha256(
        (ROOT / "utils/training_metrics_logger.py").read_bytes()
        + (ROOT / "utils/performance_logging.py").read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.detach().cpu().tolist() if torch.is_tensor(value) else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def validate_logging_state(payload):
    state = unpack(payload["collector"])
    log = state.get("full_logger")
    required = {"schema", "identity", "origin_run_root", "performance",
                "watermark", "cpu_threads", "last_transition_step"}
    if not isinstance(log, dict) or set(log) != required or log["schema"] != SCHEMA:
        raise ValueError("Missing full logger state/schema; old generation resume forbidden")
    identity = log["identity"]
    if identity != {
        "run_id": state.get("run_id"), "variant": payload["variant"],
        "seed": payload["seed"], "source_sha256": source_sha256(),
        "contract_sha256": CONTRACT_SHA256,
        "tag_definition_sha256": tag_definition_sha256(),
        "log_every": unpack(payload["runtime_config"])["log_every"],
    }:
        raise ValueError("Full logger identity/source/tag-definition mismatch")
    if (log["last_transition_step"] != state["step_count"]
            or log["performance"]["episode_length"] != state["episode_length"]
            or log["performance"]["episode_return"] != state["episode_return"]
            or not isinstance(log["cpu_threads"], int) or log["cpu_threads"] < 1):
        raise ValueError("Full logger episode/counter mismatch")
    prefix_bytes(Path(log["origin_run_root"]) / "metrics.jsonl", log["watermark"])
    return log


class FullLoggerSession(base.ExtendableSession):
    """The learner is inherited; only observer/lifecycle hooks are added."""
    def __init__(self, cfg, variant, seed, run_root):
        super().__init__(cfg, variant, seed, run_root)
        self.journal = None
        self.metrics_logger = None
        self.logging_enabled = False
        self.logger_origin = str(self.run_root)

    def attach_logging(self, *, writer=None, run_id=None, restored=None, enabled=True):
        if self.journal is not None:
            raise ValueError("Full logger already owns this session")
        self.logging_enabled = bool(enabled)
        if not enabled:
            self.trainer.writer = None
            return
        if run_id is not None:
            prior = self.state.get("run_id")
            if prior is not None and prior != run_id:
                raise ValueError("Wrong run_id for full logger")
            self.state["run_id"] = run_id
        identity = {
            "run_id": self.state.get("run_id"), "variant": self.variant,
            "seed": self.seed, "source_sha256": source_sha256(),
            "contract_sha256": CONTRACT_SHA256,
            "tag_definition_sha256": tag_definition_sha256(), "log_every": self.cfg.log_every,
        }
        self.logger_origin = restored["origin_run_root"] if restored else str(self.run_root)
        self.journal = MetricJournalWriter(
            self.run_root / "metrics.jsonl", identity, tb_writer=writer,
            watermark=restored["watermark"] if restored else None,
            origin=Path(self.logger_origin) / "metrics.jsonl" if restored else None)
        self.metrics_logger = TrainingMetricsLogger(
            self.journal, self.cfg.log_every, marl=True, fedavg=True, privacy=True)
        if restored:
            self.metrics_logger.performance.load_state_dict(restored["performance"])
        self.trainer.writer = self.journal
        self.trainer.metrics_logger = self.metrics_logger
        self._sync_logger_state()

    def set_tensorboard(self, writer):
        if not self.logging_enabled or self.journal is None:
            raise ValueError("No full logger attached")
        self.journal.tb_writer = writer

    def _sync_logger_state(self):
        if self.logging_enabled:
            self.state["full_logger"] = {
                "schema": SCHEMA, "identity": dict(self.journal.identity),
                "origin_run_root": self.logger_origin,
                "performance": self.metrics_logger.performance.state_dict(),
                "watermark": self.journal.watermark(),
                "cpu_threads": torch.get_num_threads(),
                "last_transition_step": self.state["step_count"],
            }

    def _record_transition_metrics(self, **t):
        if not self.logging_enabled:
            return
        info = t["info"]
        if info.get("raw_prod") is None or info.get("raw_cost") is None:
            raise ValueError("Full logger requires actual raw production and cost")
        step = t["step"]
        perf = self.metrics_logger.performance
        self.journal.append("transition", step=step, producer="record_action+record_transition",
            **plain({k: v for k, v in t.items() if k not in ("info", "step")}),
            prev_action=plain(self.state["prev_actions"]),
            raw_prod=float(info["raw_prod"]), raw_cost=float(info["raw_cost"]),
            shutdown=bool(info.get("shutdown", False)), mv=plain(info["mv"]),
            rms_count=[r.count for r in self.obs_rms],
            pending_window_denominator=len(perf.shutdown_buffer) + 1,
            logging_instance_owner="one_shared_instance")
        self.metrics_logger.record_action(step=step, action_parts=t["actions"],
            mv_indices=[a.mv_indices for a in AGENT_CFGS], info=info,
            delta_mv_max=self.cfg.delta_mv_max)
        self.metrics_logger.record_transition(step=step, reward=t["reward"],
            raw_prod=info["raw_prod"], raw_cost=info["raw_cost"],
            shutdown=info.get("shutdown", False),
            done=bool(t["terminated"] or t["truncated"]))

    def update(self):
        if self.logging_enabled:
            collected = len(self.state["buffers"][0]["obs"])
            skipped = collected < self.cfg.min_rollout_steps and not self.cfg.allow_short_final_rollout
            self.journal.append("call", api="record_rollout", step=self.state["step_count"],
                length=collected, episodes=self.state["episodes_in_rollout"])
            self.metrics_logger.record_rollout(step=self.state["step_count"],
                length=collected, episodes=self.state["episodes_in_rollout"],
                skipped_short=skipped, emit_length=skipped)
        before = self.state["update_count"]
        result = super().update()
        if self.logging_enabled and self.state["update_count"] != before:
            self.journal.append("call", api="record_update", step=self.state["step_count"],
                                inherited_update_tags=len(result))
            self.metrics_logger.record_update({
                "train/update_count": float(self.state["update_count"]),
                "train/updates_per_env_step": float(self.state["update_count"]) /
                    max(float(self.state["step_count"]), 1.0),
            }, step=self.state["step_count"])
        return result

    def finish_metrics(self):
        if not self.logging_enabled:
            return
        remaining = len(self.state["buffers"][0]["obs"])
        if remaining:
            # Observe the short tail; preserve it for a future exact extension.
            self.journal.append("call", api="record_rollout", step=self.state["step_count"],
                                length=remaining, tail_preserved=True)
            self.metrics_logger.record_rollout(step=self.state["step_count"], length=remaining,
                episodes=self.state["episodes_in_rollout"], skipped_short=True, emit_length=True)
        # Final reporting must not discard an incomplete window needed by an
        # explicit later extension. The journal records this terminal report;
        # the primitive tracker still owns its uncompleted natural window.
        pending = self.metrics_logger.performance.state_dict()
        self.metrics_logger.flush(step=max(self.state["step_count"], 1))
        self.metrics_logger.performance.load_state_dict(pending)
        self._sync_logger_state()

    def checkpoint(self, *, final=False):
        if not self.logging_enabled:
            raise ValueError("Full logger checkpoint cannot omit logger state")
        # A restored branch owns its copied prefix and subsequent suffix.
        # Publish this physical journal; never modify its parent's journal.
        self.logger_origin = str(self.run_root)
        self._sync_logger_state()
        return super().checkpoint(final=final)

    def snapshot_payload(self):
        self._sync_logger_state()
        return checkpoint_payload(self.trainer, self.env, self.obs_rms, self.state,
            seed=self.seed, source_sha256=source_sha256(), runtime_config=asdict(self.cfg))

    def continuation_digest(self):
        return content_digest(self.snapshot_payload())

    def _checkpoint_probe_module(self):
        return "scripts.train_a0_globalbi_full_logger"

    @classmethod
    def resume(cls, cfg, variant, seed, run_root, path, *, _candidate_probe=False):
        payload = load_payload(Path(path), variant=variant, source_sha256=source_sha256())
        log = validate_logging_state(payload)
        if payload["seed"] != seed:
            raise ValueError("Wrong full logger resume seed")
        obj = super().resume(cfg, variant, seed, run_root, path, _candidate_probe=_candidate_probe)
        obj.attach_logging(restored=log)
        return obj

    def close_logging(self):
        if self.journal is not None:
            self.journal.close()

    @classmethod
    def extend(cls, cfg, variant, seed, run_root, parent_final):
        payload = load_payload(Path(parent_final), variant=variant, source_sha256=source_sha256())
        log = validate_logging_state(payload)
        obj = super().extend(cfg, variant, seed, run_root, parent_final)
        obj.attach_logging(restored=log)
        return obj


def canonical_tensorboard(session):
    session.journal.flush()
    folder = session.run_root / "final/tensorboard_verified"
    if folder.exists():
        raise ValueError("Refusing to overwrite canonical full TensorBoard")
    writer = SummaryWriter(log_dir=str(folder))
    counts = {}
    try:
        for row in scalar_timeline(session.journal.path):
            writer.add_scalar(row["tag"], row["value"], row["step"])
            counts[row["tag"]] = counts.get(row["tag"], 0) + 1
    finally:
        writer.flush(); writer.close()
    paths = [{"path": str(p), "sha256": file_sha256(p), "size_bytes": p.stat().st_size}
             for p in sorted(folder.glob("events.out.tfevents.*"))]
    save_json_atomic(folder / "timeline_manifest.json", {
        "schema": SCHEMA, "identity": session.journal.identity,
        "journal": session.journal.watermark(), "scalar_counts": counts, "events": paths})
    return paths, counts


def _verify_candidate_entry(argv):
    path, output, variant, seed = Path(argv[0]), Path(argv[1]), argv[2], int(argv[3])
    started, cpu = time.monotonic(), time.process_time()
    payload = load_payload(path, variant=variant, source_sha256=source_sha256())
    log = validate_logging_state(payload)
    torch.set_num_threads(log["cpu_threads"])
    cfg = unpack(payload["runtime_config"])
    cfg["model_cfg"] = SplitFLModelConfig(**cfg["model_cfg"])
    cfg = ExperimentConfig(**cfg)
    fresh = FullLoggerSession.resume(cfg, variant, seed, output.parent / "child_reload",
                                     path, _candidate_probe=True)
    if not _equal(payload, fresh.snapshot_payload()):
        raise ValueError("Fresh full-logger candidate state mismatch")
    row = fresh.step()
    digest = fresh.continuation_digest()
    fresh.close_logging()
    output.write_text(json.dumps({"reload_exact": True, "step": row,
        "continuation_digest": digest, "process_accounting": ACCOUNTING,
        "wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu})+"\n")


def run(args):
    if args.extend_from and args.resume:
        raise ValueError("Choose extension or same-generation resume")
    if args.total_steps <= 0 or args.rollout_horizon <= 0:
        raise ValueError("Positive target/horizon required")
    if args.test_checkpoint_interval is not None and (not args.test_only or args.total_steps > 128):
        raise ValueError("Short checkpoint interval requires test-only target<=128")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise ValueError("Full logger generation is CPU-only")
    torch.set_num_threads(args.cpu_threads)
    cfg = replace(ExperimentConfig(), total_steps=args.total_steps,
        rollout_horizon=args.rollout_horizon, min_rollout_steps=args.min_rollout_steps,
        max_steps=args.max_steps)
    attempt = Path(args.run_root) / "tensorboard_attempts" / f"pid_{os.getpid()}"
    writer = SummaryWriter(log_dir=str(attempt))
    session = None
    started, cpu = time.monotonic(), time.process_time()
    try:
        if args.resume:
            path = (discover_latest_verified(args.run_root, variant=args.variant, seed=args.seed,
                        source_sha256=source_sha256()) if str(args.resume) == "auto" else args.resume)
            session = FullLoggerSession.resume(cfg, args.variant, args.seed, args.run_root, path)
            if session.state.get("run_id") != args.run_id:
                raise ValueError("Resume run_id mismatch")
            session.set_tensorboard(writer)
            lineage = session.state["extension_lineage"]
            if lineage["execution_sha256"] != base.execution_sha256() or lineage["target_history"] != [cfg.total_steps]:
                raise ValueError("Execution/target lineage mismatch")
            bundle_sha = lineage["initial_bundle_sha256"]
        elif args.extend_from:
            session = FullLoggerSession.extend(cfg, args.variant, args.seed, args.run_root, args.extend_from)
            if session.state.get("run_id") != args.run_id:
                raise ValueError("Extension run_id mismatch")
            session.set_tensorboard(writer)
            bundle_sha = session.state["extension_lineage"]["initial_bundle_sha256"]
        else:
            session = FullLoggerSession(cfg, args.variant, args.seed, args.run_root)
            bundle_sha = base._initial_state_check(session, args.initial_bundle)
            session.state["extension_lineage"] = {"generation": 0,
                "target_history": [cfg.total_steps], "parent_checkpoint_sha256": None,
                "parent_manifest_sha256": None, "initial_bundle_sha256": bundle_sha,
                "execution_sha256": base.execution_sha256()}
            session.attach_logging(writer=writer, run_id=args.run_id)
            # A verified step-zero state permits exact recovery before first50k.
            initial = session.checkpoint()
            save_json_atomic(session.run_root / "fresh_initial_state.json", {
                "run_id": args.run_id, "variant": args.variant, "seed": args.seed,
                "env_step": 0, "optimizer_steps": {"actor": 0, "bi": 0, "server": 0, "privacy": 0},
                "rms_counts": [r.count for r in session.obs_rms], "checkpoint": initial,
                "initial_bundle_sha256": bundle_sha, "schema": SCHEMA})
        if session.state["step_count"] >= cfg.total_steps:
            raise ValueError("Target already reached")
        milestones = ({50000, 100000, 150000} if args.test_checkpoint_interval is None
            else set(range(args.test_checkpoint_interval, cfg.total_steps, args.test_checkpoint_interval)))
        while session.state["step_count"] < cfg.total_steps:
            session.step()
            step = session.state["step_count"]
            if step in milestones:
                session.checkpoint()
            if step % args.heartbeat_steps == 0:
                save_json_atomic(session.run_root / "heartbeat.json", {
                    "step": step, "update": session.state["update_count"],
                    "pid": os.getpid(), "time_unix": time.time(), "variant": args.variant,
                    "seed": args.seed, "logger_schema": SCHEMA,
                    "obs_rms_count": [r.count for r in session.obs_rms]})
            if base.STOP_REQUESTED:
                if step not in milestones:
                    session.checkpoint()
                return {"status": "INTERRUPTED", "env_steps": step,
                        "run_root": str(session.run_root), "process_accounting": ACCOUNTING}
        session.finish_metrics()
        final = session.checkpoint(final=True)
        frozen = base._frozen_evaluator(session)
        tb, counts = canonical_tensorboard(session)
        result = {
            "status": "COMPLETED", "variant": args.variant, "seed": args.seed,
            "run_id": args.run_id, "run_root": str(session.run_root),
            "env_steps": session.state["step_count"], "ppo_updates": session.state["update_count"],
            "collector_cursor": len(session.state["buffers"][0]["obs"]),
            "completed_episodes": session.state["completed_episodes"],
            "shutdown_count": session.state["shutdown_count"],
            "episode_log_sha256": file_sha256(session.run_root / "episodes.jsonl"),
            "final": final, "frozen": frozen, "initial_bundle_sha256": bundle_sha,
            "tensorboard_verified": tb, "scalar_counts": counts,
            "metric_journal": session.journal.watermark(), "logger_schema": SCHEMA,
            "lineage": session.state["extension_lineage"],
            "process_accounting": ACCOUNTING, "wall_seconds": time.monotonic()-started,
            "cpu_seconds": time.process_time()-cpu,
        }
        save_json_atomic(session.run_root / "run_summary.json", result)
        return result
    finally:
        try:
            if session is not None:
                session.close_logging()
        finally:
            writer.flush(); writer.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--phase6-verify-candidate":
        _verify_candidate_entry(sys.argv[2:])
    else:
        signal.signal(signal.SIGTERM, base._signal_stop)
        signal.signal(signal.SIGINT, base._signal_stop)
        print(json.dumps(run(base.parse_args()), indent=2), flush=True)
