"""Opt-in, CPU-only Phase 6 runner with a target-independent rollout boundary.

The experiment target stops collection; it does not cause a short PPO update or
discard a partial horizon. A later target is a new run generation, so the old
final checkpoint and frozen evaluator weights remain immutable.
"""
from __future__ import annotations

import argparse
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

from scripts.experiment_fl_fedavg10_privacy_full import ExperimentConfig
from scripts.train_fl_split_shallow_hybrid import AGENT_CFGS, SplitFLModelConfig
from scripts.train_a0_globalbi_phase4 import (
    Phase4Session, ACCOUNTING, _equal, optimizer_step_totals, source_sha256,
    validate_collector,
)
from scripts.phase4_exact_resume import (
    file_sha256, pack, restore_checkpoint, save_atomic, save_json_atomic,
    unpack, verify_published_checkpoint,
)
from trainers.fl_split_ppo_trainer_globalbi_phase4 import CONTRACT_SHA256


STOP_REQUESTED = False


def _signal_stop(_signal_number, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def execution_sha256() -> str:
    return file_sha256(Path(__file__))


def _science_config(config: dict) -> dict:
    config = dict(config)
    config.pop("total_steps")
    return config


def _restore_append_log(run_root: Path, name: str, expected: str) -> None:
    """Archive work after the last verified snapshot before exact replay.

    The checkpoint, not a later uncommitted log suffix, owns the continuation
    point. The suffix remains available for accounting and divergence audits.
    """
    path = run_root / name
    if not path.exists():
        archive_folder = run_root / "abandoned_attempt_logs"
        if expected and not any(archive_folder.glob(f"{name}.*")):
            raise ValueError(f"Checkpoint log missing: {name}")
        return
    actual = path.read_text()
    if actual == expected:
        return
    if not actual.startswith(expected):
        raise ValueError(f"Existing {name} is not a prefix-compatible checkpoint log")
    folder = run_root / "abandoned_attempt_logs"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"{name}.{time.time_ns()}.{file_sha256(path)[:16]}"
    os.replace(path, archive)
    from scripts.phase4_exact_resume import fsync_directory
    fsync_directory(folder)
    fsync_directory(run_root)
    temp = path.with_name(f".{name}.restore.tmp")
    with temp.open("w") as f:
        f.write(expected)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fsync_directory(run_root)
    record = {"log": name, "archive": str(archive),
              "archive_sha256": file_sha256(archive),
              "committed_prefix_bytes": len(expected.encode()),
              "discarded_suffix_bytes": len(actual.encode()) - len(expected.encode())}
    with (folder / "archive_events.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())


class ExtendableSession(Phase4Session):
    def __init__(self, cfg, variant, seed, run_root):
        super().__init__(cfg, variant, seed, run_root)
        self.state["episode_history"] = []
        self._sync_episode_log()

    def _sync_episode_log(self):
        path = self.run_root / "episodes.jsonl"
        temp = path.with_name(".episodes.jsonl.tmp")
        with temp.open("w") as f:
            for episode in self.state["episode_history"]:
                f.write(json.dumps(episode, sort_keys=True) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
        from scripts.phase4_exact_resume import fsync_directory
        fsync_directory(path.parent)

    def step(self, *, record_trace=False):
        before_return = self.state["episode_return"]
        before_profit = self.state["episode_profit"]
        before_length = self.state["episode_length"]
        before_shutdowns = self.state["shutdown_count"]
        row = super().step(record_trace=record_trace)
        if row["done"]:
            self.state["episode_history"].append({
                "episode": self.state["completed_episodes"], "end_step": row["step"],
                "return": before_return + row["reward"],
                "profit": before_profit + (row["raw_profit"] or 0.0),
                "length": before_length + 1,
                "terminated": row["terminated"], "truncated": row["truncated"],
                "shutdown": self.state["shutdown_count"] > before_shutdowns,
            })
            self._sync_episode_log()
        # Phase 4 omits the reset only at its final target. Make the state at a
        # stop identical to a longer uninterrupted run, even if that step ends
        # an episode. The row still describes the original transition.
        if row["done"] and self.state["step_count"] == self.cfg.total_steps:
            obs, _ = self.env.reset(seed=self.state["episode_seed"])
            ACCOUNTING["env_resets"] += 1
            self.state["full_obs"] = np.asarray(obs, dtype=np.float32)
            self.state["critic_hidden_list"] = self.trainer.init_hidden(batch_size=1)
            self.state["prev_actions"] = [np.zeros(len(a.mv_indices), dtype=np.float32)
                                          for a in AGENT_CFGS]
        return row

    @classmethod
    def resume(cls, cfg, variant, seed, run_root, path, *, _candidate_probe=False):
        from scripts.phase4_exact_resume import load_payload
        if _candidate_probe:
            payload = load_payload(Path(path), variant=variant, source_sha256=source_sha256())
        else:
            _, payload = verify_published_checkpoint(Path(path), variant=variant,
                seed=seed, source_sha256=source_sha256())
        state = unpack(payload["collector"])
        if len(state.get("update_history", [])) != state.get("log_rows"):
            raise ValueError("Checkpoint update history count mismatch")
        expected_updates = "".join(json.dumps(row) + "\n" for row in state["update_history"])
        expected_episodes = "".join(json.dumps(row, sort_keys=True) + "\n"
                                    for row in state.get("episode_history", []))
        run_root = Path(run_root).resolve()
        if run_root.exists():
            _restore_append_log(run_root, "updates.jsonl", expected_updates)
            _restore_append_log(run_root, "episodes.jsonl", expected_episodes)
        obj = super().resume(cfg, variant, seed, run_root, path,
                             _candidate_probe=_candidate_probe)
        if "episode_history" not in obj.state:
            raise ValueError("Resume checkpoint lacks episode history")
        obj._sync_episode_log()
        return obj

    @classmethod
    def extend(cls, cfg, variant, seed, run_root, parent_final):
        record, payload = verify_published_checkpoint(
            Path(parent_final), variant=variant, seed=seed,
            source_sha256=source_sha256())
        if record["kind"] != "final":
            raise ValueError("Extension requires a verified final checkpoint")
        prior_cfg = unpack(payload["runtime_config"])
        prior_goal = int(prior_cfg["total_steps"])
        if _science_config(prior_cfg) != _science_config(asdict(cfg)):
            raise ValueError("Scientific/runtime configuration changed during extension")
        if cfg.total_steps <= prior_goal or cfg.total_steps <= record["step"]:
            raise ValueError("Extension target must increase beyond prior goal and reached step")
        prior_state = unpack(payload["collector"])
        lineage = prior_state.get("extension_lineage")
        if not isinstance(lineage, dict) or lineage.get("target_history", [])[-1] != prior_goal:
            raise ValueError("Parent checkpoint lacks trusted extension lineage")
        if lineage.get("execution_sha256") != execution_sha256():
            raise ValueError("Extension execution source changed")
        obj = cls(cfg, variant, seed, run_root)
        obj.state = restore_checkpoint(payload, obj.trainer, obj.env, obj.obs_rms)
        validate_collector(obj.state, obj.env)
        obj.state["extension_lineage"] = {
            **lineage, "generation": int(lineage["generation"]) + 1,
            "parent_checkpoint_sha256": record["sha256"],
            "parent_manifest_sha256": file_sha256(Path(parent_final).parent / "verified_final.json"),
            "target_history": [*lineage["target_history"], cfg.total_steps],
        }
        log_path = obj.run_root / "updates.jsonl"
        rows = obj.state["update_history"]
        if len(rows) != obj.state["log_rows"]:
            raise ValueError("Parent update history length mismatch")
        with log_path.open("x") as f:
            f.write("".join(json.dumps(row) + "\n" for row in rows))
            f.flush(); os.fsync(f.fileno())
        obj._sync_episode_log()
        return obj


def _initial_state_check(session, bundle_path: Path | None):
    if bundle_path is None:
        return None
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if bundle.get("seed") != session.seed or bundle.get("variant") != session.variant:
        raise ValueError("Initial bundle seed/variant mismatch")
    if bundle.get("source_sha256") != source_sha256() or bundle.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("Initial bundle source/contract mismatch")
    if not _equal(bundle["trainer"], session.trainer.checkpoint_state()):
        raise ValueError("Fresh initial trainer differs from preserved seed bundle")
    return file_sha256(bundle_path)


def _frozen_evaluator(session):
    path = session.run_root / "final" / "frozen_evaluator.pt"
    if path.exists():
        raise ValueError("Refusing to overwrite frozen evaluator checkpoint")
    payload = {"format": "extendable_frozen_actor_v1",
               "variant": session.variant, "seed": session.seed,
               "step": session.state["step_count"], "source_sha256": source_sha256(),
               "contract_sha256": CONTRACT_SHA256,
               "actors": [client.actor.state_dict() for client in session.trainer.clients],
               "obs_rms": pack([dict(vars(rms)) for rms in session.obs_rms]),
               "completed_ppo_updates": session.state["update_count"]}
    info = save_atomic(path, payload)
    save_json_atomic(path.with_suffix(".json"), {k: v for k, v in info.items()})
    return info


def _verified_tensorboard(session):
    folder = session.run_root / "final" / "tensorboard_verified"
    if folder.exists():
        raise ValueError("Refusing to overwrite canonical TensorBoard output")
    writer = SummaryWriter(log_dir=str(folder))
    for row in session.state["update_history"]:
        for key, value in row["logs"].items():
            writer.add_scalar(key, float(value), row["step"])
    writer.flush(); writer.close()
    events = sorted(folder.glob("events.out.tfevents.*"))
    if not events:
        raise RuntimeError("Canonical TensorBoard event file missing")
    return [{"path": str(path), "sha256": file_sha256(path),
             "size_bytes": path.stat().st_size} for path in events]


def run(args):
    if args.total_steps <= 0 or args.rollout_horizon <= 0:
        raise ValueError("Positive goal and horizon required")
    if args.test_checkpoint_interval is not None and (not args.test_only or args.total_steps > 128):
        raise ValueError("Short interim interval is test-only, goal <=128")
    if args.extend_from and args.resume:
        raise ValueError("Choose extension parent or same-generation resume")
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only run requires CUDA_VISIBLE_DEVICES='' ")
    torch.set_num_threads(args.cpu_threads)
    cfg = replace(ExperimentConfig(), total_steps=args.total_steps,
                  rollout_horizon=args.rollout_horizon,
                  min_rollout_steps=args.min_rollout_steps,
                  max_steps=args.max_steps)
    if args.resume:
        from scripts.phase4_exact_resume import discover_latest_verified
        path = (discover_latest_verified(args.run_root, variant=args.variant, seed=args.seed,
                 source_sha256=source_sha256()) if str(args.resume) == "auto" else args.resume)
        session = ExtendableSession.resume(cfg, args.variant, args.seed, args.run_root, path)
        lineage = session.state.get("extension_lineage")
        if not isinstance(lineage, dict) or lineage.get("execution_sha256") != execution_sha256():
            raise ValueError("Resume execution source/lineage mismatch")
        if lineage["target_history"][-1] != cfg.total_steps:
            raise ValueError("Resume target history mismatch")
        bundle_sha = lineage.get("initial_bundle_sha256")
    elif args.extend_from:
        session = ExtendableSession.extend(cfg, args.variant, args.seed, args.run_root, args.extend_from)
        bundle_sha = session.state["extension_lineage"].get("initial_bundle_sha256")
    else:
        session = ExtendableSession(cfg, args.variant, args.seed, args.run_root)
        bundle_sha = _initial_state_check(session, args.initial_bundle)
        session.state["extension_lineage"] = {
            "generation": 0, "target_history": [cfg.total_steps],
            "parent_checkpoint_sha256": None, "parent_manifest_sha256": None,
            "initial_bundle_sha256": bundle_sha, "execution_sha256": execution_sha256(),
        }
    if session.state["step_count"] >= cfg.total_steps:
        raise ValueError("Target already reached")
    if args.run_id is not None:
        prior_run_id = session.state.get("run_id")
        if prior_run_id is not None and prior_run_id != args.run_id:
            raise ValueError("Checkpoint run_id differs from requested run")
        session.state["run_id"] = args.run_id
    save_json_atomic(session.run_root / "heartbeat.json", {
        "step": session.state["step_count"], "update": session.state["update_count"],
        "pid": os.getpid(), "time_unix": time.time(),
        "variant": args.variant, "seed": args.seed})
    attempt_tb = session.run_root / "tensorboard_attempts" / f"pid_{os.getpid()}"
    writer = SummaryWriter(log_dir=str(attempt_tb))
    session.trainer.writer = writer
    milestones = ({50000, 100000, 150000} if args.test_checkpoint_interval is None
                  else set(range(args.test_checkpoint_interval, cfg.total_steps,
                                 args.test_checkpoint_interval)))
    started = time.monotonic()
    while session.state["step_count"] < cfg.total_steps:
        session.step()
        step = session.state["step_count"]
        if step in milestones:
            session.checkpoint()
        if step % args.heartbeat_steps == 0:
            save_json_atomic(session.run_root / "heartbeat.json", {
                "step": step, "update": session.state["update_count"],
                "pid": os.getpid(), "time_unix": time.time(),
                "variant": args.variant, "seed": args.seed})
        if STOP_REQUESTED:
            if step not in milestones:
                session.checkpoint()
            writer.flush(); writer.close()
            return {"status": "INTERRUPTED", "env_steps": step,
                    "run_root": str(session.run_root), "process_accounting": ACCOUNTING}
    # Do not call update() on a partial horizon. At 200k this retains 320
    # transitions, so 400k extension follows the same next 1024-boundary.
    writer.flush(); writer.close()
    session.trainer.writer = None
    final = session.checkpoint(final=True)
    frozen = _frozen_evaluator(session)
    canonical_tb = _verified_tensorboard(session)
    result = {"status": "COMPLETED", "variant": args.variant, "seed": args.seed,
              "run_id": args.run_id,
              "run_root": str(session.run_root), "env_steps": session.state["step_count"],
              "ppo_updates": session.state["update_count"],
              "collector_cursor": len(session.state["buffers"][0]["obs"]),
              "completed_episodes": session.state["completed_episodes"],
              "shutdown_count": session.state["shutdown_count"],
              "episode_log_sha256": file_sha256(session.run_root / "episodes.jsonl")
                  if (session.run_root / "episodes.jsonl").exists() else None,
              "final": final, "frozen": frozen, "initial_bundle_sha256": bundle_sha,
              "tensorboard_verified": canonical_tb,
              "tensorboard_attempt_dir": str(attempt_tb),
              "lineage": session.state["extension_lineage"],
              "wall_seconds": time.monotonic() - started,
              "process_accounting": ACCOUNTING,
              "final_optimizer_steps": optimizer_step_totals(session.trainer)}
    save_json_atomic(session.run_root / "run_summary.json", result)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("F", "M", "G"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-id")
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--initial-bundle", type=Path)
    p.add_argument("--extend-from", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--rollout-horizon", type=int, default=1024)
    p.add_argument("--min-rollout-steps", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--heartbeat-steps", type=int, default=64)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--test-checkpoint-interval", type=int)
    return p.parse_args(argv)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    print(json.dumps(run(parse_args()), indent=2), flush=True)
