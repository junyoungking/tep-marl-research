"""Opt-in F/M/G full-privacy runner with exact Python-TEP state checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_fl_fedavg10_privacy_full import ExperimentConfig
from scripts.train_fl_split_shallow_hybrid import (
    AGENT_CFGS, RunningMeanStd, SplitFLModelConfig, append_transition, buffers_to_rollouts,
    concat_joint_action, make_env, make_local_obs,
)
from scripts.phase4_exact_resume import (
    CheckpointTrustError, checkpoint_payload, discover_latest_verified, file_sha256,
    fsync_directory, load_payload, quarantine_file, read_verified_record,
    reconcile_interim_orphans, restore_checkpoint, restore_rng, rng_state,
    save_atomic, save_json_atomic, snapshot_env, unpack, verified_manifest_path,
    verified_record, verify_published_checkpoint,
)
from trainers.fl_split_ppo_trainer_globalbi_phase4 import CONTRACT, CONTRACT_SHA256, build_matched


SOURCE_FILES = (
    "models/fl_split_actor.py", "models/fl_split_critic.py",
    "models/bi_privacy_defense.py", "models/fl_split_bottleneck.py",
    "trainers/fl_split_ppo_trainer.py", "trainers/fl_split_ppo_trainer_privacy_full.py",
    "trainers/fl_split_ppo_trainer_dravl.py",
    "trainers/fl_split_ppo_trainer_privacy_module_only_v3.py",
    "trainers/fl_split_ppo_trainer_continue_rollout.py",
    "trainers/fl_split_ppo_trainer_globalbi_phase4.py", "scripts/train_fl_split_shallow_hybrid.py",
    "scripts/experiment_fl_fedavg10_privacy_full.py",
    "scripts/experiment_fl_fedavg10_privacy_module_only.py",
    "scripts/experiment_fl_fedavg10_privacy_module_only_v3.py",
    "scripts/checkpointing_utils.py",
    "scripts/phase4_exact_resume.py", "scripts/train_a0_globalbi_phase4.py",
    "envs/env.py", "envs/reward.py", "envs/cost.py", "envs/wrappers.py",
    "utils/fedavg.py", "utils/training_metrics_logger.py", "utils/performance_logging.py",
    "utils/globalbi_metric_journal.py", "scripts/train_a0_globalbi_full_logger.py",
)
BACKEND_FILES = (
    ROOT.parent / "tennessee-eastman-process/src/tep/__init__.py",
    ROOT.parent / "tennessee-eastman-process/src/tep/python_backend.py",
    ROOT.parent / "tennessee-eastman-process/src/tep/simulator.py",
)
ACCOUNTING = {"env_constructs": 0, "env_resets": 0, "backend_steps": 0,
              "ppo_updates": 0, "optimizer_steps": {"actor": 0, "bi": 0, "server": 0, "privacy": 0}}


def optimizer_step_totals(trainer):
    def step_count(opt):
        return max((int(state["step"]) for state in opt.state.values() if "step" in state), default=0)
    return {"actor": sum(step_count(o) for o in trainer.actor_opts),
            "bi": sum(step_count(o) for o in trainer.local_critic_opts),
            "server": step_count(trainer.server_opt),
            "privacy": sum(step_count(o) for o in trainer.priv_opts)}


def source_sha256() -> str:
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        h.update(rel.encode()); h.update(file_sha256(ROOT / rel).encode())
    for path in BACKEND_FILES:
        h.update(str(path.resolve()).encode()); h.update(file_sha256(path).encode())
    return h.hexdigest()


def _empty_buffers():
    return [{key: [] for key in ("obs", "act", "prev_act", "logp", "val", "rew", "done")}
            for _ in AGENT_CFGS]


def _equal(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def validate_collector(state, env):
    lengths = [len(b["obs"]) for b in state["buffers"]]
    if len(set(lengths)) != 1:
        raise ValueError("Collector agent buffer lengths differ")
    if lengths[0] > 0 and not state["last_transition_done"]:
        for i in range(3):
            if not np.array_equal(state["prev_actions"][i], state["buffers"][i]["act"][-1]):
                raise ValueError("Collector previous-action timestamp mismatch")
    if any(np.asarray(x).shape != (len(AGENT_CFGS[i].mv_indices),)
           for i, x in enumerate(state["prev_actions"])):
        raise ValueError("Collector previous-action shape mismatch")
    live_obs = env.unwrapped.sim.get_measurements().astype(np.float32)
    if not np.array_equal(state["full_obs"], live_obs):
        raise ValueError("Collector current observation differs from restored TEP backend")


def assert_single_obsrms_update(before_counts, after_counts, env_step):
    if len(before_counts) != len(after_counts):
        raise RuntimeError(f"ObsRMS count vector length mismatch at env_step={env_step}")
    for agent_id, (before, after) in enumerate(zip(before_counts, after_counts)):
        expected = before + 1.0
        if after != expected:
            raise RuntimeError(
                "ObsRMS did not update exactly once at pre-action timestep: "
                f"agent_id={agent_id} env_step={env_step} "
                f"before={before!r} after={after!r} expected={expected!r} "
                f"abs_err={abs(after - expected)!r}")


class Phase4Session:
    def _record_transition_metrics(self, **transition):
        """Opt-in observer hook; legacy Phase4Session has no observer."""

    def _checkpoint_probe_module(self):
        return "scripts.train_a0_globalbi_phase4"

    def __init__(self, cfg: ExperimentConfig, variant: str, seed: int, run_root: Path):
        self.cfg, self.variant, self.seed = cfg, variant, int(seed)
        self.run_root = Path(run_root).resolve()
        if not str(self.run_root).startswith(str((ROOT / "audit_artifacts").resolve()) + os.sep):
            raise ValueError("Phase 4 outputs must be under local audit_artifacts")
        self.run_root.mkdir(parents=True, exist_ok=True)
        config_path = self.run_root / "config.json"
        resolved = json.loads(json.dumps(
            {"variant": variant, "seed": int(seed), "experiment": asdict(cfg),
             "source_sha256": source_sha256(), "contract_sha256": CONTRACT_SHA256,
             "contract": CONTRACT, "run_root": str(self.run_root)}))
        if config_path.exists():
            if json.loads(config_path.read_text()) != resolved:
                raise ValueError("Existing Phase 4 run configuration differs")
        else:
            config_path.write_text(json.dumps(resolved, indent=2) + "\n")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.trainer = build_matched(cfg, seed)[variant]
        # Constructor RNG consumption is isolated by build_matched.
        self.env, obs, _ = make_env(
            tuple(mv for agent in AGENT_CFGS for mv in agent.mv_indices), seed,
            cfg.rl_dt_seconds, cfg.max_steps, cfg.delta_mv_max,
            cfg.production_index, cfg.baseline, cfg.scale, cfg.reset_reward,
        )
        ACCOUNTING["env_constructs"] += 1
        ACCOUNTING["env_resets"] += 1
        self.obs_rms = [RunningMeanStd((len(a.obs_indices),)) for a in AGENT_CFGS]
        self.state = {
            "step_count": 0, "update_count": 0, "episode_seed": seed,
            "full_obs": np.asarray(obs, dtype=np.float32),
            "prev_actions": [np.zeros(len(a.mv_indices), dtype=np.float32) for a in AGENT_CFGS],
            "critic_hidden_list": self.trainer.init_hidden(batch_size=1),
            "buffers": _empty_buffers(), "last_transition_done": False,
            "episodes_in_rollout": 0, "episode_return": 0.0,
            "episode_profit": 0.0, "episode_length": 0,
            "completed_episodes": 0, "shutdown_count": 0,
            "last_logs": {}, "log_rows": 0, "update_history": [],
        }
        self.trace = []

    @classmethod
    def resume(cls, cfg, variant, seed, run_root, path, *, _candidate_probe=False):
        obj = cls(cfg, variant, seed, run_root)
        if _candidate_probe:
            payload = load_payload(Path(path), variant=variant, source_sha256=source_sha256())
        else:
            _, payload = verify_published_checkpoint(Path(path), variant=variant,
                seed=seed, source_sha256=source_sha256())
        if payload["seed"] != int(seed):
            raise ValueError("Wrong train seed")
        if unpack(payload["runtime_config"]) != asdict(cfg):
            raise ValueError("Runtime training configuration differs from checkpoint")
        obj.state = restore_checkpoint(payload, obj.trainer, obj.env, obj.obs_rms)
        validate_collector(obj.state, obj.env)
        history = obj.state["update_history"]
        if len(history) != obj.state["log_rows"]:
            raise ValueError("Checkpoint logging history length mismatch")
        log_path = obj.run_root / "updates.jsonl"
        serialized = "".join(json.dumps(row) + "\n" for row in history)
        if log_path.exists():
            if log_path.read_text() != serialized:
                raise ValueError("Existing Phase 4 update log differs from checkpoint")
        elif serialized:
            with log_path.open("w") as f:
                f.write(serialized)
                f.flush(); os.fsync(f.fileno())
        return obj

    def step(self, *, record_trace=False):
        state = self.state
        step_before = state["step_count"]
        raw_before = state["full_obs"].copy()
        rms_before = [r.count for r in self.obs_rms]
        local = make_local_obs(raw_before, self.obs_rms, update_rms=True)
        rms_after = [r.count for r in self.obs_rms]
        assert_single_obsrms_update(rms_before, rms_after, step_before + 1)
        if step_before + 1 == 32768:
            boundary = {"variant": self.variant, "seed": self.seed,
                        "env_step": step_before + 1, "pid": os.getpid(),
                        "source_sha256": source_sha256(),
                        "counts": [{"agent_id": i, "before": before, "after": after,
                                    "expected": before + 1.0,
                                    "abs_err": abs(after - (before + 1.0))}
                                   for i, (before, after) in enumerate(zip(rms_before, rms_after))]}
            with (self.run_root / "obsrms_boundary_32768.jsonl").open("a") as f:
                f.write(json.dumps(boundary, sort_keys=True) + "\n")
                f.flush(); os.fsync(f.fileno())
        actions, logps, values, hidden = self.trainer.act(
            local, state["prev_actions"], state["critic_hidden_list"])
        joint_action = concat_joint_action(actions)
        next_obs, reward, terminated, truncated, info = self.env.step(joint_action)
        ACCOUNTING["backend_steps"] += 1
        done = bool(terminated or truncated)
        append_transition(state["buffers"], local, actions, state["prev_actions"],
                          logps, values, float(reward), done)
        state["step_count"] += 1
        self._record_transition_metrics(
            step=state["step_count"], raw_obs=raw_before, local_obs=local,
            actions=actions, logps=logps, values=values, reward=float(reward),
            terminated=bool(terminated), truncated=bool(truncated), info=info,
            next_obs=np.asarray(next_obs, dtype=np.float32))
        state["full_obs"] = np.asarray(next_obs, dtype=np.float32)
        state["prev_actions"] = actions
        state["critic_hidden_list"] = hidden
        state["last_transition_done"] = done
        state["episode_return"] += float(reward)
        prod, cost = info.get("raw_prod"), info.get("raw_cost")
        state["episode_profit"] += float(prod - cost) if prod is not None and cost is not None else 0.0
        state["episode_length"] += 1
        row = {"step": state["step_count"], "raw_obs_before": raw_before.tolist(),
               "local_obs": [x.tolist() for x in local], "rms_before": rms_before,
               "rms_after": rms_after, "prev_action": [state["buffers"][i]["prev_act"][-1].tolist() for i in range(3)],
               "action": [x.tolist() for x in actions], "value": values,
               "reward": float(reward), "raw_profit": float(prod-cost) if prod is not None and cost is not None else None,
               "done": done, "terminated": bool(terminated), "truncated": bool(truncated),
               "next_obs": state["full_obs"].tolist(), "mv": info["mv"].tolist(),
               "prev_mv": self.env._prev_mv.tolist()}
        if record_trace:
            self.trace.append(row)
        if done:
            state["completed_episodes"] += 1
            state["shutdown_count"] += int(bool(info.get("shutdown", False)))
            state["episodes_in_rollout"] += 1
            state["episode_seed"] += 1
            state["episode_return"] = state["episode_profit"] = 0.0
            state["episode_length"] = 0
            if state["step_count"] < self.cfg.total_steps:
                new_obs, _ = self.env.reset(seed=state["episode_seed"])
                ACCOUNTING["env_resets"] += 1
                state["full_obs"] = np.asarray(new_obs, dtype=np.float32)
                state["critic_hidden_list"] = self.trainer.init_hidden(batch_size=1)
                state["prev_actions"] = [np.zeros(len(a.mv_indices), dtype=np.float32) for a in AGENT_CFGS]
        if len(state["buffers"][0]["obs"]) >= self.cfg.rollout_horizon:
            self.update()
        return row

    def update(self):
        state = self.state
        collected = len(state["buffers"][0]["obs"])
        if not collected:
            raise ValueError("No rollout to update")
        if collected < self.cfg.min_rollout_steps and not self.cfg.allow_short_final_rollout:
            state["buffers"] = _empty_buffers()
            return {"rollout/skipped_short": 1.0}
        if state["last_transition_done"]:
            last_values = [0.0] * 3
        else:
            tail = make_local_obs(state["full_obs"], self.obs_rms, update_rms=False)
            last_values, _ = self.trainer.value(tail, state["prev_actions"], state["critic_hidden_list"])
        rollouts = buffers_to_rollouts(state["buffers"])
        before_steps = optimizer_step_totals(self.trainer)
        logs = self.trainer.update(agent_rollouts=rollouts, last_values=last_values,
                                   global_step=state["step_count"])
        after_steps = optimizer_step_totals(self.trainer)
        ACCOUNTING["ppo_updates"] += 1
        for owner in ACCOUNTING["optimizer_steps"]:
            ACCOUNTING["optimizer_steps"][owner] += after_steps[owner] - before_steps[owner]
        state["update_count"] += 1
        state["last_logs"] = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        state["log_rows"] += 1
        log_row = {"step": state["step_count"], "update": state["update_count"],
                   "logs": state["last_logs"]}
        state["update_history"].append(log_row)
        with (self.run_root / "updates.jsonl").open("a") as f:
            f.write(json.dumps(log_row) + "\n")
            f.flush(); os.fsync(f.fileno())
        state["buffers"] = _empty_buffers()
        state["episodes_in_rollout"] = 0
        return logs

    def _checkpoint_event(self, event: str, **fields):
        path = self.run_root / "checkpoint_events.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"event": event, "step": self.state["step_count"],
                                "variant": self.variant, "seed": self.seed, **fields}) + "\n")
            f.flush(); os.fsync(f.fileno())

    def checkpoint(self, *, final=False):
        step = self.state["step_count"]
        validate_collector(self.state, self.env)
        folder = self.run_root / ("final" if final else "checkpoints")
        path = folder / ("train_resume.pt" if final else f"interim_step_{step}.pt")
        folder.mkdir(parents=True, exist_ok=True)
        if not final:
            reconcile_interim_orphans(self.run_root, variant=self.variant, seed=self.seed,
                                      source_sha256=source_sha256())
        previous = read_verified_record(self.run_root, final=final)
        if previous is not None:
            old_path = self.run_root / previous["checkpoint_relpath"]
            verify_published_checkpoint(old_path, variant=self.variant, seed=self.seed,
                                        source_sha256=source_sha256())
        if path.exists():
            raise CheckpointTrustError("Refusing to overwrite an existing published checkpoint")
        candidate = folder / "staging" / f"candidate_{step}_{os.urandom(8).hex()}.pt"
        payload = checkpoint_payload(self.trainer, self.env, self.obs_rms, self.state,
                                     seed=self.seed, source_sha256=source_sha256(),
                                     runtime_config=asdict(self.cfg))
        committed = False
        published = False
        self._checkpoint_event("candidate_start", path=str(candidate), final=bool(final))
        try:
            info = save_atomic(candidate, payload)
            payload_sha = info["sha256"]
            if file_sha256(candidate) != info["sha256"] or candidate.stat().st_size != info["size_bytes"]:
                raise CheckpointTrustError("Candidate changed after atomic write")
            verify_root = self.run_root / "verification" / f"step_{step}_{candidate.stem}"
            fresh = type(self).resume(self.cfg, self.variant, self.seed,
                                         verify_root / "reload", candidate,
                                         _candidate_probe=True)
            reloaded = checkpoint_payload(fresh.trainer, fresh.env, fresh.obs_rms, fresh.state,
                                          seed=self.seed, source_sha256=source_sha256(),
                                          runtime_config=asdict(self.cfg))
            if not _equal(payload, reloaded):
                raise RuntimeError("Candidate reload state mismatch")
            # This probe runs in a fresh process; the parent clone supplies an
            # independent one-step continuation for exact comparison.
            child_result = verify_root / "fresh_process_step.json"
            child_result.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run([sys.executable, "-B", "-m", self._checkpoint_probe_module(),
                "--phase6-verify-candidate", str(candidate), str(child_result),
                self.variant, str(self.seed)], cwd=ROOT, capture_output=True, text=True,
                timeout=180)
            if proc.returncode:
                raise RuntimeError(f"Fresh-process candidate probe failed: {proc.stderr.strip()}")
            left = type(self).resume(self.cfg, self.variant, self.seed,
                                        verify_root / "parent", candidate,
                                        _candidate_probe=True)
            restore_rng(payload["rng"])
            left_step = left.step()
            child = json.loads(child_result.read_text())
            if not child["reload_exact"] or not _equal(left_step, child["step"]):
                raise RuntimeError("Fresh-process candidate continuation mismatch")
            if hasattr(left, "continuation_digest") and child.get("continuation_digest") != left.continuation_digest():
                raise RuntimeError("Fresh-process model/optimizer/logger continuation mismatch")
            self._checkpoint_event("candidate_verified", path=str(candidate),
                                   sha256=info["sha256"], size_bytes=info["size_bytes"])
            os.replace(candidate, path)
            fsync_directory(candidate.parent)
            fsync_directory(folder)
            published = True
            info = {"path": str(path), "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size}
            if info["sha256"] != payload_sha:
                raise CheckpointTrustError("Published checkpoint differs from verified candidate")
            record = verified_record(self.run_root, path, info, step=step,
                update_count=self.state["update_count"],
                collector_cursor=len(self.state["buffers"][0]["obs"]),
                variant=self.variant, seed=self.seed, source_sha256=source_sha256(),
                final=final,
                previous=({"checkpoint_relpath": previous["checkpoint_relpath"],
                           "step": previous["step"], "sha256": previous["sha256"]}
                          if previous else None))
            manifest_path = verified_manifest_path(self.run_root, final=final)
            save_json_atomic(manifest_path, record)
            verify_published_checkpoint(path, variant=self.variant, seed=self.seed,
                                        source_sha256=source_sha256())
            committed = True
            self._checkpoint_event("manifest_committed", path=str(path),
                                   manifest=str(manifest_path),
                                   manifest_sha256=file_sha256(manifest_path),
                                   checkpoint_sha256=info["sha256"])
            if previous is not None and not final:
                old_path = self.run_root / previous["checkpoint_relpath"]
                if old_path != path and old_path.exists():
                    old_path.unlink()
                    fsync_directory(old_path.parent)
                    self._checkpoint_event("previous_interim_removed", path=str(old_path))
            return {**info, "step": step, "fresh_reload_exact": True,
                    "one_step_continuation_exact": True, "final": bool(final),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": file_sha256(manifest_path)}
        except Exception as exc:
            # A committed manifest is authoritative even when old-file cleanup
            # fails. Before commit, the preceding manifest remains authoritative.
            if not committed:
                manifest_path = verified_manifest_path(self.run_root, final=final)
                try:
                    current = read_verified_record(self.run_root, final=final)
                    if current is not None and current.get("checkpoint_relpath") == str(path.relative_to(self.run_root)):
                        if previous is not None:
                            save_json_atomic(manifest_path, previous)
                        else:
                            quarantine_folder = self.run_root / "checkpoints/quarantine"
                            quarantine_folder.mkdir(parents=True, exist_ok=True)
                            os.replace(manifest_path, quarantine_folder / f"uncommitted_manifest_{step}.json")
                            fsync_directory(manifest_path.parent)
                            fsync_directory(quarantine_folder)
                except Exception as rollback_exc:
                    self._checkpoint_event("manifest_rollback_failed",
                                           error=f"{type(rollback_exc).__name__}: {rollback_exc}")
                for untrusted in (candidate, path if published else candidate):
                    if untrusted.exists():
                        quarantine_file(untrusted, self.run_root,
                                        reason=f"candidate_failed:{type(exc).__name__}:{exc}")
            self._checkpoint_event("candidate_failed", error=f"{type(exc).__name__}: {exc}",
                                   committed=committed, candidate=str(candidate))
            raise
        finally:
            restore_rng(payload["rng"])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("F", "M", "G"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--rollout-horizon", type=int, default=1024)
    p.add_argument("--min-rollout-steps", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--checkpoint-steps", type=int, nargs="*", default=(50000, 100000, 150000))
    p.add_argument("--resume", type=Path)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--test-checkpoint-interval", type=int)
    return p.parse_args(argv)


def _verify_candidate_entry(argv):
    """Internal fresh-process probe; never publishes or selects a checkpoint."""
    if len(argv) != 4:
        raise ValueError("Candidate probe requires path, output, variant, seed")
    path, output, variant, seed = Path(argv[0]), Path(argv[1]), argv[2], int(argv[3])
    start_wall, start_cpu = time.monotonic(), time.process_time()
    payload = load_payload(path, variant=variant, source_sha256=source_sha256())
    if payload["seed"] != seed:
        raise ValueError("Candidate probe seed mismatch")
    cfg_values = unpack(payload["runtime_config"])
    cfg_values["model_cfg"] = SplitFLModelConfig(**cfg_values["model_cfg"])
    cfg = ExperimentConfig(**cfg_values)
    fresh = Phase4Session.resume(cfg, variant, seed, output.parent / "child_reload",
                                 path, _candidate_probe=True)
    reloaded = checkpoint_payload(fresh.trainer, fresh.env, fresh.obs_rms, fresh.state,
                                  seed=seed, source_sha256=source_sha256(),
                                  runtime_config=asdict(cfg))
    exact = _equal(payload, reloaded)
    if not exact:
        raise RuntimeError("Fresh-process candidate reload state mismatch")
    step = fresh.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"reload_exact": exact, "step": step,
        "process_accounting": ACCOUNTING, "wall_seconds": time.monotonic() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu}) + "\n")


def main(argv=None):
    args = parse_args(argv)
    if args.test_checkpoint_interval is not None and (not args.test_only or args.total_steps > 128):
        raise ValueError("Short checkpoint interval requires --test-only and <=128 total steps")
    if args.total_steps <= 0 or args.rollout_horizon <= 0:
        raise ValueError("Positive total steps and horizon are required")
    cfg = replace(ExperimentConfig(), total_steps=args.total_steps,
                  rollout_horizon=args.rollout_horizon, min_rollout_steps=args.min_rollout_steps,
                  max_steps=args.max_steps)
    steps = set(args.checkpoint_steps)
    if args.test_checkpoint_interval is not None:
        steps = set(range(args.test_checkpoint_interval, args.total_steps, args.test_checkpoint_interval))
    if args.resume is not None and str(args.resume) == "auto":
        reconcile_interim_orphans(args.run_root, variant=args.variant, seed=args.seed,
                                  source_sha256=source_sha256())
    resume_path = (discover_latest_verified(args.run_root, variant=args.variant,
                   seed=args.seed, source_sha256=source_sha256())
                   if args.resume is not None and str(args.resume) == "auto" else args.resume)
    if resume_path is not None:
        reconcile_interim_orphans(args.run_root, variant=args.variant, seed=args.seed,
                                  source_sha256=source_sha256())
    session = (Phase4Session.resume(cfg, args.variant, args.seed, args.run_root, resume_path)
               if resume_path else Phase4Session(cfg, args.variant, args.seed, args.run_root))
    start = time.monotonic()
    checkpoints = []
    while session.state["step_count"] < cfg.total_steps:
        session.step()
        if session.state["step_count"] in steps:
            checkpoints.append(session.checkpoint())
    if session.state["buffers"][0]["obs"]:
        session.update()
    checkpoints.append(session.checkpoint(final=True))
    summary = {"variant": args.variant, "seed": args.seed,
               "run_root": str(session.run_root),
               "env_steps": session.state["step_count"],
               "ppo_updates": session.state["update_count"],
               "wall_seconds": time.monotonic() - start, "checkpoints": checkpoints,
               "source_sha256": source_sha256(), "process_accounting": ACCOUNTING,
               "final_optimizer_steps": optimizer_step_totals(session.trainer)}
    (session.run_root / "phase4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--phase6-verify-candidate":
        _verify_candidate_entry(sys.argv[2:])
    else:
        main()
