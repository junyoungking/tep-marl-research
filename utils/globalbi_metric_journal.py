"""Single-writer scalar/transition journal for the fresh full-logger generation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

SCHEMA = "globalbi_full_metrics_v1"


def content_digest(value):
    """Hash computation contents, independent of checkpoint ZIP metadata."""
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            visit(("tensor", str(x.dtype), tuple(x.shape)))
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            visit(("ndarray", str(x.dtype), tuple(x.shape)))
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            h.update(b"{")
            for key in sorted(x, key=str):
                visit(key); visit(x[key])
            h.update(b"}")
        elif isinstance(x, (list, tuple)):
            h.update(b"[" if isinstance(x, list) else b"(")
            for item in x:
                visit(item)
            h.update(b"]")
        else:
            h.update(json.dumps(x, sort_keys=True, allow_nan=False).encode() + b"\0")
    visit(value)
    return h.hexdigest()


def prefix_bytes(origin: Path, watermark: dict) -> bytes:
    required = {"schema", "bytes", "seq", "sha256", "last_scalar_steps"}
    if set(watermark) != required or watermark["schema"] != SCHEMA:
        raise ValueError("Metric journal watermark schema mismatch")
    if not isinstance(watermark["bytes"], int) or watermark["bytes"] <= 0:
        raise ValueError("Metric journal byte watermark invalid")
    with origin.open("rb") as f:
        prefix = f.read(watermark["bytes"])
    if len(prefix) != watermark["bytes"] or not prefix.endswith(b"\n"):
        raise ValueError("Metric journal prefix incomplete")
    if hashlib.sha256(prefix).hexdigest() != watermark["sha256"]:
        raise ValueError("Metric journal prefix SHA mismatch")
    first = json.loads(prefix.split(b"\n", 1)[0])
    last = json.loads(prefix.rstrip(b"\n").rsplit(b"\n", 1)[-1])
    if first.get("kind") != "header" or first.get("schema") != SCHEMA or last.get("seq") != watermark["seq"]:
        raise ValueError("Metric journal sequence/header mismatch")
    actual_last = {}
    for expected_seq, line in enumerate(prefix.splitlines()):
        record = json.loads(line)
        if record.get("seq") != expected_seq:
            raise ValueError("Metric journal committed sequence gap")
        if record.get("kind") == "scalar":
            tag, step = record["tag"], record["step"]
            if (tag not in LEGACY_156_TAGS or not isinstance(step, int)
                    or step <= actual_last.get(tag, -1)):
                raise ValueError("Metric journal committed scalar duplicate/order/tag mismatch")
            actual_last[tag] = step
    if actual_last != watermark["last_scalar_steps"]:
        raise ValueError("Metric journal scalar watermark mismatch")
    return prefix


class MetricJournalWriter:
    """One writer adapter used by both the runner and shared PPO metric logger."""
    def __init__(self, path: Path, identity: dict, *, tb_writer=None, watermark=None,
                 origin=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tb_writer = tb_writer
        self.identity = identity
        self.last_scalar_steps = {}
        self.seq = -1
        self.bytes = 0
        self.hash = hashlib.sha256()
        self.scalar_counts = {}
        if watermark is None:
            if self.path.exists():
                raise ValueError("Fresh metric journal already exists")
            self.file = self.path.open("xb")
            self.append("header", schema=SCHEMA, **identity)
        else:
            prefix = prefix_bytes(Path(origin), watermark)
            first = json.loads(prefix.split(b"\n", 1)[0])
            for k, v in identity.items():
                if first.get(k) != v:
                    raise ValueError(f"Metric journal identity mismatch: {k}")
            if self.path.exists():
                with self.path.open("rb") as f:
                    actual_prefix = f.read(len(prefix))
                if actual_prefix != prefix:
                    raise ValueError("Local metric journal committed prefix differs")
                if self.path.stat().st_size > len(prefix):
                    folder = self.path.parent / "abandoned_attempt_logs"
                    folder.mkdir(exist_ok=True)
                    archive = folder / f"metrics.jsonl.{time.time_ns()}"
                    os.replace(self.path, archive)
                    from scripts.phase4_exact_resume import fsync_directory
                    fsync_directory(folder)
                    fsync_directory(self.path.parent)
            if not self.path.exists():
                temp = self.path.with_name(".metrics.restore.tmp")
                with temp.open("xb") as f:
                    f.write(prefix); f.flush(); os.fsync(f.fileno())
                os.replace(temp, self.path)
                from scripts.phase4_exact_resume import fsync_directory
                fsync_directory(self.path.parent)
            self.seq = watermark["seq"]
            self.bytes = len(prefix)
            self.hash.update(prefix)
            self.last_scalar_steps = dict(watermark["last_scalar_steps"])
            self.file = self.path.open("ab")
        self.closed = False

    def append(self, kind, **data):
        self.seq += 1
        record = {"seq": self.seq, "kind": kind, **data}
        encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        self.file.write(encoded)
        self.hash.update(encoded)
        self.bytes += len(encoded)

    def add_scalar(self, tag, value, step, **kwargs):
        tag, step, value = str(tag), int(step), float(value)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite metric from producer: {tag} step={step}")
        if step <= self.last_scalar_steps.get(tag, -1):
            raise ValueError(f"Duplicate/out-of-order scalar: {tag} step={step}")
        self.append("scalar", tag=tag, value=value, step=step)
        self.last_scalar_steps[tag] = step
        self.scalar_counts[tag] = self.scalar_counts.get(tag, 0) + 1
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step, **kwargs)

    def flush(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def watermark(self):
        self.flush()
        return {"schema": SCHEMA, "seq": self.seq, "bytes": self.bytes,
                "sha256": self.hash.hexdigest(),
                "last_scalar_steps": dict(self.last_scalar_steps)}

    def close(self):
        if not self.closed:
            self.flush()
            self.file.close()
            if self.tb_writer is not None:
                self.tb_writer.close()
            self.closed = True


def scalar_timeline(path: Path):
    expected = 0
    last = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("seq") != expected:
                raise ValueError("Metric journal sequence gap")
            expected += 1
            if row["kind"] == "scalar":
                key, step = row["tag"], row["step"]
                if step <= last.get(key, -1):
                    raise ValueError("Metric journal duplicate scalar")
                last[key] = step
                yield row


# Preserved original156-tag surface; provenance in the fresh-generation audit.
LEGACY_156_TAGS = (
    'action/absolute/mv_1/value',
    'action/absolute/mv_6/value',
    'action/absolute/mv_8/value',
    'action/absolute_mean',
    'action/delta/agent_0/mv_1/value',
    'action/delta/agent_1/mv_8/value',
    'action/delta/agent_2/mv_6/value',
    'action/delta_abs_mean',
    'action/policy/agent_0/mv_1/value',
    'action/policy/agent_1/mv_8/value',
    'action/policy/agent_2/mv_6/value',
    'action/policy_abs_mean',
    'action/policy_max_abs',
    'agent_0/adv_mean_raw',
    'agent_0/adv_std_normed',
    'agent_0/adv_std_raw',
    'agent_0/approx_kl',
    'agent_0/approx_kl_post',
    'agent_0/clipfrac',
    'agent_0/debug/actor_grad_norm',
    'agent_0/debug/actor_param_delta',
    'agent_0/entropy',
    'agent_0/loss/priv_ar_raw',
    'agent_0/loss/priv_dcor_weighted',
    'agent_0/loss/priv_nr_raw',
    'agent_0/loss/priv_nr_weighted',
    'agent_0/loss/privacy_backward_total',
    'agent_0/loss/recon_attacker',
    'agent_0/loss_actor',
    'agent_0/loss_critic',
    'agent_0/privacy/ar_lambda_eff',
    'agent_0/privacy/dcor_log',
    'agent_0/privacy/dcor_raw',
    'agent_0/privacy/recon_mse',
    'agent_0/ratio_std',
    'agent_1/adv_mean_raw',
    'agent_1/adv_std_normed',
    'agent_1/adv_std_raw',
    'agent_1/approx_kl',
    'agent_1/approx_kl_post',
    'agent_1/clipfrac',
    'agent_1/debug/actor_grad_norm',
    'agent_1/debug/actor_param_delta',
    'agent_1/entropy',
    'agent_1/loss/priv_ar_raw',
    'agent_1/loss/priv_dcor_weighted',
    'agent_1/loss/priv_nr_raw',
    'agent_1/loss/priv_nr_weighted',
    'agent_1/loss/privacy_backward_total',
    'agent_1/loss/recon_attacker',
    'agent_1/loss_actor',
    'agent_1/loss_critic',
    'agent_1/privacy/ar_lambda_eff',
    'agent_1/privacy/dcor_log',
    'agent_1/privacy/dcor_raw',
    'agent_1/privacy/recon_mse',
    'agent_1/ratio_std',
    'agent_2/adv_mean_raw',
    'agent_2/adv_std_normed',
    'agent_2/adv_std_raw',
    'agent_2/approx_kl',
    'agent_2/approx_kl_post',
    'agent_2/clipfrac',
    'agent_2/debug/actor_grad_norm',
    'agent_2/debug/actor_param_delta',
    'agent_2/entropy',
    'agent_2/loss/priv_ar_raw',
    'agent_2/loss/priv_dcor_weighted',
    'agent_2/loss/priv_nr_raw',
    'agent_2/loss/priv_nr_weighted',
    'agent_2/loss/privacy_backward_total',
    'agent_2/loss/recon_attacker',
    'agent_2/loss_actor',
    'agent_2/loss_critic',
    'agent_2/privacy/ar_lambda_eff',
    'agent_2/privacy/dcor_log',
    'agent_2/privacy/dcor_raw',
    'agent_2/privacy/recon_mse',
    'agent_2/ratio_std',
    'critic/value_loss_agent_0',
    'critic/value_loss_agent_1',
    'critic/value_loss_agent_2',
    'debug/actor_grad_norm',
    'debug/actor_logp_delta_post',
    'debug/actor_param_delta',
    'debug/actor_param_delta_min_epoch',
    'debug/actor_post_kl_available',
    'debug/critic_dL_dBI_requires_grad_rate',
    'debug/critic_gru_done_resets',
    'debug/critic_gru_reset_enabled',
    'debug/critic_local_changed_on_handoff',
    'debug/critic_server_changed_during_local_step',
    'debug/critic_server_changed_on_server_step',
    'debug/critic_server_grads_created_during_local',
    'debug/hidden_resets',
    'debug/local_actor_reward_value_free',
    'debug/privacy_reconstructor_detached',
    'debug/server_advantage_owner',
    'episode/length',
    'episode/return',
    'fedavg/applied',
    'fedavg/round',
    'fedavg/scope_enhancer',
    'fedavg/scope_gru',
    'grad_norm/bi_encoder_0',
    'grad_norm/bi_encoder_1',
    'grad_norm/bi_encoder_2',
    'grad_norm/priv_defense_0',
    'grad_norm/priv_defense_1',
    'grad_norm/priv_defense_2',
    'loss/actor',
    'loss/critic',
    'loss/critic_first_epoch',
    'loss/critic_last_epoch',
    'loss/entropy',
    'loss/privacy_backward_total',
    'loss/privacy_recon_attacker_total',
    'loss/privacy_total',
    'loss/recon_attacker',
    'privacy/attacker_applied',
    'privacy/bi_grad_norm_priv_0',
    'privacy/bi_grad_norm_priv_1',
    'privacy/bi_grad_norm_priv_2',
    'privacy/bi_grad_norm_td_0',
    'privacy/bi_grad_norm_td_1',
    'privacy/bi_grad_norm_td_2',
    'privacy/bi_grad_ratio_0',
    'privacy/bi_grad_ratio_1',
    'privacy/bi_grad_ratio_2',
    'privacy/bi_grad_task_0',
    'privacy/bi_grad_task_1',
    'privacy/bi_grad_task_2',
    'privacy/full_privacy_applied',
    'privacy/recon_mse',
    'privacy/reconstructor_grad_0',
    'privacy/reconstructor_grad_1',
    'privacy/reconstructor_grad_2',
    'ratio/privacy_backward_to_critic',
    'raw/cost',
    'raw/prod',
    'raw/profit',
    'raw/shutdown_rate',
    'rollout/episodes_per_update',
    'rollout/length',
    'rollout/mean_episode_len',
    'rollout/num_episodes',
    'rollout/skipped_short',
    'train/actor_epochs_used',
    'train/approx_kl',
    'train/approx_kl_epoch0',
    'train/approx_kl_last',
    'train/approx_kl_post',
    'train/clipfrac',
    'train/critic_epochs_used',
    'train/update_count',
    'train/updates_per_env_step',
)
