# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Observe unchanged duplex methods without changing their return values."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
from pathlib import Path

TARGETS = {
    "vllm_omni.clients.duplex": "client",
    "vllm_omni.engine.duplex.session.runner": "runner",
    "vllm_omni.engine.duplex.session.model_channel": "channel",
    "vllm_omni.engine.duplex.session.append_task": "append",
    "vllm_omni.engine.duplex.session.emitter": "emitter",
    "vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin": "plugin",
}
LOCK = threading.Lock()
OBSERVERS: set[asyncio.Task] = set()
FAILED = False


def compact(value, depth=0, key=""):
    """Keep event identity and tokens; hash media and omit tensor contents."""
    if key.lower() in {"authorization", "hf_token", "vllm_ci_hf_token", "token", "api_key"}:
        return "redacted"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= 512 and not value.startswith("data:"):
            return value
        raw = value.encode("utf-8")
        return {"characters": len(value), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if depth >= 6:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {str(k): compact(v, depth + 1, str(k)) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {"length": len(value), "tail": compact(value[-16:], depth + 1, key)}
        return [compact(v, depth + 1, key) for v in value]
    return {"type": type(value).__name__}


def record(kind, **fields):
    global FAILED
    try:
        path = Path(os.environ["ROCM_WINDOW_TRACE_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        entry = {
            "kind": kind,
            "pid": os.getpid(),
            "epoch": time.time(),
            "monotonic": time.monotonic(),
            **compact(fields),
        }
        with LOCK, (path / f"trace-{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as exc:
        if not FAILED:
            print(f"ROCM_WINDOW_TRACE_ERROR {type(exc).__name__}", file=sys.stderr, flush=True)
            FAILED = True


def task_state(task, *, stack=False):
    if task is None:
        return None
    result = {"done": task.done(), "cancelled": task.cancelled(), "name": task.get_name()}
    if stack and not task.done():
        result["stack"] = [
            {"file": frame.f_code.co_filename, "line": frame.f_lineno, "function": frame.f_code.co_name}
            for frame in task.get_stack(limit=4)
        ]
    return result


def state(owner, *, stack=False):
    try:
        ctx = getattr(owner, "_ctx", None) or getattr(owner, "ctx", None)
        if ctx is None:
            return {}
        session, model, tasks = ctx.session, ctx.model_state, ctx.tasks
        return {
            "session_id": session.session_id,
            "epoch": session.epoch,
            "turn_id": session.turn_id,
            "state": str(session.state),
            "active_request_id": session.active_request_id,
            "active_response_id": session.active_response_id,
            "active_response_turn_id": session.active_response_turn_id,
            "stream_request_id": ctx.run.stream_request_id,
            "closing": ctx.run.closing,
            "input_since_commit": model.input_since_commit,
            "speech_since_commit": model.speech_since_commit,
            "pending_audio_bytes": model.audio_buffer.pending_byte_count,
            "reserved_audio": model.audio_buffer.has_reserved(),
            "retained_commit": model.committed_audio_payload is not None,
            "deferred_response_create": model.deferred_response_create,
            "continuation_owner_id": model.continuation_owner_id,
            "continuation_units": model.continuation_units,
            "pending_silence_owner_id": model.pending_silence_owner_id,
            "last_native_submit_monotonic": model.last_native_submit_monotonic,
            "silence_deadline_monotonic": model.silence_deadline_monotonic,
            "append_tasks": len(tasks.append_tasks),
            "append_tail": task_state(tasks.append_tail, stack=stack),
            "pending_silence": task_state(model.pending_silence_task, stack=stack),
            "worker": task_state(getattr(ctx.services, "_worker", None), stack=stack),
            "background_tasks": [
                task_state(task, stack=stack) for task in list(getattr(ctx.services, "_background_tasks", ()))[:16]
            ],
        }
    except Exception as exc:
        return {"snapshot_error": type(exc).__name__}


def call_data(args, kwargs):
    data = {"kwargs": kwargs}
    if args:
        first = args[0]
        if isinstance(first, dict):
            data["payload"] = first
        elif hasattr(first, "stage_id"):
            data["stage"] = {
                "stage_id": first.stage_id,
                "request_id": getattr(first, "request_id", None),
            }
    return data


async def observe_after_commit(owner):
    for index in range(26):
        record("server.commit_observation", index=index, state=state(owner, stack=True))
        if owner.run.closing:
            return
        await asyncio.sleep(10)


def wrap_async(cls, name):
    original = getattr(cls, name)

    @functools.wraps(original)
    async def wrapped(self, *args, **kwargs):
        record(f"{cls.__name__}.{name}.enter", state=state(self), **call_data(args, kwargs))
        try:
            result = await original(self, *args, **kwargs)
            record(f"{cls.__name__}.{name}.exit", state=state(self), result=result)
            if name == "_start_append" and isinstance(result, asyncio.Task):

                def completed(task):
                    record("server.append_completed", state=state(self), task=task_state(task))

                result.add_done_callback(completed)
            if name == "_on_commit":
                observer = asyncio.create_task(observe_after_commit(self), name="window-trace-observer")
                OBSERVERS.add(observer)
                observer.add_done_callback(OBSERVERS.discard)
            return result
        except BaseException as exc:
            record(f"{cls.__name__}.{name}.raise", state=state(self), error=type(exc).__name__)
            raise

    setattr(cls, name, wrapped)


def patch(module, role):
    if role == "client":
        original = module.EventCollector.add

        @functools.wraps(original)
        def add(self, event, **kwargs):
            raw = event.raw if isinstance(event, module.DuplexEvent) else event
            record("client.received", collector=id(self), event=raw)
            return original(self, event, **kwargs)

        module.EventCollector.add = add
        wrap_async(module.DuplexClient, "_send_command")
    elif role == "runner":
        for name in ("_on_commit", "_start_append", "_schedule_silence_continuation"):
            wrap_async(module.DuplexSessionRunner, name)
        original = module.DuplexSessionRunner.emit

        @functools.wraps(original)
        def emit(self, payload):
            record("server.emitted", state=state(self), event=payload)
            return original(self, payload)

        module.DuplexSessionRunner.emit = emit
    elif role == "channel":
        for name in ("on_stage_output_item", "_on_model_listen", "maybe_continue_response"):
            wrap_async(module.ModelChannel, name)
    elif role == "append":
        wrap_async(module.AppendAttempt, "_submit")
    elif role == "emitter":
        original = module.SessionEmitter.emit

        @functools.wraps(original)
        def emit(self, payload):
            record("server.projected_event", state=state(self), event=payload)
            return original(self, payload)

        module.SessionEmitter.emit = emit
    elif role == "plugin":
        original = module.MiniCPMO45DuplexPlugin.decide_output

        @functools.wraps(original)
        def decide(self, **kwargs):
            result = original(self, **kwargs)
            output = kwargs["output"]
            completions = getattr(output, "outputs", ()) or ()
            completion = completions[0] if completions else None
            record(
                "server.model_decision",
                stage_id=kwargs["stage_id"],
                segment_finished=kwargs["segment_finished"],
                segment_token_ids=kwargs["segment_token_ids"],
                metadata=kwargs["segment_output_metadata"],
                stop_reason=getattr(completion, "stop_reason", None),
                token_ids=getattr(completion, "token_ids", None),
                action=str(result.action) if result is not None else None,
            )
            return result

        module.MiniCPMO45DuplexPlugin.decide_output = decide
    record("hooks.installed", module=module.__name__, role=role)


class TraceLoader(importlib.abc.Loader):
    def __init__(self, loader, role):
        self.loader, self.role = loader, role

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        patch(module, self.role)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class TraceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = TraceLoader(spec.loader, TARGETS[fullname])
        return spec


def install():
    sys.meta_path.insert(0, TraceFinder())
    record("process.trace_enabled")


def pytest_runtest_logstart(nodeid, location):
    record("test.start", nodeid=nodeid)


def pytest_runtest_logreport(report):
    record("test.report", nodeid=report.nodeid, phase=report.when, outcome=report.outcome, duration=report.duration)
