#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run unchanged continuous-window tests with process-local observation hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SOURCE = "ad7c78e1589347c6b0580faf69afd107fdbaf5b1"
NODE = "tests/e2e/online_serving/test_minicpmo_4_5_window.py::test_window_continuous_input"
SOURCE_HASHES = {
    "tests/e2e/online_serving/test_minicpmo_4_5_window.py": (
        "1ea8d035f6d0f5913c3c7950137fe514cb826ec6f83fa97a13cd0e6a7b4946ab"
    ),
    "tests/e2e/online_serving/helpers/minicpmo_window_e2e.py": (
        "819661687ef96c511c375bf47937fd9accd20be208d0844272f24b22357b4d26"
    ),
    "tests/e2e/online_serving/helpers/minicpmo_4_5_duplex.py": (
        "efb2855c007f483567b0aced00307deebd513103ae8e8b1ddc83ccddfdf45673"
    ),
    "vllm_omni/clients/duplex.py": "8e8db727e7b4e17ca499fdff20dcc96453653439e20ae53fc6f0451ae9f0c449",
    "vllm_omni/engine/duplex/session/runner.py": "b318b32fb65870688e603985c43c2178a78f6e64ec45d0c2cc4f2cc4868c4358",
    "vllm_omni/engine/duplex/session/model_channel.py": (
        "9eb911992b89160a7f28abccf99197135f6f8e1819e84d87d48c80a973ac7dd9"
    ),
    "vllm_omni/engine/duplex/session/append_task.py": (
        "7fdc36ecb0988b7cf647396b1b89df45243ee27ddd86bbc75a8ec93808564981"
    ),
    "vllm_omni/engine/duplex/session/emitter.py": ("4a489202ba18882e624b8f5b6ff2626fbd543900ed1c9f69bdad2defcc5cbadf"),
    "vllm_omni/model_executor/models/minicpmo_4_5/duplex/plugin.py": (
        "2729d20df5b7a1e9594f0a46b9e2f8a8e9ae2543bab2e15362e7032bfb4e80a3"
    ),
    "vllm_omni/model_executor/models/minicpmo_4_5/duplex/policy.py": (
        "683ec0e7622310075ec50ba20aa09c84743db0b7ea4551cc6fc403ce85de2351"
    ),
}


def bounded_read(path: Path, limit: int = 8192) -> str:
    try:
        with path.open("rb") as handle:
            return handle.read(limit).decode("utf-8", "replace")
    except OSError as exc:
        return f"unavailable:{exc.errno}"


def snapshot() -> dict:
    memory = {}
    for base in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
        for name in (
            "memory.current",
            "memory.max",
            "memory.peak",
            "memory.events",
            "memory.events.local",
            "memory.stat",
            "memory.swap.current",
            "memory.swap.max",
            "memory.usage_in_bytes",
            "memory.limit_in_bytes",
            "memory.max_usage_in_bytes",
            "memory.failcnt",
            "memory.oom_control",
        ):
            path = base / name
            if path.exists():
                memory[str(path)] = bounded_read(path)
    processes = []
    scanned = 0
    # Read status only; command lines and environment are excluded.
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            scanned += 1
            if scanned > 512:
                break
            fields = {}
            for line in bounded_read(Path(entry.path) / "status").splitlines():
                key, _, value = line.partition(":")
                if key in {"Name", "Pid", "PPid", "VmRSS", "VmHWM", "VmSize", "Threads"}:
                    fields[key] = value.strip()
            if fields:
                processes.append(fields)
    processes.sort(key=lambda item: int(item.get("VmRSS", "0").split()[0]), reverse=True)
    return {
        "epoch": time.time(),
        "cgroup": memory,
        "proc_meminfo": bounded_read(Path("/proc/meminfo")),
        "cgroup_membership": bounded_read(Path("/proc/self/cgroup")),
        "processes_scanned": scanned,
        "largest_processes": processes[:24],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, expected in SOURCE_HASHES.items():
        digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Production image source changed: {filename}: {digest}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "window_trace",
        "-s",
        "-v",
        "-ra",
        "--tb=short",
        NODE,
        "--run-level",
        "full_model",
        "--durations=0",
        f"--junitxml={args.output / 'pytest.xml'}",
    ]
    os.environ["ROCM_WINDOW_TRACE_DIR"] = str(args.output / "traces")
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    plugin_path = str(Path(__file__).resolve().parent / "minicpm_window_capture")
    os.environ["PYTHONPATH"] = plugin_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    identity = {
        "node": NODE,
        "production_test_unchanged": True,
        "observation_hooks": True,
        "completion_timeout_seconds": 240,
        "source_sha256": SOURCE_HASHES,
        "command": command,
        "image_source": SOURCE,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "DIFFUSION_ATTENTION_BACKEND",
                "VLLM_CI_QWEN3_OMNI_SEED",
                "PYTHONHASHSEED",
                "VLLM_CI_EXPECTED_GPU_COUNT",
                "BUILDKITE_COMMIT",
                "BUILDKITE_JOB_ID",
            )
        },
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2))
    print(json.dumps(identity), flush=True)
    stopped = threading.Event()
    output_lock = threading.Lock()

    def monitor() -> None:
        with (args.output / "memory.jsonl").open("w") as handle:
            for _ in range(200):
                record = json.dumps(snapshot())
                handle.write(record + "\n")
                handle.flush()
                with output_lock:
                    print("ROCM_WINDOW_MEMORY " + record, flush=True)
                if stopped.wait(15):
                    return

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    assert child.stdout is not None
    terminating_at: float | None = None
    interrupted = False

    def stop_group(sig: int) -> None:
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    def terminate(_signum: int, _frame: object) -> None:
        nonlocal terminating_at, interrupted
        interrupted = True
        if terminating_at is None:
            stop_group(signal.SIGTERM)
            terminating_at = time.monotonic()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    started = time.monotonic()
    deadline = started + 50 * 60
    try:
        with selectors.DefaultSelector() as selector, (args.output / "pytest.log").open("wb") as log:
            selector.register(child.stdout, selectors.EVENT_READ)
            while selector.get_map():
                now = time.monotonic()
                # Descendants can retain stdout after the pytest parent exits.
                if now >= deadline and terminating_at is None:
                    stop_group(signal.SIGTERM)
                    terminating_at = now
                elif terminating_at is not None and now - terminating_at >= 60:
                    stop_group(signal.SIGKILL)
                for key, _events in selector.select(timeout=1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(chunk)
                    log.flush()
                    with output_lock:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
            code = child.wait(timeout=65)
    finally:
        stop_group(signal.SIGKILL)
        if child.poll() is None:
            child.wait(timeout=10)
        child.stdout.close()
        stopped.set()
        watcher.join(timeout=5)
    final = {
        "returncode": code,
        "elapsed_seconds": time.monotonic() - started,
        "timed_out": terminating_at is not None and not interrupted,
        "interrupted": interrupted,
        "memory": snapshot(),
    }
    (args.output / "result.json").write_text(json.dumps(final, indent=2))
    print("ROCM_WINDOW_RESULT " + json.dumps(final), flush=True)
    if final["timed_out"]:
        return 124
    if interrupted:
        return 143
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
