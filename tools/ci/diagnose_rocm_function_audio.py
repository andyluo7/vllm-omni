#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run the unchanged Function Expansion suite with audio retention and telemetry."""

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

TEST = Path("tests/e2e/online_serving/test_qwen3_omni_expansion.py")
TEST_SHA256 = "678983311bbbacbd5da37fe52c96e94e9c976564de2e57733e98ca11c0506739"
CASES = ("function_expansion",)
SOURCE_HASHES = {
    str(TEST): TEST_SHA256,
    "tests/helpers/assertions.py": "2c95d3912086900e5e5a90d566f843436838819b7fdb17994b8ad9820657724d",
    "tests/helpers/media.py": "8637b47968273a07f4abced80c424aca3c1bec181acecb24cdf39ac213129320",
    "tests/helpers/client.py": "de47d6155aa2a1b97bf946186778d610f7bcdbff4f053a3f6e806ce56e7cc048",
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
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, expected in SOURCE_HASHES.items():
        digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Function Expansion source changed: {filename}: {digest}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        "-v",
        "-ra",
        "--tb=short",
        str(TEST),
        "-p",
        "rocm_function_audio_capture",
        "-m",
        "full_model and rocm and MI325 and cards_2",
        "--run-level",
        "full_model",
        "--durations=0",
        f"--junitxml={args.output / 'pytest.xml'}",
    ]
    os.environ["ROCM_FUNCTION_AUDIO_OUTPUT_DIR"] = str(args.output / "requests")
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    os.environ["VLLM_CI_QWEN3_OMNI_SEED"] = "42"
    os.environ["QWEN3_OMNI_FUNCTION_ARTIFACT_DIR"] = str(args.output)
    plugin_path = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = plugin_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    identity = {
        "case": args.case,
        "source_sha256": SOURCE_HASHES,
        "command": command,
        "image_source": "d7f463a646641c022dd929e5b3286f15f51d9f01",
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
            for _ in range(690):
                record = json.dumps(snapshot())
                handle.write(record + "\n")
                handle.flush()
                with output_lock:
                    print("ROCM_FUNCTION_MEMORY " + record, flush=True)
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
    deadline = started + 10200
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
    print("ROCM_FUNCTION_RESULT " + json.dumps(final), flush=True)
    if final["timed_out"]:
        return 124
    if interrupted:
        return 143
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
