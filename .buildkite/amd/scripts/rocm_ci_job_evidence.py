#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Supervise one native AMD job and validate its retained execution evidence."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from xml.etree.ElementTree import ParseError

from rocm_ci_evidence import _junit_counts, capture_processes, check_cleanup

# Keep the original shell command and first-error policy. Pytest options,
# selections, deadlines and per-job teardown remain in the source pipelines.
COMMAND_SHELL = """
set -E
test_status=0
trap '
    command_status=$?
    if (( test_status == 0 || (test_status == 5 && command_status != 5) )); then
        test_status=${command_status}
    fi
' ERR
eval "$1"
exit "${test_status}"
"""


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def begin(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        directory / "job-start.json",
        {
            "started_at_epoch": time.time(),
            "commit": os.environ.get("BUILDKITE_COMMIT", "unknown"),
            "job_id": os.environ.get("BUILDKITE_JOB_ID", "local"),
            "expected_gpus": os.environ.get("VLLM_CI_EXPECTED_GPU_COUNT", "1"),
            "runtime": os.environ.get("AMD_CI_RUNTIME", "unknown"),
        },
    )
    capture_processes(directory / "processes-before.txt")


def _signal_group(process: subprocess.Popen, signum: int) -> bool:
    try:
        os.killpg(process.pid, signum)
        return True
    except ProcessLookupError:
        return False


def run(directory: Path, commands: str) -> int:
    received: list[int] = []
    handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, lambda value, frame: received.append(value))
    started = time.monotonic()
    process = subprocess.Popen(
        ["/bin/bash", "-o", "pipefail", "-c", COMMAND_SHELL, "_", commands],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = None
    exited_at = None
    actions = []
    incomplete_log = False
    try:
        with (directory / "job.log").open("wb") as output:
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if received and deadline is None:
                    _signal_group(process, received[0])
                    actions.append({"reason": "job_signal", "signal": received[0]})
                    deadline = now + 5
                if process.poll() is not None and exited_at is None:
                    exited_at = now
                if exited_at is not None and now - exited_at >= 1 and deadline is None:
                    if _signal_group(process, signal.SIGTERM):
                        actions.append({"reason": "surviving_process_group", "signal": int(signal.SIGTERM)})
                    deadline = now + 5
                if deadline is not None and now >= deadline:
                    if _signal_group(process, signal.SIGKILL):
                        actions.append({"reason": "termination_deadline", "signal": int(signal.SIGKILL)})
                    if now >= deadline + 2 and selector.get_map():
                        incomplete_log = True
                        break
                for key, _ in selector.select(timeout=0.2):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output.write(chunk)
                    output.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            # A daemon can close its inherited output stream but still retain
            # the process group. Terminate only this job's owned group.
            if _signal_group(process, signal.SIGTERM):
                actions.append({"reason": "surviving_process_group", "signal": int(signal.SIGTERM)})
                until = time.monotonic() + 5
                while time.monotonic() < until:
                    if not _signal_group(process, 0):
                        break
                    time.sleep(0.1)
                if _signal_group(process, signal.SIGKILL):
                    actions.append({"reason": "termination_deadline", "signal": int(signal.SIGKILL)})
        code = process.wait(timeout=5)
    finally:
        selector.close()
        process.stdout.close()
        if process.poll() is None:
            _signal_group(process, signal.SIGKILL)
            process.wait(timeout=5)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    # Preserve the conventional shell signal status and the original failure.
    exit_code = 128 + received[0] if received else (128 - code if code < 0 else code)
    _write_json(
        directory / "command-result.json",
        {
            "exit_status": exit_code,
            "runtime_seconds": time.monotonic() - started,
            "termination_actions": actions,
            "incomplete_log": incomplete_log,
        },
    )
    return exit_code


def finish(directory: Path, exit_code: int) -> int:
    problems = []
    invocations = []
    totals = {"selected": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0, "deselected": 0}
    for path in sorted((directory / "pytest").glob("*/result.json")):
        try:
            invocation = json.loads(path.read_text(encoding="utf-8"))
            invocation["report_directory"] = str(path.parent.relative_to(directory))
            if invocation["state"] != "finished":
                problems.append(f"unfinished pytest invocation: {invocation['report_directory']}")
            counts = _junit_counts([path.parent / "pytest.xml"])
            counts["deselected"] = invocation["deselected"]
            invocation["counts"] = counts
            for key in totals:
                totals[key] += counts[key]
            if counts["passed"] + counts["failed"] == 0:
                problems.append(f"pytest invocation executed no tests: {invocation['report_directory']}")
            if counts["failed"] or counts["errors"] or invocation.get("exit_status") != 0:
                problems.append(f"pytest invocation failed: {invocation['report_directory']}")
            invocations.append(invocation)
        except (OSError, ValueError, KeyError, RuntimeError, ParseError) as error:
            problems.append(f"invalid pytest evidence at {path.parent.name}: {error}")
    if not invocations:
        problems.append("no complete pytest invocation reports")
    for name in ("job-start.json", "environment.txt", "processes-before.txt", "job.log", "command-result.json"):
        if not (directory / name).is_file():
            problems.append(f"missing required evidence: {name}")
    if (directory / "command-result.json").is_file():
        command = json.loads((directory / "command-result.json").read_text(encoding="utf-8"))
        if command["incomplete_log"]:
            problems.append("command output stream did not terminate")
        if any(action["reason"] == "surviving_process_group" for action in command["termination_actions"]):
            problems.append("test commands left a surviving process group")
    try:
        check_cleanup(
            directory / "processes-before.txt", directory / "processes-after.txt", directory / "process-cleanup.txt", 1
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        problems.append(f"cleanup failed: {error}")
        if not (directory / "process-cleanup.txt").exists():
            (directory / "process-cleanup.txt").write_text("status=FAIL cleanup_evidence_unavailable=1\n")
    started = json.loads((directory / "job-start.json").read_text()) if (directory / "job-start.json").exists() else {}
    result = {
        "status": "FAIL" if problems or exit_code else "PASS",
        "exit_status": exit_code,
        "runtime_seconds": time.time() - started.get("started_at_epoch", time.time()),
        "counts": totals,
        "invocations": invocations,
        "problems": problems,
    }
    _write_json(directory / "job-result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "exit_status", "counts", "problems")}))
    # Evidence failure must not hide the original command/preflight status.
    return exit_code or int(bool(problems))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("begin", "run", "finish"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--commands")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    if args.action == "begin":
        begin(args.directory)
    elif args.action == "run":
        if args.commands is None:
            parser.error("run requires --commands")
        sys.exit(run(args.directory, args.commands))
    else:
        sys.exit(finish(args.directory, args.exit_code))


if __name__ == "__main__":
    main()
