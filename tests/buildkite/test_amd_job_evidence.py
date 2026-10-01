# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPTS = Path(".buildkite/amd/scripts").resolve()
RUNNER = SCRIPTS / "run-amd-test.sh"


def _fixture(tmp_path: Path, commands: str, **overrides: str) -> dict[str, str]:
    # Exercise the actual runner/plugin/supervisor. Stub only GPU and mounted
    # cache preflight, since local contract tests do not have an AMD device.
    (tmp_path / "bin").mkdir()
    for name in ("rocminfo", "findmnt"):
        path = tmp_path / "bin" / name
        path.write_text("#!/bin/sh\nprintf '/persistent test nfs\\n'\n")
        path.chmod(0o755)
    # A CI pod has an isolated process namespace. Scope this local ps probe
    # to the test process tree and its deliberately detached child, so macOS
    # background services cannot masquerade as leaked test processes.
    ps = tmp_path / "bin/ps"
    ps.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess\n"
        "from pathlib import Path\n"
        "rows = subprocess.check_output(['/bin/ps', '-eo', 'pid=,ppid=,lstart=,stat=,comm='], text=True).splitlines()\n"
        "owned = {int(os.environ['ROCM_CI_CONTRACT_ROOT_PID'])}\n"
        "detached = Path('detached.pid')\n"
        "if detached.exists(): owned.add(int(detached.read_text()))\n"
        "while True:\n"
        "    extra = {int(row.split()[0]) for row in rows if int(row.split()[1]) in owned}\n"
        "    if extra <= owned: break\n"
        "    owned.update(extra)\n"
        "for row in rows:\n"
        "    fields = row.split(None, 8)\n"
        "    if int(fields[0]) in owned and int(fields[0]) != os.getpid() and fields[8] not in ('/bin/ps', 'ps'):\n"
        "        print(row)\n"
    )
    ps.chmod(0o755)
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        "__version__ = 'test'\n"
        "version = SimpleNamespace(hip='test')\n"
        "cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, "
        "get_device_name=lambda i: 'stub', get_device_properties=lambda i: "
        "SimpleNamespace(total_memory=1))\n"
        "accelerator = SimpleNamespace(device_count=lambda: 1)\n"
    )
    (tmp_path / "test_sample.py").write_text(
        "import os, signal, time\n"
        "import subprocess, sys\n"
        "import pytest\n"
        "def test_ok():\n    assert True\n"
        "def test_fail():\n    assert False\n"
        "def test_nested_expected_failure():\n"
        "    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'test_sample.py::test_fail'])\n"
        "    assert result.returncode == 1\n"
        "def test_abort():\n    os.kill(os.getpid(), signal.SIGABRT)\n"
        "def test_wait():\n"
        "    open('waiting.pid', 'w').write(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "@pytest.mark.skip(reason='contract probe')\n"
        "def test_skip():\n    pass\n"
    )
    environment = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "PYTHONPATH": str(tmp_path),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_PLUGINS": "",
        "VLLM_CI_DOCKER_DISABLED": "1",
        "VLLM_CI_EXPECTED_GPU_COUNT": "1",
        "VLLM_CI_REQUIRE_PERSISTENT_HF_CACHE": "1",
        "VLLM_CI_JOB_EVIDENCE": "1",
        "BUILDKITE_BUILD_CHECKOUT_PATH": str(tmp_path),
        "BUILDKITE_COMMIT": "contract-probe",
        "BUILDKITE_JOB_ID": f"contract-{tmp_path.name}",
        "ROCM_CI_CONTRACT_ROOT_PID": str(os.getpid()),
        "HF_HOME": str(tmp_path / "hf"),
        "TEST_COMMANDS": commands,
    }
    environment.pop("ROCM_CI_PYTEST_OWNER_PID", None)
    environment.update(overrides)
    return environment


def _run(tmp_path: Path, commands: str, **overrides: str) -> tuple[subprocess.CompletedProcess, dict]:
    result = subprocess.run(
        ["bash", str(RUNNER)],
        env=_fixture(tmp_path, commands, **overrides),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=25,
    )
    report = json.loads((tmp_path / "artifacts/rocm-job/job-result.json").read_text())
    return result, report


def test_nested_repeated_pytest_preserves_explicit_junit_and_selectors(tmp_path: Path) -> None:
    commands = (
        "pytest -q test_sample.py::test_ok --junitxml=original.xml\n"
        "bash -c 'for scope in test_ok test_ok; do "
        "python3 -m pytest -q test_sample.py::$scope; done'\n"
    )
    result, report = _run(tmp_path, commands)
    assert result.returncode == 0, result.stdout
    assert report["status"] == "PASS"
    assert report["counts"]["passed"] == 3
    assert len(report["invocations"]) == 3
    assert (tmp_path / "original.xml").is_file()
    assert len(list((tmp_path / "artifacts/rocm-job/pytest").glob("*/pytest.xml"))) == 3
    original_suite = ElementTree.parse(tmp_path / "original.xml").getroot().find("testsuite")
    assert original_suite is not None
    assert original_suite.attrib["tests"] == "1"


def test_concurrent_pytest_invocations_keep_separate_reports(tmp_path: Path) -> None:
    result, report = _run(tmp_path, "pytest -q test_sample.py::test_ok &\npytest -q test_sample.py::test_ok &\nwait")
    assert result.returncode == 0, result.stdout
    assert len(report["invocations"]) == 2
    assert report["counts"]["passed"] == 2


def test_xdist_controller_reports_every_test_once(tmp_path: Path) -> None:
    pytest.importorskip("xdist")
    result, report = _run(tmp_path, "pytest -p xdist.plugin -n 2 -q test_sample.py::test_ok")
    assert result.returncode == 0, result.stdout
    assert len(report["invocations"]) == 1
    assert report["counts"]["passed"] == 1
    assert report["invocations"][0]["selected"] == 1


def test_expected_child_pytest_failure_belongs_to_parent_assertion(tmp_path: Path) -> None:
    result, report = _run(tmp_path, "pytest -q test_sample.py::test_nested_expected_failure")
    assert result.returncode == 0, result.stdout
    assert len(report["invocations"]) == 1
    assert report["counts"]["passed"] == 1


@pytest.mark.parametrize(
    ("commands", "expected_status", "expected_passed"),
    [
        ("pytest -q test_sample.py::test_fail\npytest -q test_sample.py::test_ok", 1, 1),
        ("pytest -q test_sample.py::test_abort", 134, 0),
        ("pytest -q test_sample.py -k absent", 5, 0),
        ("pytest -q test_sample.py::test_skip", 1, 0),
        ("exit 7", 7, 0),
    ],
)
def test_failed_aborted_empty_and_pre_pytest_commands_retain_status(
    tmp_path: Path, commands: str, expected_status: int, expected_passed: int
) -> None:
    result, report = _run(tmp_path, commands)
    assert result.returncode == expected_status, result.stdout
    assert report["status"] == "FAIL"
    assert report["counts"]["passed"] == expected_passed
    assert (tmp_path / "artifacts/rocm-job/process-cleanup.txt").is_file()
    assert (tmp_path / "artifacts/rocm-job/job.log").is_file()
    if expected_status == 134:
        assert "unfinished pytest invocation" in "\n".join(report["problems"])


def test_native_preflight_failure_retains_final_report(tmp_path: Path) -> None:
    result, report = _run(tmp_path, "pytest -q test_sample.py::test_ok", VLLM_CI_DOCKER_DISABLED="0")
    assert result.returncode == 1
    assert report["status"] == "FAIL"
    assert report["invocations"] == []
    assert "missing required evidence: environment.txt" in report["problems"]
    assert (tmp_path / "artifacts/rocm-job/process-cleanup.txt").is_file()


def test_detached_process_cannot_pass_cleanup(tmp_path: Path) -> None:
    commands = (
        "pytest -q test_sample.py::test_ok\n"
        f'{sys.executable} -c "import subprocess; '
        "p=subprocess.Popen(['sleep','60'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "open('detached.pid','w').write(str(p.pid))\""
    )
    try:
        result, report = _run(tmp_path, commands)
        assert result.returncode == 1, result.stdout
        assert any(problem.startswith("cleanup failed") for problem in report["problems"])
    finally:
        pid_file = tmp_path / "detached.pid"
        if pid_file.exists():
            os.kill(int(pid_file.read_text()), signal.SIGTERM)


def test_surviving_owned_process_is_terminated_and_fails_job(tmp_path: Path) -> None:
    result, report = _run(tmp_path, "pytest -q test_sample.py::test_ok\nsleep 60 &")
    assert result.returncode == 1, result.stdout
    assert "test commands left a surviving process group" in report["problems"]


def test_runner_sigterm_reaps_owned_test_group_and_retains_partial_report(tmp_path: Path) -> None:
    environment = _fixture(tmp_path, "pytest -q test_sample.py::test_wait")
    output = tmp_path / "runner-output.txt"
    with output.open("w") as log:
        process = subprocess.Popen(["bash", str(RUNNER)], env=environment, cwd=tmp_path, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / "waiting.pid").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (tmp_path / "waiting.pid").is_file(), output.read_text()
            child_pid = int((tmp_path / "waiting.pid").read_text())
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=15) == 143, output.read_text()
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
            report = json.loads((tmp_path / "artifacts/rocm-job/job-result.json").read_text())
            assert report["exit_status"] == 143
            assert report["status"] == "FAIL"
            assert "unfinished pytest invocation" in "\n".join(report["problems"])
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=15)


def test_template_retains_common_artifacts_for_grouped_and_ungrouped_jobs() -> None:
    template = Path(".buildkite/amd/test-template-amd-omni.j2").read_text()
    assert template.count('VLLM_CI_JOB_EVIDENCE: "1"') == 2
    assert template.count('"artifacts/rocm-job/**/*"') == 2
    assert template.count("{% for path in step.artifact_paths %}") == 2
