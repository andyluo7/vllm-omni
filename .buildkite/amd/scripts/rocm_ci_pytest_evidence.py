# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Retain an independent report for every pytest invocation in an AMD job."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest
from _pytest.junitxml import LogXML


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class InvocationEvidence:
    def __init__(self, config: pytest.Config, root: Path) -> None:
        self.directory = root / "pytest" / f"{os.getpid()}-{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.started = time.monotonic()
        self.result = {
            "state": "started",
            "pid": os.getpid(),
            "arguments": list(config.invocation_params.args),
            "selected": 0,
            "deselected": 0,
            "started_at_epoch": time.time(),
        }
        _write_json(self.directory / "result.json", self.result)
        # A separate built-in reporter preserves an explicitly requested JUnit
        # path, including jobs that run pytest repeatedly in a shell loop.
        self.junit = LogXML(
            str(self.directory / "pytest.xml"),
            config.option.junitprefix,
            config.getini("junit_suite_name"),
            config.getini("junit_logging"),
            config.getini("junit_duration_report"),
            config.getini("junit_family"),
            config.getini("junit_log_passing_tests"),
        )
        config.pluginmanager.register(self.junit, "rocm_ci_independent_junit")

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.result["deselected"] += len(items)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.result["selected"] = len(session.items)
        _write_json(self.directory / "result.json", self.result)
        (self.directory / "nodeids.json").write_text(
            json.dumps([item.nodeid for item in session.items], indent=2) + "\n", encoding="utf-8"
        )

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids: list[str]) -> None:
        # The xdist controller has no local session.items. Every worker has
        # the same collection; retain it once without multiplying counts.
        self.result["selected"] = len(ids)
        _write_json(self.directory / "result.json", self.result)
        (self.directory / "nodeids.json").write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # Flush each completed phase so an abort retains the last known test
        # even when pytest cannot finish its JUnit document.
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"nodeid": report.nodeid, "phase": report.when, "outcome": report.outcome}) + "\n")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self.result.update(
            state="finished", exit_status=int(exitstatus), runtime_seconds=time.monotonic() - self.started
        )
        _write_json(self.directory / "result.json", self.result)


def pytest_configure(config: pytest.Config) -> None:
    root = os.environ.get("ROCM_CI_JOB_EVIDENCE_DIR")
    owner = os.environ.get("ROCM_CI_PYTEST_OWNER_PID")
    # Only the controller writes reports under xdist. Workers forward their
    # reports to it; they must not create empty, misleading JUnit documents.
    if root and owner in (None, str(os.getpid())) and not hasattr(config, "workerinput"):
        # Pytester and other test fixtures deliberately spawn failing/empty
        # pytest children. Those belong to the outer test's assertion, while
        # fresh shell invocations inherit the original unclaimed environment.
        os.environ["ROCM_CI_PYTEST_OWNER_PID"] = str(os.getpid())
        config.pluginmanager.register(InvocationEvidence(config, Path(root)), "rocm_ci_invocation_evidence")
