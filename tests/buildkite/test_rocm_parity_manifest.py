# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / ".buildkite/ci/rocm_cuda_parity.yaml"
SCHEMA_PATH = ROOT / ".buildkite/ci/rocm_cuda_parity.schema.json"
REPORT_PATH = ROOT / ".buildkite/scripts/generate_rocm_parity_report.py"

SPEC = importlib.util.spec_from_file_location("generate_rocm_parity_report", REPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def _load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_schema_is_valid_json_and_tracks_report_statuses() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    status_enum = set(schema["$defs"]["row"]["properties"]["status"]["enum"])
    assert status_enum == REPORT.ALL_STATUSES
    assert schema["properties"]["rows"]["minItems"] == 61
    assert schema["properties"]["rows"]["maxItems"] == 61


def test_manifest_accounts_for_every_issue_row_exactly_once() -> None:
    rows = REPORT.validate_manifest(_load_manifest())
    assert [row["issue_item"] for row in rows] == list(range(1, 62))
    assert len({row["id"] for row in rows}) == 61


def test_current_report_is_reviewable_but_not_final() -> None:
    manifest = _load_manifest()
    rows = REPORT.validate_manifest(manifest)
    report = REPORT.render_markdown(manifest, rows)
    assert "| Item | Status | Owner | Planned PR | Title |" in report
    assert "| 1 | partial | @andyluo7 | #7398 |" in report
    assert REPORT.main(["--manifest", str(MANIFEST_PATH), "--schema", str(SCHEMA_PATH), "--final"]) == 1


def test_final_mode_accepts_only_evidenced_terminal_rows() -> None:
    manifest = _load_manifest()
    row = deepcopy(manifest["rows"][0])
    row.update(
        status="covered",
        pr=7398,
        head_sha="a" * 40,
        builds=[
            {
                "number": 1,
                "url": "https://buildkite.com/example/builds/1",
                "named_job": "job",
                "state": "passed",
                "soft_failed": False,
            },
            {
                "number": 2,
                "url": "https://buildkite.com/example/builds/2",
                "named_job": "job",
                "state": "passed",
                "soft_failed": False,
            },
        ],
        artifacts=["pytest.xml"],
    )
    manifest["rows"] = [{**row, "issue_item": item, "id": f"row-{item}"} for item in range(1, 62)]
    rows = REPORT.validate_manifest(manifest)
    assert all(item["status"] in REPORT.FINAL_STATUSES for item in rows)


def test_covered_row_rejects_soft_failed_evidence() -> None:
    manifest = _load_manifest()
    manifest["rows"][0].update(
        status="covered",
        head_sha="a" * 40,
        builds=[
            {
                "number": number,
                "url": f"https://buildkite.com/example/builds/{number}",
                "named_job": "job",
                "state": "passed",
                "soft_failed": True,
            }
            for number in (1, 2)
        ],
        artifacts=["pytest.xml"],
    )
    with pytest.raises(ValueError, match="two non-soft-failed passing builds"):
        REPORT.validate_manifest(manifest)
