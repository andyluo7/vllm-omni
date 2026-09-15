#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate the CUDA/ROCm parity ledger and render a reviewable report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / ".buildkite/ci/rocm_cuda_parity.yaml"
DEFAULT_SCHEMA = ROOT / ".buildkite/ci/rocm_cuda_parity.schema.json"
FINAL_STATUSES = {
    "covered",
    "stale-reference",
    "duplicate",
    "approved-exception",
    "cuda-only",
}
INCOMPLETE_STATUSES = {"missing", "partial", "unknown"}
ALL_STATUSES = FINAL_STATUSES | INCOMPLETE_STATUSES


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate invariants that make the ledger safe to use as a closeout gate."""
    if manifest.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if manifest.get("issue") != "https://github.com/vllm-project/vllm-omni/issues/5731":
        raise ValueError("issue must identify vllm-project/vllm-omni#5731")
    snapshot = manifest.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a mapping")
    sha = snapshot.get("upstream_sha")
    if not isinstance(sha, str) or len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("snapshot.upstream_sha must be a full lowercase Git SHA")

    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 61:
        raise ValueError(
            f"rows must contain exactly 61 entries, got {len(rows) if isinstance(rows, list) else 'non-list'}"
        )
    expected_items = list(range(1, 62))
    actual_items = [row.get("issue_item") for row in rows if isinstance(row, dict)]
    if actual_items != expected_items:
        raise ValueError(f"issue_item values must be ordered 1..61, got {actual_items}")

    ids: set[str] = set()
    required = {
        "issue_item",
        "id",
        "title",
        "cuda_source",
        "cuda_command",
        "rocm_suite",
        "rocm_queue",
        "gpu_count",
        "status",
        "owner",
        "planned_pr",
        "pr",
        "head_sha",
        "builds",
        "artifacts",
        "resolution",
    }
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every row must be a mapping")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"item {row.get('issue_item')} is missing fields: {sorted(missing)}")
        row_id = row["id"]
        if row_id in ids:
            raise ValueError(f"duplicate row id: {row_id}")
        ids.add(row_id)
        if row["status"] not in ALL_STATUSES:
            raise ValueError(f"item {row['issue_item']} has invalid status {row['status']!r}")
        if not row["owner"]:
            raise ValueError(f"item {row['issue_item']} has no owner")
        if row["rocm_queue"] is None and row["gpu_count"] is not None:
            raise ValueError(f"item {row['issue_item']} has gpu_count without rocm_queue")
        if row["rocm_queue"] is not None:
            expected_gpu_count = int(str(row["rocm_queue"]).rsplit("_", 1)[1])
            if row["gpu_count"] != expected_gpu_count:
                raise ValueError(
                    f"item {row['issue_item']} queue {row['rocm_queue']} requires gpu_count={expected_gpu_count}"
                )
        if row["status"] == "covered":
            if row["pr"] is None or row["head_sha"] is None:
                raise ValueError(f"covered item {row['issue_item']} must identify a PR and exact head SHA")
            qualifying = [
                build
                for build in row["builds"]
                if build.get("state") == "passed" and not build.get("soft_failed", False)
            ]
            if len(qualifying) < 2:
                raise ValueError(f"covered item {row['issue_item']} needs two non-soft-failed passing builds")
            if not row["artifacts"]:
                raise ValueError(f"covered item {row['issue_item']} must retain artifacts")
        if row["status"] in FINAL_STATUSES - {"covered"} and not row["resolution"]:
            raise ValueError(f"terminal item {row['issue_item']} must explain its resolution")
    return rows


def render_markdown(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["status"] for row in rows)
    lines = [
        "# CUDA/ROCm CI parity report",
        "",
        f"Issue: {manifest['issue']}",
        f"Snapshot: `{manifest['snapshot']['upstream_sha']}` ({manifest['snapshot']['recorded_at']})",
        "",
        "## Status summary",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ]
    for status in sorted(ALL_STATUSES):
        lines.append(f"| {status} | {counts.get(status, 0)} |")
    lines.extend(
        [
            "",
            "## Issue rows",
            "",
            "| Item | Status | Owner | Planned PR | Title |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        title = str(row["title"]).replace("|", "\\|")
        lines.append(
            f"| {row['issue_item']} | {row['status']} | @{row['owner']} | {row['planned_pr'] or '-'} | {title} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final", action="store_true", help="fail unless all 61 rows have terminal status")
    args = parser.parse_args(argv)

    # Loading the checked-in schema catches malformed JSON and keeps the CLI
    # coupled to the documented contract without adding a jsonschema runtime dependency.
    with args.schema.open(encoding="utf-8") as stream:
        json.load(stream)
    manifest = _load_yaml(args.manifest)
    rows = validate_manifest(manifest)
    incomplete = [row for row in rows if row["status"] in INCOMPLETE_STATUSES]
    if args.final and incomplete:
        print(
            "ROCm parity is incomplete: " + ", ".join(f"#{row['issue_item']}={row['status']}" for row in incomplete),
            file=sys.stderr,
        )
        return 1

    output = json.dumps(manifest, indent=2) + "\n" if args.format == "json" else render_markdown(manifest, rows)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
