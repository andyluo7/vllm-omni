#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render parity audio diagnostics against the exact production image."""

import argparse
import os
import subprocess
from pathlib import Path

import yaml

SOURCE = "bbc0f5718b4439d068c92121ac4d52e4bd8adf30"
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", default="minijinja-cli")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    result = subprocess.run(
        [
            args.cli,
            str(ROOT / ".buildkite/amd/test-template-amd-omni.j2"),
            str(ROOT / ".buildkite/amd/test-function-audio-diagnostic.yml"),
            "-D",
            "mirror_hw=amdproduction",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    pipeline = yaml.safe_load(result.stdout)
    pipeline["steps"] = [group for group in pipeline["steps"] if group["group"] != "AMD Tests"]
    steps = [step for group in pipeline["steps"] for step in group["steps"]]
    assert len(steps) == 2
    for step in steps:
        step.pop("depends_on", None)
        pod = step["plugins"][0]["kubernetes"]["podSpecPatch"]
        pod["containers"][0]["image"] = f"rocm/vllm-omni:{SOURCE}"
        cards = str(step["env"]["VLLM_CI_EXPECTED_GPU_COUNT"])
        assert cards in {"1", "2"}
        assert step["agents"]["queue"] == f"amd_mi300_{cards}"
        assert pod["containers"][0]["resources"] == {
            "limits": {"amd.com/gpu": cards},
            "requests": {"amd.com/gpu": cards},
        }
        assert pod["automountServiceAccountToken"] is False
        assert step["soft_fail"] is True and step["timeout_in_minutes"] == (130 if cards == "1" else 100)
        assert (
            "$$BUILDKITE_BUILD_CHECKOUT_PATH/tools/ci/diagnose_rocm_function_audio.py" in step["env"]["TEST_COMMANDS"]
        )
    args.output.write_text(yaml.safe_dump(pipeline, sort_keys=False))
    if args.upload:
        # Buildkite expands $$ once when uploading TEST_COMMANDS.
        subprocess.run(["buildkite-agent", "artifact", "upload", str(args.output)], check=True, timeout=60)
        subprocess.run(["buildkite-agent", "pipeline", "upload", str(args.output)], check=True, timeout=60)
    commit = os.environ.get("BUILDKITE_COMMIT", "local")
    print(f"Rendered native one/two-GPU parity audio diagnostics using image {SOURCE}; commit={commit}")


if __name__ == "__main__":
    main()
