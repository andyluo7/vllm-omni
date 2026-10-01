#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render a single diagnostic using the exact existing production image."""

import argparse
import subprocess
from pathlib import Path

import yaml

SOURCE = "ad7c78e1589347c6b0580faf69afd107fdbaf5b1"
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", default="minijinja-cli")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    rendered = subprocess.run(
        [
            args.cli,
            str(ROOT / ".buildkite/amd/test-template-amd-omni.j2"),
            str(ROOT / ".buildkite/amd/test-window-diagnostic.yml"),
            "-D",
            "mirror_hw=amdproduction",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    pipeline = yaml.safe_load(rendered.stdout)
    pipeline["steps"] = [group for group in pipeline["steps"] if group["group"] != "AMD Tests"]
    steps = [step for group in pipeline["steps"] for step in group["steps"]]
    assert len(steps) == 1
    step = steps[0]
    step.pop("depends_on", None)
    pod = step["plugins"][0]["kubernetes"]["podSpecPatch"]
    pod["containers"][0]["image"] = f"rocm/vllm-omni:{SOURCE}"
    assert step["agents"]["queue"] == "amd_mi300_1"
    assert pod["containers"][0]["resources"] == {
        "limits": {"amd.com/gpu": "1"},
        "requests": {"amd.com/gpu": "1"},
    }
    assert pod["automountServiceAccountToken"] is False
    assert step["env"]["VLLM_CI_DOCKER_DISABLED"] == "1"
    assert step["soft_fail"] is True and step["timeout_in_minutes"] == 55
    assert "$$BUILDKITE_BUILD_CHECKOUT_PATH/tools/ci/diagnose_rocm_window.py" in step["env"]["TEST_COMMANDS"]
    assert "retry" not in step
    args.output.write_text(yaml.safe_dump(pipeline, sort_keys=False))
    if args.upload:
        subprocess.run(["buildkite-agent", "artifact", "upload", str(args.output)], check=True, timeout=60)
        subprocess.run(["buildkite-agent", "pipeline", "upload", str(args.output)], check=True, timeout=60)
    print(f"Rendered one native ROCm window diagnostic; image={SOURCE}")


if __name__ == "__main__":
    main()
