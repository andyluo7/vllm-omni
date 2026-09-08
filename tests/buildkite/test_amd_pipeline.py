# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")


def _find_step(label: str, pipeline_path: Path = AMD_MERGE_PIPELINE) -> dict:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == label:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline.get("steps", []))
    assert step is not None, f"missing AMD pipeline step: {label}"
    return step


def test_qwen3_tts_base_preserves_advanced_model_arguments() -> None:
    step = _find_step("Qwen3-TTS Base E2E Test")
    commands = step["commands"]

    assert all("bash -c" not in command for command in commands)
    pytest_command = next(command for command in commands if "pytest" in command)
    argv = split(pytest_command)

    marker_index = argv.index("-m")
    run_level_index = argv.index("--run-level")
    assert argv[marker_index + 1] == "advanced_model and cuda"
    assert argv[run_level_index + 1] == "advanced_model"


def test_ltx2_ulysses_uses_two_gpu_amd_lane() -> None:
    model_step = _find_step("Diffusion · Model Test", AMD_READY_PIPELINE)
    assert model_step["agent_pool"] == "mi300_1"
    model_command = next(command for command in model_step["commands"] if "pytest" in command)
    model_argv = split(model_command)
    model_marker_index = model_argv.index("-m")
    assert model_argv[model_marker_index + 1] == "core_model and cuda and not cards_2"

    sp_step = _find_step("Diffusion Sequence Parallelism Test", AMD_READY_PIPELINE)
    assert sp_step["agent_pool"] == "mi300_2"
    ltx_command = next(command for command in sp_step["commands"] if "test_ltx2_transformer_ulysses.py" in command)
    ltx_argv = split(ltx_command)
    marker_index = ltx_argv.index("-m")
    run_level_index = ltx_argv.index("--run-level")
    assert ltx_argv[marker_index + 1] == "core_model and rocm and cards_2"
    assert ltx_argv[run_level_index + 1] == "core_model"
