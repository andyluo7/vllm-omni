# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
AMD_NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
AMD_TEMPLATE = Path(".buildkite/amd/test-template-amd-omni.j2")
AR_PAGED_ATTENTION_LABEL = "ROCm · AR Diffusion Paged Attention GPU Test"
AITER_COVERAGE_LABEL = "ROCm · FlashAttention/AITER GPU Coverage"
DIFFUSION_GROUP = ":card_index_dividers: Diffusion Test"
AR_PAGED_ATTENTION_MARKERS = "core_model and rocm and MI325 and cards_1"
MAMMOTH_ATTENTION_TEST = "tests/diffusion/models/mammoth_moda2/test_dit_attention_cuda.py"


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


def _find_group(label: str, pipeline_path: Path) -> dict:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    group = next((step for step in pipeline.get("steps", []) if step.get("group") == label), None)
    assert group is not None, f"missing AMD pipeline group: {label}"
    return group


def test_ar_paged_attention_gpu_lane_is_blocking_and_pinned() -> None:
    lane_definitions = []
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        group = _find_group(DIFFUSION_GROUP, pipeline_path)
        step = next((item for item in group["steps"] if item.get("label") == AR_PAGED_ATTENTION_LABEL), None)
        assert step is not None, f"missing {AR_PAGED_ATTENTION_LABEL} from {DIFFUSION_GROUP} in {pipeline_path}"
        lane_definitions.append(step)

        assert step["grade"] == "Blocking"
        assert step["timeout_in_minutes"] == 20
        assert "export VLLM_OMNI_AR_FA_REQUIRED=1" in step["commands"]

        pytest_command = next(command for command in step["commands"] if "pytest" in command)
        argv = split(pytest_command)
        assert argv[:4] == ["timeout", "--signal=TERM", "--kill-after=2m", "15m"]
        assert "tests/diffusion/ar_diffusion/test_paged_attention.py" in argv
        assert argv[argv.index("-m") + 1] == AR_PAGED_ATTENTION_MARKERS
        assert argv[argv.index("--run-level") + 1] == "core_model"

    assert lane_definitions[0] == lane_definitions[1]


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


def test_qwen3_accuracy_defers_artifact_path_expansion() -> None:
    step = _find_step("Qwen3-Omni Accuracy", AMD_NIGHTLY_PIPELINE)
    staging_command = next(command for command in step["commands"] if "artifact_dir=" in command)

    # Dynamic pipelines are interpolated once during upload. Double dollars
    # preserve these variables for the GPU job's runtime shell.
    assert '"$$PWD"' in staging_command
    assert '"$${BUILDKITE_BUILD_CHECKOUT_PATH:?}"' in staging_command
    assert '"$$artifact_dir"' in staging_command
    assert step["artifact_paths"] == ["tests/e2e/accuracy/qwen3_omni/results/qwen_omni_acc/*.json"]


def test_ready_diffusion_cpu_suite_is_sharded() -> None:
    step = _find_step("Simple · Diffusion Test · Shard %N/%t", AMD_READY_PIPELINE)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)

    assert step["parallelism"] == 4
    assert step["timeout_in_minutes"] == 45
    assert "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in pytest_command
    assert "--shard-id=$$BUILDKITE_PARALLEL_JOB" in pytest_command


def test_mammoth_attention_tests_reuse_ready_aiter_cache() -> None:
    aiter_step = _find_step(AITER_COVERAGE_LABEL, AMD_READY_PIPELINE)
    model_step = _find_step("Diffusion · Model Test", AMD_READY_PIPELINE)

    assert aiter_step["grade"] == "NonBlocking"
    assert aiter_step["timeout_in_minutes"] == 60
    assert 'export AITER_JIT_DIR="/tmp/vllm-omni-aiter-$$BUILDKITE_JOB_ID"' in aiter_step["commands"]
    assert 'export TORCH_EXTENSIONS_DIR="/tmp/vllm-omni-torch-extensions-$$BUILDKITE_JOB_ID"' in aiter_step["commands"]

    pytest_commands = [command for command in aiter_step["commands"] if "pytest" in command]
    warmup_index = next(
        index for index, command in enumerate(pytest_commands) if "tests/diffusion/attention/" in command
    )
    mammoth_index = next(index for index, command in enumerate(pytest_commands) if MAMMOTH_ATTENTION_TEST in command)
    assert warmup_index < mammoth_index

    mammoth_argv = split(pytest_commands[mammoth_index])
    assert mammoth_argv[:4] == ["timeout", "--signal=TERM", "--kill-after=2m", "10m"]
    assert mammoth_argv[mammoth_argv.index("-n") + 1] == "1"
    assert MAMMOTH_ATTENTION_TEST in mammoth_argv
    assert mammoth_argv[mammoth_argv.index("--run-level") + 1] == "core_model"

    model_pytest_argv = [split(command) for command in model_step["commands"] if "pytest" in command]
    generic_argv = next(argv for argv in model_pytest_argv if "tests/diffusion/models/" in argv)
    assert f"--ignore={MAMMOTH_ATTENTION_TEST}" in generic_argv

    blocking_mammoth_argv = next(argv for argv in model_pytest_argv if MAMMOTH_ATTENTION_TEST in argv)
    assert blocking_mammoth_argv[:3] == ["timeout", "10m", "env"]
    assert "DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in blocking_mammoth_argv
    assert MAMMOTH_ATTENTION_TEST in blocking_mammoth_argv
    assert blocking_mammoth_argv[blocking_mammoth_argv.index("--run-level") + 1] == "core_model"


def test_z_image_merge_timeout_covers_cold_aiter_compile() -> None:
    step = _find_step("Diffusion Model Test")
    pytest_command = next(command for command in step["commands"] if "test_z_image.py" in command)

    assert split(pytest_command)[:2] == ["timeout", "55m"]


def test_cosyvoice_ready_smoke_uses_sdpa() -> None:
    step = _find_step("CosyVoice3-TTS E2E Smoke (SDPA)", AMD_READY_PIPELINE)

    assert step["grade"] == "Blocking"
    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}
    assert "export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in step["commands"]

    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    assert "tests/e2e/online_serving/test_cosyvoice3_tts_expansion.py::test_voice_clone_zh_002" in pytest_command


def test_cosyvoice_full_default_backend_suite_runs_nightly() -> None:
    step = _find_step("CosyVoice3-TTS E2E Test", AMD_NIGHTLY_PIPELINE)

    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 90
    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}
    assert all("DIFFUSION_ATTENTION_BACKEND" not in command for command in step["commands"])

    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    assert "tests/e2e/online_serving/test_cosyvoice3_tts_expansion.py" in pytest_command
    assert "::" not in pytest_command


def test_amd_template_preserves_step_retry_policy() -> None:
    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    # Both grouped and top-level AMD steps must preserve an explicit retry
    # policy when the source suite is rendered into the uploaded pipeline.
    assert template.count("{% if step.retry %}") == 2
    assert template.count("{% for retry_rule in step.retry.automatic %}") == 2
