# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
STAGE_CONFIG_HELPER = Path("tests/helpers/stage_config.py")
FUNCTION_TEST = Path("tests/e2e/online_serving/test_qwen3_omni_expansion.py")
ASSERTIONS = Path("tests/helpers/assertions.py")
ACCURACY_DRIVER = Path("tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py")


def _find_step(label: str) -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == label:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline["steps"])
    assert step is not None, f"missing AMD pipeline step: {label}"
    return step


def _load_stage_config_helper(monkeypatch: pytest.MonkeyPatch):
    stub = types.ModuleType("vllm_omni.config.stage_config")
    setattr(stub, "load_deploy_config", lambda config_path: config_path)
    monkeypatch.setitem(sys.modules, "vllm_omni.config.stage_config", stub)
    spec = importlib.util.spec_from_file_location("nightly_stage_config", STAGE_CONFIG_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen3_ci_overlay_pins_only_the_talker_sampling_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    production_path = Path("vllm_omni/deploy/qwen3_omni_moe.yaml")
    production_before = production_path.read_bytes()
    module = _load_stage_config_helper(monkeypatch)
    generated = Path(module.get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
    overlay = yaml.safe_load(generated.read_text(encoding="utf-8"))
    talker = next(stage for stage in overlay["stages"] if stage["stage_id"] == 1)

    assert talker["default_sampling_params"] == {"max_tokens": 1000, "seed": 42}
    # Upstream now pins a production seed too. Generating a CI overlay must
    # preserve the complete production configuration whatever its seed is.
    assert production_path.read_bytes() == production_before


def test_qwen3_ci_overlay_retains_long_output_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_stage_config_helper(monkeypatch)
    generated = Path(module.get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
    overlay = yaml.safe_load(generated.read_text(encoding="utf-8"))
    thinker = next(stage for stage in overlay["stages"] if stage["stage_id"] == 0)
    function_source = FUNCTION_TEST.read_text(encoding="utf-8")

    assert thinker["default_sampling_params"]["max_tokens"] == 512
    assert "assert word_count >= 200" in function_source


def test_long_output_request_expands_only_its_downstream_audio_budgets() -> None:
    function_source = FUNCTION_TEST.read_text(encoding="utf-8")

    assert '"sampling_params_list": LONG_OUTPUT_SAMPLING_PARAMS' in function_source
    assert '"max_tokens": 3072' in function_source
    assert '"max_tokens": 6144' in function_source
    assert "assert word_count >= 200" in function_source


def test_qwen3_ci_overlay_reserves_rocm_thinker_encoder_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_stage_config_helper(monkeypatch)
    generated = Path(module.get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
    overlay = yaml.safe_load(generated.read_text(encoding="utf-8"))
    rocm_thinker = next(stage for stage in overlay["platforms"]["rocm"]["stages"] if stage["stage_id"] == 0)

    assert rocm_thinker == {
        "stage_id": 0,
        "gpu_memory_utilization": None,
        "kv_cache_memory_bytes": 80 * 1024**3,
    }
    production = yaml.safe_load(Path("vllm_omni/deploy/qwen3_omni_moe.yaml").read_text(encoding="utf-8"))
    production_thinker = next(stage for stage in production["stages"] if stage["stage_id"] == 0)
    assert production_thinker["gpu_memory_utilization"] == 0.9
    assert "kv_cache_memory_bytes" not in production_thinker


def test_function_expansion_uses_the_seeded_ci_overlay() -> None:
    source = FUNCTION_TEST.read_text(encoding="utf-8")
    assert 'get_deploy_config_path("ci/qwen3_omni_moe.yaml")' in source
    assert 'get_deploy_config_path("qwen3_omni_moe.yaml")' not in source


@pytest.mark.parametrize(
    ("label", "artifact_dir"),
    [
        ("Qwen3-Omni Function Expansion", "artifacts/rocm-qwen3-omni-function"),
        ("Qwen3-Omni Accuracy", "artifacts/rocm-qwen3-omni-accuracy"),
    ],
)
def test_unstable_jobs_pin_seed_and_retain_diagnostics(label: str, artifact_dir: str) -> None:
    step = _find_step(label)
    commands = step["commands"]
    pytest_commands = [command for command in commands if "pytest " in command]

    assert step["grade"] == "NonBlocking"
    assert len(pytest_commands) == 1
    assert "export VLLM_CI_QWEN3_OMNI_SEED=42" in commands
    assert "export PYTHONHASHSEED=42" in commands
    assert any(path == f"{artifact_dir}/**/*" for path in step["artifact_paths"])
    assert "--junitxml=" in pytest_commands[0]
    assert "pytest-summary.txt" in "\n".join(commands)
    assert "reproducibility.txt" in "\n".join(commands)


def test_function_and_accuracy_selection_are_unchanged() -> None:
    function = next(
        command for command in _find_step("Qwen3-Omni Function Expansion")["commands"] if "pytest " in command
    )
    accuracy = next(command for command in _find_step("Qwen3-Omni Accuracy")["commands"] if "pytest " in command)
    function_argv = split(function)
    accuracy_argv = split(accuracy)

    assert "tests/e2e/online_serving/test_qwen3_omni_expansion.py" in function_argv
    assert function_argv[function_argv.index("-m") + 1] == "full_model and rocm and MI325 and cards_2"
    assert "tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py" in accuracy_argv
    assert accuracy_argv[accuracy_argv.index("-m") + 1] == "full_model and rocm and MI325 and cards_2"


@pytest.mark.parametrize(
    ("label", "artifact_dir"),
    [
        ("Qwen3-Omni Function Expansion", "artifacts/rocm-qwen3-omni-function"),
        ("Qwen3-Omni Accuracy", "artifacts/rocm-qwen3-omni-accuracy"),
        ("CosyVoice3-TTS E2E Test", "artifacts/rocm-cosyvoice3-nightly"),
        ("Qwen3-Omni Documentation Examples", "artifacts/rocm-qwen3-omni-documentation"),
        ("Qwen3-Omni AITER-on Smoke", "artifacts/rocm-qwen3-omni-aiter-smoke"),
    ],
)
def test_nightly_model_jobs_report_runtime_execution_and_cleanup(label: str, artifact_dir: str) -> None:
    step = _find_step(label)
    commands = step["commands"]
    assert f"{artifact_dir}/**/*" in step["artifact_paths"]
    pytest_index = next(index for index, command in enumerate(commands) if "pytest -s" in command)
    environment_index = next(
        index for index, command in enumerate(commands) if "rocm_ci_evidence.py environment" in command
    )
    before_index = next(index for index, command in enumerate(commands) if "rocm_ci_evidence.py processes" in command)
    result_index = next(
        index for index, command in enumerate(commands) if "rocm_ci_evidence.py pytest-result" in command
    )
    cleanup_index = next(index for index, command in enumerate(commands) if "rocm_ci_evidence.py cleanup" in command)
    assert environment_index < before_index < pytest_index < result_index < cleanup_index
    assert "--junitxml=" in commands[pytest_index]
    assert "tee " in commands[pytest_index]
    assert "runtime_seconds=" in commands[pytest_index + 1]
    assert "--xml " in commands[result_index] and "--log " in commands[result_index]
    assert "--before " in commands[cleanup_index] and "--after " in commands[cleanup_index]
    assert step["grade"] == "NonBlocking"


def test_cosyvoice_evidence_preserves_full_scope_and_gpu_hang_retry() -> None:
    step = _find_step("CosyVoice3-TTS E2E Test")
    command = next(command for command in step["commands"] if "pytest -s" in command)
    argv = split(command)
    assert "tests/e2e/online_serving/test_cosyvoice3_tts_expansion.py" in argv
    assert argv[argv.index("-m") + 1] == "slow"
    assert argv[argv.index("--run-level") + 1] == "core_model"
    assert "--collect-only" not in argv
    assert step["agent_pool"] == "mi300_1"
    assert step["timeout_in_minutes"] == 90
    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}


@pytest.mark.parametrize(
    ("label", "node", "budget"),
    [
        ("Qwen3-Omni Documentation Examples", "tests/examples/online_serving/test_qwen3_omni.py", 120),
        (
            "Qwen3-Omni AITER-on Smoke",
            "tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_to_text_audio_001",
            90,
        ),
    ],
)
def test_documentation_and_aiter_evidence_preserves_selection_and_budget(label: str, node: str, budget: int) -> None:
    step = _find_step(label)
    argv = split(next(command for command in step["commands"] if "pytest -s" in command))
    assert argv[0] == "pytest"
    assert node in argv
    assert argv[argv.index("-m") + 1] == "full_model and rocm and MI325 and cards_2"
    assert argv[argv.index("--run-level") + 1] == "full_model"
    assert "--collect-only" not in argv
    assert step["agent_pool"] == "mi300_2"
    assert step["timeout_in_minutes"] == budget
    assert "retry" not in step
    if label == "Qwen3-Omni Documentation Examples":
        assert "qwen3-omni-doc-artifacts/**/*" in step["artifact_paths"]
        assert 'export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"' in step["commands"]
    else:
        assert "export VLLM_ROCM_USE_AITER=1" in step["commands"]


def test_quality_thresholds_are_not_weakened() -> None:
    assertions_source = ASSERTIONS.read_text(encoding="utf-8")
    accuracy_source = ACCURACY_DRIVER.read_text(encoding="utf-8")
    assert 'request_config.get("similarity_threshold", 0.8)' in assertions_source
    assert "default=0.35" in accuracy_source
