#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Run vLLM-Omni ROCm tests directly in the MI300 Kubernetes pod. The pod's
# container image is selected by test-template-amd-omni.j2; MI300 has no DinD.
set -euo pipefail

: "${PYTHONFAULTHANDLER:=1}"
: "${HF_HOME:=/home/buildkite-agent/huggingface}"
: "${HF_HUB_DOWNLOAD_TIMEOUT:=300}"
: "${HF_HUB_ETAG_TIMEOUT:=60}"
: "${MIOPEN_DEBUG_CONV_DIRECT:=0}"
: "${MIOPEN_DEBUG_CONV_GEMM:=0}"
: "${VLLM_ROCM_USE_AITER:=0}"
export PYTHONFAULTHANDLER HF_HOME HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT
export MIOPEN_DEBUG_CONV_DIRECT MIOPEN_DEBUG_CONV_GEMM VLLM_ROCM_USE_AITER
export PYTORCH_ROCM_ARCH=""
export PYTHONPATH="${PYTHONPATH:-..}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${VLLM_CI_JOB_EVIDENCE:-0}" == "1" ]]; then
    ROCM_CI_JOB_EVIDENCE_DIR="${BUILDKITE_BUILD_CHECKOUT_PATH:-${PWD}}/artifacts/rocm-job"
    export ROCM_CI_JOB_EVIDENCE_DIR
    export PYTHONPATH="${script_dir}:${PYTHONPATH}"
    export PYTEST_PLUGINS="rocm_ci_pytest_evidence${PYTEST_PLUGINS:+,${PYTEST_PLUGINS}}"
    mkdir -p "${ROCM_CI_JOB_EVIDENCE_DIR}"
    supervisor_pid=""
    # shellcheck disable=SC2329 # Invoked by the EXIT trap.
    finalize_job_evidence() {
        local original_status=$1
        trap - EXIT TERM INT HUP
        set +e
        python3 "${script_dir}/rocm_ci_job_evidence.py" finish \
            --directory "${ROCM_CI_JOB_EVIDENCE_DIR}" --exit-code "${original_status}"
        local evidence_status=$?
        if (( original_status != 0 )); then
            exit "${original_status}"
        fi
        exit "${evidence_status}"
    }
    # shellcheck disable=SC2329 # Invoked by signal traps.
    forward_job_signal() {
        local signal_name=$1
        local signal_status=$2
        trap - TERM INT HUP
        if [[ -n "${supervisor_pid}" ]]; then
            kill -"${signal_name}" "${supervisor_pid}" 2>/dev/null || true
            wait "${supervisor_pid}" || true
        fi
        exit "${signal_status}"
    }
    trap 'finalize_job_evidence "$?"' EXIT
    trap 'forward_job_signal TERM 143' TERM
    trap 'forward_job_signal INT 130' INT
    trap 'forward_job_signal HUP 129' HUP
    python3 "${script_dir}/rocm_ci_job_evidence.py" begin --directory "${ROCM_CI_JOB_EVIDENCE_DIR}"
fi

if [[ "${VLLM_CI_DOCKER_DISABLED:-0}" != "1" ]]; then
    echo "Error: MI300 CI must run natively with Docker disabled." >&2
    exit 1
fi

if [[ -n "${TEST_COMMANDS:-}" ]]; then
    commands="${TEST_COMMANDS}"
else
    commands="$*"
fi
if [[ -z "${commands}" ]]; then
    echo "Error: No test commands provided." >&2
    exit 1
fi

echo "--- Native in-pod ROCm CI"

job_id="${BUILDKITE_JOB_ID:-${BUILDKITE_PARALLEL_JOB:-local}}"
job_id="${job_id//[^A-Za-z0-9_.-]/_}"
native_cache_root="/tmp/vllm-omni-native-${job_id}"
TMPDIR="${native_cache_root}/tmp"
TORCHINDUCTOR_CACHE_DIR="${native_cache_root}/torchinductor"
TRITON_CACHE_DIR="${native_cache_root}/triton"
VLLM_CACHE_ROOT="${native_cache_root}/vllm"
XDG_CACHE_HOME="${native_cache_root}/xdg"
HF_DATASETS_CACHE="${native_cache_root}/huggingface/datasets"
export TMPDIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT
export XDG_CACHE_HOME HF_DATASETS_CACHE
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
    "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${HF_HOME}" \
    "${HF_DATASETS_CACHE}"

if [[ "${VLLM_CI_REQUIRE_PERSISTENT_HF_CACHE:-0}" == "1" ]]; then
    if ! command -v findmnt >/dev/null 2>&1; then
        echo "Error: findmnt is required to verify the Hugging Face cache mount." >&2
        exit 1
    fi
    hf_mount=$(findmnt -n -T "${HF_HOME}" -o TARGET 2>/dev/null || true)
    if [[ -z "${hf_mount}" || "${hf_mount}" == "/" ]]; then
        echo "Error: MI300 CI requires a persistent volume at or above ${HF_HOME}." >&2
        exit 1
    fi
fi

rocminfo

python3 - <<'PY'
import os

import torch

expected = int(os.environ.get("VLLM_CI_EXPECTED_GPU_COUNT", "1"))
assert torch.version.hip, "PyTorch is not a ROCm build"
assert torch.cuda.is_available(), "ROCm GPU is not available to PyTorch"
actual = torch.cuda.device_count()
assert actual == expected, f"Expected {expected} ROCm GPU(s), found {actual}"
PY

if [[ "${VLLM_CI_JOB_EVIDENCE:-0}" == "1" ]]; then
    python3 "${script_dir}/rocm_ci_evidence.py" environment \
        --output "${ROCM_CI_JOB_EVIDENCE_DIR}/environment.txt"
fi

echo "Commands:${commands}"
if [[ "${VLLM_CI_JOB_EVIDENCE:-0}" == "1" ]]; then
    python3 "${script_dir}/rocm_ci_job_evidence.py" run \
        --directory "${ROCM_CI_JOB_EVIDENCE_DIR}" --commands "${commands}" &
    supervisor_pid=$!
    if wait "${supervisor_pid}"; then
        exit_code=0
    else
        exit_code=$?
    fi
    supervisor_pid=""
elif /bin/bash -o pipefail -c '
set -E
test_status=0
trap '\''
    command_status=$?
    # Keep running subsequent commands, but retain a failure for the final exit.
    # Prefer a test failure over pytest "no tests collected" when both occur.
    if (( test_status == 0 || (test_status == 5 && command_status != 5) )); then
        test_status=${command_status}
    fi
'\'' ERR
eval "$1"
exit "${test_status}"
' _ "${commands}"; then
    exit 0
else
    exit_code=$?
fi

if [[ ${exit_code} -eq 5 && "${VLLM_CI_ALLOW_NO_TESTS:-0}" == "1" ]]; then
    echo "Pytest collected no tests; VLLM_CI_ALLOW_NO_TESTS=1, treating exit code 5 as success."
    exit 0
fi

exit "${exit_code}"
