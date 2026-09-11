# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.ci"
ROCM_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.rocm"


def _docker_arg(path: Path, name: str) -> str:
    prefix = f"ARG {name}="
    matches = [
        line.removeprefix(prefix) for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)
    ]
    assert len(matches) == 1, f"expected one {name} declaration in {path}, found {len(matches)}"
    return matches[0]


def test_rocm_base_tracks_ci_vllm_release() -> None:
    ci_release = _docker_arg(CI_DOCKERFILE, "VLLM_BASE_TAG")
    rocm_base = _docker_arg(ROCM_DOCKERFILE, "BASE_IMAGE")
    rocm_source_ref = _docker_arg(ROCM_DOCKERFILE, "VLLM_VERSION_OR_COMMIT_HASH")

    assert rocm_base.rsplit(":", 1)[-1] == ci_release
    assert rocm_source_ref == ci_release


def test_rocm_image_fails_fast_on_missing_vllm_api() -> None:
    dockerfile = ROCM_DOCKERFILE.read_text(encoding="utf-8")
    canary = 'RUN python3 -c "import vllm; from vllm.v1.kv_cache_interface import compute_layout_strides;'

    assert any(line.startswith(canary) for line in dockerfile.splitlines())
