# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ROCm CI setup for deterministic tiny diffusion attention."""

from __future__ import annotations

import os


def _configure_rocm_ci() -> None:
    if os.environ.get("VLLM_OMNI_ROCM_CI_FORCE_MATH_SDPA") != "1":
        return

    try:
        import torch
    except ModuleNotFoundError:
        return

    if torch.version.hip:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


_configure_rocm_ci()
