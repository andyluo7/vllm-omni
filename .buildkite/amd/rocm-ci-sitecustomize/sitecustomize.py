# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ROCm CI process isolation for deterministic diffusion smoke tests."""

from __future__ import annotations

import os
from pathlib import Path


def _configure_rocm_ci() -> None:
    if os.environ.get("VLLM_OMNI_ROCM_CI_DETERMINISTIC") != "1":
        return

    miopen_root = os.environ.get("VLLM_OMNI_ROCM_CI_MIOPEN_ROOT")
    if miopen_root:
        process_root = Path(miopen_root) / f"pid-{os.getpid()}"
        user_db = process_root / "db"
        kernel_cache = process_root / "cache"
        user_db.mkdir(parents=True, exist_ok=True)
        kernel_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MIOPEN_USER_DB_PATH"] = str(user_db)
        os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(kernel_cache)

    try:
        import torch
    except ModuleNotFoundError:
        return

    if torch.version.hip:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


_configure_rocm_ci()
