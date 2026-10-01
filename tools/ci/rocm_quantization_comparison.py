# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit diagnostic variants; original quality configurations stay intact."""

import json
import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if "test_quantization_quality[" not in item.nodeid:
        yield
        return
    module = item.module
    mode = os.environ["ROCM_QUANT_COMPARISON"]
    original = module._build_omni_kwargs
    if mode == "skinny_disabled":
        assert item.callspec.params["config"].id == "fp8_z_image"
        assert os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] == "0"
    elif mode == "unpinned_offload":
        assert item.callspec.params["config"].id == "fp8_flux2_dev_text_encoder"
    else:
        raise ValueError(f"Unknown diagnostic comparison: {mode}")

    def kwargs(config, model):
        values = original(config, model)
        if mode == "unpinned_offload":
            assert values["enable_cpu_offload"] is True
            # Both BF16 and quantized arms use the same model-level offload
            # plan; only host pinning changes for this controlled comparison.
            assert "pin_cpu_memory" not in values
            values["pin_cpu_memory"] = False
        print("ROCM_QUANT_COMPARISON " + json.dumps({"mode": mode, "omni_kwargs": values}), flush=True)
        return values

    module._build_omni_kwargs = kwargs
    try:
        yield
    finally:
        module._build_omni_kwargs = original
