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
    elif mode in (
        "unpinned_offload",
        "flux_movement_trace",
        "flux_blocking_cpu_copy",
        "flux_host_cache_release",
        "flux_packed_host_copy",
    ):
        assert item.callspec.params["config"].id == "fp8_flux2_dev_text_encoder"
    elif mode in (
        "fp8_routing_trace",
        "zimage_encoder_bf16",
        "zimage_transformer_bf16",
        "zimage_per_token",
        "zimage_ptpc",
        "zimage_block128",
    ):
        assert item.callspec.params["config"].id == "fp8_z_image"
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
        elif mode in (
            "flux_movement_trace",
            "flux_blocking_cpu_copy",
            "flux_host_cache_release",
            "flux_packed_host_copy",
            "fp8_routing_trace",
            "zimage_encoder_bf16",
            "zimage_transformer_bf16",
            "zimage_per_token",
            "zimage_ptpc",
            "zimage_block128",
        ):
            assert "worker_extension_cls" not in values
            values["worker_extension_cls"] = "rocm_quantization_worker_trace.RocmQuantizationTrace"
        print("ROCM_QUANT_COMPARISON " + json.dumps({"mode": mode, "omni_kwargs": values}), flush=True)
        return values

    module._build_omni_kwargs = kwargs
    try:
        yield
    finally:
        module._build_omni_kwargs = original
