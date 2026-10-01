#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the selected real FP8 block kernel before allocating any model."""

import argparse
import json
from pathlib import Path


def verify(evidence: dict, block_size: int = 128) -> dict:
    from vllm.config import VllmConfig, set_current_vllm_config

    # Match upstream's default_vllm_config fixture for standalone CustomOps,
    # including the compiled cast and kernel construction/forward execution.
    with set_current_vllm_config(VllmConfig()):
        return _verify_kernel(evidence, block_size)


def _verify_kernel(evidence: dict, block_size: int) -> dict:
    import torch
    from rocm_quantization_worker_trace import _source_identity
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        create_fp8_quant_key,
    )
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    evidence.update(
        torch_version=torch.__version__, torch_git_version=torch.version.git_version, hip_version=torch.version.hip
    )
    assert current_platform.is_rocm()
    assert block_size in (32, 64, 128)
    torch.manual_seed(42)
    weight = torch.randn(256, 384, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(2, 384, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    quantized, scales = per_block_cast_to_fp8(weight, block_size=[block_size, block_size], use_ue8m0=False)
    assert quantized.dtype == current_platform.fp8_dtype()
    assert tuple(quantized.shape) == (256, 384)
    assert tuple(scales.shape) == (256 // block_size, 384 // block_size)
    kernel = init_fp8_linear_kernel(
        activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, block_size)),
        weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(block_size, block_size)),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(256, 384),
    )
    evidence.update(
        kernel=type(kernel).__name__,
        weight_dtype=str(quantized.dtype),
        weight_shape=list(quantized.shape),
        scale_shape=list(scales.shape),
        block_size=block_size,
        sources=[_source_identity(obj) for obj in (type(kernel), per_block_cast_to_fp8)],
    )
    assert type(kernel).__name__ == "TritonFp8BlockScaledMMKernel"
    assert kernel.apply_input_quant
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(quantized, requires_grad=False))
    layer.register_parameter("weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False))
    layer.input_scale = None
    kernel.process_weights_after_loading(layer)
    assert layer.weight.dtype == current_platform.fp8_dtype()
    assert tuple(layer.weight.shape) == (256, 384)
    output = kernel.apply_weights(layer, inputs, bias)
    torch.accelerator.synchronize()
    assert tuple(output.shape) == (2, 256) and torch.isfinite(output).all()
    restored = layer.weight.float() * layer.weight_scale_inv.float().repeat_interleave(block_size, 0).repeat_interleave(
        block_size, 1
    )
    reference = inputs.float() @ restored.t() + bias.float()
    relative_l2 = (
        torch.linalg.vector_norm(output.float() - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    ).item()
    evidence.update(relative_l2=relative_l2, activation_fp8_required=True, finite=True)
    # A small capability sanity bound; the original model LPIPS gate stays 0.15.
    assert relative_l2 < 0.1, "Small FP8 block GEMM capability probe has excessive error"
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, choices=(32, 64, 128), default=128)
    args = parser.parse_args()
    evidence: dict = {}
    try:
        result = {"passed": True, **verify(evidence, args.block_size)}
    except Exception as exc:
        result = {"passed": False, "error": type(exc).__name__ + ": " + str(exc), **evidence}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("ROCM_FP8_BLOCK_PREFLIGHT " + json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
