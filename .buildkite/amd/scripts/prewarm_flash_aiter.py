#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prebuild one BF16 AITER MHA module used by READY attention coverage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.platforms import current_omni_platform

DENSE_MODULE = "mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale.so"
VARLEN_MODULE = "mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale.so"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("dense", "varlen"))
    return parser.parse_args()


def _inputs(variant: str, device: torch.device) -> tuple[torch.Tensor, AttentionMetadata, str]:
    dtype = torch.bfloat16
    num_heads = 8
    head_dim = 64

    if variant == "dense":
        sequence_length = 64
        query = torch.randn(1, sequence_length, num_heads, head_dim, device=device, dtype=dtype)
        return query, AttentionMetadata(attn_mask=None), DENSE_MODULE

    # Match test_padding_equivalence: 48 image tokens and 16 encoder tokens,
    # with ten padding positions after each segment.
    sequence_length = 84
    query = torch.randn(1, sequence_length, num_heads, head_dim, device=device, dtype=dtype)
    attention_mask = torch.zeros(1, sequence_length, dtype=torch.bool, device=device)
    attention_mask[:, :48] = True
    attention_mask[:, 58:74] = True
    return query, AttentionMetadata(attn_mask=attention_mask), VARLEN_MODULE


def main() -> None:
    args = _parse_args()
    if not current_omni_platform.is_rocm():
        raise RuntimeError("The FlashAttention/AITER prewarm must run on ROCm")
    if not fa.IS_AITER:
        raise RuntimeError("ROCm FlashAttention did not select AITER")

    cache_dir_value = os.environ.get("AITER_JIT_DIR")
    if not cache_dir_value:
        raise RuntimeError("AITER_JIT_DIR must identify the job-local cache")
    cache_dir = Path(cache_dir_value)

    device = torch.device(current_omni_platform.device_type)
    torch.manual_seed(42)
    query, metadata, expected_module = _inputs(args.variant, device)
    key = query.clone()
    value = query.clone()

    impl = FlashAttentionImpl(
        num_heads=8,
        head_size=64,
        softmax_scale=64**-0.5,
        causal=False,
    )
    output = impl.forward(query=query, key=key, value=value, attn_metadata=metadata)
    current_omni_platform.synchronize()

    if output.shape != query.shape:
        raise RuntimeError(f"Unexpected AITER output shape: {output.shape}")
    if not torch.isfinite(output).all():
        raise RuntimeError("AITER prewarm produced non-finite output")

    module_path = cache_dir / expected_module
    if not module_path.is_file():
        raise RuntimeError(f"Expected AITER module was not built: {module_path}")
    print(f"Prebuilt {args.variant} AITER module: {module_path}", flush=True)


if __name__ == "__main__":
    main()
