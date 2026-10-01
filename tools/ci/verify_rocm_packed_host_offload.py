#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small real ROCm data/layout/alias/allocator preflight for packed D2H storage."""

import argparse
import gc
import json
from pathlib import Path


def verify(evidence: dict) -> None:
    import torch
    from rocm_packed_host_offload import packed_to_cpu
    from torch import nn

    from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook

    if not torch.version.hip:
        raise RuntimeError("This preflight requires the exact ROCm image")
    stats = getattr(torch.cuda, "host_memory_stats", None)
    release = getattr(torch._C, "_host_emptyCache", None)
    if not callable(stats) or not callable(release):
        raise RuntimeError("Pinned allocation counters and cache release must be available")
    evidence.update(
        torch_version=torch.__version__,
        torch_git_version=torch.version.git_version,
        hip_version=torch.version.hip,
        device=torch.cuda.get_device_name(),
    )
    release()
    initial = stats()
    evidence["initial_stats"] = initial
    module = nn.Module()
    raw = torch.arange(64 * 1024, dtype=torch.float32, device="cuda")
    module.weight = nn.Parameter(raw[:32768].view(128, 256), requires_grad=False)
    module.alias = nn.Parameter(raw[1024:33792].view(128, 256).transpose(0, 1), requires_grad=False)
    module.register_buffer("shared", raw[2048:4096])
    module.register_buffer("counter", torch.arange(1025, dtype=torch.int64, device="cuda"))
    module.register_buffer("padding", torch.arange(17, dtype=torch.float32, device="cuda"))
    originals = {name: tensor.detach().cpu().clone() for name, tensor in module.named_parameters()}
    originals.update({name: tensor.cpu().clone() for name, tensor in module.named_buffers()})
    layouts = {
        name: (tuple(tensor.shape), tuple(tensor.stride()))
        for name, tensor in [*module.named_parameters(), *module.named_buffers()]
    }
    del raw
    traces = []

    def emit(kind, **fields):
        traces.append({"kind": kind, **fields})

    assert packed_to_cpu(
        module, non_blocking=True, pin_memory=True, original_move=SequentialOffloadHook._move_params, emit=emit
    )
    plan = traces[0]
    active = stats()
    evidence.update(plan=plan, active_stats=active, traces=traces)
    assert active["allocated_bytes.current"] - initial["allocated_bytes.current"] == plan["pinned_allocation_bytes"]
    assert active["allocations.current"] - initial["allocations.current"] == plan["slabs"]
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        assert tensor.is_pinned() and tensor.device.type == "cpu"
        assert torch.equal(tensor, originals[name]), name
        assert (tuple(tensor.shape), tuple(tensor.stride())) == layouts[name], name
    assert module.weight.untyped_storage().data_ptr() == module.alias.untyped_storage().data_ptr()
    assert module.weight.untyped_storage().data_ptr() == module.shared.untyped_storage().data_ptr()
    assert module.alias.storage_offset() - module.weight.storage_offset() == 1024
    assert module.shared.storage_offset() - module.weight.storage_offset() == 2048
    assert not packed_to_cpu(
        module, non_blocking=True, pin_memory=True, original_move=SequentialOffloadHook._move_params
    )
    # Reuse the original CPU-to-GPU path; mutate live values before the next
    # offload to ensure the diagnostic copies current state instead of masters.
    assert SequentialOffloadHook._move_params(module, torch.device("cuda"), non_blocking=False)
    module.weight.add_(7)
    originals["weight"].add_(7)
    assert packed_to_cpu(
        module, non_blocking=False, pin_memory=True, original_move=SequentialOffloadHook._move_params, emit=emit
    )
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        assert torch.equal(tensor, originals[name]), name
        assert (tuple(tensor.shape), tuple(tensor.stride())) == layouts[name], name
    del tensor, module
    gc.collect()
    release()
    final = stats()
    evidence["final_stats"] = final
    assert final["allocated_bytes.current"] == initial["allocated_bytes.current"]
    assert final["allocations.current"] == initial["allocations.current"]
    evidence.update(
        data_layout_aliases_verified=True,
        nonblocking_completion_verified=True,
        current_mutated_values_preserved=True,
        pinned_allocation_counters_verified=True,
        released_back_to_initial_allocation=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict = {}
    try:
        verify(result)
        result["passed"] = True
    except Exception as exc:
        result.update(passed=False, error=type(exc).__name__ + ": " + str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("ROCM_PACKED_HOST_PREFLIGHT " + json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
