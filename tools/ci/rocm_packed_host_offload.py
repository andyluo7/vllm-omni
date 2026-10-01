# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diagnostic D2H copies into packed pinned storage, with original tensor layouts."""

from dataclasses import dataclass
from itertools import chain

ALIGNMENT = 256
SLAB_BYTES = 4 * 1024 * 1024 * 1024


def plan_slabs(sizes: list[int], *, slab_bytes: int = SLAB_BYTES) -> tuple[list[int], list[tuple[int, int]]]:
    """Pack physical storage spans; return power-of-two slab sizes and placements."""
    if slab_bytes < ALIGNMENT or slab_bytes & (slab_bytes - 1):
        raise ValueError("Slab capacity must be a power of two at least 256 bytes")
    if any(size <= 0 for size in sizes):
        raise ValueError("Only nonempty physical storage spans can be packed")
    used: list[int] = []
    capacities: list[int] = []
    placements = [(-1, -1)] * len(sizes)
    # Largest-first packing lets small scale/bias buffers occupy matrix padding.
    for index in sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True):
        size = sizes[index]
        aligned = (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        candidates = [i for i, capacity in enumerate(capacities) if used[i] + aligned <= capacity]
        if candidates:
            slab = min(candidates, key=lambda i: capacities[i] - used[i] - aligned)
        else:
            slab = len(used)
            used.append(0)
            capacities.append(max(slab_bytes, 1 << (aligned - 1).bit_length()))
        placements[index] = (slab, used[slab])
        used[slab] += aligned
    allocations = [1 << (size - 1).bit_length() for size in used]
    return allocations, placements


@dataclass(frozen=True)
class Binding:
    target: object
    dtype: object
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int


def packed_to_cpu(module, *, non_blocking: bool, pin_memory: bool, original_move, emit=None):
    """Copy current GPU values directly into pinned slabs; never retain stale masters."""
    import torch

    from vllm_omni.diffusion.offloader.tensor_utils import is_dtensor, set_tensor_storage

    target_device = torch.device("cpu")
    if not pin_memory:
        return original_move(module, target_device, non_blocking=non_blocking, pin_memory=pin_memory)
    targets = []
    seen = set()
    for tensor in chain(module.parameters(), module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if len(seen) > 16384:
            raise ValueError("Diagnostic module exceeds the bounded tensor inventory")
        # Distributed, sparse and meta tensor movement stays on the original path.
        if is_dtensor(tensor) or tensor.layout != torch.strided or tensor.is_meta:
            return original_move(module, target_device, non_blocking=non_blocking, pin_memory=pin_memory)
        if tensor.device.type != "cpu":
            if tensor.device.type != "cuda":
                return original_move(module, target_device, non_blocking=non_blocking, pin_memory=pin_memory)
            targets.append(tensor)
    if not targets:
        return False
    grouped = {}
    for target in targets:
        local = target.detach()
        storage = local.untyped_storage()
        key = (local.device.index, storage.data_ptr(), storage.nbytes())
        if storage.nbytes() == 0:
            # Avoid zero-storage ambiguity; preserve the ordinary empty-tensor path.
            return original_move(module, target_device, non_blocking=non_blocking, pin_memory=pin_memory)
        if key not in grouped:
            grouped[key] = (local, [])
        grouped[key][1].append(
            Binding(target, local.dtype, tuple(local.shape), tuple(local.stride()), local.storage_offset())
        )
    groups = list(grouped.values())
    sizes = [source.untyped_storage().nbytes() for source, _ in groups]
    allocations, placements = plan_slabs(sizes)
    if emit is not None:
        emit(
            "packed_host_plan",
            tensor_count=len(targets),
            storage_count=len(groups),
            physical_bytes=sum(sizes),
            pinned_allocation_bytes=sum(allocations),
            original_per_storage_rounded_bytes=sum(1 << (size - 1).bit_length() for size in sizes),
            slabs=len(allocations),
            largest_slab_bytes=max(allocations),
        )
    slabs = [torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True) for size in allocations]
    views = []
    try:
        for (source, bindings), size, (slab, offset) in zip(groups, sizes, placements, strict=True):
            raw = torch.empty(0, dtype=torch.uint8, device=source.device).set_(
                source.untyped_storage(), 0, (size,), (1,)
            )
            backing = slabs[slab].narrow(0, offset, size)
            backing.copy_(raw, non_blocking=non_blocking)
            for binding in bindings:
                # Offsets are measured in elements of each target's original dtype.
                element_offset = offset // torch.empty(0, dtype=binding.dtype).element_size() + binding.offset
                view = torch.empty(0, dtype=binding.dtype, device="cpu").set_(
                    slabs[slab].untyped_storage(), element_offset, binding.shape, binding.stride
                )
                views.append((binding.target, view))
    finally:
        # CPU weights must be readable before rebinding. Even a failed copy keeps
        # all source/backing tensors alive until outstanding D2H work completes.
        for device_index in {source.device.index for source, _ in groups}:
            torch.cuda.current_stream(device_index).synchronize()
    for target, view in views:
        set_tensor_storage(target, view)
    if emit is not None:
        emit(
            "packed_host_complete",
            pinned_allocation_bytes=sum(allocations),
            tensors_pinned=all(target.is_pinned() for target, _ in views),
            copy_completion_verified=True,
        )
    return True
