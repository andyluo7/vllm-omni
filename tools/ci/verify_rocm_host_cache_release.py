#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Verify pinned-host cache release in the exact image with a 4 MiB allocation."""

import argparse
import gc
import json
from pathlib import Path


def verify(evidence=None) -> dict:
    import torch
    from rocm_quantization_worker_trace import _resolve_host_cache_release, _source_identity

    evidence = {} if evidence is None else evidence
    evidence.update(
        torch_version=torch.__version__,
        torch_git_version=torch.version.git_version,
        hip_version=torch.version.hip,
        release_operation="torch._C._host_emptyCache",
    )
    release = _resolve_host_cache_release()
    stats = getattr(torch.cuda, "host_memory_stats", None)
    if not callable(stats):
        raise RuntimeError("Pinned-host allocator statistics are required to verify cache release")
    if not torch.version.hip:
        raise RuntimeError("This capability check requires the ROCm image")
    torch.cuda.init()
    evidence["cuda_device"] = torch.cuda.get_device_name()
    evidence["stats_source"] = _source_identity(stats)
    release()
    initial = stats()
    evidence["statistics"] = {"initial": initial}
    probe_bytes = 4 * 1024 * 1024
    probe = torch.empty(probe_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    probe.fill_(37)
    assert probe.is_pinned() and probe[0].item() == 37 and probe[-1].item() == 37
    allocated = stats()
    evidence["statistics"]["allocated"] = allocated
    release()
    active_retained = stats()
    evidence["statistics"]["active_retained"] = active_retained
    # Inactive cache release must preserve a live pinned allocation.
    assert probe.is_pinned() and probe[0].item() == 37 and probe[-1].item() == 37
    assert active_retained["active_bytes.current"] >= initial["active_bytes.current"] + probe_bytes, (
        "Pinned probe is not represented in active allocator bytes"
    )
    del probe
    gc.collect()
    inactive = stats()
    evidence["statistics"]["inactive"] = inactive
    release()
    released = stats()
    evidence["statistics"]["released"] = released
    assert inactive["allocated_bytes.current"] - released["allocated_bytes.current"] >= probe_bytes, (
        "Pinned cache operation did not release the inactive probe allocation"
    )
    assert released["active_bytes.current"] == initial["active_bytes.current"], (
        "Active bytes did not return to baseline"
    )
    return {
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "hip_version": torch.version.hip,
        "cuda_device": torch.cuda.get_device_name(),
        "stats_source": _source_identity(stats),
        "release_operation": "torch._C._host_emptyCache",
        "probe_bytes": probe_bytes,
        "live_allocation_preserved": True,
        "inactive_allocation_released": True,
        "statistics": {
            "initial": initial,
            "allocated": allocated,
            "active_retained": active_retained,
            "inactive": inactive,
            "released": released,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {}
    try:
        result = {"passed": True, **verify(evidence)}
    except Exception as exc:
        result = {"passed": False, "error": type(exc).__name__ + ": " + str(exc), **evidence}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("ROCM_HOST_CACHE_PREFLIGHT " + json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
