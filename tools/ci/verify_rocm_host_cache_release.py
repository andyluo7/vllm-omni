#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Verify pinned-host cache release while an independent live buffer survives."""

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
    assert allocated["allocated_bytes.current"] == initial["allocated_bytes.current"] + probe_bytes
    assert allocated["allocations.current"] == initial["allocations.current"] + 1
    assert allocated["num_host_alloc"] == initial["num_host_alloc"] + 1
    release()
    active_retained = stats()
    evidence["statistics"]["active_retained"] = active_retained
    # Inactive cache release must preserve a live pinned allocation.
    assert probe.is_pinned() and probe[0].item() == 37 and probe[-1].item() == 37
    assert active_retained["active_bytes.current"] >= initial["active_bytes.current"] + probe_bytes, (
        "Pinned probe is not represented in active allocator bytes"
    )
    assert active_retained["allocated_bytes.current"] == allocated["allocated_bytes.current"]
    assert active_retained["num_host_free"] == initial["num_host_free"]
    # Keep a distinct allocation alive while releasing the original probe.
    # This checks that a partially occupied allocator frees only inactive blocks.
    guard_bytes = 2 * 1024 * 1024
    guard = torch.empty(guard_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    guard.fill_(53)
    guarded = stats()
    evidence["statistics"]["guarded"] = guarded
    assert guarded["allocated_bytes.current"] == allocated["allocated_bytes.current"] + guard_bytes
    assert guarded["allocations.current"] == initial["allocations.current"] + 2
    assert guarded["num_host_alloc"] == initial["num_host_alloc"] + 2
    del probe
    gc.collect()
    inactive = stats()
    evidence["statistics"]["inactive"] = inactive
    release()
    released = stats()
    evidence["statistics"]["released"] = released
    assert inactive["allocated_bytes.current"] - released["allocated_bytes.current"] == probe_bytes, (
        "Pinned cache operation did not release the inactive probe allocation"
    )
    assert released["allocated_bytes.current"] == initial["allocated_bytes.current"] + guard_bytes
    assert released["allocations.current"] == initial["allocations.current"] + 1
    assert released["num_host_free"] == initial["num_host_free"] + 1
    assert guard.is_pinned() and guard[0].item() == 53 and guard[-1].item() == 53
    del guard
    gc.collect()
    release()
    final = stats()
    evidence["statistics"]["final"] = final
    assert final["allocated_bytes.current"] == initial["allocated_bytes.current"], (
        "Allocated bytes did not return to baseline"
    )
    assert final["allocations.current"] == initial["allocations.current"]
    assert final["num_host_free"] == initial["num_host_free"] + 2
    assert final["allocated_bytes.freed"] == initial["allocated_bytes.freed"] + probe_bytes + guard_bytes
    active_stats_match = final["active_bytes.current"] == initial["active_bytes.current"]
    evidence["active_byte_accounting_matches_lifetime"] = active_stats_match
    if not active_stats_match:
        # This exact image's CachingHostAllocator::free has no active-stat
        # decrement in the no-stream path. free_from_pool does decrement the
        # independent allocation/free counters checked above. Preserve the
        # anomalous stats; do not interpret them as a live allocation.
        assert torch.version.git_version == "6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5"
        assert evidence["stats_source"]["sha256"] == (
            "e6da277701a58d2b7c6a50ffe5680abb4a35f9b8f610c60a04a270f2758746e4"
        )
        assert final["active_bytes.current"] == initial["active_bytes.current"] + probe_bytes + guard_bytes
        evidence["accounting_defect"] = "Exact Torch no-stream free path omits active-byte decrement"
    return {
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "hip_version": torch.version.hip,
        "cuda_device": torch.cuda.get_device_name(),
        "stats_source": _source_identity(stats),
        "release_operation": "torch._C._host_emptyCache",
        "probe_bytes": probe_bytes,
        "guard_bytes": guard_bytes,
        "live_allocation_preserved": True,
        "independent_live_guard_preserved": True,
        "inactive_allocation_released": True,
        "allocation_and_free_counters_verified": True,
        "active_byte_accounting_matches_lifetime": active_stats_match,
        "accounting_defect": evidence.get("accounting_defect"),
        "statistics": {
            "initial": initial,
            "allocated": allocated,
            "active_retained": active_retained,
            "guarded": guarded,
            "inactive": inactive,
            "released": released,
            "final": final,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence: dict = {}
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
