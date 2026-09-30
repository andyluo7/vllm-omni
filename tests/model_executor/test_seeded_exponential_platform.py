# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.utils import seeded_exponential

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
@pytest.mark.parametrize("is_nvidia", [False, True], ids=["rocm", "cuda"])
def test_cuda_tensor_requires_nvidia_distribution_contract(monkeypatch, is_nvidia: bool) -> None:
    monkeypatch.setattr(seeded_exponential.current_platform, "is_cuda", lambda: is_nvidia)
    # HIP tensors expose the same Torch device predicate as NVIDIA tensors.
    q = Mock(spec=torch.Tensor)
    q.is_cuda = True
    q.dtype = torch.float32
    q.dim.return_value = 2
    q.is_contiguous.return_value = True
    generators = {0: object(), 1: object()}

    assert seeded_exponential.batched_seeded_exponential_supported(q, generators) is is_nvidia


@pytest.mark.cpu
def test_shared_generator_keeps_ordered_torch_draws(monkeypatch) -> None:
    monkeypatch.setattr(seeded_exponential.current_platform, "is_cuda", lambda: True)
    q = Mock(spec=torch.Tensor)
    q.is_cuda = True
    q.dtype = torch.float32
    q.dim.return_value = 2
    q.is_contiguous.return_value = True
    shared = object()

    assert not seeded_exponential.batched_seeded_exponential_supported(q, {0: shared, 1: shared})


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("seeded_rows", [(0, 1, 2), (0, 2)], ids=["all_seeded", "mixed"])
@pytest.mark.parametrize("use_fp64_gumbel", [False, True])
def test_torch_fallback_preserves_samples_and_generator_state(monkeypatch, seeded_rows, use_fp64_gumbel) -> None:
    from vllm.v1.sample.ops import topk_topp_sampler as sampler_ops

    import vllm_omni.patch  # noqa: F401

    original = sampler_ops.random_sample.__wrapped__
    monkeypatch.setattr(seeded_exponential.current_platform, "is_cuda", lambda: False)

    def reject_kernel(*args, **kwargs):
        raise AssertionError("the CUDA distribution kernel must not run on the fallback path")

    monkeypatch.setattr(seeded_exponential, "fill_exponential_rows", reject_kernel)
    probs = torch.arange(1, 258, device="cuda", dtype=torch.float32).expand(3, -1).contiguous()
    probs /= probs.sum(dim=-1, keepdim=True)
    expected_gens = {row: torch.Generator(device="cuda").manual_seed(42 + row) for row in seeded_rows}
    actual_gens = {row: torch.Generator(device="cuda").manual_seed(42 + row) for row in seeded_rows}
    expected_default = torch.cuda.get_rng_state()
    actual_default = expected_default.clone()
    try:
        for _ in range(5):
            torch.cuda.set_rng_state(expected_default)
            expected = original(probs, expected_gens, use_fp64_gumbel)
            expected_default = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(actual_default)
            actual = sampler_ops.random_sample(probs, actual_gens, use_fp64_gumbel)
            actual_default = torch.cuda.get_rng_state()
            assert torch.equal(actual, expected)
            assert torch.equal(actual_default, expected_default)
            for row in seeded_rows:
                assert torch.equal(actual_gens[row].get_state(), expected_gens[row].get_state())
    finally:
        torch.cuda.set_rng_state(expected_default)
