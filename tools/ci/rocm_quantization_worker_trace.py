# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Temporary worker-side evidence collection; all model calls stay original."""

import functools
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

_INSTALLED = set()


def _emit(kind, **fields):
    print(
        "ROCM_QUANT_TRACE " + json.dumps({"kind": kind, "epoch": time.time(), "pid": os.getpid(), **fields}), flush=True
    )


def _source_identity(obj):
    path = Path(inspect.getfile(obj))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _memory():
    fields: dict[str, Any] = {}
    for name in ("memory.current", "memory.peak", "memory.events", "memory.stat"):
        try:
            with (Path("/sys/fs/cgroup") / name).open("rb") as handle:
                fields[name] = handle.read(8192).decode("utf-8", "replace")
        except OSError:
            pass
    # ROCm exposes pinned allocator statistics through this Torch namespace.
    # Keep absence of the optional telemetry API visible in the diagnostic.
    import torch

    stats = getattr(torch.cuda, "host_memory_stats", None)
    if stats is not None:
        try:
            fields["pinned_host_allocator"] = {
                key: value for key, value in stats().items() if key.endswith((".current", ".peak"))
            }
        except Exception as exc:
            fields["pinned_host_allocator_error"] = type(exc).__name__ + ": " + str(exc)
    else:
        fields["pinned_host_allocator_error"] = "host_memory_stats unavailable"
    return fields


def _module_summary(module):
    totals: dict[str, int] = {}
    seen = set()
    tensors = 0
    for iterator in (module.parameters(), module.buffers()):
        for tensor in iterator:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            tensors += 1
            if tensors > 16384:
                return {"tensor_bytes": totals, "tensors": tensors - 1, "truncated": True}
            key = str(tensor.device) + "/" + str(tensor.dtype)
            totals[key] = totals.get(key, 0) + tensor.numel() * tensor.element_size()
    return {"class": type(module).__name__, "tensor_bytes": totals, "tensors": tensors, "truncated": False}


def _components(pipeline):
    return {
        name: _module_summary(getattr(pipeline, name))
        for name in ("transformer", "text_encoder", "vae")
        if getattr(pipeline, name, None) is not None
    }


def _fp8_component(prefix):
    # Z-Image builds AutoModelForCausalLM for its text encoder. Its lm_head
    # runs before the diffusion transformer and must not consume that sample.
    return "encoder" if prefix.startswith("model.") or prefix == "lm_head" else "transformer"


def _resolve_host_cache_release():
    import torch

    release = getattr(torch._C, "_host_emptyCache", None)
    if not callable(release):
        raise RuntimeError(f"Torch {torch.__version__} has no pinned-host cache release operation")
    return release


def _install_flux_trace(*, blocking_cpu_copy=False, release_host_cache=False, packed_host_copy=False):
    from vllm_omni.diffusion.models.flux2.pipeline_flux2 import Flux2Pipeline
    from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend, SequentialOffloadHook

    _emit("runtime_sources", sources=[_source_identity(obj) for obj in (Flux2Pipeline, SequentialOffloadHook)])
    original_load = Flux2Pipeline.load_weights

    @functools.wraps(original_load)
    def load(pipeline, *args, **kwargs):
        _emit("flux_load_before", components=_components(pipeline), memory=_memory())
        result = original_load(pipeline, *args, **kwargs)
        _emit("flux_load_after", components=_components(pipeline), memory=_memory())
        return result

    Flux2Pipeline.load_weights = load
    original_enable = ModelLevelOffloadBackend.enable

    @functools.wraps(original_enable)
    def enable(backend, pipeline):
        _emit(
            "offload_enable_before",
            components=_components(pipeline),
            config=repr(backend.config)[:4096],
            memory=_memory(),
        )
        result = original_enable(backend, pipeline)
        _emit("offload_enable_after", components=_components(pipeline), memory=_memory())
        return result

    ModelLevelOffloadBackend.enable = enable
    original_move = SequentialOffloadHook._move_params
    move_count = 0
    release = _resolve_host_cache_release() if release_host_cache else None

    def release_cache(phase, move):
        if release is None:
            return
        before = _memory()
        release()
        if move <= 64:
            _emit("host_cache_release", phase=phase, move=move, before=before, after=_memory())

    @functools.wraps(original_move)
    def move(module, target_device, **kwargs):
        nonlocal move_count
        move_count += 1
        requested_kwargs = dict(kwargs)
        if blocking_cpu_copy and str(target_device) == "cpu":
            kwargs = {**kwargs, "non_blocking": False}
        trace = move_count <= 64
        if trace:
            _emit(
                "move_before",
                move=move_count,
                target=str(target_device),
                requested_kwargs=requested_kwargs,
                kwargs=kwargs,
                module=_module_summary(module),
                memory=_memory(),
            )
        release_cache("before_move", move_count)
        if packed_host_copy and str(target_device) == "cpu":
            from rocm_packed_host_offload import packed_to_cpu

            result = packed_to_cpu(module, original_move=original_move, emit=_emit, **kwargs)
        else:
            result = original_move(module, target_device, **kwargs)
        release_cache("after_move", move_count)
        if trace:
            _emit(
                "move_after",
                move=move_count,
                target=str(target_device),
                moved=result,
                module=_module_summary(module),
                memory=_memory(),
            )
        return result

    SequentialOffloadHook._move_params = staticmethod(move)


def _install_fp8_trace(*, bf16_component=None):
    import torch
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod

    _emit("runtime_sources", sources=[_source_identity(obj) for obj in (Fp8Config, Fp8PerTensorOnlineLinearMethod)])
    original_route = Fp8Config.get_quant_method
    route_count = 0

    @functools.wraps(original_route)
    def route(config, layer, prefix):
        nonlocal route_count
        result = original_route(config, layer, prefix)
        original_method = type(result).__name__
        if isinstance(result, Fp8PerTensorOnlineLinearMethod) and _fp8_component(prefix) == bf16_component:
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod

            result = UnquantizedLinearMethod()
        route_count += 1
        if route_count <= 1024:
            _emit(
                "fp8_route",
                prefix=prefix,
                layer=type(layer).__name__,
                method=type(result).__name__,
                original_method=original_method,
                bf16_component=bf16_component,
                ignored_layers=config.ignored_layers,
                match_mode=str(config.ignored_layers_match_mode),
            )
        if isinstance(result, Fp8PerTensorOnlineLinearMethod):
            result._rocm_diagnostic_prefix = prefix
        return result

    Fp8Config.get_quant_method = route
    original_process = Fp8PerTensorOnlineLinearMethod.process_weights_after_loading
    original_apply = Fp8PerTensorOnlineLinearMethod.apply
    sampled_weights = set()
    sampled_activations = set()

    def metric(actual, reference):
        delta = actual.float() - reference.float()
        return {
            "max_abs_error": delta.abs().max().item(),
            "relative_l2": (
                torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            ).item(),
            "finite": bool(torch.isfinite(actual).all().item()),
        }

    @functools.wraps(original_process)
    def process(method, layer):
        prefix = getattr(method, "_rocm_diagnostic_prefix", "unknown")
        family = _fp8_component(prefix)
        sample = family not in sampled_weights and not getattr(
            layer, "_already_called_process_weights_after_loading", False
        )
        if sample:
            sampled_weights.add(family)
            shape = layer.weight.shape
            rows, cols = min(16, shape[0]), min(256, shape[1])
            reference = layer.weight[:rows, :cols].float()
        result = original_process(method, layer)
        if sample:
            scale = layer.weight_scale
            assert scale.numel() == 1
            restored = layer.weight[:cols, :rows].float().t() * scale.float()
            _emit(
                "fp8_weight_sample",
                prefix=prefix,
                family=family,
                original_shape=list(shape),
                quantized_dtype=str(layer.weight.dtype),
                scale=scale.item(),
                samples=reference.numel(),
                error=metric(restored, reference),
                kernel=type(method.fp8_linear).__name__,
                activation_quant_key=str(method.activation_quant_key),
                weight_quant_key=str(method.weight_quant_key),
            )
        return result

    Fp8PerTensorOnlineLinearMethod.process_weights_after_loading = process

    @functools.wraps(original_apply)
    def apply(method, layer, x, bias=None):
        result = original_apply(method, layer, x, bias)
        prefix = getattr(method, "_rocm_diagnostic_prefix", "unknown")
        family = _fp8_component(prefix)
        if family not in sampled_activations and x.numel() and layer.weight_scale.numel() == 1:
            sampled_activations.add(family)
            with torch.no_grad():
                # Sample two input rows and sixteen output channels. This FP32
                # reference avoids allocating a full dequantized model and
                # measures activation+GEMM error with FP8 weights fixed.
                channels = min(16, layer.weight.shape[1])
                inputs = x.reshape(-1, x.shape[-1])[:2].float()
                weights = layer.weight[:, :channels].float() * layer.weight_scale.float()
                reference = inputs @ weights
                if bias is not None:
                    reference += bias[:channels].float()
                actual = result.reshape(-1, result.shape[-1])[:2, :channels]
                _emit(
                    "fp8_activation_gemm_sample",
                    prefix=prefix,
                    family=family,
                    input_shape=list(x.shape),
                    sampled_rows=inputs.shape[0],
                    sampled_output_channels=channels,
                    error=metric(actual, reference),
                    kernel=type(method.fp8_linear).__name__,
                )
        return result

    Fp8PerTensorOnlineLinearMethod.apply = apply


def _install_fp8_scaling_comparison(mode):
    import torch
    from vllm import envs
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
        Fp8PtpcOnlineLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
    from vllm.platforms import current_platform

    assert current_platform.is_rocm(), "Scaling comparisons require the original ROCm runtime"
    assert not envs.VLLM_BATCH_INVARIANT, "Scaling comparisons must retain FP8 activation compute"
    assert mode in ("zimage_per_token", "zimage_ptpc")
    runtime_objects = (Fp8Config, Fp8PerTensorOnlineLinearMethod, Fp8PtpcOnlineLinearMethod)
    _emit("runtime_sources", sources=[_source_identity(obj) for obj in runtime_objects])
    original_route = Fp8Config.get_quant_method
    route_count = 0
    sampled_weights = set()
    sampled_activations = set()

    def metric(actual, reference):
        delta = actual.float() - reference.float()
        return {
            "max_abs_error": delta.abs().max().item(),
            "relative_l2": (
                torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            ).item(),
            "finite": bool(torch.isfinite(actual).all().item()),
        }

    @functools.wraps(original_route)
    def route(config, layer, prefix):
        nonlocal route_count
        result = original_route(config, layer, prefix)
        original_method = type(result).__name__
        if isinstance(result, Fp8PerTensorOnlineLinearMethod):
            if mode == "zimage_ptpc":
                result = Fp8PtpcOnlineLinearMethod()
            else:
                result.activation_quant_key = kFp8DynamicTokenSym
            family = _fp8_component(prefix)
            original_process = result.process_weights_after_loading
            original_apply = result.apply

            def process(current_layer):
                sample = family not in sampled_weights and not getattr(
                    current_layer, "_already_called_process_weights_after_loading", False
                )
                if sample:
                    sampled_weights.add(family)
                    shape = current_layer.weight.shape
                    rows, cols = min(16, shape[0]), min(256, shape[1])
                    reference = current_layer.weight[:rows, :cols].float()
                value = original_process(current_layer)
                assert result.activation_quant_key == kFp8DynamicTokenSym
                assert result.fp8_linear.config.activation_quant_key == kFp8DynamicTokenSym
                assert current_layer.weight.dtype == current_platform.fp8_dtype()
                if mode == "zimage_ptpc":
                    assert result.weight_quant_key == kFp8StaticChannelSym
                if sample:
                    scale = current_layer.weight_scale.float()
                    sample_scale = scale if scale.numel() == 1 else scale[:rows].reshape(-1, 1)
                    restored = current_layer.weight[:cols, :rows].float().t() * sample_scale
                    _emit(
                        "fp8_scaling_weight_sample",
                        mode=mode,
                        prefix=prefix,
                        family=family,
                        weight_shape=list(current_layer.weight.shape),
                        weight_scale_shape=list(scale.shape),
                        quantized_dtype=str(current_layer.weight.dtype),
                        activation_quant_key=str(result.activation_quant_key),
                        weight_quant_key=str(result.weight_quant_key),
                        kernel=type(result.fp8_linear).__name__,
                        samples=reference.numel(),
                        error=metric(restored, reference),
                    )
                return value

            def apply(current_layer, x, bias=None):
                value = original_apply(current_layer, x, bias)
                if family not in sampled_activations and x.numel():
                    sampled_activations.add(family)
                    with torch.no_grad():
                        channels = min(16, current_layer.weight.shape[1])
                        inputs = x.reshape(-1, x.shape[-1])[:2].float()
                        scale = current_layer.weight_scale.float()
                        sample_scale = scale if scale.numel() == 1 else scale[:channels].reshape(1, -1)
                        weights = current_layer.weight[:, :channels].float() * sample_scale
                        reference = inputs @ weights
                        if bias is not None:
                            reference += bias[:channels].float()
                        actual = value.reshape(-1, value.shape[-1])[:2, :channels]
                        _emit(
                            "fp8_scaling_activation_gemm_sample",
                            mode=mode,
                            prefix=prefix,
                            family=family,
                            input_shape=list(x.shape),
                            kernel=type(result.fp8_linear).__name__,
                            error=metric(actual, reference),
                        )
                return value

            result.process_weights_after_loading = process
            result.apply = apply
        route_count += 1
        if route_count <= 1024:
            _emit(
                "fp8_scaling_route",
                mode=mode,
                prefix=prefix,
                original_method=original_method,
                method=type(result).__name__,
                ignored_layers=config.ignored_layers,
            )
        return result

    Fp8Config.get_quant_method = route


def _install_fp8_block_comparison(block_size=128):
    import torch
    from vllm import envs
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerBlockOnlineLinearMethod,
        Fp8PerTensorOnlineLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key
    from vllm.platforms import current_platform

    assert current_platform.is_rocm(), "Block comparison requires the original ROCm image"
    assert not envs.VLLM_BATCH_INVARIANT, "Block comparison must retain FP8 activation compute"
    assert block_size in (32, 64, 128)
    _emit("runtime_sources", sources=[_source_identity(obj) for obj in (Fp8Config, Fp8PerBlockOnlineLinearMethod)])
    original_route = Fp8Config.get_quant_method
    sampled_weights = set()
    sampled_activations = set()
    routes = 0

    def metric(actual, reference):
        delta = actual.float() - reference.float()
        return {
            "max_abs_error": delta.abs().max().item(),
            "relative_l2": (
                torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            ).item(),
            "finite": bool(torch.isfinite(actual).all().item()),
        }

    @functools.wraps(original_route)
    def route(config, layer, prefix):
        nonlocal routes
        original = original_route(config, layer, prefix)
        result = original
        if isinstance(original, Fp8PerTensorOnlineLinearMethod):
            result = Fp8PerBlockOnlineLinearMethod()
            result.weight_block_size = [block_size, block_size]
            result.activation_quant_key = create_fp8_quant_key(static=False, group_shape=GroupShape(1, block_size))
            result.weight_quant_key = create_fp8_quant_key(static=True, group_shape=GroupShape(block_size, block_size))
            family = _fp8_component(prefix)
            process_original = result.process_weights_after_loading
            apply_original = result.apply

            def process(current_layer):
                sample = family not in sampled_weights and not getattr(
                    current_layer, "_already_called_process_weights_after_loading", False
                )
                if sample:
                    sampled_weights.add(family)
                    rows, cols = min(16, current_layer.weight.shape[0]), min(256, current_layer.weight.shape[1])
                    reference = current_layer.weight[:rows, :cols].detach().float().clone()
                value = process_original(current_layer)
                assert result.fp8_linear.apply_input_quant, "Block kernel must compute FP8 activations"
                assert current_layer.weight.dtype == current_platform.fp8_dtype()
                assert type(result.fp8_linear).__name__ == "TritonFp8BlockScaledMMKernel"
                assert current_layer.weight_block_size == [block_size, block_size]
                if sample:
                    scales = current_layer.weight_scale_inv.float()
                    row_ids = torch.arange(rows, device=current_layer.weight.device) // block_size
                    col_ids = torch.arange(cols, device=current_layer.weight.device) // block_size
                    restored = current_layer.weight[:rows, :cols].float() * scales[row_ids[:, None], col_ids[None, :]]
                    _emit(
                        "fp8_block_weight_sample",
                        prefix=prefix,
                        family=family,
                        weight_shape=list(current_layer.weight.shape),
                        weight_scale_shape=list(scales.shape),
                        weight_dtype=str(current_layer.weight.dtype),
                        activation_quant_key=str(result.activation_quant_key),
                        weight_quant_key=str(result.weight_quant_key),
                        kernel=type(result.fp8_linear).__name__,
                        block_size=block_size,
                        error=metric(restored, reference),
                    )
                return value

            def apply(current_layer, x, bias=None):
                value = apply_original(current_layer, x, bias)
                if family not in sampled_activations and x.numel():
                    sampled_activations.add(family)
                    with torch.no_grad():
                        channels = min(16, current_layer.weight.shape[0])
                        columns = current_layer.weight.shape[1]
                        row_ids = torch.arange(channels, device=current_layer.weight.device) // block_size
                        col_ids = torch.arange(columns, device=current_layer.weight.device) // block_size
                        scales = current_layer.weight_scale_inv.float()
                        weights = current_layer.weight[:channels].float() * scales[row_ids[:, None], col_ids[None, :]]
                        reference = x.reshape(-1, columns)[:2].float() @ weights.t()
                        if bias is not None:
                            reference += bias[:channels].float()
                        actual = value.reshape(-1, value.shape[-1])[:2, :channels]
                        _emit(
                            "fp8_block_activation_gemm_sample",
                            prefix=prefix,
                            family=family,
                            input_shape=list(x.shape),
                            kernel=type(result.fp8_linear).__name__,
                            block_size=block_size,
                            error=metric(actual, reference),
                        )
                return value

            result.process_weights_after_loading = process
            result.apply = apply
        routes += 1
        if routes <= 1024:
            _emit(
                "fp8_block_route",
                prefix=prefix,
                original_method=type(original).__name__,
                method=type(result).__name__,
                ignored_layers=config.ignored_layers,
                block_size=block_size,
            )
        return result

    Fp8Config.get_quant_method = route


class RocmQuantizationTrace:
    def __init__(self, *args, **kwargs):
        mode = os.environ["ROCM_QUANT_COMPARISON"]
        if mode not in _INSTALLED:
            if mode in (
                "flux_movement_trace",
                "flux_blocking_cpu_copy",
                "flux_host_cache_release",
                "flux_packed_host_copy",
            ):
                _install_flux_trace(
                    blocking_cpu_copy=mode == "flux_blocking_cpu_copy",
                    release_host_cache=mode in ("flux_host_cache_release", "flux_packed_host_copy"),
                    packed_host_copy=mode == "flux_packed_host_copy",
                )
            elif mode in ("fp8_routing_trace", "zimage_encoder_bf16", "zimage_transformer_bf16"):
                component = {
                    "fp8_routing_trace": None,
                    "zimage_encoder_bf16": "encoder",
                    "zimage_transformer_bf16": "transformer",
                }[mode]
                _install_fp8_trace(bf16_component=component)
            elif mode in ("zimage_per_token", "zimage_ptpc"):
                _install_fp8_scaling_comparison(mode)
            elif mode in ("zimage_block32", "zimage_block64", "zimage_block128"):
                _install_fp8_block_comparison(block_size=int(mode.removeprefix("zimage_block")))
            else:
                raise ValueError(mode)
            _INSTALLED.add(mode)
        super().__init__(*args, **kwargs)
