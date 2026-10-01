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
    fields = {}
    for name in ("memory.current", "memory.peak", "memory.events", "memory.stat"):
        try:
            with (Path("/sys/fs/cgroup") / name).open("rb") as handle:
                fields[name] = handle.read(8192).decode("utf-8", "replace")
        except OSError:
            pass
    return fields


def _module_summary(module):
    totals = {}
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


def _install_flux_trace():
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

    @functools.wraps(original_move)
    def move(module, target_device, **kwargs):
        nonlocal move_count
        move_count += 1
        trace = move_count <= 64
        if trace:
            _emit(
                "move_before",
                move=move_count,
                target=str(target_device),
                kwargs=kwargs,
                module=_module_summary(module),
                memory=_memory(),
            )
        result = original_move(module, target_device, **kwargs)
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


def _install_fp8_trace():
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
        route_count += 1
        if route_count <= 1024:
            _emit(
                "fp8_route",
                prefix=prefix,
                layer=type(layer).__name__,
                method=type(result).__name__,
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


class RocmQuantizationTrace:
    def __init__(self, *args, **kwargs):
        mode = os.environ["ROCM_QUANT_COMPARISON"]
        if mode not in _INSTALLED:
            if mode == "flux_movement_trace":
                _install_flux_trace()
            elif mode == "fp8_routing_trace":
                _install_fp8_trace()
            else:
                raise ValueError(mode)
            _INSTALLED.add(mode)
        super().__init__(*args, **kwargs)
