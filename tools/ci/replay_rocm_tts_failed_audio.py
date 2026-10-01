# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay a retained failing TTS WAV in its original ROCm image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

SOURCE = "ad7c78e1589347c6b0580faf69afd107fdbaf5b1"
SOURCE_HASHES = {
    "tests/helpers/assertions.py": "591c4beac29332b1f0d47fcf60792fe415fc12f5a78545c24a66b84d3cd9d05c",
    "tests/helpers/media.py": "8637b47968273a07f4abced80c424aca3c1bec181acecb24cdf39ac213129320",
}
SETTINGS = {"temperature": 0.0, "word_timestamps": True, "condition_on_previous_text": False, "language": None}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def download(path: Path, spec: dict) -> None:
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
    temporary = path.with_suffix(".partial")
    with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
        size = 0
        while chunk := response.read(65536):
            size += len(chunk)
            if size > 4 * 1024 * 1024:
                raise ValueError("Replay input exceeds 4 MiB")
            output.write(chunk)
    assert size == spec["bytes"] and digest(temporary) == spec["sha256"]
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("/app/vllm-omni"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["image_source"] == SOURCE
    assert manifest["build"] == 13155
    assert manifest["threshold"] == 0.9
    args.output.mkdir(parents=True, exist_ok=True)
    for relative, expected in SOURCE_HASHES.items():
        assert digest(args.source_root / relative) == expected, relative
    for name in ("response.wav", "transcription.json"):
        download(args.output / name, manifest["inputs"][name])
    original = json.loads((args.output / "transcription.json").read_text())
    assert original["audio_sha256"] == manifest["inputs"]["response.wav"]["sha256"]
    assert original["test_call_outcome"] == "failed"
    assert original["similarity"] < 0.9 and original["asr_model"] == "small"
    assert original["request_settings"]["sample_rate"] == 8000
    assert "test_sample_rate_001[async_chunk]" in original["test"]
    if args.prepare_only:
        print("Replay input and production source hashes verified", flush=True)
        return

    sys.path.insert(0, str(args.source_root))
    import soundfile as sf
    import torch
    import whisper

    from tests.helpers import media
    from vllm_omni.platforms import current_omni_platform

    assert torch.version.hip, "Replay requires the original ROCm backend"
    assert current_omni_platform.get_device_count() == 1
    assert importlib.metadata.version("openai-whisper") == "20250625"
    raw = (args.output / "response.wav").read_bytes()
    samples, sample_rate = sf.read(io.BytesIO(raw))
    assert sample_rate == 8000
    # Match convert_audio_bytes_to_text: decode and write PCM_16 before ASR.
    materialized = args.output / "asr-materialized.wav"
    sf.write(materialized, samples, sample_rate, format="WAV", subtype="PCM_16")
    expected = original["expected_text"]
    result = {
        "image_source": SOURCE,
        "diagnostic_commit": os.environ.get("BUILDKITE_COMMIT"),
        "job_id": os.environ.get("BUILDKITE_JOB_ID"),
        "input_manifest": manifest,
        "source_hashes": SOURCE_HASHES,
        "original": original,
        "torch_version": torch.__version__,
        "rocm": torch.version.hip,
        "whisper_version": importlib.metadata.version("openai-whisper"),
        "expected_gpus": 1,
        "visible_gpus": current_omni_platform.get_device_count(),
        "materialized_sha256": digest(materialized),
        "sample_rate": sample_rate,
        "actual_seconds": len(samples) / sample_rate,
        "original_settings": SETTINGS,
        "quality_threshold": 0.9,
        "no_model_server_running": True,
        "comparability_limit": (
            "Fixed WAV and original helper settings; original ASR worker device "
            "and co-resident serving load were not recorded."
        ),
        "cases": [],
    }
    output = args.output / "replay-results.json"

    def save() -> None:
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for number in (1, 2):
            started = time.monotonic()
            transcript = media.convert_audio_bytes_to_text(raw)
            similarity = media.cosine_similarity_text(transcript.strip().lower(), expected.strip().lower())
            result["cases"].append(
                {
                    "name": f"original_helper_{number}",
                    "transcript": transcript,
                    "similarity": similarity,
                    "passes_threshold": similarity > 0.9,
                    "asr_device_index": media.whisper_resident_device_index(),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            save()
            print(json.dumps(result["cases"][-1]), flush=True)
    finally:
        media.release_audio_transcriber()

    media._WHISPER_LOADED_DEVICE = "cuda:0"
    for model_size, cases in (
        (
            "small",
            [
                ("original_greedy", {}),
                ("fp32_greedy", {"fp16": False}),
                ("temperature_fallback", {"temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)}),
            ],
        ),
        ("large-v3", [("independent_large_greedy", {})]),
    ):
        model = media._get_whisper_model(model_size)
        model_sha = whisper._MODELS[model_size].split("/")[-2]
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "whisper"
        assert digest(cache / f"{model_size}.pt") == model_sha
        for number in (1, 2):
            for name, extra in cases:
                torch.manual_seed(0)
                started = time.monotonic()
                response = model.transcribe(str(materialized), **{**SETTINGS, **extra})
                similarity = media.cosine_similarity_text(response["text"].strip().lower(), expected.strip().lower())
                result["cases"].append(
                    {
                        "name": f"{name}_{number}",
                        "model": model_size,
                        "model_sha256": model_sha,
                        "settings": {**SETTINGS, **extra},
                        "torch_seed": 0,
                        "similarity": similarity,
                        "passes_threshold": similarity > 0.9,
                        "elapsed_seconds": time.monotonic() - started,
                        "response": response,
                    }
                )
                save()
                print(json.dumps({k: v for k, v in result["cases"][-1].items() if k != "response"}), flush=True)
        media._WHISPER_MODELS.pop(model_size)
        del model
        current_omni_platform.empty_cache()
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
