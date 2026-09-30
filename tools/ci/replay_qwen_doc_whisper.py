# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay CUDA #16446's retained audio with original and candidate ASR settings."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    "https://buildkite.com/organizations/vllm/pipelines/vllm-omni/builds/16446/"
    "jobs/01a0f427-9bb4-4a66-afaa-5fc7aff96e4f/artifacts/"
)
INPUTS = {
    "audio.wav": (
        "01a0f462-ae17-4721-9aa6-5bd495be2c54",
        "f50677e5b9d2f289e7307f7d6817f7b0e72e0d5362e8870b575675db8632c6b9",
    ),
    "transcription.json": (
        "01a0f462-ae17-44a9-beef-3ef4518f2914",
        "13d7e9eb112fc0598f553dc7c0fa1e1da2120f825ae37d95ff7fe6ce57cccd1b",
    ),
}
SOURCE_HASHES = {
    "tests/helpers/media.py": "c0d4b0ed631bc049288c23a0db2731a57ee0a4397a65622905225cf0030f901f",
    "tests/examples/helpers.py": "0ecacba66fb9805530d36e2660a5357993aad5749b1aef9df630afa104b0c662",
}
SETTINGS = {
    "temperature": 0.0,
    "word_timestamps": True,
    "condition_on_previous_text": False,
    "language": None,
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download(path: Path, artifact_id: str, expected_sha: str) -> None:
    if path.exists() and digest(path) == expected_sha:
        return
    temporary = path.with_suffix(path.suffix + ".download")
    for attempt in range(3):
        try:
            request = urllib.request.Request(ARTIFACT_ROOT + artifact_id, headers={"User-Agent": "curl/8.7.1"})
            with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
                size = 0
                while chunk := response.read(65536):
                    size += len(chunk)
                    if size > 4 * 1024 * 1024:
                        raise ValueError("Diagnostic input exceeds 4 MiB")
                    output.write(chunk)
            if digest(temporary) != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {path.name}")
            temporary.replace(path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("qwen-doc-whisper-replay"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for relative, expected in SOURCE_HASHES.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Source helper differs from c30: {relative}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (artifact_id, sha) in INPUTS.items():
        download(args.output_dir / filename, artifact_id, sha)
    evidence = json.loads((args.output_dir / "transcription.json").read_text())
    if evidence["asr_model"] != "small":
        raise ValueError("Expected the retained Whisper-small transcription")
    print("Retained inputs and source helper hashes verified", flush=True)
    if args.prepare_only:
        return

    sys.path.insert(0, str(ROOT))
    import torch
    import whisper

    from tests.examples.helpers import extract_content_after_keyword, strip_trailing_audio_saved_line
    from tests.helpers import media
    from vllm_omni.platforms import current_omni_platform

    expected_text = strip_trailing_audio_saved_line(
        extract_content_after_keyword("Chat completion output from text:", evidence["client_output"])
    )
    device = torch.device(args.device)
    if device.type != "cuda" or torch.version.hip:
        raise ValueError("This diagnostic requires the original CUDA backend")
    current_omni_platform.set_device(device)
    metadata = {
        "checkout_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip(),
        "reference_head": "c30e63d04dc9a54c5c4b43235eb2ce535fe68df3",
        "source_helper_sha256": SOURCE_HASHES,
        "input_sha256": {filename: sha for filename, (_, sha) in INPUTS.items()},
        "whisper_version": importlib.metadata.version("openai-whisper"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "gpu_capability": torch.cuda.get_device_capability(device),
        "torch_threads": torch.get_num_threads(),
        "expected_text": expected_text,
        "original_transcript": evidence["transcript"],
        "original_similarity": media.cosine_similarity_text(evidence["transcript"].lower(), expected_text.lower()),
        "quality_threshold": 0.8,
        "settings": SETTINGS,
        "results": [],
    }
    result_path = args.output_dir / "replay-results.json"

    def save() -> None:
        result_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save()
    if metadata["whisper_version"] != "20250625":
        raise ValueError("The diagnostic image must use the repository's pinned OpenAI Whisper version")
    # Use the real helper's model loader and download lock. Pin its device to
    # the device observed in the failed Documentation log, without a server.
    media._WHISPER_LOADED_DEVICE = str(device)
    model = media._get_whisper_model("small")
    model_sha = whisper._MODELS["small"].split("/")[-2]
    checkpoint = Path.home() / ".cache" / "whisper" / "small.pt"
    actual_sha = digest(checkpoint)
    if actual_sha != model_sha:
        raise ValueError("Whisper-small checkpoint differs from its upstream model hash")
    metadata["model_sha256"] = actual_sha
    save()
    cases = [
        ("original_greedy_1", {}),
        ("candidate_beam_5_1", {"beam_size": 5}),
        ("original_greedy_2", {}),
        ("candidate_beam_5_2", {"beam_size": 5}),
    ]
    for name, extra in cases:
        started = time.monotonic()
        try:
            response = model.transcribe(str(args.output_dir / "audio.wav"), **SETTINGS, **extra)
            similarity = media.cosine_similarity_text(response["text"].lower(), expected_text.lower())
            metadata["results"].append(
                {
                    "name": name,
                    "extra_settings": extra,
                    "elapsed_seconds": time.monotonic() - started,
                    "similarity": similarity,
                    "passes_existing_quality_threshold": similarity > 0.8,
                    "response": response,
                }
            )
            print(f"{name}: similarity={similarity:.10f}, elapsed={time.monotonic() - started:.2f}s", flush=True)
        except BaseException as exc:
            metadata["results"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            save()
    # Exit 0 means the experiment ran. Quality outcomes are recorded above;
    # this diagnostic does not replace the original Documentation CI result.


if __name__ == "__main__":
    main()
