# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pytest diagnostic plugin retaining exact response bytes before original ASR."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    from tests.helpers import assertions

    original = assertions._resolve_audio_transcript
    output = Path(os.environ["ROCM_FUNCTION_AUDIO_OUTPUT_DIR"])
    safe_test = re.sub(r"[^A-Za-z0-9_.-]", "_", item.nodeid)

    def capture(response, request_config, run_level, *, speech_api):
        raw = getattr(response, "audio_bytes", None)
        if not raw:
            return original(response, request_config, run_level, speech_api=speech_api)
        folder = output / safe_test / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        # Preserve wire/parsed response bytes before the existing helper's
        # PCM16 conversion and Whisper call. Each concurrent response has its
        # own directory; original assertions and transcript settings survive.
        (folder / "response.wav").write_bytes(raw)
        evidence = {
            "test": item.nodeid,
            "audio_sha256": hashlib.sha256(raw).hexdigest(),
            "audio_bytes": len(raw),
            "text_output": getattr(response, "text_content", None),
            "response_audio_format": getattr(response, "audio_format", None),
            "run_level": run_level,
            "speech_api": speech_api,
            "model": request_config.get("model"),
            "asr_model": request_config.get("transcript_model", "small"),
            "asr_language": request_config.get("transcript_language"),
            "asr_temperature": 0.0,
            "similarity_threshold": request_config.get("similarity_threshold", 0.8),
            "sampling_params": (request_config.get("extra_body") or {}).get("sampling_params"),
        }
        metadata = folder / "transcription.json"
        metadata.write_text(json.dumps(evidence, indent=2))
        try:
            transcript = original(response, request_config, run_level, speech_api=speech_api)
            evidence["transcript"] = transcript
            text = getattr(response, "text_content", None)
            if isinstance(text, str) and isinstance(transcript, str):
                evidence["similarity"] = assertions.cosine_similarity_text(transcript.lower(), text.strip().lower())
            return transcript
        except BaseException as exc:
            evidence["asr_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            metadata.write_text(json.dumps(evidence, indent=2))
            print(f"ROCM_FUNCTION_AUDIO_EVIDENCE {folder}", flush=True)

    assertions._resolve_audio_transcript = capture
    try:
        yield
    finally:
        assertions._resolve_audio_transcript = original
