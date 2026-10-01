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
from threading import Lock

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    from tests.helpers import assertions

    original = assertions._resolve_audio_transcript
    output = Path(os.environ["ROCM_FUNCTION_AUDIO_OUTPUT_DIR"])
    safe_test = re.sub(r"[^A-Za-z0-9_.-]", "_", item.nodeid)
    records = []
    records_lock = Lock()

    def capture(response, request_config, run_level, *, speech_api):
        raw = getattr(response, "audio_bytes", None)
        if not raw:
            return original(response, request_config, run_level, speech_api=speech_api)
        folder = output / safe_test / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        # Preserve wire/parsed response bytes before the existing helper's
        # PCM16 conversion and Whisper call. Each concurrent response has its
        # own directory; original assertions and transcript settings survive.
        is_wav = raw.startswith(b"RIFF") and raw[8:12] == b"WAVE"
        audio_name = "response.wav" if is_wav else "response.bin"
        (folder / audio_name).write_bytes(raw)
        evidence = {
            "test": item.nodeid,
            "audio_file": audio_name,
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
            "similarity_threshold": 0.9 if speech_api else request_config.get("similarity_threshold", 0.8),
            "expected_text": request_config.get("transcript_expected_text", request_config.get("input"))
            if speech_api
            else getattr(response, "text_content", None),
            "request_settings": {
                key: request_config[key]
                for key in (
                    "input",
                    "voice",
                    "speaker",
                    "sample_rate",
                    "response_format",
                    "stream",
                    "modalities",
                    "transcript_expected_text",
                    "transcript_pcm_sample_rate",
                    "transcript_escalation_model",
                    "sampling_params_list",
                    "extra_body",
                    "key_words",
                )
                if key in request_config
            },
        }
        metadata = folder / "transcription.json"
        metadata.write_text(json.dumps(evidence, indent=2))
        with records_lock:
            records.append((metadata, evidence))
        try:
            transcript = original(response, request_config, run_level, speech_api=speech_api)
            evidence["transcript"] = transcript
            text = evidence["expected_text"]
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
        outcome = yield
        failure = outcome.excinfo
        for metadata, evidence in records:
            evidence["test_call_outcome"] = "failed" if failure else "passed"
            if failure:
                evidence["test_call_error"] = f"{failure[0].__name__}: {failure[1]}"
            metadata.write_text(json.dumps(evidence, indent=2))
    finally:
        assertions._resolve_audio_transcript = original
