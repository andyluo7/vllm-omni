# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Retain speech response bytes before quality checks in opted-in CI jobs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

_CURRENT: ContextVar[dict[str, Any] | None] = ContextVar("speech_response_evidence", default=None)
_REQUEST_KEYS = (
    "input",
    "model",
    "voice",
    "speaker",
    "sample_rate",
    "expected_sample_rate",
    "response_format",
    "stream",
    "transcript_expected_text",
    "transcript_pcm_sample_rate",
    "transcript_model",
    "transcript_language",
    "transcript_escalation_model",
    "min_hnr_db",
    "min_audio_bytes",
    "sampling_params_list",
)


def update_speech_evidence(**values: Any) -> None:
    """Record the values already computed by the original quality checks."""
    evidence = _CURRENT.get()
    if evidence is not None:
        # JSON cannot represent infinity; retain it explicitly for silence.
        evidence.update(
            {
                key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in values.items()
            }
        )


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def retain_speech_response(assertion: Callable[..., Any]) -> Callable[..., Any]:
    """Keep each concurrent response separate and preserve the assertion verdict."""

    @wraps(assertion)
    def wrapped(response: Any, request_config: dict[str, Any], run_level: str | None = None) -> Any:
        output = os.environ.get("VLLM_CI_SPEECH_EVIDENCE_DIR")
        if not output:
            return assertion(response, request_config, run_level)
        test = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
        safe_test = re.sub(r"[^A-Za-z0-9_.-]", "_", test)[:180]
        directory = Path(output) / safe_test / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        raw = getattr(response, "audio_bytes", None)
        evidence = {
            "test": test,
            "commit": os.environ.get("BUILDKITE_COMMIT", "unknown"),
            "job_id": os.environ.get("BUILDKITE_JOB_ID", "unknown"),
            "run_level": run_level,
            "response_success": getattr(response, "success", None),
            "response_audio_format": getattr(response, "audio_format", None),
            "request_settings": {key: request_config[key] for key in _REQUEST_KEYS if key in request_config},
            "asr_model": request_config.get("transcript_model", "small"),
            "asr_language": request_config.get("transcript_language"),
            "existing_audio_transcript": getattr(response, "audio_content", None),
            "asr_temperature": 0.0,
            "similarity_threshold": 0.9,
            "outcome": "started",
        }
        if raw:
            is_wav = raw.startswith(b"RIFF") and raw[8:12] == b"WAVE"
            name = (
                "response.wav"
                if is_wav
                else ("response.pcm" if request_config.get("response_format") == "pcm" else "response.bin")
            )
            (directory / name).write_bytes(raw)
            evidence.update(audio_file=name, audio_bytes=len(raw), audio_sha256=hashlib.sha256(raw).hexdigest())
        else:
            evidence.update(audio_file=None, audio_bytes=0)
        metadata = directory / "result.json"
        _write(metadata, evidence)
        token = _CURRENT.set(evidence)
        assertion_error = None
        try:
            result = assertion(response, request_config, run_level)
            evidence["outcome"] = "passed"
            return result
        except BaseException as error:
            assertion_error = error
            evidence.update(outcome="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            _CURRENT.reset(token)
            try:
                _write(metadata, evidence)
            except (OSError, TypeError, ValueError) as error:
                if assertion_error is None:
                    raise
                # The existing assertion failure still controls the verdict.
                print(f"Failed to finalize speech evidence at {metadata}: {error}", flush=True)

    return wrapped
