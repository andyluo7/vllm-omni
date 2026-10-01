# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("speech_evidence_contract", ROOT / "tests/helpers/audio_evidence.py")
assert SPEC is not None and SPEC.loader is not None
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


def _actual_assertions():
    # Run the actual assertion functions. Stub only missing local media/ASR
    # dependencies; fixture callbacks below exercise original failure ordering.
    tree = ast.parse((ROOT / "tests/helpers/assertions.py").read_text())
    names = {
        "_assert_pcm_int16_speech_hnr",
        "_assert_transcript_matches",
        "assert_audio_speech_response",
    }
    selected = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[]
    )

    class Array:
        def astype(self, *_args):
            return self

        def __truediv__(self, _value):
            return self

    def similarity(transcript, expected):
        return 1.0 if transcript == expected else 0.681

    namespace = {
        "Any": object,
        "np": SimpleNamespace(frombuffer=lambda *_args, **_kwargs: Array(), int16=int, float32=float),
        "io": io,
        "sf": SimpleNamespace(info=lambda _stream: SimpleNamespace(samplerate=8000)),
        "_MIN_PCM_SPEECH_HNR_DB": 1.0,
        "_PCM_SPEECH_SAMPLE_RATE_HZ": 24000,
        "retain_speech_response": EVIDENCE.retain_speech_response,
        "update_speech_evidence": EVIDENCE.update_speech_evidence,
        "_compute_pcm_hnr_db": lambda *_args, **_kwargs: 0.9996,
        "_speech_pcm_sample_rate": lambda config: int(config.get("expected_sample_rate", 24000)),
        "_speech_audio_for_transcription": lambda raw, _config: raw,
        "_resolve_audio_transcript": lambda *_args, **_kwargs: "These wristbands should be encoded as 8 kHz audio.",
        "cosine_similarity_text": similarity,
        "_short_transcript_contains_expected": lambda *_args: False,
        "_assert_preset_voice_gender_from_audio": lambda *_args, **_kwargs: None,
    }
    exec(compile(selected, "actual-speech-assertions", "exec"), namespace)
    return namespace


def _records(root):
    return [json.loads(path.read_text()) for path in root.glob("*/*/result.json")]


def test_hnr_failure_retains_wire_pcm_before_asr_and_exact_value(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CI_SPEECH_EVIDENCE_DIR", str(tmp_path))
    namespace = _actual_assertions()

    def no_asr(*_args, **_kwargs):
        pytest.fail("HNR failure must preserve the original early exit before ASR")

    namespace["_resolve_audio_transcript"] = no_asr
    raw = b"\x01\x00" * 128
    response = SimpleNamespace(success=True, audio_bytes=raw, audio_format="audio/pcm")
    with pytest.raises(AssertionError, match="below the configured floor"):
        namespace["assert_audio_speech_response"](
            response, {"response_format": "pcm", "input": "original"}, "full_model"
        )
    record = _records(tmp_path)[0]
    assert record["outcome"] == "failed"
    assert record["hnr_db"] == 0.9996 and record["hnr_floor_db"] == 1.0
    assert record["pcm_sample_rate"] == 24000
    assert record["audio_sha256"] == hashlib.sha256(raw).hexdigest()
    assert next(tmp_path.glob("*/*/response.pcm")).read_bytes() == raw
    assert "transcript" not in record


def test_transcript_failure_preserves_wav_request_and_unchanged_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CI_SPEECH_EVIDENCE_DIR", str(tmp_path))
    namespace = _actual_assertions()
    raw = b"RIFF\x00\x00\x00\x00WAVEoriginal"
    response = SimpleNamespace(success=True, audio_bytes=raw, audio_format="audio/wav")
    settings = {
        "response_format": "wav",
        "sample_rate": 8000,
        "input": "This response should be encoded as eight kilohertz audio.",
        "authorization": "must not be saved",
    }
    with pytest.raises(AssertionError, match="Transcript doesn't match input"):
        namespace["assert_audio_speech_response"](response, settings, "full_model")
    record = _records(tmp_path)[0]
    assert record["similarity"] == 0.681 and record["similarity_threshold"] == 0.9
    assert record["transcript"] == "These wristbands should be encoded as 8 kHz audio."
    assert record["request_settings"]["sample_rate"] == 8000
    assert "authorization" not in record["request_settings"]
    assert next(tmp_path.glob("*/*/response.wav")).read_bytes() == raw


def test_concurrent_requests_keep_separate_metadata_and_original_returns(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CI_SPEECH_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_batch (call)")

    @EVIDENCE.retain_speech_response
    def assertion(response, request_config, run_level=None):
        EVIDENCE.update_speech_evidence(transcript=request_config["input"], similarity=1.0)
        return response.audio_bytes

    def invoke(number):
        raw = str(number).encode()
        return assertion(SimpleNamespace(success=True, audio_bytes=raw), {"input": str(number)}, "full_model")

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(invoke, range(8))) == [str(n).encode() for n in range(8)]
    records = _records(tmp_path)
    assert len(records) == 8
    assert {r["transcript"] for r in records} == {str(n) for n in range(8)}
    assert all(r["transcript"] == r["request_settings"]["input"] and r["outcome"] == "passed" for r in records)
    assert EVIDENCE._CURRENT.get() is None


def test_asr_failure_and_missing_audio_keep_original_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CI_SPEECH_EVIDENCE_DIR", str(tmp_path))

    @EVIDENCE.retain_speech_response
    def assertion(response, config, run_level=None):
        if response.audio_bytes:
            raise RuntimeError("original ASR error")
        raise AssertionError("original missing audio")

    for raw, error, message in [
        (b"raw", RuntimeError, "original ASR error"),
        (None, AssertionError, "original missing audio"),
    ]:
        with pytest.raises(error, match=message):
            assertion(SimpleNamespace(success=False, audio_bytes=raw), {})
    records = _records(tmp_path)
    assert len(records) == 2 and all(r["outcome"] == "failed" for r in records)
    assert sum(r["audio_file"] is None for r in records) == 1


def test_silence_metadata_is_valid_json_and_final_write_failure_keeps_assertion(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CI_SPEECH_EVIDENCE_DIR", str(tmp_path))

    @EVIDENCE.retain_speech_response
    def assertion(response, config, run_level=None):
        EVIDENCE.update_speech_evidence(hnr_db=float("-inf"))
        raise AssertionError("original silence failure")

    with pytest.raises(AssertionError, match="original silence failure"):
        assertion(SimpleNamespace(audio_bytes=b"\x00\x00"), {})
    assert _records(tmp_path)[0]["hnr_db"] == "-inf"
    original = EVIDENCE._write

    def fail_final(path, value):
        if value["outcome"] == "failed":
            raise OSError("injected final write error")
        original(path, value)

    monkeypatch.setattr(EVIDENCE, "_write", fail_final)
    with pytest.raises(AssertionError, match="original silence failure"):
        assertion(SimpleNamespace(audio_bytes=b"\x00\x00"), {})
    assert EVIDENCE._CURRENT.get() is None


def test_disabled_evidence_does_not_touch_files_or_change_arguments(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_CI_SPEECH_EVIDENCE_DIR", raising=False)
    seen = []

    @EVIDENCE.retain_speech_response
    def assertion(response, config, run_level=None):
        seen.append((response, config, run_level))
        return 7

    response, config = object(), {"input": "unchanged"}
    assert assertion(response, config, "core_model") == 7
    assert seen == [(response, config, "core_model")]
    assert not list(tmp_path.iterdir())


def test_actual_native_runner_upload_scope_retains_original_speech_failure(tmp_path):
    spec = importlib.util.spec_from_file_location("runner_contract", ROOT / "tests/buildkite/test_amd_job_evidence.py")
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    commands = "pytest -q test_sample.py::test_speech_failure"
    environment = runner._fixture(tmp_path, commands)
    (tmp_path / "audio_evidence.py").write_bytes((ROOT / "tests/helpers/audio_evidence.py").read_bytes())
    with (tmp_path / "test_sample.py").open("a") as test:
        test.write(
            "\nfrom types import SimpleNamespace\n"
            "from audio_evidence import retain_speech_response, update_speech_evidence\n"
            "@retain_speech_response\n"
            "def check_speech(response, config, run_level=None):\n"
            "    update_speech_evidence(hnr_db=0.9996, hnr_floor_db=1.0)\n"
            "    raise AssertionError('original HNR failure')\n"
            "def test_speech_failure():\n"
            "    check_speech(SimpleNamespace(success=True, audio_bytes=b'\\x00\\x00'), "
            "{'response_format':'pcm', 'input':'original'}, 'full_model')\n"
        )
    result = subprocess.run(
        ["bash", str(runner.RUNNER)],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=25,
    )
    assert result.returncode == 1, result.stdout
    report = json.loads((tmp_path / "artifacts/rocm-job/job-result.json").read_text())
    assert report["exit_status"] == 1 and report["counts"]["failed"] == 1
    records = _records(tmp_path / "artifacts/rocm-job/speech")
    assert len(records) == 1 and records[0]["error"] == "original HNR failure"
    assert records[0]["hnr_db"] == 0.9996
