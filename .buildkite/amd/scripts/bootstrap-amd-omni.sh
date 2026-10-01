#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Diagnostic only: fixed retained WAV in the original ROCm image.
set -euo pipefail
curl -sSfL https://github.com/mitsuhiko/minijinja/releases/download/2.3.1/minijinja-cli-installer.sh | sh
# shellcheck disable=SC1091
source /var/lib/buildkite-agent/.cargo/env
python3 tools/ci/upload_rocm_tts_failed_audio_replay.py \
    --output tts-failed-audio-replay-pipeline.yml --upload
