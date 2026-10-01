#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Validation draft only: replay production parity lanes in the unchanged image.
set -euo pipefail
curl -sSfL https://github.com/mitsuhiko/minijinja/releases/download/2.3.1/minijinja-cli-installer.sh | sh
# shellcheck disable=SC1091
source /var/lib/buildkite-agent/.cargo/env
python3 tools/ci/upload_rocm_function_audio_diagnostic.py \
    --output parity-audio-diagnostic-pipeline.yml --upload
