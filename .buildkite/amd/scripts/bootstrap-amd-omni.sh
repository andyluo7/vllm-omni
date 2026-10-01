#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Validation draft only: upload one focused job against the exact d7 image.
set -euo pipefail
curl -sSfL https://github.com/mitsuhiko/minijinja/releases/download/2.3.1/minijinja-cli-installer.sh | sh
# shellcheck disable=SC1091
source /var/lib/buildkite-agent/.cargo/env
python3 tools/ci/upload_rocm_quantization_diagnostic.py \
    --output quantization-diagnostic-pipeline.yml --upload
