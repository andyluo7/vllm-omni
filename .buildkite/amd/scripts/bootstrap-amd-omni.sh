#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Validation draft only: run the window diagnostic in the exact final image.
set -euo pipefail
curl -sSfL https://github.com/mitsuhiko/minijinja/releases/download/2.3.1/minijinja-cli-installer.sh | sh
# shellcheck disable=SC1091
source /var/lib/buildkite-agent/.cargo/env
python3 tools/ci/upload_rocm_window_diagnostic.py \
    --output window-diagnostic-pipeline.yml --upload
