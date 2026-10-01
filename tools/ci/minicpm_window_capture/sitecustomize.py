# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Install observation hooks only in the dedicated window diagnostic."""

import os

if os.environ.get("ROCM_WINDOW_TRACE_DIR"):
    from window_trace import install

    install()
