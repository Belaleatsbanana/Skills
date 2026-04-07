#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Deprecated: use ./dockerfiles/sandbox/execeval_sandbox.sh run [image_tag]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/execeval_sandbox.sh" run "${1:-nemo-skills-sandbox-execeval}"
