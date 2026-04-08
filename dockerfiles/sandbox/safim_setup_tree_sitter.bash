#!/usr/bin/env bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Tree-sitter grammars for SAFIM evaluation, pinned to the same commits as upstream:
# https://github.com/gonglinyuan/safim/blob/main/setup_tree_sitter.bash
#
# Installed under SAFIM_TREE_SITTER_ROOT (default /opt/safim-tree-sitter).

set -euo pipefail

ROOT="${SAFIM_TREE_SITTER_ROOT:-/opt/safim-tree-sitter}"
mkdir -p "${ROOT}"
cd "${ROOT}"

git clone https://github.com/tree-sitter/tree-sitter-python
cd tree-sitter-python || exit 1
git checkout 62827156d01c74dc1538266344e788da74536b8a || exit 1
cd ..
git clone https://github.com/tree-sitter/tree-sitter-java
cd tree-sitter-java || exit 1
git checkout 3c24aa9365985830421a3a7b6791b415961ea770 || exit 1
cd ..
git clone https://github.com/tree-sitter/tree-sitter-cpp
cd tree-sitter-cpp || exit 1
git checkout -f 03fa93db133d6048a77d4de154a7b17ea8b9d076
cd ..
git clone https://github.com/tree-sitter/tree-sitter-c-sharp
cd tree-sitter-c-sharp || exit 1
git checkout fcacbeb4af6bcdcfb4527978a997bb03f4fe086d
cd ..
