# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility to clone a git repo at a specific commit on the cluster login node."""

import os
import re


def _extract_repo_name(repo_url: str) -> str:
    """Extract repository name from a git URL.

    Handles both HTTPS and SSH URLs:
        https://github.com/NVIDIA/NeMo.git  -> NeMo
        git@github.com:NVIDIA/NeMo.git      -> NeMo
        https://github.com/NVIDIA/NeMo      -> NeMo
    """
    # Remove trailing .git if present
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # Take the last path component
    return url.rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def _validate_commit_id(commit_id: str) -> None:
    """Validate that commit_id looks like a hex SHA (not a branch name or tag)."""
    if not commit_id:
        raise ValueError(
            "code_commit is required when code_repo is set. "
            "Provide a full commit SHA (e.g. code_commit: abc123def456)."
        )
    if not re.fullmatch(r'[0-9a-fA-F]{7,40}', commit_id):
        raise ValueError(
            f"code_commit must be a hex commit SHA (7-40 hex chars), got: '{commit_id}'. "
            f"Branch names and tags are not allowed because they are mutable and would "
            f"break caching. Resolve to a SHA first with: git rev-parse origin/<branch>"
        )


def resolve_code_repo(repo_url: str, commit_id: str, cache_base_dir: str, tunnel) -> str:
    """Clone a git repo at a specific commit on the cluster, return the path.

    The clone is executed on the cluster login node via the provided tunnel.
    A deterministic cache directory is used so that repeated calls with the
    same repo+commit are instant (no re-clone).

    Args:
        repo_url: Git repository URL (HTTPS or SSH).
        commit_id: Git commit SHA to checkout.
        cache_base_dir: Base directory on the cluster filesystem for the cache
            (e.g. the output_dir from config).
        tunnel: A nemo_run SSHTunnel or LocalTunnel instance.

    Returns:
        The absolute path to the cloned repo root on the cluster filesystem.

    Raises:
        ValueError: If commit_id is missing or invalid.
        RuntimeError: If git clone or checkout fails.
    """
    _validate_commit_id(commit_id)

    repo_name = _extract_repo_name(repo_url)
    commit_short = commit_id[:12]
    target_dir = os.path.join(cache_base_dir, "code_repos", repo_name, commit_short)
    sentinel = os.path.join(target_dir, "clone_done")

    # Check if already cached
    check = tunnel.run(f"test -f {sentinel}", hide=True, warn=True)
    if check.return_code == 0:
        print(f"  Using cached clone: {target_dir}")
        return target_dir

    # Clone the repo
    print(f"  Cloning {repo_url} at {commit_id[:12]} to {target_dir} ...")
    clone_cmd = (
        f"mkdir -p {target_dir} && "
        f"git clone --no-checkout {repo_url} {target_dir} && "
        f"cd {target_dir} && "
        f"git checkout {commit_id} && "
        f"touch {sentinel}"
    )
    result = tunnel.run(clone_cmd, hide=True, warn=True)

    if result.return_code != 0:
        # Clean up partial clone on failure
        tunnel.run(f"rm -rf {target_dir}", hide=True, warn=True)
        error_output = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Failed to clone {repo_url} at commit {commit_id}.\n"
            f"Command ran on login node, exit code {result.return_code}.\n"
            f"Output:\n{error_output}"
        )

    print(f"  Clone complete: {target_dir}")
    return target_dir
