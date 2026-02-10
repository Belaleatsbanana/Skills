# Installation & Dependency Groups

NeMo Skills is organized into dependency groups so that downstream integrations
(NeMo Gym, NeMo RL, custom agents) can install only what they need.

## Default installation

The base `pip install` gives you **core only** (inference, evaluation, math/code
grading, MCP tool calling). This is all you need for NeMo Gym/RL integration or
custom agent code:

```bash
pip install git+https://github.com/NVIDIA-NeMo/Skills.git
# or, from a local clone:
pip install -e .
```

For the **full install** (cluster orchestration, all benchmarks — equivalent to
the old default), use the `all` extra:

```bash
pip install -e ".[all]"
```

## Extras (dependency groups)

Each extra maps to a requirements file under `requirements/`.

| Extra | Requirements file | What it provides |
|-------|-------------------|------------------|
| `core` | `requirements/core.txt` | Agent runtime: inference, evaluation, tool calling (MCP), prompt formatting, math/code grading. No cluster orchestration. Same as base install. |
| `pipeline` | `requirements/pipeline.txt` | CLI (`ns` command), cluster management, experiment tracking (`nemo_run`, `typer`, `wandb`). |
| `benchmarks` | `requirements/benchmarks.txt` | Assorted benchmark dependencies (BFCL, BIRD, translation, etc.). |
| `all` | `requirements/main.txt` | Everything (core + pipeline + benchmarks). Use this if unsure. |
| `dev` | `requirements/common-tests.txt`, `requirements/common-dev.txt` | Development and testing tools (`pytest`, `ruff`, `pre-commit`). |

### Examples

```bash
# Core only — for NeMo Gym/RL integration or custom agent code
pip install -e .
# (or equivalently: pip install -e ".[core]")

# Core + pipeline (cluster orchestration, no benchmark deps)
pip install -e ".[pipeline]"

# Full install (everything)
pip install -e ".[all]"

# Development (everything + dev tools)
pip install -e ".[all,dev]"
```

## Core / Pipeline architecture boundary

The codebase enforces a one-way dependency rule:

```
Pipeline → Core   ✅  (pipeline modules can import from core)
Core → Pipeline   ❌  (core modules must NOT import from pipeline)
```

**Core** modules live under:

- `nemo_skills/inference/`
- `nemo_skills/evaluation/`
- `nemo_skills/dataset/`
- `nemo_skills/prompt/`
- `nemo_skills/mcp/`
- `nemo_skills/code_execution/`
- `nemo_skills/conversion/`
- `nemo_skills/training/`
- `nemo_skills/utils.py`, `nemo_skills/file_utils.py`

**Pipeline** modules live under:

- `nemo_skills/pipeline/`

This boundary is enforced by `tests/test_dependency_isolation.py` which creates
fresh virtualenvs and verifies that core modules import successfully without
pipeline dependencies installed.

### Dataset loading example

The boundary shows up concretely in dataset loading:

```python
# Core: local-only dataset loading (no cluster deps)
from nemo_skills.dataset.utils import get_dataset_module
module, path, on_cluster = get_dataset_module("gsm8k")

# Pipeline: cluster-aware dataset loading (SSH tunnels, mount resolution)
from nemo_skills.pipeline.dataset import get_dataset_module
module, path, on_cluster = get_dataset_module("gsm8k", cluster_config=cfg)
```

If you pass `cluster_config` to the core version, it will emit a
`DeprecationWarning` and redirect to the pipeline version.
