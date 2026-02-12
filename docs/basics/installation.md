# Installation & Dependency Groups

NeMo Skills provides two installable packages:

- **`nemo-skills`** (root) — full install with CLI, cluster orchestration, benchmarks
- **`nemo-skills-core`** (`core/` subdirectory) — lightweight runtime only

## Default installation

`pip install nemo-skills` gives you **everything** (inference, evaluation, CLI,
cluster orchestration, benchmarks):

```bash
pip install git+https://github.com/NVIDIA-NeMo/Skills.git
# or, from a local clone:
pip install -e .
```

## Lightweight installation

If you only need inference, evaluation, and tool calling (no cluster orchestration):

```bash
pip install "nemo-skills-core @ git+https://github.com/NVIDIA-NeMo/Skills.git#subdirectory=core"
# or, from a local clone:
pip install -e core/
```

## Extras (dependency groups)

Each extra maps to a requirements file under `requirements/`.

| Extra | Requirements file | What it provides |
|-------|-------------------|------------------|
| `core` | `requirements/core.txt` | Agent runtime: inference, evaluation, tool calling (MCP), prompt formatting, math/code grading. No cluster orchestration. |
| `pipeline` | `requirements/pipeline.txt` | CLI (`ns` command), cluster management, experiment tracking (`nemo_run`, `typer`, `wandb`). |
| `benchmarks` | `requirements/benchmarks.txt` | Assorted benchmark dependencies (BFCL, BIRD, translation, etc.). |
| `dev` | `requirements/common-tests.txt`, `requirements/common-dev.txt` | Development and testing tools (`pytest`, `ruff`, `pre-commit`). |

### Examples

```bash
# Full install (default)
pip install -e .

# Core only — lightweight runtime for downstream integrations
pip install -e core/

# Development (everything + dev tools)
pip install -e ".[dev]"
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
