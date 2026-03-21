#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import argparse
import ast
import html
import json
import random
import sys
from pathlib import Path

from flask import Flask, render_template_string, request

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "local/tmp/results/azure-openai-gpt-4-1/ipython-tool/default/eval-results/dsbench_da/output-rs0.jsonl"
)
DEFAULT_FILTER = 'row["generation"] == ""'

CUSTOM_CSS = """
.ns-results-viewer {max-width: 1600px; margin: 0 auto;}
.ns-results-viewer .row-summary {
  background: #f6f7fb;
  border: 1px solid #d9dce7;
  border-radius: 12px;
  padding: 12px 16px;
  margin-bottom: 16px;
}
.ns-results-viewer .summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px 16px;
}
.ns-results-viewer .summary-card {
  background: white;
  border: 1px solid #e4e7f0;
  border-radius: 10px;
  padding: 10px 12px;
}
.ns-results-viewer .summary-card .label {
  font-size: 12px;
  color: #5c6475;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.ns-results-viewer .summary-card .value {
  margin-top: 4px;
  font-size: 14px;
  word-break: break-word;
}
.ns-results-viewer .tree-root {
  border: 1px solid #d9dce7;
  border-radius: 12px;
  padding: 10px 12px;
  background: white;
}
.ns-results-viewer details {
  margin: 6px 0;
  padding-left: 10px;
  border-left: 2px solid #e4e7f0;
}
.ns-results-viewer summary {
  cursor: pointer;
  font-family: ui-monospace, SFMono-Regular, SFMono-Regular, Consolas, monospace;
}
.ns-results-viewer .node-meta {color: #6b7280;}
.ns-results-viewer .node-summary {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.ns-results-viewer .node-actions {
  display: inline-flex;
  gap: 4px;
  margin-left: 8px;
}
.ns-results-viewer .node-actions button {
  border: 0;
  background: #e5ebf7;
  color: #334155;
  border-radius: 6px;
  padding: 1px 6px;
  font-size: 12px;
  line-height: 1.4;
  cursor: pointer;
}
.ns-results-viewer .scalar {
  margin: 4px 0 8px 16px;
  background: #f8f9fc;
  border-radius: 8px;
  padding: 8px 10px;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, SFMono-Regular, Consolas, monospace;
  font-size: 13px;
}
.ns-results-viewer .scalar-list,
.ns-results-viewer .scalar-dict {
  margin: 4px 0 8px 16px;
  background: #f8f9fc;
  border-radius: 8px;
  padding: 8px 10px;
  font-family: ui-monospace, SFMono-Regular, SFMono-Regular, Consolas, monospace;
  font-size: 13px;
}
.ns-results-viewer .scalar-list {
  padding-left: 28px;
}
.ns-results-viewer .scalar-list li {
  margin: 4px 0;
  white-space: pre-wrap;
  word-break: break-word;
}
.ns-results-viewer .scalar-dict-row {
  display: grid;
  grid-template-columns: minmax(120px, 220px) minmax(0, 1fr);
  gap: 10px;
  padding: 3px 0;
}
.ns-results-viewer .mixed-dict-row {
  margin: 6px 0;
  padding-left: 16px;
}
.ns-results-viewer .scalar-dict-key {
  color: #475569;
}
.ns-results-viewer .scalar-dict-value {
  white-space: pre-wrap;
  word-break: break-word;
}
.ns-results-viewer .scalar-preview summary {
  cursor: pointer;
  color: #475569;
}
.ns-results-viewer .scalar-preview .full-value {
  margin-top: 6px;
  white-space: pre-wrap;
  word-break: break-word;
}
.ns-results-viewer .empty {color: #8b93a7; font-style: italic;}
.ns-results-viewer .status {padding-top: 6px;}
"""

ALLOWED_FUNCS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "sum": sum,
    "min": min,
    "max": max,
    "any": any,
    "all": all,
    "sorted": sorted,
}

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Call,
    ast.Load,
    ast.Name,
    ast.Subscript,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Slice,
)


def load_jsonl(file_path: str) -> list[dict]:
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    files_to_load = resolve_input_files(path)
    rows = []
    for current_file in files_to_load:
        with current_file.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def resolve_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        matches = sorted(p for p in path.glob("output-rs*.jsonl") if p.is_file())
        if not matches:
            raise FileNotFoundError(f"No files matching output-rs*.jsonl found in directory: {path}")
        return matches
    raise FileNotFoundError(f"Unsupported input path: {path}")


def validate_filter_expression(expression: str) -> ast.AST:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python expression: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported syntax in filter: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"row", "True", "False", "None", *ALLOWED_FUNCS}:
            raise ValueError(f"Unsupported name in filter: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ValueError("Only a small set of safe builtins is allowed in filters")

    return tree


def evaluate_filter(tree: ast.AST, row: dict) -> bool:
    return bool(eval(compile(tree, "<filter>", "eval"), {"__builtins__": {}}, {"row": row, **ALLOWED_FUNCS}))


def escape_scalar(value) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
    return html.escape(rendered)


def summarize_scalar(value) -> str:
    if value is None:
        return "None"
    if value == "":
        return '""'
    if isinstance(value, str):
        compact = value.replace("\n", "\\n")
        if len(compact) > 80:
            compact = compact[:77] + "..."
        return compact
    if isinstance(value, (int, float, bool)):
        return str(value)
    return type(value).__name__


def render_full_scalar(value) -> str:
    if value is None:
        return "None"
    if value == "":
        return '""'
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, indent=2)


def is_basic_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def render_scalar_list(items: list) -> str:
    entries = "".join(f"<li>{html.escape(render_full_scalar(item))}</li>" for item in items)
    return f"<ul class='scalar-list'>{entries}</ul>"


def render_scalar_dict(items: dict) -> str:
    entries = "".join(
        "<div class='scalar-dict-row'>"
        f"<div class='scalar-dict-key'>{html.escape(str(key))}</div>"
        f"<div class='scalar-dict-value'>{render_scalar_value(value)}</div>"
        "</div>"
        for key, value in items.items()
    )
    return f"<div class='scalar-dict'>{entries}</div>"


def render_scalar_dict_row(key, value) -> str:
    return (
        "<div class='mixed-dict-row'>"
        "<div class='scalar-dict-row'>"
        f"<div class='scalar-dict-key'>{html.escape(str(key))}</div>"
        f"<div class='scalar-dict-value'>{render_scalar_value(value)}</div>"
        "</div>"
        "</div>"
    )


def render_scalar_value(value) -> str:
    rendered = render_full_scalar(value)
    if isinstance(value, str) and (len(value) > 400 or value.count("\n") >= 6):
        preview = value.replace("\n", "\\n")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        return (
            "<details class='scalar-preview'>"
            f"<summary>{html.escape(preview)}</summary>"
            f"<div class='full-value'>{html.escape(rendered)}</div>"
            "</details>"
        )
    return html.escape(rendered)


def make_node_summary(label: str, meta: str) -> str:
    return (
        "<span class='node-summary'>"
        f"<span>{html.escape(label)}</span>"
        f"<span class='node-meta'>{html.escape(meta)}</span>"
        "<span class='node-actions'>"
        "<button type='button' onclick='toggleNodeSubtree(event, this, true)'>⏬</button>"
        "<button type='button' onclick='toggleNodeSubtree(event, this, false)'>⏫</button>"
        "</span>"
        "</span>"
    )


def tree_html(value, label: str = "value", open_by_default: bool = False) -> str:
    if isinstance(value, dict):
        if not value:
            return (
                f'<div class="tree-root"><details {"open" if open_by_default else ""}>'
                f"<summary>{make_node_summary(label, 'dict(0)')}</summary>"
                "<div class='scalar empty'>{}</div></details></div>"
            )
        if all(is_basic_scalar(item) for item in value.values()):
            return (
                f'<div class="tree-root"><details {"open" if open_by_default else ""}>'
                f"<summary>{make_node_summary(label, f'dict({len(value)})')}</summary>"
                f"{render_scalar_dict(value)}</details></div>"
            )
        children = ""
        for key, item in value.items():
            if is_basic_scalar(item):
                children += render_scalar_dict_row(key, item)
            else:
                children += tree_html(item, label=str(key))
        return (
            f'<div class="tree-root"><details {"open" if open_by_default else ""}>'
            f"<summary>{make_node_summary(label, f'dict({len(value)})')}</summary>"
            f"{children}</details></div>"
        )
    if isinstance(value, list):
        if not value:
            return (
                f'<div class="tree-root"><details {"open" if open_by_default else ""}>'
                f"<summary>{make_node_summary(label, 'list(0)')}</summary>"
                "<div class='scalar empty'>[]</div></details></div>"
            )
        if all(is_basic_scalar(item) for item in value):
            return (
                f'<div class="tree-root"><details {"open" if open_by_default else ""}>'
                f"<summary>{make_node_summary(label, f'list({len(value)})')}</summary>"
                f"{render_scalar_list(value)}</details></div>"
            )
        children = "".join(tree_html(item, label=f"[{idx}]") for idx, item in enumerate(value))
        return (
            f'<div class="tree-root"><details {"open" if open_by_default else ""}>'
            f"<summary>{make_node_summary(label, f'list({len(value)})')}</summary>"
            f"{children}</details></div>"
        )
    return (
        f'<details {"open" if open_by_default else ""}>'
        f"<summary>{html.escape(label)} <span class='node-meta'>{html.escape(type(value).__name__)}</span></summary>"
        f"<div class='scalar'>{escape_scalar(value)}</div></details>"
    )


def render_conversation(conversation) -> str:
    if not isinstance(conversation, list):
        return tree_html(conversation, label="conversation", open_by_default=True)
    if not conversation:
        return tree_html(conversation, label="conversation", open_by_default=True)

    children = []
    for idx, message in enumerate(conversation):
        if isinstance(message, dict):
            role = message.get("role", "unknown")
            remaining = {key: value for key, value in message.items() if key != "role"}
            summary = f'[{idx}] {role}'
            if "tool_call_id" in message:
                summary += f' <span class="node-meta">{html.escape(str(message["tool_call_id"]))}</span>'
            children.append(
                '<details>'
                f"<summary>{summary} <span class='node-meta'>message</span></summary>"
                f"{tree_html(remaining, label='fields')}"
                "</details>"
            )
        else:
            children.append(tree_html(message, label=f"[{idx}]"))
    return (
        '<div class="tree-root"><details open>'
        f"<summary>conversation <span class='node-meta'>list({len(conversation)})</span></summary>"
        f"{''.join(children)}</details></div>"
    )


def render_row(row: dict, file_path: str, matched_position: int, matched_total: int, row_index: int) -> tuple[str, str]:
    conversation = row.get("conversation")
    remaining = {key: value for key, value in row.items() if key != "conversation"}
    summary_fields = {
        "file": file_path,
        "row_index": row_index,
        "match_position": f"{matched_position + 1} / {matched_total}",
        "question_id": row.get("question_id"),
        "task_name": row.get("task_name"),
        "finish_reason": summarize_scalar(row.get("finish_reason")),
        "generation": summarize_scalar(row.get("generation")),
        "num_tool_calls": summarize_scalar(row.get("num_tool_calls")),
        "symbolic_correct": summarize_scalar(row.get("symbolic_correct")),
    }
    summary_html = "".join(
        (
            "<div class='summary-card'>"
            f"<div class='label'>{html.escape(str(key))}</div>"
            f"<div class='value'>{html.escape(str(value))}</div>"
            "</div>"
        )
        for key, value in summary_fields.items()
    )
    row_summary = (
        "<div class='row-summary'>"
        "<div class='summary-grid'>"
        f"{summary_html}"
        "</div>"
        "</div>"
    )
    conversation_html = row_summary + render_conversation(conversation)
    details_html = tree_html(remaining, label="row_without_conversation", open_by_default=True)
    return conversation_html, details_html


def build_status(file_path: str, rows: list[dict], matches: list[int], expression: str) -> str:
    total = len(rows)
    matched = len(matches)
    percentage = 0.0 if total == 0 else (matched / total) * 100
    return (
        "<div class='status'>"
        f"<strong>{matched}</strong> matching rows out of <strong>{total}</strong> total "
        f"(<strong>{percentage:.1f}%</strong>)"
        f" for <code>{html.escape(expression)}</code><br>"
        f"<span>{html.escape(file_path)}</span>"
        "</div>"
    )


def view_for_position(
    file_path: str, rows: list[dict], matches: list[int], position: int, expression: str
) -> tuple[int, int, str, str, str]:
    if not matches:
        empty_message = "<div class='scalar empty'>No rows matched the current filter.</div>"
        return 0, 0, build_status(file_path, rows, matches, expression), empty_message, empty_message

    position = max(0, min(position, len(matches) - 1))
    row_index = matches[position]
    conversation_html, details_html = render_row(rows[row_index], file_path, position, len(matches), row_index)
    return position, row_index, build_status(file_path, rows, matches, expression), conversation_html, details_html


def load_and_filter(file_path: str, expression: str) -> tuple[list[dict], list[int], int, int, str, str, str]:
    rows = load_jsonl(file_path)
    if not expression.strip():
        expression = "True"
    tree = validate_filter_expression(expression)
    matches = [idx for idx, row in enumerate(rows) if evaluate_filter(tree, row)]
    position, row_index, status_html, conversation_html, details_html = view_for_position(
        file_path, rows, matches, 0, expression
    )
    return rows, matches, position, row_index, status_html, conversation_html, details_html


def move_position(
    file_path: str, expression: str, rows: list[dict], matches: list[int], position: int, step: int
) -> tuple[int, int, str, str, str]:
    if not matches:
        return view_for_position(file_path, rows, matches, 0, expression)
    new_position = (position + step) % len(matches)
    return view_for_position(file_path, rows, matches, new_position, expression)


def random_position(
    file_path: str, expression: str, rows: list[dict], matches: list[int]
) -> tuple[int, int, str, str, str]:
    if not matches:
        return view_for_position(file_path, rows, matches, 0, expression)
    return view_for_position(file_path, rows, matches, random.randint(0, len(matches) - 1), expression)


def jump_to_position(
    file_path: str, expression: str, rows: list[dict], matches: list[int], requested_position: float
) -> tuple[int, int, str, str, str]:
    position = int(requested_position) if requested_position is not None else 0
    if position > 0:
        position -= 1
    return view_for_position(file_path, rows, matches, position, expression)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ns eval results viewer</title>
  <style>
    body {
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      margin: 0;
      background: #eef1f7;
      color: #1f2937;
    }
    .page {
      max-width: 1680px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      background: linear-gradient(135deg, #183153, #245b8f);
      color: white;
      border-radius: 18px;
      padding: 22px 24px;
      margin-bottom: 18px;
      box-shadow: 0 18px 50px rgba(16, 40, 74, 0.18);
    }
    .hero h1 {margin: 0 0 8px 0;}
    .hero p {margin: 0; max-width: 980px;}
    .panel {
      background: white;
      border-radius: 18px;
      border: 1px solid #dce3f0;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      margin-bottom: 18px;
    }
    .controls {
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 14px;
      margin-bottom: 14px;
    }
    .field label {
      display: block;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
      color: #526077;
    }
    .field input {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #cfd7e6;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      background: #fcfdff;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: end;
    }
    .actions button {
      border: 0;
      border-radius: 10px;
      background: #1d4ed8;
      color: white;
      padding: 10px 14px;
      font-size: 14px;
      cursor: pointer;
    }
    .actions button.secondary {background: #475569;}
    .actions button.ghost {background: #0f766e;}
    .actions .mini {
      width: 120px;
      background: #fcfdff;
      color: #111827;
      border: 1px solid #cfd7e6;
    }
    .hint {
      font-size: 13px;
      color: #5b6578;
      margin-top: 8px;
    }
    .split {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
      gap: 18px;
    }
    .error {
      background: #fff1f2;
      color: #9f1239;
      border: 1px solid #fecdd3;
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 18px;
    }
    code {
      background: #e5ebf7;
      padding: 1px 5px;
      border-radius: 6px;
    }
    @media (max-width: 1000px) {
      .controls, .split {
        grid-template-columns: 1fr;
      }
    }
    {{ css|safe }}
  </style>
  <script>
    function toggleNodeSubtree(event, button, expanded) {
      event.preventDefault();
      event.stopPropagation();
      const summary = button.closest("summary");
      if (!summary) return;
      const details = summary.parentElement;
      if (!details) return;
      if (expanded) {
        details.open = true;
      }
      details.querySelectorAll("details").forEach((node) => {
        if (node === details) return;
        if (node.classList.contains("scalar-preview")) return;
        node.open = expanded;
      });
    }
  </script>
</head>
<body>
  <div class="page ns-results-viewer">
    <div class="hero">
      <h1>ns eval results viewer</h1>
      <p>Inspect JSONL output with a collapsible <code>conversation</code> tree, Python-style row filters, and navigation across matching rows.</p>
    </div>

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

    <form method="get" class="panel">
      <div class="controls">
        <div class="field">
          <label for="file_path">Results JSONL</label>
          <input id="file_path" name="file_path" value="{{ file_path }}">
        </div>
        <div class="field">
          <label for="filter_expr">Row filter</label>
          <input id="filter_expr" name="filter_expr" value="{{ filter_expr }}">
        </div>
      </div>

      <div class="actions">
        <button type="submit" name="action" value="apply">Apply filter</button>
        <button type="submit" name="action" value="reset" class="secondary">Reset filter</button>
        <button type="submit" name="action" value="prev" class="secondary">Previous match</button>
        <button type="submit" name="action" value="next" class="secondary">Next match</button>
        <button type="submit" name="action" value="random" class="ghost">Random match</button>
        <div class="field">
          <label for="position">Match position</label>
          <input id="position" name="position" type="number" min="1" value="{{ position_display }}" class="mini">
        </div>
        <button type="submit" name="action" value="go" class="secondary">Go</button>
      </div>

      <div class="hint">
        Path can be a single JSONL file or a directory containing <code>output-rs*.jsonl</code>.
        Example filter: <code>row["generation"] == ""</code>
      </div>
    </form>

    {% if status_html %}
    <div class="panel">{{ status_html|safe }}</div>
    {% endif %}

    {% if conversation_html or details_html %}
    <div id="results-root" class="split">
      <div class="panel">
        <h2>Conversation</h2>
        {{ conversation_html|safe }}
      </div>
      <div class="panel">
        <h2>Other Fields</h2>
        {{ details_html|safe }}
      </div>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""


def create_app(initial_file_path: str = "") -> Flask:
    app = Flask(__name__)
    app.config["INITIAL_FILE_PATH"] = initial_file_path

    @app.get("/")
    def index():
        file_path = request.args.get("file_path", app.config["INITIAL_FILE_PATH"])
        action = request.args.get("action", "apply")
        filter_expr = request.args.get("filter_expr", DEFAULT_FILTER)
        if action == "reset":
            filter_expr = "True"

        try:
            if not file_path.strip():
                return render_template_string(
                    PAGE_TEMPLATE,
                    css=CUSTOM_CSS,
                    error="",
                    file_path="",
                    filter_expr=filter_expr,
                    position_display=1,
                    row_index=-1,
                    status_html="",
                    conversation_html="",
                    details_html="",
                )
            rows = load_jsonl(file_path)
            expression = filter_expr.strip() or "True"
            tree = validate_filter_expression(expression)
            matches = [idx for idx, row in enumerate(rows) if evaluate_filter(tree, row)]

            requested_position = request.args.get("position", "1")
            try:
                position = max(int(requested_position or "1") - 1, 0)
            except ValueError:
                position = 0

            if action == "next" and matches:
                position = (position + 1) % len(matches)
            elif action == "prev" and matches:
                position = (position - 1) % len(matches)
            elif action == "random" and matches:
                position = random.randint(0, len(matches) - 1)

            position, row_index, status_html, conversation_html, details_html = view_for_position(
                file_path, rows, matches, position, expression
            )
            return render_template_string(
                PAGE_TEMPLATE,
                css=CUSTOM_CSS,
                error="",
                file_path=file_path,
                filter_expr=expression,
                position_display=max(position + 1, 1),
                row_index=row_index,
                status_html=status_html,
                conversation_html=conversation_html,
                details_html=details_html,
            )
        except Exception as exc:
            return render_template_string(
                PAGE_TEMPLATE,
                css=CUSTOM_CSS,
                error=str(exc),
                file_path=file_path,
                filter_expr=filter_expr,
                position_display=1,
                row_index=-1,
                status_html="",
                conversation_html="",
                details_html="",
            )

    return app


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View ns eval JSONL outputs in a local web app.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="",
        help="Optional JSONL file or directory containing output-rs*.jsonl files.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the local web server.")
    parser.add_argument("--port", type=int, default=7861, help="Port to bind the local web server.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    app = create_app(initial_file_path=args.input_path)
    app.run(host=args.host, port=args.port, debug=False)
