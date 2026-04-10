# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Tree-sitter helpers for SAFIM-style completion post-processing (mirrors upstream safim ``ast_utils``).
# Uses ``ast_utils`` from an installed ``safim`` package when present; otherwise loads grammars from
# ``SAFIM_TREE_SITTER_LIB`` (same layout as dockerfiles/sandbox execeval).

from __future__ import annotations

import os
from pathlib import Path

try:
    from ast_utils import ErrorCheckVisitor, get_parser
except ImportError:
    from tree_sitter import Language, Parser

    _TS_LANG: dict | None = None

    def _ts_lang_map() -> dict:
        global _TS_LANG
        if _TS_LANG is not None:
            return _TS_LANG
        lib = os.environ.get("SAFIM_TREE_SITTER_LIB", "")
        if not lib or not Path(lib).is_file():
            raise ImportError(
                "SAFIM tree-sitter is unavailable: either `pip install` a safim package that provides "
                "`ast_utils`, or set SAFIM_TREE_SITTER_LIB to a combined grammar .so "
                "(see dockerfiles/Dockerfile.sandbox.execeval)."
            )
        _TS_LANG = {
            "python": Language(lib, "python"),
            "java": Language(lib, "java"),
            "cpp": Language(lib, "cpp"),
            "csharp": Language(lib, "c_sharp"),
        }
        return _TS_LANG

    class ASTVisitor:
        def __init__(self, with_ndtypes=False, print_debug_outputs=False):
            self.with_ndtypes = with_ndtypes
            self.print_debug_outputs = print_debug_outputs
            self.stack = []
            self.ndtypes = []

        def enter(self, node) -> bool:
            return True

        def leave(self, node):
            pass

        def enter_leaf(self, node):
            pass

        def print_stack(self, node):
            depth = len(self.stack)
            print(" " * depth * 2 + node.type)

        def on_enter(self, node) -> bool:
            if self.print_debug_outputs:
                self.print_stack(node)
            if self.with_ndtypes:
                self.ndtypes.append((node.start_byte, True, node.type))
            enter_fn = getattr(self, "enter_%s" % node.type, self.enter)
            r = enter_fn(node)
            if node.child_count == 0:
                self.enter_leaf(node)
            self.stack.append(node.type)
            return r

        def on_leave(self, node):
            assert self.stack.pop() == node.type
            leave_fn = getattr(self, "leave_%s" % node.type, self.leave)
            leave_fn(node)
            if self.with_ndtypes:
                self.ndtypes.append((node.end_byte, False, node.type))

        def walk(self, root_node):
            if root_node is None:
                return

            cursor = root_node.walk()
            has_next = True

            while has_next:
                current_node = cursor.node

                if self.on_enter(current_node):
                    has_next = cursor.goto_first_child()
                else:
                    has_next = False

                if not has_next:
                    self.on_leave(current_node)
                    has_next = cursor.goto_next_sibling()

                while not has_next and cursor.goto_parent():
                    self.on_leave(cursor.node)
                    has_next = cursor.goto_next_sibling()

        def __call__(self, root_node):
            return self.walk(root_node)

    class ErrorCheckVisitor(ASTVisitor):
        def __init__(self, with_ndtypes=False):
            super().__init__(with_ndtypes)
            self.error_cnt = 0

        def enter_ERROR(self, node):
            if node.text.decode("utf-8") != ";":
                self.error_cnt += 1

    def get_parser(lang: str):
        parser = Parser()
        parser.set_language(_ts_lang_map()[lang])
        return parser
