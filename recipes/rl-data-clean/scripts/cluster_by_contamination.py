#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Union-Find clustering: group problems that the LLM judged as same (contaminated=true) into clusters for dedup.
#
# Usage:
#   python cluster_by_contamination.py output.jsonl
#   python cluster_by_contamination.py output.jsonl --out-dedup dedup.jsonl
#   python cluster_by_contamination.py output.jsonl --out-clusters clusters.jsonl

import argparse
import json
import os
import sys
from collections import defaultdict


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def ensure(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self.ensure(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        rx, ry = self.rank[px], self.rank[py]
        if rx < ry:
            self.parent[px] = py
        elif rx > ry:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

    def get_components(self):
        comp = defaultdict(set)
        for x in self.parent:
            root = self.find(x)
            comp[root].add(x)
        return dict(comp)


def main():
    ap = argparse.ArgumentParser(description="Cluster problems by LLM contamination judgment (union-find)")
    ap.add_argument("input", default="output.jsonl", nargs="?", help="input jsonl (check_contamination output)")
    ap.add_argument("--out-dedup", default=None, help="write one problem per cluster (dedup list) to this jsonl")
    ap.add_argument("--out-clusters", default=None, help="write cluster_id per problem to this jsonl")
    ap.add_argument("--out-cluster-sizes", default=None, help="write cluster size distribution to this json file")
    ap.add_argument(
        "--repr",
        choices=["first", "smallest_id"],
        default="first",
        help="which problem to keep as representative when writing dedup",
    )
    ap.add_argument("--top-k", type=int, default=None, help="only use first k similar items per problem; default all")
    ap.add_argument(
        "--stats-only",
        action="store_true",
        help="only print one JSON line to stdout: n_clusters, n_nodes, n_edges, n_lines",
    )
    ap.add_argument(
        "--sweep-k",
        type=int,
        default=None,
        metavar="K_MAX",
        help="sweep top_k=1..K_MAX in one pass, print k -> n_clusters table",
    )
    args = ap.parse_args()

    if args.sweep_k is not None:
        k_max = args.sweep_k
        all_problems = set()
        edges = []
        n_lines = 0
        print(f"Pass: reading input and collecting edges for sweep_k 1..{k_max} ...", file=sys.stderr)
        with open(args.input) as f:
            for line in f:
                n_lines += 1
                if n_lines % 50000 == 0:
                    print(f"  lines {n_lines}, edges {len(edges)}", file=sys.stderr)
                d = json.loads(line)
                p = d.get("problem")
                if not p:
                    continue
                all_problems.add(p)
                for i, j in enumerate(d.get("judgments", [])):
                    if j.get("contaminated") is not True:
                        continue
                    q = j.get("similar_item")
                    if not q or q == p:
                        continue
                    edges.append((p, q, i + 1))
        all_nodes = all_problems | {q for _, q, _ in edges}
        print(f"Nodes: {len(all_nodes)}, edges: {len(edges)}, n_lines: {n_lines}", file=sys.stderr)
        print("k\tn_clusters", file=sys.stderr)
        print("k\tn_clusters")
        for k in range(1, k_max + 1):
            uf = UnionFind()
            for x in all_nodes:
                uf.ensure(x)
            for p, q, k_min in edges:
                if k_min <= k:
                    uf.union(p, q)
            n_comp = len(uf.get_components())
            print(f"{k}\t{n_comp}", file=sys.stderr)
            print(f"{k}\t{n_comp}")
        return 0

    uf = UnionFind()
    problem_order = []
    print("Pass 1: building union-find from contaminated judgments ...", file=sys.stderr)
    n_lines = 0
    n_edges = 0
    with open(args.input) as f:
        for line in f:
            n_lines += 1
            if n_lines % 50000 == 0:
                print(f"  processed {n_lines} lines, edges so far {n_edges}", file=sys.stderr)
            d = json.loads(line)
            p = d.get("problem")
            if not p:
                continue
            uf.ensure(p)
            if p not in problem_order:
                problem_order.append(p)
            judgments = d.get("judgments", [])
            if args.top_k is not None:
                judgments = judgments[: args.top_k]
            for j in judgments:
                if j.get("contaminated") is not True:
                    continue
                q = j.get("similar_item")
                if not q or q == p:
                    continue
                uf.ensure(q)
                uf.union(p, q)
                n_edges += 1

    print(f"Total lines: {n_lines}, edges (contaminated=true): {n_edges}", file=sys.stderr)
    comp = uf.get_components()
    n_comp = len(comp)
    n_nodes = sum(len(s) for s in comp.values())
    print(f"Nodes: {n_nodes}, clusters: {n_comp}", file=sys.stderr)

    if args.stats_only:
        out = {"n_lines": n_lines, "n_edges": n_edges, "n_nodes": n_nodes, "n_clusters": n_comp}
        if args.top_k is not None:
            out["top_k"] = args.top_k
        print(json.dumps(out))
        return 0

    sizes = [len(s) for s in comp.values()]
    size_dist = defaultdict(int)
    for s in sizes:
        size_dist[s] += 1
    print("Cluster size distribution (size -> count):", file=sys.stderr)
    for size in sorted(size_dist.keys()):
        print(f"  size {size}: {size_dist[size]} clusters", file=sys.stderr)

    if args.out_cluster_sizes:
        with open(args.out_cluster_sizes, "w") as out:
            json.dump(
                {"n_clusters": n_comp, "n_nodes": n_nodes, "size_distribution": dict(size_dist), "sizes": sizes},
                out,
                indent=2,
            )
        print(f"Wrote cluster sizes to {args.out_cluster_sizes}", file=sys.stderr)

    root_to_repr = {}
    for root, members in comp.items():
        if args.repr == "first":
            repr_p = None
            for p in problem_order:
                if p in members:
                    repr_p = p
                    break
            if repr_p is None:
                repr_p = min(members)
        else:
            repr_p = min(members)
        root_to_repr[root] = repr_p

    if args.out_dedup:
        out_path = args.out_dedup
        tmp_path = out_path + ".tmp"
        chunk_size = 5000
        with open(tmp_path, "w") as out:
            buf = []
            for i, root in enumerate(comp):
                repr_p = root_to_repr[root]
                members = comp[root]
                similar_items = [x for x in members if x != repr_p]
                buf.append(json.dumps({"problem": repr_p, "similar_items": similar_items}, ensure_ascii=False) + "\n")
                if len(buf) >= chunk_size:
                    out.write("".join(buf))
                    buf = []
                    if (i + 1) % 50000 == 0:
                        print(f"  written {i + 1}/{len(comp)}", file=sys.stderr)
            if buf:
                out.write("".join(buf))
        os.rename(tmp_path, out_path)
        print(f"Wrote {len(comp)} deduplicated records to {out_path}", file=sys.stderr)

    if args.out_clusters:
        repr_to_id = {v: i for i, v in enumerate(sorted(root_to_repr.values()))}
        problem_to_cid = {}
        for root, members in comp.items():
            cid = repr_to_id[root_to_repr[root]]
            for p in members:
                problem_to_cid[p] = cid
        with open(args.out_clusters, "w") as out:
            for p, cid in problem_to_cid.items():
                out.write(json.dumps({"problem": p, "cluster_id": cid}, ensure_ascii=False) + "\n")
        print(f"Wrote problem -> cluster_id to {args.out_clusters}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
