#!/usr/bin/env python3
"""Overlap between the innovation training corpus and the three taste benchmarks.

All three benchmarks are built on arXiv / OpenReview papers and our innovation
corpus is reverse-engineered from real papers, so this has to be measured, not
assumed (TASTE_EVAL_SHORTLIST_zh.md "数据污染风险").

Training identity = `methods.json` (slug/title/arxiv of every method the
innovation corpus was derived from).  Emits the contaminated ids so score.py
can be re-run with them excluded.

    python decontam_check.py [--out contaminated_ids.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benches  # noqa: E402


def norm_arxiv(a):
    if a is None:
        return None
    a = re.sub(r"^arxiv[:/]*", "", str(a).strip(), flags=re.I)
    a = re.sub(r"v\d+$", "", a)
    return a or None


def norm_title(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=os.path.join(ROOT, "methods.json"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, ".cache", "taste_eval"))
    ap.add_argument("--giants-n", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    methods = json.load(open(a.methods))
    train_ax = {norm_arxiv(m.get("arxiv")) for m in methods} - {None}
    train_ti = {norm_title(m.get("title")) for m in methods} - {""}
    report = {"train_methods": len(methods), "train_arxiv_ids": len(train_ax)}
    bad = {}

    g = benches.load_giants(os.path.join(a.data_dir, "giants_test.parquet"), n=a.giants_n)
    hit = [t["id"] for t in g if norm_arxiv(t["meta"]["arxiv_id"]) in train_ax]
    report["giants"] = {"n": len(g), "contaminated": len(hit)}
    bad["giants"] = hit

    sj = [json.loads(l) for l in open(os.path.join(a.data_dir, "scijudge_test.jsonl"))]
    hit = []
    for i, r in enumerate(sj):
        if norm_arxiv(r["paper_a_arxiv_id"]) in train_ax or norm_arxiv(r["paper_b_arxiv_id"]) in train_ax:
            hit.append(f"{i:05d}_{r.get('paper_a_arxiv_id')}")
    report["scijudge"] = {"n_pairs": len(sj), "contaminated_pairs": len(hit)}
    bad["scijudge"] = hit

    sb = [json.loads(l) for l in open(os.path.join(a.data_dir, "soundness.jsonl"))]
    hit = [r["pair_id"] for r in sb if norm_title(r["title"]) in train_ti]
    report["soundness"] = {"n": len(sb), "contaminated": len(hit)}
    bad["soundness"] = hit

    print(json.dumps(report, indent=2))
    if a.out:
        json.dump(bad, open(a.out, "w"), indent=2)
        print(f"[decontam] ids written to {a.out}")


if __name__ == "__main__":
    main()
