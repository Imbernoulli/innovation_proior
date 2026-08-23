#!/usr/bin/env python3
"""RQ-overlap matrix for MLS-Bench training variants (tv1..tv5).

For each env it compares every variant research question
  * against the EVAL task's "## Research Question" (gate: max_shared_sentence
    < 0.5 AND 0 verbatim sentences), and
  * pairwise against every sibling variant (gate: max_shared_sentence < 0.5,
    0 verbatim sentences either direction).

Same sentence/overlap definitions as scripts/mlsbench_check_train_tasks.py.

Usage:
  python scripts/mlsbench_rq_overlap_matrix.py --base ml-clustering-algorithm
  python scripts/mlsbench_rq_overlap_matrix.py --all [--out matrix.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_ROOT = Path("/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev")
SUFFIXES = ["tv1", "tv2", "tv3", "tv4", "tv5"]


def _rq_section(md_path: Path) -> str:
    if not md_path.exists():
        return ""
    lines = md_path.read_text().split("\n")
    out, on = [], False
    for ln in lines:
        if ln.strip().lower().startswith("## research question"):
            on = True
            continue
        if on and ln.startswith("## "):
            break
        if on:
            out.append(ln)
    return "\n".join(out).strip()


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def overlap(a: str, b: str) -> dict:
    sa, sb = _sentences(a), _sentences(b)
    best, best_pair = 0.0, ("", "")
    verbatim = 0
    for x in sa:
        for y in sb:
            r = difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio()
            if r > best:
                best, best_pair = r, (x[:110], y[:110])
        if x.lower() in b.lower():
            verbatim += 1
    wa = set(re.findall(r"[a-z]{4,}", a.lower()))
    wb = set(re.findall(r"[a-z]{4,}", b.lower()))
    jac = len(wa & wb) / max(1, len(wa | wb))
    return {
        "max_shared_sentence": round(best, 3),
        "verbatim_sentences": verbatim,
        "word_jaccard": round(jac, 3),
        "closest_pair": best_pair,
    }


def variant_rq(base: str, sfx: str) -> str:
    p = (PROJECT_ROOT / "training_tasks" / f"mlsbench_{sfx}"
         / f"{base}_{sfx}" / "research_question.md")
    return p.read_text().strip() if p.exists() else ""


def check_env(base: str, gate: float) -> dict:
    eval_rq = _rq_section(DEV_ROOT / "tasks" / base / "task_description.md")
    rqs = {s: variant_rq(base, s) for s in SUFFIXES}
    present = [s for s, t in rqs.items() if t]
    rec = {"base": base, "variants_present": present, "vs_eval": {}, "pairwise": {}, "violations": []}
    for s in present:
        o = overlap(rqs[s], eval_rq)
        rec["vs_eval"][s] = o
        if o["max_shared_sentence"] >= gate or o["verbatim_sentences"] > 0:
            rec["violations"].append(f"{s} vs eval: {o['max_shared_sentence']}/{o['verbatim_sentences']}")
    for i, s1 in enumerate(present):
        for s2 in present[i + 1:]:
            o = overlap(rqs[s1], rqs[s2])
            rec["pairwise"][f"{s1}-{s2}"] = o
            if o["max_shared_sentence"] >= gate or o["verbatim_sentences"] > 0:
                rec["violations"].append(f"{s1}-{s2}: {o['max_shared_sentence']}/{o['verbatim_sentences']}")
    rec["ok"] = not rec["violations"]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bases = args.base or []
    if args.all or not bases:
        seen = set()
        for sfx in SUFFIXES:
            root = PROJECT_ROOT / "training_tasks" / f"mlsbench_{sfx}"
            if root.is_dir():
                for d in sorted(root.iterdir()):
                    if d.is_dir() and d.name.endswith(f"_{sfx}"):
                        seen.add(d.name[: -len(sfx) - 1])
        bases = sorted(seen)

    results = []
    for b in bases:
        rec = check_env(b, args.gate)
        results.append(rec)
        flag = "OK " if rec["ok"] else "BAD"
        ve = {s: o["max_shared_sentence"] for s, o in rec["vs_eval"].items()}
        pmax = max([o["max_shared_sentence"] for o in rec["pairwise"].values()] or [0.0])
        print(f"[{flag}] {b:42s} vs_eval={ve} pair_max={pmax}"
              + (f"  VIOLATIONS: {rec['violations']}" if rec["violations"] else ""))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"-> {args.out}")
    return 0 if all(r["ok"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
