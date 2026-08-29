#!/usr/bin/env python3
"""Does a <think> DERIVE the method, or RETRIEVE it?

The corpus teaches a model to play a scientist at a point in time and reach a
method from the constraints available then. A strong base model does not need
to derive: it recognises the setup and restates the method it already knows. The
two look similar in a similarity metric and are opposite in kind, so the signal
we want is not "did it get the method right" -- it will -- but WHERE the method's
modern name shows up.

A scientist deriving PCA in 1901 has no name for it until the end, if ever. The
hand-written corpus behaves that way: 49% of its 701 rungs never name the method
at all, and where a name appears the median position is 52.5% of the way in, with
only 11% naming it inside the first tenth.

So per trace we report: whether the rung's own method name appears, and at what
fraction of the text it first appears. Compare base against trained on the SAME
rows; the corpus's own distribution is the target.
"""
import argparse, glob, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import THINK


def method_pattern(slug, method):
    """Match the method's modern name, in slug or prose form."""
    pats = []
    name = re.split(r"[（(]", method or "")[0].strip()
    if len(name) >= 3:
        pats.append(re.escape(name))
    if slug and len(slug) >= 3:
        pats.append(re.escape(slug).replace(r"\_", "[ _-]?").replace(r"\-", "[ _-]?"))
    return re.compile("|".join(pats), re.I) if pats else None


def rung_index():
    """(task, n) -> (slug, method) from the trajectory metadata."""
    idx = {}
    for f in glob.glob("trajectories/*/meta.json"):
        task = f.split("/")[1]
        for s in (json.load(open(f)).get("steps") or []):
            idx[(task, s.get("n"))] = (s.get("slug"), s.get("method"))
    return idx


def scan(path, src, idx):
    """path: a distilled jsonl keyed by _id (source line index); src: the source corpus."""
    want = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            want[r["_id"]] = r
    # the source row carries no task/rung, so recover them from the system+question
    # by matching the trajectory whose reasoning file this row came from is not
    # available -- instead take the method name from the row's own answer heading.
    out = []
    for i, l in enumerate(open(src)):
        rid = f"{i:05d}"
        if rid not in want:
            continue
        for j, m in enumerate(want[rid]["conversations"]):
            if m["from"] != "gpt" or not m.get("loss"):
                continue
            mt = THINK.search(m["value"])
            if not mt:
                continue
            out.append((rid, j, mt.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distill", action="append", required=True,
                    help="repeatable: label=path")
    ap.add_argument("--src", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--names", default=None,
                    help="json: row-id -> [slug, method]; built by --emit-names if absent")
    a = ap.parse_args()

    names = json.load(open(a.names)) if a.names and os.path.exists(a.names) else {}
    print(f"{'arm':16s}{'n':>6}{'名字出现率':>12}{'首现位置中位':>14}{'前10%就点名':>13}")
    for spec in a.distill:
        lab, path = spec.split("=", 1)
        rows = scan(path, a.src, None)
        named, pos = 0, []
        for rid, j, think in rows:
            nm = names.get(rid)
            if not nm:
                continue
            pat = method_pattern(nm[0], nm[1])
            if not pat:
                continue
            m = pat.search(think)
            if m:
                named += 1
                pos.append(m.start() / max(1, len(think)))
        n = sum(1 for rid, _, _ in rows if names.get(rid))
        if not n:
            print(f"{lab:16s}{len(rows):6d}  (缺 --names 映射，无法判定)")
            continue
        early = sum(1 for p in pos if p < 0.1)
        print(f"{lab:16s}{n:6d}{named/n*100:11.1f}%{(st.median(pos) if pos else float('nan'))*100:13.1f}%"
              f"{(early/len(pos)*100 if pos else 0):12.1f}%")


if __name__ == "__main__":
    main()
