#!/usr/bin/env python3
"""Produce the TRAINING-FINAL innovation SFT file from the freshly rebuilt
innovation_prior corpus, applying two fixes that belong to the training side:

1. User ruling (2026-08-18): the system prompt convention is TIME ONLY.
   "It is now year YYYY." stays; the persona sentence ("You are a good
   researcher.") and the delivery clause are r1-era legacy and are stripped.
   Task-specific instructions that follow (agentic tool workflow, v4 C++
   contract) are kept -- they are task setup, not persona.
2. Corpus bug: 6 trajectory dirs added by the recent traj-build commits
   (convnext/deit/lion/preact/roberta/wide-resnet) were never registered in
   trajectories.json, so build_sft.py rendered "It is now year None." into 32
   rows. Years are assigned here by matching each row's initial context to the
   trajectory dir, using the corpus convention (arXiv year; cross-checked:
   methods.json already has wide-resnet=2016).

Input is NEVER modified. Output: LF-innov/data/innovation_final_timeonly.jsonl
"""
import json, os, re, sys

SRC = "/scratch/gpfs/CHIJ/bohan/fs/innovation_prior"
OUT = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/LF-innov/data/innovation_final_timeonly.jsonl"

PERSONA = "You are a good researcher."
DELIVERY = ("When you write code, deliver a single, self-contained, runnable implementation that "
            "respects any stated input/output contract; if an idea is not converging within the "
            "budget, fall back to the simplest correct approach and ship that.")
# arXiv years, matching methods.json convention (resnet=2015, adam=2014, wide-resnet=2016)
UNREGISTERED_YEARS = {
    "convnext-modernization": 2022,
    "deit-data-efficient-vit": 2020,
    "lion-program-search": 2023,
    "preact-identity-mappings": 2016,
    "roberta-pretraining-recipe": 2019,
    "wide-resnet-widening": 2016,
}

def init_prefix(task):
    d = os.path.join(SRC, "trajectories", task)
    meta = json.load(open(os.path.join(d, "meta.json")))
    p = os.path.join(d, meta.get("initial_context_file", "00-initial-context.md"))
    return open(p).read()[:300]

def main():
    prefixes = {task: init_prefix(task) for task in UNREGISTERED_YEARS}
    rows, fixed_none, stripped = [], 0, 0
    for line in open(os.path.join(SRC, "sft", "innovation_sft.jsonl")):
        r = json.loads(line)
        s = r.get("system") or ""
        if "It is now year None." in s:
            first = next(t["value"] for t in r["conversations"] if t["from"] == "human")
            task = [t for t, pre in prefixes.items() if first.startswith(pre)]
            assert len(task) == 1, f"ambiguous/unmatched None-year row: {first[:80]!r} -> {task}"
            s = s.replace("It is now year None.", f"It is now year {UNREGISTERED_YEARS[task[0]]}.")
            fixed_none += 1
        before = s
        s = s.replace(PERSONA, "").replace(DELIVERY, "")
        s = re.sub(r"[ \t]+", " ", s).replace(" .", ".").strip()
        if s != before.strip():
            stripped += 1
        r["system"] = s
        rows.append(r)

    bad = [r["system"][:80] for r in rows
           if not re.match(r"^It is now year \d{4}\.", r["system"])
           or "good researcher" in r["system"] or "ship that" in r["system"]]
    assert not bad, f"{len(bad)} rows failed the time-only invariant: {bad[:3]}"
    assert fixed_none == 32, f"expected 32 None-year fixes, got {fixed_none}"

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    yrs = Counter(int(re.match(r"^It is now year (\d{4})\.", r["system"]).group(1)) for r in rows)
    print(f"rows={len(rows)} fixed_none={fixed_none} stripped={stripped}")
    print(f"year span {min(yrs)}-{max(yrs)}, unique {len(yrs)}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
