#!/usr/bin/env python3
"""Hidden naive-baseline anchor for the GiantsBench judge (Lit2Test defense).

Takes a real generation file and emits a control file in the same shape whose
"prediction" for item i is the GOLD INSIGHT OF A DIFFERENT ITEM (deterministic
derangement over sorted ids).  Every control item is a fluent, well-written,
on-distribution research insight that is simply about the wrong papers -- so a
judge that is reading substance must score it near the floor.  If it does not,
the whole GiantsBench column is uninterpretable and gets thrown away.

    python make_control.py --gen giants_taste_base.jsonl --out giants_control.jsonl
"""
from __future__ import annotations

import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = {}
for line in open(a.gen):
    if line.strip():
        r = json.loads(line)
        rows[r["id"]] = r
ids = sorted(rows)
n = len(ids)
with open(a.out, "w") as f:
    for k, i in enumerate(ids):
        donor = rows[ids[(k + 1) % n]]["meta"]["gold"]
        r = dict(rows[i])
        r["output"] = f"<think>\n(control)\n</think>\n\n<insight>\n{donor}\n</insight>"
        r["finish_reason"] = "control"
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[control] {n} items written to {a.out}")
