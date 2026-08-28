#!/usr/bin/env python3
"""Build the case files the method-consistency judges read.

Splits into `--chunks` files so several judges can run in parallel on the same
rubric. Two modes decide which rows are eligible:

  --holdout-ids FILE   only rows the teacher never trained on (the memorisation
                       control; the sampler walks the whole source by index, so
                       held-out rows get distilled too and this is free)
  default              any row that was actually regenerated

Rows where the regeneration fell back to the hand-written think are skipped —
judging those would compare the original against itself and inflate agreement.
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import THINK

FENCE = re.compile(r"```.*?```", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--distill", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="mc")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--plain-only", action="store_true", default=True)
    ap.add_argument("--holdout-ids", default=None)
    ap.add_argument("--seed", type=int, default=20260828)
    a = ap.parse_args()

    keep = set(json.load(open(a.holdout_ids))) if a.holdout_ids else None
    D = {}
    for l in open(a.distill):
        if l.strip():
            r = json.loads(l); D[r["_id"]] = r

    cases = []
    for i, l in enumerate(open(a.src)):
        rid = f"{i:05d}"
        if rid not in D or (keep is not None and rid not in keep):
            continue
        o = json.loads(l)
        if a.plain_only and any(m["from"] in ("observation", "function_call")
                                for m in o["conversations"]):
            continue
        for j, m in enumerate(D[rid]["conversations"]):
            if m["from"] != "gpt" or not m.get("loss"):
                continue
            dm = THINK.search(m["value"]); om = THINK.search(o["conversations"][j]["value"])
            if not dm or not om:
                continue
            if dm.group(1).strip() == om.group(1).strip():
                continue                       # fell back; judging it proves nothing
            cases.append((rid, o.get("system", ""), o["conversations"][0]["value"],
                          FENCE.sub("\n[代码块略]\n",
                                    o["conversations"][j]["value"][om.end():]).strip(),
                          dm.group(1).strip()))

    rng = random.Random(a.seed); rng.shuffle(cases)
    sel = cases[: a.n]
    os.makedirs(a.outdir, exist_ok=True)
    print(f"eligible {len(cases)}, writing {len(sel)} into {a.chunks} chunks")
    for k in range(a.chunks):
        part = sel[k::a.chunks]
        if not part:
            continue
        p = os.path.join(a.outdir, f"{a.prefix}_{k}.md")
        with open(p, "w") as f:
            for rid, sp, q, gold, dist in part:
                f.write(f"\n\n{'='*100}\n# CASE {rid}\n{'='*100}\n")
                f.write(f"\n## SYSTEM\n{sp}\n\n## 题目（截前 2500）\n{q[:2500]}\n")
                f.write(f"\n## GOLD ANSWER（代码块已略，截前 3500）\n{gold[:3500]}\n")
                f.write(f"\n## DISTILLED REASONING（开头 2500 + 结尾 2500）\n{dist[:2500]}"
                        f"\n\n[……中间略……]\n\n{dist[-2500:] if len(dist)>5000 else ''}\n")
        print(f"  {p}: {len(part)} cases, {os.path.getsize(p)//1024} KB")


if __name__ == "__main__":
    main()
