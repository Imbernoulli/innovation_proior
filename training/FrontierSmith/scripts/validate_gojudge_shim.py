#!/usr/bin/env python3
"""Fidelity validation for the userns-free judge backend (gojudge_shim.py).

Re-scores banked pre-outage FrontierCS generations (samples.jsonl records that
were scored by the REAL go-judge before 2026-07-28) through a judge stack now
running on GJ_BACKEND=shim, using the *exact same* scoring entrypoint the eval
driver uses (verl frontiercs.compute_score: same C++ extraction, same /submit
+ /result flow, same hardtest normalization). Per-sample old-vs-new diff is
the acceptance test for the shim.

Pass criteria (reported, not enforced): aggregate avg@5 delta within eval
noise (<1.0 FCS point) and per-sample large diffs (|d|>0.5) restricted to
timing-borderline cases.

Usage:
  FRONTIERCS_JUDGE_FAIL_SOFT=0 python scripts/validate_gojudge_shim.py \
      --samples <samples.jsonl> --judge-url http://127.0.0.1:PORT \
      --workers 12 --out validation.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "verl"))

os.environ.setdefault("FRONTIERCS_JUDGE_FAIL_SOFT", "0")

from verl.utils.reward_score import frontiercs  # noqa: E402


def load_jsonl(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        buf = ""
        for line in f:
            buf += line
            try:
                recs.append(json.loads(buf))
                buf = ""
            except json.JSONDecodeError:
                continue
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = [r for r in load_jsonl(args.samples)
            if r.get("data_source") == "frontiercs"]
    if args.limit:
        recs = recs[: args.limit]
    print(f"loaded {len(recs)} frontiercs samples from {args.samples}", flush=True)

    lock = threading.Lock()
    out_f = args.out.open("w", encoding="utf-8")
    done = [0]

    def score_one(idx_rec):
        idx, r = idx_rec
        old = (r.get("metrics") or {}).get("score")
        old_err = r.get("error")
        row = {
            "i": idx,
            "ground_truth": str(r.get("ground_truth")),
            "problem_idx": r.get("problem_idx"),
            "sample_idx": r.get("sample_idx"),
            "old": old,
            "old_error": bool(old_err),
        }
        try:
            new = frontiercs.compute_score(
                "frontiercs", r.get("text") or "", r.get("ground_truth"),
                judge_url=args.judge_url,
            )
            row["new"] = float(new)
        except Exception as e:  # JudgeInfraError etc.
            row["new"] = None
            row["new_infra_error"] = repr(e)[:300]
        with lock:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  {done[0]}/{len(recs)} scored", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [f.result() for f in as_completed(
            ex.submit(score_one, (i, r)) for i, r in enumerate(recs))]
    out_f.close()

    # ---- summary ----
    ok = [r for r in rows if r["new"] is not None and r["old"] is not None
          and not r["old_error"]]
    infra = [r for r in rows if r["new"] is None]
    oerr = [r for r in rows if r["old_error"]]
    print(f"\n=== SUMMARY ===\ncompared={len(ok)}  new_infra_err={len(infra)}  "
          f"old_had_error(excluded)={len(oerr)}")
    diffs = [(r["new"] - r["old"]) for r in ok]
    exact = sum(1 for d in diffs if abs(d) < 1e-6)
    close = sum(1 for d in diffs if abs(d) < 0.5)
    print(f"exact match: {exact}/{len(ok)} ({100 * exact / max(len(ok), 1):.1f}%)")
    print(f"|d| < 0.5  : {close}/{len(ok)} ({100 * close / max(len(ok), 1):.1f}%)")
    print(f"mean signed delta: {sum(diffs) / max(len(diffs), 1):+.4f}")
    print(f"old mean: {sum(r['old'] for r in ok) / max(len(ok), 1):.4f}   "
          f"new mean: {sum(r['new'] for r in ok) / max(len(ok), 1):.4f}")

    # avg@5 both ways (official: mean over problems of mean-of-samples)
    def avg_at_5(key):
        per_prob = collections.defaultdict(list)
        for r in ok:
            per_prob[r["ground_truth"]].append(r[key])
        return sum(sum(v) / len(v) for v in per_prob.values()) / max(len(per_prob), 1)

    print(f"avg@5 old: {avg_at_5('old'):.4f}   avg@5 new: {avg_at_5('new'):.4f}")

    big = sorted((r for r in ok if abs(r["new"] - r["old"]) >= 0.5),
                 key=lambda r: -abs(r["new"] - r["old"]))
    print(f"\nsamples with |d| >= 0.5: {len(big)}")
    for r in big[:30]:
        print(f"  gt={r['ground_truth']:>6} sample={r['sample_idx']} "
              f"old={r['old']:.3f} new={r['new']:.3f}")
    zero_to_pos = sum(1 for r in ok if r["old"] == 0 and r["new"] > 0.5)
    pos_to_zero = sum(1 for r in ok if r["old"] > 0.5 and r["new"] == 0)
    print(f"direction: 0->pos {zero_to_pos}, pos->0 {pos_to_zero}")


if __name__ == "__main__":
    main()
