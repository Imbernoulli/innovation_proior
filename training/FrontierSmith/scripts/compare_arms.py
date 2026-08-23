#!/usr/bin/env python3
"""Compare RL arms against their base arm across benchmarks, honestly.

Prints, per benchmark:
  * the headline metrics the field reports (avg@k, best@k, pass@1, pass@k, worst@k)
  * a PAIRED comparison against base over the shared problem set
  * the two data-quality numbers that decide whether the comparison is even valid

Why the data-quality numbers are not optional here. An errored sample (judge or
evaluator infra failure) is written with score 0 and an `error` field. In the
official metric that sample counts as 0 inside a fixed-size window -- no
denominator change. BUT a problem is dropped entirely when its variant 0 errored
(`if 0 not in variant_scores: continue`), which DOES change the problem set. If
arms drop different problems, their means are computed over different problems
and are not directly comparable. So we report:
    err            - errored samples per arm
    dropped_probs  - problems lost because sample_idx 0 errored
    shared_probs   - problems present in BOTH arms (the paired test uses only these)

Statistics: a t-test on these heavy-tailed per-problem scores has poor power
(FrontierCS per-sample sd is ~13 against a mean of ~7), so a sign test and
Wilcoxon are reported alongside. All three are shown because disagreement between
them is itself informative.

Usage:
  python3 scripts/compare_arms.py --prefix rlv12 --suffix _s20 \
      --arms base loraIM soupNEW10 soupWD03_20
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

OUT = Path("outputs")
TRACKS = {
    "FCS": ("thinking_32k_both_vllm", "frontiercs"),
    "ALE": ("thinking_32k_both_vllm", "alebench"),
    "Research": ("research_thinking_32k_vllm", "frontiercs_research"),
}


def load(tag: str, suffix: str, source: str):
    """-> (per_problem_scores, n_err, dropped_problems)"""
    d = OUT / f"cc_eval_{tag}_{suffix}"
    recs: dict[tuple, dict] = {}
    for shard in ("shard_0", "shard_1"):
        p = d / shard / "samples.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a job killed mid-write leaves one torn line
                if r.get("data_source") != source:
                    continue
                recs[(r["ground_truth"], int(r["sample_idx"]))] = r

    per = defaultdict(dict)
    n_err = 0
    for (gt, si), r in recs.items():
        if r.get("error"):
            n_err += 1
            continue
        v = (r.get("metrics") or {}).get("score")
        if v is not None:
            per[gt][si] = float(v)

    # Mirror the official rule: a problem without variant 0 is dropped.
    dropped = [gt for gt, sv in per.items() if 0 not in sv]
    kept = {gt: sv for gt, sv in per.items() if 0 in sv}
    return kept, n_err, dropped


def panel(per: dict[str, dict[int, float]], k: int = 5) -> dict:
    if not per:
        return {}
    allv = [v for sv in per.values() for v in sv.values()]
    best = [max(sv.values()) for sv in per.values()]
    worst = [min(sv.values()) for sv in per.values()]
    return {
        "avg": sum(allv) / len(allv),
        "best@k": sum(best) / len(best),
        "worst@k": sum(worst) / len(worst),
        "pass@1": sum(1 for v in allv if v > 0) / len(allv),
        "pass@k": sum(1 for b in best if b > 0) / len(best),
        "n_prob": len(per),
    }


def _var(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def paired(a: dict, b: dict) -> dict:
    """arm a vs base b over shared problems, using per-problem means."""
    keys = sorted(set(a) & set(b))
    d = [sum(a[k].values()) / len(a[k]) - sum(b[k].values()) / len(b[k]) for k in keys]
    n = len(d)
    if n < 2:
        return {"n": n}
    mu = sum(d) / n
    sd = math.sqrt(_var(d))
    t = mu / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    p_t = math.erfc(abs(t) / math.sqrt(2))
    w = sum(1 for x in d if x > 0)
    l = sum(1 for x in d if x < 0)
    # exact two-sided sign test
    m = w + l
    p_s = (
        min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(w, l) + 1)) / 2**m)
        if m
        else float("nan")
    )
    return {"n": n, "mean_diff": mu, "t": t, "p_t": p_t, "win": w, "loss": l, "p_sign": p_s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="e.g. rlv12")
    ap.add_argument("--suffix", default="_s20")
    ap.add_argument("--arms", nargs="+", required=True, help="first is treated as base")
    args = ap.parse_args()

    base_arm = args.arms[0]
    for track, (suf, source) in TRACKS.items():
        print(f"\n{'='*84}\n{track}\n{'='*84}")
        data = {}
        for arm in args.arms:
            tag = f"{args.prefix}_{arm}{args.suffix}"
            per, err, dropped = load(tag, suf, source)
            data[arm] = (per, err, dropped)
        if not data[base_arm][0]:
            print("  (no base data yet)")
            continue

        print(f"  {'arm':13s} {'avg':>8s} {'best@5':>8s} {'worst@5':>8s} "
              f"{'pass@1':>7s} {'pass@5':>7s} {'#prob':>6s} {'err':>5s} {'drop':>5s}")
        for arm in args.arms:
            per, err, dropped = data[arm]
            m = panel(per)
            if not m:
                print(f"  {arm:13s}  (no data)")
                continue
            print(f"  {arm:13s} {m['avg']:8.3f} {m['best@k']:8.3f} {m['worst@k']:8.3f} "
                  f"{m['pass@1']:6.1%} {m['pass@k']:6.1%} {m['n_prob']:6d} {err:5d} {len(dropped):5d}")

        print(f"\n  paired vs {base_arm} (per-problem means, shared problems only):")
        for arm in args.arms[1:]:
            if not data[arm][0]:
                continue
            r = paired(data[arm][0], data[base_arm][0])
            if r.get("n", 0) < 2:
                print(f"    {arm:13s} too few shared problems")
                continue
            star = "*" if min(r["p_t"], r["p_sign"]) < 0.05 else " "
            print(f"    {arm:13s} n={r['n']:3d} diff={r['mean_diff']:+7.3f} "
                  f"t-p={r['p_t']:.3f} sign {r['win']}W/{r['loss']}L p={r['p_sign']:.3f} {star}")

        drops = {a: set(data[a][2]) for a in args.arms if data[a][0]}
        if any(drops.values()):
            uneq = set().union(*drops.values()) - set.intersection(*drops.values()) if len(drops) > 1 else set()
            if uneq:
                print(f"  ⚠ arms dropped DIFFERENT problems ({len(uneq)} not common); "
                      f"means are over different problem sets -- the paired test above "
                      f"restricts to shared problems, the panel above does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
