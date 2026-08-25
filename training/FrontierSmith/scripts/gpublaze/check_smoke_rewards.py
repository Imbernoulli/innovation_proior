#!/usr/bin/env python3
"""Verify a 1-step RL smoke produced NON-DEGENERATE rewards.

Reads the trainer's rollout dump (<ROLLOUT_DIR>/<step>.jsonl, written by
ray_trainer._dump_generations: one row per sample with `score` = sum of
token_level_scores) and checks the batch is neither all-zero nor all-identical.

  .venv-gpublaze/bin/python scripts/gpublaze/check_smoke_rewards.py \
      outputs/rl_multisource_rollout/ms_smoke_gpublaze/1.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def main() -> int:
    path = sys.argv[1]
    scores, by_gt = [], defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            s = float(d["score"])
            scores.append(s)
            by_gt[str(d.get("gt"))].append(s)

    n = len(scores)
    nz = sum(1 for s in scores if abs(s) > 1e-9)
    all_same = max(scores) - min(scores) < 1e-9
    print(f"samples={n} nonzero={nz} min={min(scores):.4f} max={max(scores):.4f} mean={sum(scores)/n:.4f}")
    for gt, ss in sorted(by_gt.items()):
        spread = max(ss) - min(ss)
        print(f"  group {gt}: n={len(ss)} scores={[round(s,4) for s in ss]} spread={spread:.4f}")
    n_spread = sum(1 for ss in by_gt.values() if max(ss) - min(ss) > 1e-9)
    print(f"groups with intra-group spread (GRPO signal): {n_spread}/{len(by_gt)}")

    if nz == 0:
        print("FAIL: all-zero rewards (degenerate)"); return 1
    if all_same:
        print("FAIL: all rewards identical (degenerate)"); return 1
    print("PASS: rewards non-degenerate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
