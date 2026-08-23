#!/usr/bin/env python3
"""Produce the *_y26 variants of the RL training parquets: every prompt gains the
time-conditioning prefix "It is now year 2026. You are a good researcher."

Why (user directive, 2026-08-17): the SFT corpus conditions on time -- each
record's system prompt opens with the method's real historical year -- and
evaluation now conditions on the present (EVAL_RESEARCHER_YEAR=2026 plus the
full training template; MLS via MLSBENCH_SYS_PREFIX). RL was the one stage
whose rollouts carried NO such conditioning (verified: zero of the rlv12
rollout inputs contain "It is now year"), so the policy was optimised for 20
steps in a prompt regime that neither SFT nor eval uses. This closes that gap
for the next RL run.

Mechanics mirror the eval-side injection exactly:
  * first message is system  -> prefix + "\n\n" + existing content (MLS rows
    keep their tool definitions, the year line just opens the block)
  * no system message        -> insert one carrying just the prefix
Nothing else in the parquet is touched. Output is a NEW file next to the input
(suffix _y26), because running jobs snapshot their data path -- never edit a
parquet a queued run might read.

Usage:
  python3 scripts/add_year_prefix_to_rl_parquet.py data/multisource_rl/train.parquet
  # -> data/multisource_rl/train_y26.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "It is now year 2026. You are a good researcher."


def convert(path: Path) -> Path:
    df = pd.read_parquet(path)
    if "prompt" not in df.columns:
        raise SystemExit(f"{path}: no 'prompt' column")

    n_sys, n_ins = 0, 0
    new_prompts = []
    for p in df["prompt"]:
        msgs = [dict(m) for m in list(p)]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = PREFIX + "\n\n" + str(msgs[0].get("content", ""))
            n_sys += 1
        else:
            msgs = [{"role": "system", "content": PREFIX}, *msgs]
            n_ins += 1
        new_prompts.append(np.array(msgs, dtype=object))
    out_df = df.copy()
    out_df["prompt"] = new_prompts

    out = path.with_name(path.stem + "_y26" + path.suffix)
    out_df.to_parquet(out, index=False)

    # Verify by reading back: every row must start with the prefix.
    chk = pd.read_parquet(out)
    bad = 0
    for p in chk["prompt"]:
        first = dict(list(p)[0])
        if first.get("role") != "system" or not str(first.get("content", "")).startswith(PREFIX):
            bad += 1
    if bad:
        out.unlink()
        raise SystemExit(f"{path}: verification failed on {bad} rows; output removed")
    print(f"  {path.name}: {len(df)} rows -> {out.name}  (prefixed existing system: {n_sys}, inserted: {n_ins})  VERIFIED")
    return out


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]] or [
        Path("data/multisource_rl/train.parquet"),
        Path("data/multisource_rl/train_synth32.parquet"),
    ]
    for p in paths:
        if not p.exists():
            print(f"  skip {p}: missing")
            continue
        convert(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
