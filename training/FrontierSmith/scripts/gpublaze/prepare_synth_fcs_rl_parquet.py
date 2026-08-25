#!/usr/bin/env python3
"""Build the gpublaze RL mix: frontiersmith_synth + FCS algorithmic, y26-conditioned.

Composition per coordinator ruling 2026-08-23:
  * frontiersmith_synth (1300)  -- reward double-direction verified on this box
  * frontiercs algorithmic      -- official problems, local go-judge on :8082
  * NO mlsbench_rl rows (the _tv train-task variants are not on this machine)
  * NO research rows (2026-08-04 ruling: research is EVAL-ONLY)

System prefix: the PURE time sentence "It is now year 2026." on every row
(2026-08-18 ruling dropped the persona/delivery clauses; the historical
scripts/add_year_prefix_to_rl_parquet.py predates that ruling and still adds
"You are a good researcher" -- deliberately NOT reused).

Schema mirrors data/multisource_rl/train.parquet:
  prompt / data_source / agent_name / reward_model / extra_info
Both sources route as single_turn_agent; rewards dispatch on data_source.

Outputs (under data/multisource_rl/):
  train_synth_fcs.parquet          full mix
  train_synth_fcs_smoke.parquet    8 rows (4 synth + 4 fcs), for 1-step smokes
  train.parquet                    copy of the full mix (the launcher default and
                                   check_multisource_reward_routing.py both read
                                   this fixed path)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FS = Path(__file__).resolve().parent.parent.parent
PREFIX = "It is now year 2026."


def add_y26(prompt) -> np.ndarray:
    msgs = [dict(m) for m in list(prompt)]
    if msgs and msgs[0].get("role") == "system":
        if not str(msgs[0].get("content", "")).startswith(PREFIX):
            msgs[0]["content"] = PREFIX + "\n\n" + str(msgs[0].get("content", ""))
    else:
        msgs = [{"role": "system", "content": PREFIX}, *msgs]
    return np.array(msgs, dtype=object)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=str(FS / "data/frontiersmith_synth/train.parquet"))
    ap.add_argument("--fcs", default=str(FS / "data/frontiercs/train.parquet"))
    ap.add_argument("--out-dir", default=str(FS / "data/multisource_rl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    synth = pd.read_parquet(args.synth)
    synth["agent_name"] = "single_turn_agent"

    fcs = pd.read_parquet(args.fcs)
    fcs["agent_name"] = "single_turn_agent"
    fcs["extra_info"] = [None] * len(fcs)

    cols = ["prompt", "data_source", "agent_name", "reward_model", "extra_info"]
    mix = pd.concat([synth[cols], fcs[cols]], ignore_index=True)
    mix = mix.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    mix["prompt"] = mix["prompt"].map(add_y26)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = out_dir / "train_synth_fcs.parquet"
    mix.to_parquet(full, index=False)

    smoke = pd.concat(
        [
            mix[mix.data_source == "frontiersmith_synth"].head(4),
            mix[mix.data_source == "frontiercs"].head(4),
        ],
        ignore_index=True,
    )
    smoke.to_parquet(out_dir / "train_synth_fcs_smoke.parquet", index=False)
    mix.to_parquet(out_dir / "train.parquet", index=False)

    # read-back verification
    chk = pd.read_parquet(full)
    assert all(dict(p[0])["role"] == "system" and dict(p[0])["content"].startswith(PREFIX) for p in chk["prompt"])
    assert set(chk["agent_name"]) == {"single_turn_agent"}
    print(f"wrote {len(mix)} rows -> {full}")
    print("  by source:", mix["data_source"].value_counts().to_dict())
    print(f"  smoke: 8 rows -> {out_dir/'train_synth_fcs_smoke.parquet'}")
    print(f"  copied to {out_dir/'train.parquet'} (launcher/checker default path)")


if __name__ == "__main__":
    main()
