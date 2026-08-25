#!/usr/bin/env python3
"""Drop FrontierCS prompts whose judge never returns, from the RL train parquet.

Problem 153 is the only one that has ever timed out: across ms_qwen35_4b_grpo_
v6fresh (1x), v7house (5x) and base_4g (2x) every single
"FrontierCS judge infra error ... timed out after 2400.0s" names problem 153 and
nothing else. Each occurrence burns the full FRONTIERCS_JUDGE_MAX_WAIT (2400s =
40min) and then FAIL_SOFTs the sample to 0.0 -- the identical outcome it would
have had immediately, so the wait buys nothing and can sit on the critical path
of a step whose generation phase is only ~20min.

It is 1 of 188 FrontierCS rows (1 of 1488 total, 0.07%), so removing it does not
meaningfully change the training distribution.

  usage: drop_pathological_fcs_rows.py [--ids 153,...] [--in P] [--out P]

Both RL arms must train on the SAME parquet for base-vs-soup to be comparable;
switch them together or not at all.
"""
import argparse, pyarrow.parquet as pq, pyarrow as pa

ap = argparse.ArgumentParser()
ap.add_argument("--ids", default="153")
ap.add_argument("--in", dest="src", default="data/multisource_rl/train_synth_fcs.parquet")
ap.add_argument("--out", dest="dst", default="data/multisource_rl/train_synth_fcs_no153.parquet")
a = ap.parse_args()

drop = {s.strip() for s in a.ids.split(",") if s.strip()}
df = pq.read_table(a.src).to_pandas()
before = len(df)


def is_dropped(row):
    if "frontiercs" not in str(row["data_source"]).lower():
        return False
    gt = (row["reward_model"] or {}).get("ground_truth")
    return str(gt) in drop


mask = df.apply(is_dropped, axis=1)
kept = df[~mask]
pq.write_table(pa.Table.from_pandas(kept, preserve_index=False), a.dst)
print(f"{a.dst}: {before} -> {len(kept)} rows (dropped {int(mask.sum())}: ids {sorted(drop)})")
