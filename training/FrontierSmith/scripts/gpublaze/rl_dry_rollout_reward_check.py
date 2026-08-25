#!/usr/bin/env python3
"""Single-GPU DRY validation of the RL rollout+reward loop (no trainer).

Why: on 1 GPU verl's FSDP degrades to NO_SHARD and every offload knob is
inert, so the full colocated trainer cannot fit 80G at the 43k context -- the
coordinator's fallback is exactly this: validate ROLLOUT (protocol sampling
against the RL venv's patched vLLM) + REWARD (default_compute_score dispatch)
end to end, and leave the true 1-step train smoke to 2 GPUs.

Reads the 8-row smoke parquet (y26 prompts), samples n=2 per prompt with the
RL protocol (temp 1.0 / top_p 0.95 / top_k 20 / presence 1.5 / 32k budget),
scores every rollout via verl's default_compute_score (synth -> bwrap harness,
frontiercs -> local judge), applies FS_PERTASK_REWARD_NORM, and asserts the
batch reward vector is neither all-zero nor all-identical.

  FRONTIERSMITH_SYNTH_ROOT=... FRONTIERCS_JUDGE_URL=http://127.0.0.1:8082 \
  .venv-gpublaze/bin/python scripts/gpublaze/rl_dry_rollout_reward_check.py \
      --base-url http://127.0.0.1:PORT/v1 --model TAG
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

FS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(FS / "verl"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(FS / "data/multisource_rl/train_synth_fcs_smoke.parquet"))
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--out", default=str(FS / "outputs/rl_dry_rollout_reward/dry_samples.jsonl"))
    args = ap.parse_args()

    from openai import OpenAI

    from verl.utils.reward_score import default_compute_score, pertask_norm

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    df = pd.read_parquet(args.data)
    print(f"[dry] {len(df)} prompts x n={args.n}, protocol: temp1.0 top_p0.95 top_k20 presence1.5 max_tokens={args.max_tokens}")

    def gen(row):
        msgs = [dict(m) for m in row["prompt"]]
        r = client.chat.completions.create(
            model=args.model, messages=msgs, n=args.n,
            temperature=1.0, top_p=0.95, presence_penalty=1.5,
            max_tokens=args.max_tokens,
            extra_body={"top_k": 20, "min_p": 0.0},
        )
        return [c.message.content or "" for c in r.choices]

    rows = list(df.iloc[i] for i in range(len(df)))
    with ThreadPoolExecutor(max_workers=8) as ex:
        gens = list(ex.map(gen, rows))

    results = []
    for row, outs in zip(rows, gens):
        gt = row["reward_model"]["ground_truth"]
        ei = row["extra_info"]
        ei = dict(ei) if ei is not None and not isinstance(ei, float) else None
        for j, text in enumerate(outs):
            raw = default_compute_score(
                data_source=row["data_source"], solution_str=text,
                ground_truth=gt, extra_info=ei)
            rawf = float(raw if not isinstance(raw, dict) else raw.get("score", 0.0))
            norm = pertask_norm.maybe_normalize(row["data_source"], rawf, {})["reward"]
            results.append({"data_source": row["data_source"], "gt": str(gt), "sample": j,
                            "gen_tokens": len(text.split()), "raw": rawf, "reward": float(norm)})
            print(f"  {row['data_source']:22s} {gt!s:12s} #{j} raw={rawf:8.3f} norm={norm:.4f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    rewards = [r["reward"] for r in results]
    nz = sum(1 for r in rewards if abs(r) > 1e-9)
    spread = max(rewards) - min(rewards)
    by_gt = {}
    for r in results:
        by_gt.setdefault(r["gt"], []).append(r["reward"])
    groups_with_spread = sum(1 for v in by_gt.values() if max(v) - min(v) > 1e-9)
    print(f"[dry] batch={len(rewards)} nonzero={nz} spread={spread:.4f} "
          f"groups_with_intragroup_spread={groups_with_spread}/{len(by_gt)}  -> {out}")
    if nz == 0:
        print("FAIL: all-zero rewards"); return 1
    if spread < 1e-9:
        print("FAIL: all rewards identical"); return 1
    print("PASS: rollout+reward non-degenerate")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("FS_PERTASK_REWARD_NORM", "1")
    raise SystemExit(main())
