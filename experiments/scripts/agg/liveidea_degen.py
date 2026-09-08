"""Objective degeneration counts on liveidea_gen (the free-form idea task).

The case-study agents reported by eye that RL-stage arms truncate and loop; this
measures it. Three independent signals, none of them a judgement call:
  unparsed      -- the client could not find the four required fields
  hit the cap   -- finish_reason == "length": the draw never terminated on its own
  looped        -- some 200-char window occurs 3+ times in the answer body
"""
import json, os, collections

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
ARMS = ["base9b_v2c", "ft01mix_a10", "ft03nm_a20", "lo32nm_a10",
        "rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_ft03nm_a20_s20", "rlv5_lo32nm_a10_s20"]
W = 200


def looped(t):
    seen = collections.Counter()
    for i in range(0, max(len(t) - W, 0), 20):
        seen[t[i:i + W]] += 1
        if seen[t[i:i + W]] >= 3:
            return True
    return False


print("| 模型 | 抽样数 | 未解析出四段式 | 打满 token 上限 | 出现整段复读 | 三者任一 |")
print("|---|---|---|---|---|---|")
for a in ARMS:
    f = f"{D}/outputs/cc_gen_{a}/samples.jsonl"
    n = up = cap = lp = bad = 0
    for l in open(f):
        r = json.loads(l)
        if r["task"] != "liveidea_gen":
            continue
        n += 1
        u = bool(r.get("unparsed")); c = r.get("finish_reason") == "length"; g = looped(r.get("text") or "")
        up += u; cap += c; lp += g; bad += (u or c or g)
    print(f"| `{a}` | {n} | {up} ({100*up/n:.1f}%) | {cap} ({100*cap/n:.1f}%) | "
          f"{lp} ({100*lp/n:.1f}%) | **{bad} ({100*bad/n:.1f}%)** |")
