"""review_weakness human-reference blind judge: per-arm table + item-level paired arm differences.
Every arm is judged against the SAME held-out human review on the SAME 60 items, so arm A - arm B
is a paired difference per item; bootstrap resamples items (10000, seed fixed)."""
import json, os, random, sys
from collections import defaultdict
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
def load(h, arm):
    p = f"{D}/cc_judge_rwhuman_{h}_{arm}__vs__human_rw_{h}/judgements.jsonl"
    if not os.path.exists(p): return None
    by = defaultdict(lambda: [0, 0]); rows = 0; unp = 0; agree = 0; ach = []; bch = []
    for line in open(p):
        r = json.loads(line); rows += 1
        if r["outcome"] == "a_wins": by[r["id"]][0] += 1; agree += 1
        elif r["outcome"] == "b_wins": by[r["id"]][1] += 1; agree += 1
        elif r["outcome"] == "unparsed": unp += 1
        ach.append(r["a_chars"]); bch.append(r["b_chars"])
    if rows == 0: return None
    items = {i: w / (w + l) for i, (w, l) in by.items() if w + l}
    return dict(rows=rows, items=items, unp=unp / rows, agree=agree / rows, ach=sum(ach) / len(ach), bch=sum(bch) / len(bch))
def boot(xs, f=lambda v: sum(v) / len(v)):
    n = len(xs); rng = random.Random(20260907)
    bs = sorted(f([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(10000))
    return f(xs), bs[250], bs[9750], sum(1 for b in bs if b > 0) / 10000
ARMS9 = ["base9b_v2c", "ft01mix_a10", "lo32nm_a10", "rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20"]
ARMS4 = ["base4b", "4b_lo32nm_a10", "rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20"]
res = {}
for h in ["h0", "h1"]:
    print(f"== {h}: model win vs held-out human review (human win = 1 - win)")
    for a in ARMS9 + ARMS4:
        s = load(h, a)
        if not s or s["rows"] < 240: print(f"  {a:28s} MISSING/partial"); continue
        res[(h, a)] = s
        xs = list(s["items"].values()); m, lo, hi, _ = boot(xs); p = boot([x - 0.5 for x in xs])[3]
        star = "*" if lo > 0.5 or hi < 0.5 else " "
        print(f"  {a:28s} rows={s['rows']} items={len(xs)} win={m*100:.1f}%{star} CI[{lo*100:.1f},{hi*100:.1f}] P(>50)={p:.3f}"
              f" agree={s['agree']*100:.1f}% unparsed={s['unp']*100:.1f}% chars={s['ach']:.0f}/{s['bch']:.0f}")
print()
print("== paired arm differences (A - B, item-level, both vs same human; * = 95% CI excludes 0)")
PAIRS = [("h0", "ft01mix_a10", "base9b_v2c"), ("h0", "lo32nm_a10", "base9b_v2c"),
         ("h0", "rlv5_ft01mix_a10_s20", "base9b_v2c"), ("h0", "rlv5_lo32nm_a10_s20", "base9b_v2c"),
         ("h0", "rlv5_ft01mix_a10_s20", "ft01mix_a10"), ("h0", "rlv5_lo32nm_a10_s20", "lo32nm_a10"),
         ("h0", "4b_lo32nm_a10", "base4b"), ("h0", "rlv5_4b_ft01mix_a10_s20", "base4b"), ("h0", "rlv5_4b_lo32nm_a10_s20", "base4b"),
         ("h0", "rlv5_4b_lo32nm_a10_s20", "4b_lo32nm_a10"), ("h0", "base9b_v2c", "base4b"),
         ("h1", "rlv5_ft01mix_a10_s20", "base9b_v2c"), ("h1", "rlv5_4b_lo32nm_a10_s20", "base4b"), ("h1", "base9b_v2c", "base4b")]
for h, a, b in PAIRS:
    if (h, a) not in res or (h, b) not in res: print(f"  {h} {a} - {b}: MISSING"); continue
    ia, ib = res[(h, a)]["items"], res[(h, b)]["items"]; common = sorted(set(ia) & set(ib))
    d = [ia[i] - ib[i] for i in common]
    m, lo, hi, p = boot(d)
    wins = sum(1 for x in d if x > 0); loss = sum(1 for x in d if x < 0)
    star = "*" if lo > 0 or hi < 0 else " "
    print(f"  {h} {a:26s} - {b:14s} n={len(d)} diff={m*100:+.1f}pp{star} CI[{lo*100:+.1f},{hi*100:+.1f}] P(>0)={p:.3f} W/L={wins}/{loss}")
print()
print("== h0 vs h1 stability (same arm, different held-out reviewer)")
for a in ["base9b_v2c", "rlv5_ft01mix_a10_s20", "base4b", "rlv5_4b_lo32nm_a10_s20"]:
    if ("h0", a) in res and ("h1", a) in res:
        x0 = res[("h0", a)]["items"]; x1 = res[("h1", a)]["items"]; c = sorted(set(x0) & set(x1))
        m0 = sum(x0[i] for i in c) / len(c); m1 = sum(x1[i] for i in c) / len(c)
        print(f"  {a:28s} h0={m0*100:.1f}% h1={m1*100:.1f}% (n={len(c)} common items)")
