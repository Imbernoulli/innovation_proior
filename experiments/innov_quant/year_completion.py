#!/usr/bin/env python3
"""The year question, asked again with the highest-power metric we have.

Score is a noisy measure on FrontierCS: heavy-tailed, and a same-year re-run already
moves an arm mean by about a point. Completion rate is much better behaved -- 860 binary
draws per cell, and section 20.1 got Stouffer Z=+18.19 out of it on the RL contrast. So
if the test-time year prompt changes the model's behaviour at all, completion rate is
where it should be visible.

A draw completes when its text contains </think> or it stopped short of the 32768 cap;
otherwise it ran out of budget mid-thought and never produced an answer.

T1 is the pre-registered contrast: NEAR = {2025, 2026} vs FAR = {<=2010} U {>=2050},
paired per problem, aggregated over (arm x bench) cells by sign test + Stouffer.

The noise floor runs the SAME machinery over same-protocol re-runs -- y1950 vs y1950r2,
y2025 vs y2025r2, and each main run against its _y2026 replicate. Nothing differs between
those pairs except that they were run twice, so whatever they show is the floor any year
effect has to clear.
"""
import json, glob, os, collections, math
import numpy as np
from scipy.stats import wilcoxon, binomtest, norm

OUT = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
YEARS = [2000, 2026, 2050, 2075]      # 2025 still running at time of writing
NEAR = {2025, 2026}
ARMS = ["ft01mix_a10", "rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "4b_ft01mix_a10", "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
REPL = [("rlv5_lo32nm_a10_s20_y1950", "rlv5_lo32nm_a10_s20_y1950r2", "y1950 复跑"),
        ("rlv5_lo32nm_a10_s20_y2025", "rlv5_lo32nm_a10_s20_y2025r2", "y2025 复跑"),
        ("base9b_v2c", "base9b_v2c_y2026", "9B base 复跑"),
        ("lo32nm_a10", "lo32nm_a10_y2026", "9B lo32nm 复跑"),
        ("rlv5_lo32nm_a10_s20", "rlv5_lo32nm_a10_s20_y2026", "9B RL(lo32nm) 复跑"),
        ("base4b", "base4b_y2026", "4B base 复跑"),
        ("4b_lo32nm_a10", "4b_lo32nm_a10_y2026", "4B lo32nm 复跑"),
        ("rlv5_4b_lo32nm_a10_s20", "rlv5_4b_lo32nm_a10_s20_y2026", "4B RL(lo32nm) 复跑")]
BEN = ["frontiercs", "alebench"]


def comp(tag):
    """-> {bench: {problem: completion rate over its draws}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(os.path.join(OUT, f"cc_eval_{tag}_thinking_32k_both_vllm",
                                    "shard_*", "samples.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("error") is not None:
                continue
            gt = r.get("ground_truth")
            gt = gt if isinstance(gt, str) else json.dumps(gt, sort_keys=True)
            t, ct = r.get("text") or "", r.get("completion_tokens") or 0
            out[r.get("data_source")][gt].append(1.0 if ("</think>" in t or ct < 32700) else 0.0)
    return {b: {k: float(np.mean(v)) for k, v in d.items()} for b, d in out.items()}


def cell(d):
    nz = d[d != 0]
    if len(nz) < 5:
        return None
    _, p = wilcoxon(nz)
    if not np.isfinite(p):
        return None
    z = (1.0 if np.median(nz) > 0 else -1.0) * abs(norm.ppf(max(p, 1e-12) / 2))
    return dict(n=len(d), mean=float(d.mean()), pos=int((d > 0).sum()),
                neg=int((d < 0).sum()), p=float(p), z=float(z))


def agg(cells, label):
    cells = [c for c in cells if c]
    if not cells:
        return
    k, n = sum(1 for c in cells if c["mean"] > 0), len(cells)
    Z = sum(c["z"] for c in cells) / math.sqrt(n)
    print(f"| **{label}** | **{n}** | **{k}/{n} 正** | **{binomtest(k, n, 0.5).pvalue:.4f}** | **{Z:+.2f}** |")


def main():
    print("## T1 完成率:NEAR − FAR\n")
    print("| arm | bench | n题 | Δ完成率 | +/− | p | Z |")
    print("|---|---|---|---|---|---|---|")
    cells, curves = collections.defaultdict(list), {}
    for a in ARMS:
        D = {y: comp(a if y == 2026 else f"{a}_y{y}") for y in YEARS}
        for b in BEN:
            pp = {y: D[y].get(b, {}) for y in YEARS}
            if not all(pp[y] for y in YEARS):
                continue
            common = sorted(set.intersection(*[set(pp[y]) for y in YEARS]))
            if len(common) < 8:
                continue
            curves[(a, b)] = {y: float(np.mean([pp[y][k] for k in common])) for y in YEARS}
            A = np.array([np.mean([pp[y][k] for y in YEARS if y in NEAR]) for k in common])
            B = np.array([np.mean([pp[y][k] for y in YEARS if y <= 2010 or y >= 2050]) for k in common])
            c = cell(A - B)
            if not c:
                continue                      # an arm pinned at 1.000 everywhere has no variance
            cells["ALL"].append(c)
            cells[b].append(c)
            cells["4B" if a.startswith(("4b_", "rlv5_4b")) else "9B"].append(c)
            print(f"| {a} | {b} | {c['n']} | {c['mean']:+.4f} | {c['pos']}/{c['neg']} | {c['p']:.4f} | {c['z']:+.2f} |")
    print("\n| 聚合 | 格子 | 方向 | 符号 p | Stouffer Z |")
    print("|---|---|---|---|---|")
    for k in ("ALL", "9B", "4B", "frontiercs", "alebench"):
        agg(cells[k], f"T1 {k}")

    print("\n## 逐年完成率(每条臂取四年公共题)\n")
    print("| arm | bench | " + " | ".join(str(y) for y in YEARS) + " | 峰值 |")
    print("|---|---|" + "---|" * (len(YEARS) + 1))
    for (a, b), m in curves.items():
        print(f"| {a} | {b} | " + " | ".join(f"{m[y]:.3f}" for y in YEARS) +
              f" | {max(m, key=m.get)} |")

    print("\n## 噪声底:同协议复跑\n")
    print("| 对比 | bench | n题 | Δ完成率 | +/− | p | Z |")
    print("|---|---|---|---|---|---|---|")
    nf = []
    for A, B, lab in REPL:
        ca, cb = comp(A), comp(B)
        for b in BEN:
            if b not in ca or b not in cb:
                continue
            ks = sorted(set(ca[b]) & set(cb[b]))
            if len(ks) < 8:
                continue
            c = cell(np.array([ca[b][k] - cb[b][k] for k in ks]))
            if not c:
                continue
            nf.append(c)
            print(f"| {lab} | {b} | {c['n']} | {c['mean']:+.4f} | {c['pos']}/{c['neg']} | {c['p']:.4f} | {c['z']:+.2f} |")
    print("\n| 聚合 | 格子 | 方向 | 符号 p | Stouffer Z |")
    print("|---|---|---|---|---|")
    agg(nf, "噪声底")
    a = [abs(c["mean"]) for c in nf]
    print(f"\n逐格 |Δ完成率| 中位数 **{np.median(a):.4f}**,最大 **{max(a):.4f}**。")


if __name__ == "__main__":
    main()
