#!/usr/bin/env python3
"""FrontierCS-research 的年份扫描 —— 此前每一张年份表都漏掉的第二块。

年份脚本(year_shape / year_completion / year_curves)把 tag 解析到
`cc_eval_<tag>_thinking_32k_both_vllm/`,而 `SOURCE=both` 只产 frontiercs + alebench。
research 走的是 `year_sweep_submit.sh` 第 62 行的独立作业,落在
`cc_eval_<tag>_research_thinking_32k_vllm/shard_{0,1}/samples.jsonl`。
141 个 research 作业早就 COMPLETED,数据一直在,只是没有任何读取器看过那个位置。
这和 §33 的 MLS 是同一类错:目录解析静默吞掉一整条 bench。

覆盖:ft01mix 线 6 臂 × 2000/2025/2050/2075,每格 64 题 × 5 抽样 = 320
(只缺 rlv5_4b_ft01mix_a10_s20 的 y2050,还在队列里)。

对照设计与 §33 对齐,好让三条 bench 可以并排读:
  NEAR = 2025,FAR = mean(2000, 2075),按题配对(题的单位是 ground_truth)。
  噪声底 = 同臂 2000 vs 2075 —— 两个都是"远年",倒 U 假设不预测它们有差。
去重按 (ground_truth, sample_idx) 后出现的覆盖先出现的:9B 的 research 分片有重复。
完成率定义与 dump2.py 一致:`text.rfind("</think>") >= 0`。
"""
import collections
import glob
import json
import os

import numpy as np

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
ARMS = [("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"),
        ("rlv5_ft01mix_a10_s20", "9B RL(先验)"),
        ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"),
        ("rlv5_4b_ft01mix_a10_s20", "4B RL(先验)")]
YEARS = [2000, 2025, 2050, 2075]
NEAR, FAR = 2025, (2000, 2075)
BOOT = 10000


def load(arm, year):
    """-> {problem: {"score":[...], "done":[...]}},按 (题, 抽样) 去重。"""
    d = os.path.join(D, f"cc_eval_{arm}_y{year}_research_thinking_32k_vllm")
    seen = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*", "samples.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            seen[(str(r.get("ground_truth")), r.get("sample_idx"))] = r
    out = collections.defaultdict(lambda: {"score": [], "done": []})
    for (gt, _), r in seen.items():
        out[gt]["score"].append(float((r.get("metrics") or {}).get("score") or 0.0))
        out[gt]["done"].append(1.0 if (r.get("text") or "").rfind("</think>") >= 0 else 0.0)
    return dict(out)


def boot(d, rng):
    d = np.asarray(d, float)
    if len(d) < 2:
        return (float("nan"),) * 4
    bs = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
    se = bs.std(ddof=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), \
        (d.mean() / se if se > 0 else float("nan"))


def paired(store, a_year, b_years, key):
    if a_year not in store or any(y not in store for y in b_years):
        return None
    common = set(store[a_year])
    for y in b_years:
        common &= set(store[y])
    common = sorted(common)
    if not common:
        return None
    d = [np.mean(store[a_year][t][key]) - np.mean([np.mean(store[y][t][key]) for y in b_years])
         for t in common]
    return common, d


def stouffer(zs):
    zs = [z for z in zs if np.isfinite(z)]
    return sum(zs) / np.sqrt(len(zs)) if zs else float("nan")


def signtest(ds):
    from scipy.stats import binomtest
    pos = sum(1 for x in ds if x > 0)
    n = sum(1 for x in ds if x != 0)
    return pos, len(ds), (binomtest(pos, n, 0.5).pvalue if n else float("nan"))


def main():
    rng = np.random.default_rng(0)
    store = {a: {y: s for y in YEARS if (s := load(a, y))} for a, _ in ARMS}
    out = ["# FrontierCS-research 的年份扫描(ft01mix 线)", ""]

    out += ["## 0. 每格进入分析的题数 / 抽样数(§27:先看分母)", "",
            "| 臂 | " + " | ".join(str(y) for y in YEARS) + " |", "|---|---|---|---|---|"]
    for a, lab in ARMS:
        row = []
        for y in YEARS:
            s = store[a].get(y)
            row.append("—" if not s else f"{len(s)}题/{sum(len(v['score']) for v in s.values())}抽样")
        out.append(f"| {lab} `{a}` | " + " | ".join(row) + " |")
    usable = [(a, lab) for a, lab in ARMS if NEAR in store[a] and all(y in store[a] for y in FAR)]
    out += ["", f"可用臂 **{len(usable)}/6**。", ""]

    for title, key in [("1. 分数 NEAR(2025) − FAR(2000/2075 均值)", "score"),
                       ("2. 完成率(`</think>` 闭合)NEAR − FAR", "done")]:
        out += [f"## {title}", "", "| 臂 | n题 | Δ | 95% CI | +/− | Z |", "|---|---|---|---|---|---|"]
        zs, ds = [], []
        for a, lab in usable:
            r = paired(store[a], NEAR, FAR, key)
            if not r:
                continue
            common, d = r
            m, lo, hi, z = boot(d, rng)
            zs.append(z)
            ds.append(m)
            out.append(f"| {lab} `{a}` | {len(common)} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                       f"{sum(1 for x in d if x > 0)}/{sum(1 for x in d if x < 0)} | {z:+.2f} |")
        p, n, sp = signtest(ds)
        out += ["", "| 聚合 | 格子 | 方向 | 符号 p | Stouffer Z |", "|---|---|---|---|---|",
                f"| **NEAR−FAR** | **{n}** | **{p}/{n} 正** | **{sp:.4f}** | **{stouffer(zs):+.2f}** |", ""]

    out += ["## 3. 噪声底:同臂 2000 vs 2075(两个都是远年,倒 U 假设不预测有差)", "",
            "| 臂 | 指标 | n题 | Δ | 95% CI | Z |", "|---|---|---|---|---|---|"]
    floors = collections.defaultdict(list)
    for a, lab in usable:
        for name, key in [("分数", "score"), ("完成率", "done")]:
            r = paired(store[a], FAR[0], (FAR[1],), key)
            if not r:
                continue
            common, d = r
            m, lo, hi, z = boot(d, rng)
            floors[name].append((m, z))
            out.append(f"| {lab} | {name} | {len(common)} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {z:+.2f} |")
    out += ["", "| 指标 | 格子 | 方向 | Stouffer Z | 逐格 \\|Δ\\| 中位 | 最大 |", "|---|---|---|---|---|---|"]
    for name, vals in floors.items():
        ms = [v[0] for v in vals]
        p, n, _ = signtest(ms)
        out.append(f"| **{name}** | **{n}** | **{p}/{n} 正** | **{stouffer([v[1] for v in vals]):+.2f}** | "
                   f"**{np.median([abs(m) for m in ms]):.4f}** | **{max(abs(m) for m in ms):.4f}** |")
    out.append("")

    out += ["## 4. 逐年原始值(每条臂限制到该臂所有年份点的公共题)", "",
            "| 臂 | 公共题 | 指标 | " + " | ".join(str(y) for y in YEARS) + " | 峰值 |",
            "|---|---|---|---|---|---|---|---|"]
    for a, lab in ARMS:
        ys = sorted(store[a])
        if not ys:
            continue
        common = set.intersection(*[set(store[a][y]) for y in ys])
        for name, key in [("分数", "score"), ("完成率", "done")]:
            cells = {y: float(np.mean([np.mean(store[a][y][t][key]) for t in common])) for y in ys}
            pk = max(cells, key=cells.get)
            row = " | ".join(f"{cells[y]:.3f}" if y in cells else "—" for y in YEARS)
            out.append(f"| {lab} `{a}` | {len(common)} | {name} | {row} | **{pk}** |")
    out.append("")

    # ---- 5. 把四个年份点当作四次准复跑,看「先验 vs base」稳不稳 ----
    # 年份只改系统提示里的一个数字,对「哪条臂更强」这个问题是无关扰动。
    # 所以同一对臂在 4 个年份点上的对比 = 4 次近似复跑,方向一致性本身就是证据。
    # 全部对比都在**同一批次内**(新四点批次),不跨 §25 的协议世代。
    PAIRS = [("rlv5_ft01mix_a10_s20", "rlv5_base_s20", "9B RL(先验) − RL(base)"),
             ("rlv5_4b_ft01mix_a10_s20", "rlv5_4b_base_s20", "4B RL(先验) − RL(base)"),
             ("rlv5_ft01mix_a10_s20", "ft01mix_a10", "9B RL(先验) − SFT"),
             ("rlv5_4b_ft01mix_a10_s20", "4b_ft01mix_a10", "4B RL(先验) − SFT")]
    out += ["## 5. 四个年份点当四次准复跑:先验 vs 对照,逐年份点", "",
            "年份只改系统提示里的一个数字,对「哪条臂更强」是无关扰动,所以同一对臂在四个",
            "年份点上的对比近似四次复跑。所有对比都在**同一批次内**,不跨 §25 的协议世代。", "",
            "| 对比 | 指标 | 年份 | n题 | Δ | 95% CI | +/− | Z |", "|---|---|---|---|---|---|---|---|"]
    for a, b, lab in PAIRS:
        for name, key in [("分数", "score"), ("完成率", "done")]:
            zs, ds = [], []
            for y in YEARS:
                if y not in store[a] or y not in store[b]:
                    continue
                common = sorted(set(store[a][y]) & set(store[b][y]))
                d = [np.mean(store[a][y][t][key]) - np.mean(store[b][y][t][key]) for t in common]
                m, lo, hi, z = boot(d, rng)
                zs.append(z)
                ds.append(m)
                star = " ★" if (lo > 0 or hi < 0) else ""
                out.append(f"| {lab} | {name} | {y} | {len(common)} | {m:+.3f}{star} | "
                           f"[{lo:+.3f}, {hi:+.3f}] | {sum(1 for x in d if x > 0)}/"
                           f"{sum(1 for x in d if x < 0)} | {z:+.2f} |")
            pcount, n, sp = signtest(ds)
            out.append(f"| **{lab} 聚合** | **{name}** | **{n} 个年份点** | | "
                       f"**{np.mean(ds):+.3f}** | | **{pcount}/{n} 正** | "
                       f"**Stouffer {stouffer(zs):+.2f}** (符号 p={sp:.4f}) |")
    out.append("")
    out += ["★ = 该年份点的 95% CI 不含 0。注意这四个点不是独立复跑(同一模型、同一题集、",
            "只换系统提示里的年份),所以 Stouffer 会偏乐观;方向一致性是主要证据,", "不是那个 p 值。", ""]

    txt = "\n".join(out) + "\n"
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "year_research.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
