#!/usr/bin/env python3
"""taste / idea / research-judgment 的**年份线** —— 只两条 RL 之后的 lora 臂。

起因(用户 2026-09-17:「taste/idea/judgement 跑完了的话跑一下年份的」,追问后限定
「就补充 lora 就可以了啊,就 rl 后的」)。年份点 2000 / 2050 / 2100,**2026 直接用已有的
`y26pp`,不重投**(铁律 d)。

和 `taste_agg.py` 的区别:那份比的是**两条臂**在同一年份下的差;这份比的是**同一条臂**
在不同年份下的差,基准年是 2026。所以配对键仍是 item id,但配对的两端是同一个模型、
同一套题、同一套采样参数,**唯一的变量是 system prompt 里的那个四位数**。

口径,逐条对齐 `taste_agg.py`,免得两张表同一个数对不上(§27):
  - 每题先对 5 抽的 `correct` 求均值,再逐 item 配对做差;
  - Wilcoxon(平局剔除)+ 4000 次 bootstrap 的 95% CI;
  - 家族内用符号检验 + Stouffer 聚合;
  - `judge3/novelty_pair` 是死任务(所有臂都在随机水平,见 §16.2),排除。

★ 半成品防护(§39):产出目录是作业边跑边写的。任何一个 `mls21-`/`idea32-`/`ideav2-`/
`judge3-` 作业还在 squeue 里,对应的 (臂, 年份) 就整格跳过并在表里标「作业还在跑」,
绝不把写了一半的 samples.jsonl 当结果读。

用法:taste_year.py  → 打印并写 taste_year.md
"""
import collections
import json
import math
import os
import subprocess
import sys

import numpy as np
from scipy.stats import binomtest, norm, wilcoxon

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
HERE = os.path.dirname(os.path.abspath(__file__))
FAM = {"idea": "cc_idea32k_{}", "ideav2": "cc_ideav2_{}", "judge3": "cc_judge3_{}"}
JOBPRE = {"idea": "idea32", "ideav2": "ideav2", "judge3": "judge3"}
ARMS = {"9B RL lora": "rlv5_lo32nm_a10_s20", "4B RL lora": "rlv5_4b_lo32nm_a10_s20"}
YEARS = [2000, 2026, 2050, 2100]
BASE_YEAR = 2026
DEAD = {("judge3", "novelty_pair")}
BOOT, SEED = 4000, 0

_BUF = []


def emit(s=""):
    print(s)
    _BUF.append(s)


def suffix(year):
    """2026 那一格用历史上已有的 `y26pp`;其余年份是这次新投的 `y<YEAR>pp`。"""
    return "y26pp" if year == 2026 else f"y{year}pp"


def running():
    """squeue 里还没结束的 (fam, arm, year),整格跳过。"""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set()
    live = set()
    for ln in out.splitlines():
        ln = ln.strip()
        for fam, pre in JOBPRE.items():
            if ln.startswith(pre + "-"):
                live.add((fam, ln[len(pre) + 1:]))
    return live


def load(fam, arm, year):
    p = os.path.join(D, FAM[fam].format(f"{arm}_{suffix(year)}"), "samples.jsonl")
    if not os.path.exists(p):
        return None
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for ln in open(p):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        c = r.get("correct")
        if c is None:
            continue
        by[r.get("task")][str(r.get("id"))].append(1.0 if c else 0.0)
    return {t: {i: float(np.mean(v)) for i, v in d.items()} for t, d in by.items()}


def cell(da, db, task, rng):
    """db − da,逐 item 配对。"""
    if not da or not db or task not in da or task not in db:
        return None
    ids = sorted(set(da[task]) & set(db[task]))
    if len(ids) < 20:
        return None
    d = np.array([db[task][i] - da[task][i] for i in ids])
    nz = d[d != 0]
    if len(nz) < 5:
        return None
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    bs = np.sort(d[idx].mean(axis=1))
    return dict(task=task, n=len(ids), mean=float(d.mean()),
                lo=float(bs[int(0.025 * BOOT)]), hi=float(bs[int(0.975 * BOOT)]),
                pos=int((d > 0).sum()), neg=int((d < 0).sum()),
                p=float(wilcoxon(nz).pvalue))


def agg(cs, label):
    if not cs:
        emit(f"| **{label}** | — | — | — | — | — |")
        return
    k, n = sum(c["mean"] > 0 for c in cs), len(cs)
    Z = sum(math.copysign(norm.isf(max(c["p"], 1e-12) / 2), c["mean"]) for c in cs) / math.sqrt(n)
    emit(f"| **{label}** | **{n} 个任务** | **{k}/{n} 正** | "
         f"**{binomtest(k, n, 0.5).pvalue:.4f}** | **{Z:+.2f}** | "
         f"**{2 * (1 - norm.cdf(abs(Z))):.4f}** |")


def main():
    rng = np.random.default_rng(SEED)
    live = running()
    emit("# taste / idea / research-judgment 的年份线(只两条 RL 之后的 lora 臂)\n")
    emit("唯一的变量是 system prompt 里的年份:`It is now year <Y>. You are a good researcher.`")
    emit(f"采样、题面、抽样数全部不变。基准年 **{BASE_YEAR}**(用已有的 `y26pp`,没有重投)。\n")

    cache = {}
    emit("## 0. 数据在不在\n")
    emit("| 臂 | family | " + " | ".join(str(y) for y in YEARS) + " |")
    emit("|---|---|" + "---:|" * len(YEARS))
    for lbl, arm in ARMS.items():
        for fam in FAM:
            cells = []
            for y in YEARS:
                tag = f"{arm}_{suffix(y)}"
                if (fam, tag) in live:
                    cells.append("作业还在跑"); cache[(fam, arm, y)] = None; continue
                d = load(fam, arm, y)
                cache[(fam, arm, y)] = d
                cells.append(str(sum(len(v) for v in d.values())) if d else "—")
            emit(f"| {lbl} | {fam} | " + " | ".join(cells) + " |")
    emit()

    for lbl, arm in ARMS.items():
        for y in YEARS:
            if y == BASE_YEAR:
                continue
            emit(f"## {lbl}:{y} − {BASE_YEAR}\n")
            base_missing = [f for f in FAM if not cache.get((f, arm, BASE_YEAR))]
            if base_missing:
                emit(f"> 基准年缺 {base_missing},跳过。\n"); continue
            miss = [f for f in FAM if not cache.get((f, arm, y))]
            if len(miss) == len(FAM):
                emit(f"> {y} 这一格还没有数据(作业在跑或未投),跳过。\n"); continue
            if miss:
                emit(f"> ⚠ 缺 {miss},下面只含已落地的家族。\n")
            emit("| family | task | n | Δ | 95% CI | +/− | Wilcoxon p |")
            emit("|---|---|---:|---:|---|:---:|---:|")
            allc = []
            for fam in ("idea", "ideav2", "judge3"):
                da, db = cache.get((fam, arm, BASE_YEAR)), cache.get((fam, arm, y))
                if not da or not db:
                    continue
                for task in sorted(da):
                    if (fam, task) in DEAD:
                        continue
                    c = cell(da, db, task, rng)
                    if not c:
                        continue
                    c["fam"] = fam
                    allc.append(c)
                    star = " ★" if (c["lo"] > 0 or c["hi"] < 0) else ""
                    emit(f"| {fam} | {task} | {c['n']} | {c['mean']:+.4f}{star} | "
                         f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['pos']}/{c['neg']} | {c['p']:.4f} |")
            emit("\n| 聚合 | 格子 | 方向 | 符号检验 p | Stouffer Z | p |")
            emit("|---|---|---|---|---|---|")
            agg(allc, "全部有效任务")
            agg([c for c in allc if c["fam"] == "judge3"], "只看 judge3(research judgment)")
            agg([c for c in allc if c["fam"] in ("idea", "ideav2")], "只看 idea+ideav2(taste)")
            emit()


main()
with open(os.path.join(HERE, "taste_year.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
