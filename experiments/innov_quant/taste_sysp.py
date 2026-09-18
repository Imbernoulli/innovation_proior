#!/usr/bin/env python3
"""加不加 system prompt:2026 年的 idea / taste / research-judgment。

起因(用户 2026-09-18:「一个是在 2026 年,就是说默认的,或者说**不加这个 System Prompt**,
这两个都试一下吧」)。两个条件:
  - `_y26pp`   = system message 是 `It is now year 2026. You are a good researcher.`
  - `_nosyspp` = **完全不发 system message**
两边的采样**逐字相同**(temperature 1.0 / top_p 0.95 / top_k 20 / min_p 0 /
presence_penalty 1.5 / max_tokens 32768 / n=5),日志里已核过:nosyspp 打印
`[idea-client] system message: (none)`,y26pp 打印那句提示。**唯一的变量就是有没有系统提示。**

★为什么不用裸 tag(无后缀)那批老产出★:它确实没有 system prompt,但跑在 09-07/09-09,
那一版 client **还没有 presence_penalty**。裸 tag 与 `_y26pp` 同时差两件事,相减分不出
是哪一个 —— 这正是 §34「裸 tag 与 _y26pp 是两个协议世代」。所以重跑了 nosyspp。

口径逐行抄 `taste_agg.py` / `taste_year.py`,免得三张表同一个数对不上(§27):
每题先对 5 抽的 `correct` 求均值 → 逐 item 配对做差 → Wilcoxon(平局剔除)
+ 4000 次 bootstrap 95% CI;家族内符号检验 + Stouffer 聚合;
`judge3/novelty_pair` 是死任务,排除。

★ 半成品防护(§39):任一 (家族, 臂) 的作业还在 squeue 里,整格跳过。
用法:taste_sysp.py  → 打印并写 taste_sysp.md
"""
import collections
import json
import math
import os
import subprocess

import numpy as np
from scipy.stats import binomtest, norm, wilcoxon

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
HERE = os.path.dirname(os.path.abspath(__file__))
FAM = {"idea": "cc_idea32k_{}", "ideav2": "cc_ideav2_{}", "judge3": "cc_judge3_{}"}
JOBPRE = {"idea": "idea32", "ideav2": "ideav2", "judge3": "judge3"}
ARMS = {"9B RL lora": "rlv5_lo32nm_a10_s20", "4B RL lora": "rlv5_4b_lo32nm_a10_s20"}
BASE_TAG, TEST_TAG = "y26pp", "nosyspp"      # 差 = 不加提示 − 加提示
DEAD = {("judge3", "novelty_pair")}
BOOT, SEED = 4000, 0
WANT = {"idea": 480, "ideav2": 360, "judge3": 1300}

_BUF = []


def emit(s=""):
    print(s)
    _BUF.append(s)


def running():
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


def load(fam, arm, tag):
    p = os.path.join(D, FAM[fam].format(f"{arm}_{tag}"), "samples.jsonl")
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
        return None
    k, n = sum(c["mean"] > 0 for c in cs), len(cs)
    Z = sum(math.copysign(norm.isf(max(c["p"], 1e-12) / 2), c["mean"]) for c in cs) / math.sqrt(n)
    emit(f"| **{label}** | **{n} 个任务** | **{k}/{n} 正** | "
         f"**{binomtest(k, n, 0.5).pvalue:.4f}** | **{Z:+.2f}** | "
         f"**{2 * (1 - norm.cdf(abs(Z))):.4f}** |")
    return dict(n=n, k=k, Z=Z, p=2 * (1 - norm.cdf(abs(Z))))


def main():
    rng = np.random.default_rng(SEED)
    live = running()
    emit("# 加不加 system prompt:2026 的 idea / taste / research-judgment"
         "(只两条 RL 之后的 lora 臂)\n")
    emit(f"差 = **不加提示(`{TEST_TAG}`) − 加 2026 提示(`{BASE_TAG}`)**。")
    emit("两边采样逐字相同(pp=1.5),**唯一变量是有没有系统提示**;")
    emit("日志已核:`nosyspp` 打印 `system message: (none)`。")
    emit("裸 tag 的老产出同时还差 presence_penalty,**不能当这个对照用**(§34)。\n")

    cache, summary = {}, {}
    emit("## 0. 数据在不在(应为 idea 480 / ideav2 360 / judge3 1300 题)\n")
    emit(f"| 臂 | family | 加提示 `{BASE_TAG}` | 不加 `{TEST_TAG}` |")
    emit("|---|---|---:|---:|")
    for lbl, arm in ARMS.items():
        for fam in FAM:
            cells = []
            for tag in (BASE_TAG, TEST_TAG):
                if (fam, f"{arm}_{tag}") in live:
                    cells.append("作业还在跑"); cache[(fam, arm, tag)] = None; continue
                d = load(fam, arm, tag)
                cache[(fam, arm, tag)] = d
                if not d:
                    cells.append("—"); continue
                n = sum(len(v) for v in d.values())
                cells.append(f"{n}" + ("" if n == WANT[fam] else f" ⚠应为{WANT[fam]}"))
            emit(f"| {lbl} | {fam} | " + " | ".join(cells) + " |")
    emit()

    for lbl, arm in ARMS.items():
        emit(f"## {lbl}:不加提示 − 加 2026 提示\n")
        miss = [f for f in FAM if not cache.get((f, arm, BASE_TAG))
                or not cache.get((f, arm, TEST_TAG))]
        if len(miss) == len(FAM):
            emit("> 还没有可比的格子(作业在跑),跳过。\n"); continue
        if miss:
            emit(f"> ⚠ 缺 {miss},下面只含两边都落地的家族。\n")
        emit("| family | task | n | Δ | 95% CI | +/− | Wilcoxon p |")
        emit("|---|---|---:|---:|---|:---:|---:|")
        allc = []
        for fam in ("idea", "ideav2", "judge3"):
            da, db = cache.get((fam, arm, BASE_TAG)), cache.get((fam, arm, TEST_TAG))
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
        summary[(lbl, "all")] = agg(allc, "全部有效任务")
        summary[(lbl, "judge3")] = agg([c for c in allc if c["fam"] == "judge3"],
                                       "只看 judge3(research judgment)")
        summary[(lbl, "taste")] = agg([c for c in allc if c["fam"] in ("idea", "ideav2")],
                                      "只看 idea+ideav2(taste)")
        emit()

    emit("## 一眼表:Stouffer Z(正 = **不加**提示更好)\n")
    emit("| 口径 | " + " | ".join(ARMS) + " |")
    emit("|---|" + "---:|" * len(ARMS))
    for key, name in (("judge3", "research judgment(judge3)"),
                      ("taste", "taste(idea+ideav2)"), ("all", "两者合并")):
        cells = []
        for lbl in ARMS:
            r = summary.get((lbl, key))
            cells.append("—" if not r else f"{r['Z']:+.2f}" + ("" if r["p"] >= 0.05 else " ★"))
        emit(f"| {name} | " + " | ".join(cells) + " |")
    emit()
    emit("★ = 该格 p < 0.05。**负号 = 那句 `It is now year 2026. You are a good researcher.` "
         "在帮忙**;正号 = 它在拖后腿。\n")


main()
with open(os.path.join(HERE, "taste_sysp.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
