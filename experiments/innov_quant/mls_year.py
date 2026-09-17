#!/usr/bin/env python3
"""MLS-21 年份扫描,只看 ft01mix 线。

年份扫描此前的三个脚本(year_shape / year_completion / year_curves)都只读 samples.jsonl,
而 MLS 不写 samples.jsonl -- 分数在 summary.json 的 mean_score。所以 MLS 的年份跑一直
没进过任何一张年份表。这个脚本补上。

三个必须先处理的口径问题:

1. 分母(§27)。summary.json 的 mean_score 除的是 n_scored,不是 21。
   base9b_v2c_y1700 的 0.3241 是只判出来的 **2 道题**的均值。这里一律按题配对,
   不用 mean_score。

2. 无效格。agent 日志里 openai.APIConnectionError 表示 vLLM serve 当时是死的,
   那一格不是模型表现,是基础设施。ft01mix_a10 的 y2000(9题) / y2075(16题) 全毁,
   该臂直接退出分析。

3. 零分的三种来源(§29 粒度规则)。score==0 拆成:
     empty  settings==[]  -> agent 没交出可判产物(多数以 "No action returned" 收尾)
     floor  settings 齐全但每档都 0 -> 真的被钳到最差基线
   前者量的是「有没有动手」,后者才是「做得好不好」。两个分开报。

对照设计:NEAR=2025,FAR=mean(2000,2075)(这两个年份点所有臂都有;2050 只有两条臂有)。
噪声底用同一条臂的 2000 vs 2075 -- 两个都是"远年",任何倒 U 假设都不预测它们有差,
所以它们之间的差就是年份点之间的跑间噪声。
"""
import collections
import glob
import json
import os
import re

import numpy as np

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
ARMS = [("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"),
        ("rlv5_ft01mix_a10_s20", "9B RL(先验)"),
        ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"),
        ("rlv5_4b_ft01mix_a10_s20", "4B RL(先验)")]
NEAR, FAR = 2025, (2000, 2075)
BOOT = 10000
RE_CONN = re.compile(r"APIConnectionError")
RE_NOACT = re.compile(r"No action returned after 3 attempts")


def load(arm):
    r"""-> {year: {task: dict(score, kind, noact, conn)}}

    补跑目录 `<tag>-fix` 必须合并进来,同名题后写覆盖 —— 这是全仓的惯例
    (year_grid / mls_audit21 / mls_align_* / mls_filestate 都这么读)。
    2026-09-17:这个函数原来用 `..._y\d{4}$` 锚定,把 `-fix` 整个挡在外面,
    于是 91 个补跑格子一个都没进来:9B SFT y2000 还是 12/21、18 个连接错误,
    而 year_grid 同一格已经是 21/21。两张表对同一个数打架就一定有一张错。
    `-alfix` 是单题跟跑,不并。
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(D, f"cc_mls21_{arm}_y[0-9][0-9][0-9][0-9]"))):
        m = re.match(rf"cc_mls21_{re.escape(arm)}_y(\d{{4}})$", os.path.basename(d))
        p = os.path.join(d, "summary.json")
        if not m or not os.path.exists(p):
            continue
        srcs = [(d, p)]
        fx = os.path.join(d + "-fix", "summary.json")
        if os.path.exists(fx):
            srcs.append((d + "-fix", fx))
        cell = {}
        for dd, pp in srcs:
          for t in json.load(open(pp))["tasks"]:
              sc, se = t.get("score"), (t.get("settings") or [])
              if sc is None:
                  kind = "nores"
              elif sc > 0:
                  kind = "pos"
              elif not se:
                  kind = "empty"
              elif all((s.get("score") or 0) == 0 for s in se):
                  kind = "floor"
              else:
                  kind = "other0"
              lg = os.path.join(dd, "task_logs", t["task"] + ".log")
              txt = open(lg, errors="replace").read() if os.path.exists(lg) else ""
              cell[t["task"]] = dict(score=sc, kind=kind,
                                     noact=bool(RE_NOACT.search(txt)),
                                     conn=bool(RE_CONN.search(txt)))
        out[int(m.group(1))] = cell
    return out


def boot(d, rng):
    """配对 bootstrap:返回 (mean, lo, hi, z)。"""
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    bs = d[idx].mean(axis=1)
    se = bs.std(ddof=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), \
        (d.mean() / se if se > 0 else float("nan"))


def contrast(store, a_year, b_years, getter):
    """按题配对:a_year 的值 减 b_years 的均值。只用三个年份点都判过的题。"""
    if a_year not in store or any(y not in store for y in b_years):
        return None
    common = set(store[a_year])
    for y in b_years:
        common &= set(store[y])
    common = sorted(t for t in common
                    if getter(store[a_year][t]) is not None
                    and all(getter(store[y][t]) is not None for y in b_years))
    if not common:
        return None
    d = [getter(store[a_year][t]) - np.mean([getter(store[y][t]) for y in b_years])
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
    store = {a: load(a) for a, _ in ARMS}
    out = ["# MLS-21 年份扫描(只看 ft01mix 线)", ""]

    # ---- 0. 每条臂进入分析的样本数 + 无效格,先于任何效应量(§27/§29) ----
    out += ["## 0. 每格的判出情况与死因拆分", "",
            "`空提交` = score 0 且 settings 为空,agent 没交出可判产物;",
            "`钳底` = settings 齐全但每档都 0,才是真的被判到最差基线;",
            "`连接错误` = 该题日志里出现 `openai.APIConnectionError`,即 vLLM serve 当时是死的。", "",
            "| 臂 | 年份 | 判出/21 | 正分 | 空提交 | 钳底 | 无产物 | 无action停 | 连接错误 |",
            "|---|---|---|---|---|---|---|---|---|"]
    bad = set()
    for a, lab in ARMS:
        for y in sorted(store[a]):
            c = collections.Counter(v["kind"] for v in store[a][y].values())
            na = sum(v["noact"] for v in store[a][y].values())
            cn = sum(v["conn"] for v in store[a][y].values())
            if cn >= 3:
                bad.add((a, y))
            flag = " ⛔" if cn >= 3 else ""
            out.append(f"| {lab} `{a}` | {y}{flag} | {21 - c['nores']} | {c['pos']} | "
                       f"{c['empty']} | {c['floor']} | {c['nores']} | {na} | {cn} |")
    out += ["", f"⛔ = 该格 ≥3 道题撞上死 serve,判为**基础设施无效**,不进任何对照:"
                f" {', '.join(f'`{a}` y{y}' for a, y in sorted(bad))}。", ""]

    usable = [(a, lab) for a, lab in ARMS
              if NEAR in store[a] and all(y in store[a] for y in FAR)
              and not any((a, y) in bad for y in (NEAR,) + FAR)]
    dropped = [lab for a, lab in ARMS if (a, lab) not in usable]
    out += [f"可用臂 **{len(usable)}/6**;剔除 {', '.join(dropped) if dropped else '无'}。", ""]

    # ---- 1. 分数:NEAR - FAR ----
    for title, getter, lo_is_good in [
            ("1. 分数 NEAR(2025) − FAR(2000/2075 均值)",
             lambda v: v["score"], False),
            ("2. 可执行性:交出产物的比例 NEAR − FAR(`空提交`/`无产物` 记 0,其余记 1)",
             lambda v: 0.0 if v["kind"] in ("empty", "nores") else 1.0, False)]:
        out += [f"## {title}", "", "| 臂 | n题 | Δ | 95% CI | +/− | Z |", "|---|---|---|---|---|---|"]
        zs, ds = [], []
        for a, lab in usable:
            r = contrast(store[a], NEAR, FAR, getter)
            if not r:
                out.append(f"| {lab} | — | — | — | — | — |")
                continue
            common, d = r
            m, lo, hi, z = boot(d, rng)
            pos = sum(1 for x in d if x > 0)
            neg = sum(1 for x in d if x < 0)
            zs.append(z)
            ds.append(m)
            out.append(f"| {lab} `{a}` | {len(common)} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                       f"{pos}/{neg} | {z:+.2f} |")
        p, n, sp = signtest(ds)
        out += ["", "| 聚合 | 格子 | 方向 | 符号 p | Stouffer Z |", "|---|---|---|---|---|",
                f"| **NEAR−FAR** | **{n}** | **{p}/{n} 正** | **{sp:.4f}** | **{stouffer(zs):+.2f}** |", ""]

    # ---- 3. 噪声底:2000 vs 2075,两个都是远年 ----
    out += ["## 3. 噪声底:同臂 2000 vs 2075(两个都是「远年」,任何倒 U 假设都不预测有差)", "",
            "| 臂 | 指标 | n题 | Δ | 95% CI | Z |", "|---|---|---|---|---|---|"]
    floors = collections.defaultdict(list)
    for a, lab in usable:
        for name, getter in [("分数", lambda v: v["score"]),
                             ("可执行性", lambda v: 0.0 if v["kind"] in ("empty", "nores") else 1.0)]:
            r = contrast(store[a], FAR[0], (FAR[1],), getter)
            if not r:
                continue
            common, d = r
            m, lo, hi, z = boot(d, rng)
            floors[name].append((m, z))
            out.append(f"| {lab} | {name} | {len(common)} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {z:+.2f} |")
    out.append("")
    out += ["| 指标 | 格子 | 方向 | Stouffer Z | 逐格 \\|Δ\\| 中位 | 最大 |", "|---|---|---|---|---|---|"]
    for name, vals in floors.items():
        ms = [v[0] for v in vals]
        p, n, sp = signtest(ms)
        out.append(f"| **{name}** | **{n}** | **{p}/{n} 正** | **{stouffer([v[1] for v in vals]):+.2f}** | "
                   f"**{np.median([abs(m) for m in ms]):.4f}** | **{max(abs(m) for m in ms):.4f}** |")
    out.append("")

    # ---- 4. 逐年原始值 ----
    # 年份列由数据决定,不写死:曾经写死 2000/2025/2050/2075,y2100 落地后峰值列指向
    # 一个表里根本没显示的年份(9B RL(先验) 与 4B SFT 都标了 2100)。读者看不到峰值
    # 所依据的那个数,就是错的表。
    YCOLS = sorted({y for a, _ in ARMS for y in store[a] if (a, y) not in bad})
    out += ["## 4. 逐年原始值(每条臂限制到该臂所有年份点都判出分的公共题)", "",
            "| 臂 | 公共题 | " + " | ".join(str(y) for y in YCOLS) + " | 峰值 |",
            "|" + "---|" * (len(YCOLS) + 3)]
    for a, lab in ARMS:
        ys = [y for y in sorted(store[a]) if (a, y) not in bad]
        if not ys:
            continue
        common = set.intersection(*[{t for t, v in store[a][y].items() if v["score"] is not None}
                                    for y in ys])
        cells = {y: np.mean([store[a][y][t]["score"] for t in common]) for y in ys} if common else {}
        pk = max(cells, key=cells.get) if cells else "—"
        row = " | ".join(f"{cells[y]:.3f}" if y in cells else "—" for y in YCOLS)
        note = " ⛔部分年份无效" if any((a, y) in bad for y in store[a]) else ""
        out.append(f"| {lab} `{a}`{note} | {len(common)} | {row} | {pk} |")
    out.append("")

    txt = "\n".join(out) + "\n"
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mls_year.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
