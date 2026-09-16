#!/usr/bin/env python3
"""裸 tag 那批 vs `_y26pp` 那批:同一套机器,同一个对比,看 ours − base 怎么变的。

用户问的是:「我记得原来都有好的结果,是不是加上年份反而不好了」。

两批到底差什么(逐行核过,不是猜的):
  裸 tag(09-07/09-08) `idea_client.py:47-49` 写明 IDEA_SYSTEM_PROMPT 未设 =
      historical behaviour (bare user prompt) —— **这批题根本没有系统提示**,
      也就没有年份那句话;采样也没有 presence_penalty(§3.2 的表:
      rlv5_ft01mix_a10_s20 每答 10369 token,strict-miss 17.7%)。
  `_y26pp`(09-15) 一次加了三样:年份系统提示
      "It is now year 2026. You are a good researcher."、presence_penalty=1.5、
      以及一个更宽容的 parser。

所以两批之差是**三个变量捆在一起**,这张表只能定位「变了多少」,
不能把它归因给年份。要拆开需要补跑缺的那一格(见文末)。
机器与 §31 逐字相同:两臂取交的 strict、penalise、逐题配对 bootstrap 10000 次。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taste_vs_base as T  # noqa: E402

BATCHES = [("裸 tag(无系统提示 / 无 pp)", ""), ("`_y26pp`(年份提示 + pp=1.5)", "_y26pp")]
CONTRASTS = [("9B RL(SFT)", "9B base"), ("4B RL(SFT)", "4B base"),
             ("9B RL(SFT)", "9B RL(base)"), ("4B RL(SFT)", "4B RL(base)")]


def load(suffix):
    store = {}
    for fam in T.FAM:
        for lab, tag in T.ARMS:
            try:
                store[(fam, lab)] = T.read(fam, tag + suffix)
            except Exception:
                store[(fam, lab)] = {}
    return store


def block(store, A, B, lens, rng):
    rows, cells = [], []
    for fam, task in T.ORDER:
        keep = T.survivors(store, fam, task, [(A, None), (B, None)]) if lens != "penalise" else set()
        a = T.acc(store, fam, A, task, lens, keep)
        b = T.acc(store, fam, B, task, lens, keep)
        c = T.cell(a, b, rng) if a and b else None
        rows.append((fam, task, c))
        if c and (fam, task) not in T.FLAGGED:
            cells.append(c)
    return rows, cells


def agg(cells):
    if not cells:
        return None
    k, n = sum(1 for c in cells if c["mean"] > 0), len(cells)
    return k, n, sum(c["z"] for c in cells) / math.sqrt(n)


def main():
    rng = np.random.default_rng(T.SEED)
    stores = {suf: load(suf) for _, suf in BATCHES}
    out = [__doc__.strip(), ""]

    # 分母先行(§27)
    out += ["## 0. 两批各自的样本量(§27)", "",
            "| 家族:任务 | 裸 tag 单位数 | `_y26pp` 单位数 |", "|---|---|---|"]
    for fam, task in T.ORDER:
        ns = []
        for _, suf in BATCHES:
            d = stores[suf].get((fam, "9B RL(SFT)"), {})
            ns.append(len(d.get(task, {})))
        out.append(f"| {fam}:{task} | {ns[0]} | {ns[1]} |")
    out.append("")

    for A, B in CONTRASTS:
        out.append(f"## {A} − {B}")
        out.append("")
        for lens in ("strict", "penalise"):
            out += [f"### lens = `{lens}`", "",
                    "| 家族:任务 | 裸 tag Δ (n) | `_y26pp` Δ (n) | 变化 |",
                    "|---|---|---|---|"]
            got = {}
            for _, suf in BATCHES:
                got[suf] = block(stores[suf], A, B, lens, rng)
            for i, (fam, task) in enumerate(T.ORDER):
                cs = [got[suf][0][i][2] for _, suf in BATCHES]
                flag = " ⚑" if (fam, task) in T.FLAGGED else ""
                def fmt(c):
                    if not c:
                        return "—"
                    star = "★" if (c["lo"] > 0 or c["hi"] < 0) else ""
                    return f"{c['mean']:+.4f}{star} ({c['n']})"
                delta = (f"{cs[1]['mean'] - cs[0]['mean']:+.4f}"
                         if cs[0] and cs[1] else "—")
                out.append(f"| {fam}:{task}{flag} | {fmt(cs[0])} | {fmt(cs[1])} | {delta} |")
            out.append("")
            out += ["| 聚合(去掉两个 ⚑) | 格子 | 同向 | Stouffer Z |", "|---|---|---|---|"]
            for name, suf in BATCHES:
                a = agg(got[suf][1])
                out.append(f"| {name} | {a[1] if a else 0} | {a[0] if a else 0}/{a[1] if a else 0} 正 | "
                           f"**{a[2]:+.2f}** |" if a else f"| {name} | 0 | — | — |")
            out.append("")
            # judge3-only
            out += ["| 只看 judge3 | 格子 | 同向 | Stouffer Z |", "|---|---|---|---|"]
            for name, suf in BATCHES:
                j = [c for (fam, task, c) in got[suf][0]
                     if fam == "judge3" and c and (fam, task) not in T.FLAGGED]
                a = agg(j)
                out.append(f"| {name} | {a[1] if a else 0} | {a[0] if a else 0}/{a[1] if a else 0} 正 | "
                           f"**{a[2]:+.2f}** |" if a else f"| {name} | 0 | — | — |")
            out.append("")

    out += ["## 这张表能说什么、不能说什么", "",
            "**不能**把任何差异归因给年份。两批差三样:年份系统提示、`presence_penalty=1.5`、",
            "更宽容的 parser。三者捆在一起。",
            "",
            "**要拆开只缺一格**:在**正确的采样协议下**(pp=1.5、同一份 parser)跑一批",
            "**不带系统提示**的(记作 `_nspp`)。那时",
            "`_y26pp` − `_nspp` 就是**纯年份效应**,其余一切相同。",
            "8 条臂 × 3 个 bench = 24 个作业,`extra_bench_submit.sh` 加 `TAGSUF=nspp`、",
            "把 `GEN/IDEA/JUDGE_SYSTEM_PROMPT` 置空即可,不改任何 RL 参数。", ""]

    txt = "\n".join(out) + "\n"
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_compare.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
