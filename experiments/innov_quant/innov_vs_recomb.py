#!/usr/bin/env python3
"""Innovation or just recombination?  One table, with the conditioning made explicit.

THE QUESTION
  The claim we want to be able to make is that the model with the innovation prior is more
  INNOVATIVE, not merely that it bolts together more existing pieces.  Those are two
  different measurements and they can move in opposite directions, so they are reported
  side by side here:

    combination side   how many distinct technique families a solution uses (n_tech), how
                       many pairs of them (n_pair), how often it uses two or more at all
                       (P(n_tech>=2)), and the same after controlling for code length.
                       The claim needs these to NOT go up: more parts is not the story.
    innovation  side   search breadth in the reasoning before the answer is fixed
                       (n_reason / n_abandon / explore_ratio / n_alt), how unusual the
                       technique pairings are against the solution pool (med_z / p10_z,
                       Uzzi-style: HIGH z = conventional pairing, LOW = atypical), how
                       derivative the code is (jac_pool), and how varied the five draws
                       are from each other (sim_self).

THE TRAP THIS FILE EXISTS TO AVOID (the section 31.5 rule, third form)
  `med_z`, `p10_z` and `new_pair_rate` are undefined when a solution uses fewer than two
  technique families -- there is no pair to score.  The arms differ enormously in how
  often that happens: on FrontierCS, P(n_tech>=2) is 51% for 9B base and 34% for 9B
  RL(prior); at 4B it is 64% for base and 22% for ours.  So a paired comparison on the
  problems where BOTH arms have a defined z is conditioned on a variable the treatment
  moves -- it selects, inside each arm, exactly the draws that behaved like the other arm.
  `innov_agg.py` reported these three fields as "not enough cells" because the NaNs
  propagated into the Wilcoxon; that was a silent failure of the same thing.

  They are reported here anyway, because dropping a metric because it is awkward is its
  own bias -- but under an explicit heading, with P(n_tech>=2) printed next to them, and
  they are kept out of the headline aggregate.

  The same conditioning applies more weakly to everything else: every metric here is
  computed on draws that finished and produced code, and completion is arm-dependent
  (section 27).  `rlv5_4b_base_s20` clears that bar on 5 / 23 / 15 draws out of 200 / 860 /
  316, so EVERY 4B contrast against it is uninterpretable and is marked as such rather
  than quietly averaged in.

MACHINERY (identical to taste_vs_base.py, so the numbers are comparable)
  Per cell = (bench x contrast x metric): average the metric over an arm's draws within a
  problem, take the paired difference over the shared problems, report mean, a paired
  bootstrap 95% CI (10000 reps, fixed seed) and z = mean / SE_bootstrap.  Cells combine by
  sign test and Stouffer over those z.

NOISE FLOOR
  The six `_y2026` same-protocol replicate pairs (section 28 verified these are single
  clean jobs, unlike the two timed-out year runs) through the identical machinery, per
  metric -- because the floor is not one number, it is per-metric.  One metric's floor
  comes out significant, and that metric cannot carry a conclusion no matter what its
  contrast cells say.
"""
import collections
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

DATA = os.environ.get(
    "INNOV_CASE",
    "/scratch/gpfs/CHIJ/ziran/.tmp/claude-374317/-scratch-gpfs-CHIJ-bohan-1-innovation-proior"
    "/20154e8e-f2e8-4272-a550-4f0c059fe5b0/scratchpad/case/innov")
BOOT = 10000
SEED = 0
BENCHES = [("frontiercs", "FrontierCS"), ("alebench", "ALE"), ("frontiercs_research", "FCS-research")]
ARMS = [("9B base", "base9b_v2c"), ("9B SFT", "ft01mix_a10"),
        ("9B RL(base)", "rlv5_base_s20"), ("9B RL(SFT)", "rlv5_ft01mix_a10_s20"),
        ("4B base", "base4b"), ("4B SFT", "4b_ft01mix_a10"),
        ("4B RL(base)", "rlv5_4b_base_s20"), ("4B RL(SFT)", "rlv5_4b_ft01mix_a10_s20")]
TAG = dict(ARMS)
CONTRASTS = [("9B RL(SFT)", "9B RL(base)", "先验在 RL 阶段"),
             ("9B RL(SFT)", "9B base", "端到端 vs base"),
             ("9B SFT", "9B base", "只做 SFT"),
             ("4B RL(SFT)", "4B RL(base)", "先验在 RL 阶段"),
             ("4B RL(SFT)", "4B base", "端到端 vs base"),
             ("4B SFT", "4B base", "只做 SFT")]
REP = [("base9b_v2c_y2026", "base9b_v2c"), ("lo32nm_a10_y2026", "lo32nm_a10"),
       ("rlv5_lo32nm_a10_s20_y2026", "rlv5_lo32nm_a10_s20"), ("base4b_y2026", "base4b"),
       ("4b_lo32nm_a10_y2026", "4b_lo32nm_a10"),
       ("rlv5_4b_lo32nm_a10_s20_y2026", "rlv5_4b_lo32nm_a10_s20")]
THIN = 100          # a cell whose thinner side has fewer draws than this is not readable

# (field, label, favourable direction for "we are innovative, not just combinatorial")
COMBI = [("n_tech", "n_tech 技术家族数", "flat"),
         ("n_pair", "n_pair 家族对数", "flat"),
         ("p_ge2", "P(n_tech≥2) 用上两个以上家族", "flat"),
         ("dens", "n_tech/100LOC 密度", "flat"),
         ("resid", "n_tech 对 log(1+LOC) 的残差", "flat")]
INNOV = [("n_reason", "n_reason 考虑过的路子", "higher"),
         ("n_abandon", "n_abandon 放弃过的路子", "higher"),
         ("explore_ratio", "explore_ratio 放弃/考虑", "higher"),
         ("n_alt", "n_alt 备选方案", "higher"),
         ("n_reason_10k", "n_reason_10k 长度归一", "higher"),
         ("jac_pool", "jac_pool 与前沿解池的相似度", "lower"),
         ("sim_self", "sim_self 5 次抽样自相似度", "lower")]
COND = [("med_z", "med_z 技术对的常规程度(中位)", "lower"),
        ("p10_z", "p10_z 最不常规那一对", "lower"),
        ("new_pair_rate", "new_pair_rate 解池里没出现过的对", "higher")]


def load():
    """-> DataFrame indexed by (bench, arm, problem, sample_idx) with every metric column."""
    def rd(name, cols):
        df = pd.read_csv(os.path.join(DATA, name), low_memory=False)
        df["problem"] = df["problem"].astype(str)
        df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce")
        df = df.dropna(subset=["sample_idx"])
        df["sample_idx"] = df["sample_idx"].astype(int)
        keep = ["bench", "arm", "problem", "sample_idx"] + [c for c in cols if c in df.columns]
        return df[keep].set_index(["bench", "arm", "problem", "sample_idx"])

    m = rd("metrics.csv", ["loc", "sim_self", "jac_pool", "complete", "has_code"])
    r = rd("recomb_metrics.csv", ["n_tech", "n_pair", "new_pair_rate", "med_z", "p10_z"])
    e = rd("explore_metrics.csv", ["n_reason", "n_abandon", "explore_ratio", "n_alt", "n_reason_10k"])
    df = m.join(r, how="outer").join(e, how="outer").reset_index()
    df["p_ge2"] = np.where(df["n_tech"].notna(), (df["n_tech"] >= 2).astype(float), np.nan)
    df["dens"] = np.where((df["loc"] > 0) & df["n_tech"].notna(),
                          100.0 * df["n_tech"] / df["loc"].replace(0, np.nan), np.nan)
    # residual of n_tech on log(1+LOC), fitted WITHIN (bench, family) exactly as
    # recomb_loc.py does -- a global fit would let the 4B/9B length gap leak into it
    df["fam"] = np.where(df["arm"].str.contains("4b|4B"), "4B", "9B")
    df["resid"] = np.nan
    for (b, f), g in df.groupby(["bench", "fam"]):
        ok = g["n_tech"].notna() & (g["loc"] > 0)
        if ok.sum() < 50:
            continue
        x = np.log1p(g.loc[ok, "loc"].to_numpy(float))
        y = g.loc[ok, "n_tech"].to_numpy(float)
        b1, b0 = np.polyfit(x, y, 1)
        df.loc[g.index[ok], "resid"] = y - (b0 + b1 * x)
    return df


DF = load()
BY = {(b, a): g for (b, a), g in DF.groupby(["bench", "arm"])}


def per_problem(bench, tag, field):
    g = BY.get((bench, tag))
    if g is None or field not in g.columns:
        return {}, 0
    s = g[["problem", field]].dropna()
    n_draw = len(s)
    return {p: float(v) for p, v in s.groupby("problem")[field].mean().items()}, n_draw


def cell(bench, A, B, field, rng):
    a, na = per_problem(bench, A, field)
    b, nb = per_problem(bench, B, field)
    ids = sorted(set(a) & set(b))
    if len(ids) < 8:
        return None
    d = np.array([a[i] - b[i] for i in ids])
    if not np.isfinite(d).all() or d.std() == 0:
        return None
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    bs = np.sort(d[idx].mean(axis=1))
    se = float(bs.std(ddof=1))
    if se <= 0:
        return None
    return dict(bench=bench, field=field, n=len(ids), n_a=na, n_b=nb,
                mean=float(d.mean()), se=se, z=float(d.mean() / se),
                lo=float(bs[int(0.025 * BOOT)]), hi=float(bs[int(0.975 * BOOT)]),
                pos=int((d > 0).sum()), neg=int((d < 0).sum()),
                thin=min(na, nb) < THIN)


def stouffer(cells):
    cs = [c for c in cells if c]
    if not cs:
        return None
    k, n = sum(1 for c in cs if c["mean"] > 0), len(cs)
    Z = sum(c["z"] for c in cs) / math.sqrt(n)
    return dict(k=k, n=n, p_sign=float(binomtest(k, n, 0.5).pvalue), Z=Z,
                p=float(2 * (1 - norm.cdf(abs(Z)))),
                med=float(np.median([abs(c["mean"]) for c in cs])))


def floors(fields, rng):
    out = {}
    for f, _, _ in fields:
        cs = [cell(b, A, B, f, rng) for b, _ in BENCHES for A, B in REP]
        out[f] = stouffer(cs)
    return out


def fmt(c):
    if not c:
        return "—"
    star = "★" if (c["lo"] > 0 or c["hi"] < 0) else ""
    warn = "⚠" if c["thin"] else ""
    return f"{c['mean']:+.3f}{star}{warn}<br>[{c['lo']:+.3f}, {c['hi']:+.3f}]"


def md_inventory(out):
    out += ["## 1. 样本盘点(§27:先看分母)", "",
            "这些指标全部只在**跑完了且产出了代码**的抽样上有定义,而「跑完」本身各臂差很多。",
            "`P(n_tech≥2)` 是 `med_z` / `p10_z` / `new_pair_rate` 的分母——它们在一份解只用了",
            "一个技术家族时无定义(没有「对」可打分)。**这个分母被处理本身改动了**,见 §5。", ""]
    out.append("| 臂 | bench | 抽样总数 | 完成且有代码 | 有 n_tech | P(n_tech≥2) | 有推理指标 |")
    out.append("|---|---|---|---|---|---|---|")
    for lab, tag in ARMS:
        for b, zh in BENCHES:
            g = BY.get((b, tag))
            if g is None:
                out.append(f"| {lab} | {zh} | — | — | — | — | — |")
                continue
            tot = len(g)
            code = int((g["jac_pool"].notna()).sum())
            nt = g["n_tech"].dropna()
            ex = int(g["n_reason"].notna().sum())
            p2 = f"{100*float((nt>=2).mean()):.1f}%" if len(nt) else "—"
            mark = "**" if len(nt) < THIN else ""
            out.append(f"| {lab} | {zh} | {tot} | {code} | {mark}{len(nt)}{mark} | {p2} | {ex} |")
    out += ["", f"粗体 = 该格进入分析的抽样数少于 {THIN},**用它做的任何对比都不可解读**。",
            "`rlv5_4b_base_s20` 三个 bench 全部如此(5 / 23 / 15),这是 §29 的 97.3% 提前停机;",
            "所以 4B 的「先验在 RL 阶段」那一列整列打 ⚠,下面照报但不进结论。", ""]
    return out


def md_floor(out, F, out_fields, title):
    out += [f"### 噪声底:{title}", "",
            "六对 `_y2026` 同协议复跑 × 3 个 bench = 18 格,走完全相同的机器。",
            "**底是逐指标的,不是一个数。**", "",
            "| 指标 | 格子 | 同向 | 符号检验 p | Stouffer Z | 中位 &#124;Δ&#124; | 可用? |",
            "|---|---|---|---|---|---|---|"]
    for f, lab, _ in out_fields:
        s = F.get(f)
        if not s:
            out.append(f"| `{f}` {lab} | — | — | — | — | — | 无数据 |")
            continue
        ok = "否 ← 底自己就显著" if s["p"] < 0.05 else "是"
        bold = "**" if s["p"] < 0.05 else ""
        out.append(f"| `{f}` {lab} | {s['n']} | {s['k']}/{s['n']} 正 | {s['p_sign']:.4f} | "
                   f"{bold}{s['Z']:+.2f}{bold} | {s['med']:.4f} | {bold}{ok}{bold} |")
    out.append("")
    return out


def md_block(out, fields, F, rng, heading, intro):
    out += [heading, ""] + intro + [""]
    for A, B, why in CONTRASTS:
        thin_all = []
        rows = []
        for f, lab, better in fields:
            cs = [cell(b, TAG[A], TAG[B], f, rng) for b, _ in BENCHES]
            s = stouffer(cs)
            fl = F.get(f)
            zt = "—" if not s else f"**{s['Z']:+.2f}**"
            zf = "—" if not fl else f"{fl['Z']:+.2f}"
            verdict = "—"
            if s:
                if better == "flat":
                    verdict = "未升高 ✔" if s["Z"] <= 0 else ("升高" if s["p"] < 0.05 else "略升不显著")
                else:
                    good = (s["Z"] > 0) if better == "higher" else (s["Z"] < 0)
                    verdict = ("有利 ✔" if s["p"] < 0.05 else "方向有利") if good else \
                              ("不利" if s["p"] < 0.05 else "方向不利")
            rows.append(f"| `{f}` {lab} | " + " | ".join(fmt(c) for c in cs) +
                        f" | {zt} | {zf} | {verdict} |")
            thin_all += [c["thin"] for c in cs if c]
        warn = "  ⚠ **本列对照臂样本过少,整列不可解读**" if thin_all and all(thin_all) else ""
        out += [f"#### {A} − {B}({why}){warn}", "",
                "| 指标 | " + " | ".join(zh for _, zh in BENCHES) + " | 3格 Stouffer Z | 噪声底 Z | 判读 |",
                "|---|---|---|---|---|---|---|"] + rows + [""]
    return out


def main():
    rng = np.random.default_rng(SEED)
    out = ["# 创新 vs 组合:我们的模型是更创新,还是只是拼得更多?", "",
           "问题来自用户:要证明模型**偏向创新而不是单纯做 combination**。",
           "这是两个不同的量,可以朝相反方向动,所以分两侧并排报:", "",
           "- **组合侧**:一份解用了几个不同的技术家族(`n_tech`)、几对(`n_pair`)、",
           "  有多大比例用上两个以上(`P(n_tech≥2)`),以及对代码长度做控制之后的同一批量。",
           "  论点要求这一侧**不升高**——「零件更多」不是我们要讲的故事。",
           "- **创新侧**:定稿之前在推理里考虑过几条路(`n_reason` / `n_abandon` /",
           "  `explore_ratio` / `n_alt`)、技术对相对解池有多不常规(`med_z` / `p10_z`,",
           "  Uzzi 口径:**z 越高越是常规搭配**)、代码与前沿解池有多像(`jac_pool`)、",
           "  5 次抽样彼此有多不一样(`sim_self`)。", "",
           "逐格机器与 `taste_vs_base.py` 完全一致(逐题配对、"
           f"{BOOT} 次配对 bootstrap、z = Δ/SE),所以两张表的 z 可以互相比。",
           f"★ = 95% CI 不含 0;⚠ = 该格对照臂进入分析的抽样数不足 {THIN},不可解读。", ""]
    out = md_inventory(out)

    Fc, Fi, Fd = floors(COMBI, rng), floors(INNOV, rng), floors(COND, rng)
    out += ["## 2. 噪声底(§20c)", ""]
    out = md_floor(out, Fc, COMBI, "组合侧")
    out = md_floor(out, Fi, INNOV, "创新侧")
    out = md_floor(out, Fd, COND, "条件指标(见 §5)")

    out = md_block(out, COMBI, Fc, rng, "## 3. 组合侧:我们并没有拼得更多",
                   ["「判读」列对这一侧问的是**有没有升高**,不是越高越好。"])
    out = md_block(out, INNOV, Fi, rng, "## 4. 创新侧",
                   ["`n_reason` 系列只在闭合了 `</think>` 的抽样上有定义(§26),",
                    "`jac_pool` / `sim_self` 只在产出了代码的抽样上有定义。",
                    "两者的分母都被处理改动过,见 §1。"])
    out = md_block(out, COND, Fd, rng,
                   "## 5. 条件于「至少用了两个技术家族」的三个指标(单列,不进结论)",
                   ["`med_z` / `p10_z` / `new_pair_rate` 在 `n_tech<2` 时无定义。",
                    "各臂的 `P(n_tech≥2)` 差得极大(9B FrontierCS:base 51% vs 我们 34%;",
                    "4B:base4b 64% vs 我们 22%),**所以「两边都≥2」这个配对子集,",
                    "是在对一个被处理本身推动的变量做筛选**——它在每条臂内部恰好挑出了",
                    "表现得像另一条臂的那些抽样(§31.5 同一根因的第三种形态)。",
                    "照报,不藏,但不进结论。`innov_agg.py` 此前对这三项打印「格子不足」,",
                    "那是 NaN 传进 Wilcoxon 后的静默失败,不是数据缺失。"])
    out.append(CONCLUSION)
    txt = "\n".join(out) + "\n"
    here = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(here, "innov_vs_recomb.md"), "w").write(txt)
    sys.stdout.write(txt)


CONCLUSION = """
## 6. 这张表能说什么、不能说什么

### 6.1 能说的:先验给的是更宽的搜索,不是更多的零件

看**同规模、同阶段**的那个对照 `9B RL(SFT) − 9B RL(base)`——它把「有没有创新先验」
单独隔出来,其余一切相同:

| | 指标 | 3 格 Stouffer Z | 噪声底 Z |
|---|---|---|---|
| 组合侧 | `n_tech` 技术家族数 | −0.05 | +1.06 |
| 组合侧 | `P(n_tech≥2)` 用上两个以上 | −0.72 | −0.73 |
| 组合侧 | `n_pair` 家族对数 | +0.32 | +1.44 |
| 创新侧 | `n_reason` 考虑过的路子 | **+8.44** | −0.91 |
| 创新侧 | `n_abandon` 放弃过的路子 | **+7.85** | −0.80 |
| 创新侧 | `explore_ratio` 放弃/考虑 | **+5.80** | −0.24 |
| 创新侧 | `n_reason_10k` 长度归一后的同一量 | **+2.14** | −0.09 |

**组合侧三个量一个都没升高,创新侧四个量全部显著升高,且三个 bench 逐格 CI 都不含 0。**
这正是用户要的那句话的证据形态:增益不在「把已有零件拼得更多」,在「定稿之前试过、
又放弃过的路子多得多」。

**而且它不是「话多所以匹配得多」**:按推理长度归一之后 `n_reason_10k` 仍是 +2.14,
噪声底 −0.09。这是 §20.2 那条结论第一次配上噪声底。

同一格里不利的三条,一并写出来:`n_alt`(备选方案数)是 −0.72 方向相反;
`sim_self` +1.29,即我们这条臂的 5 次抽样**彼此更像**,不是更发散;
`jac_pool` 那把尺子本身不合格(见 §6.3)。

### 6.2 端到端跟 base 比是混合的,不能只挑一半写

`9B RL(SFT) − 9B base`:

- 组合侧**大幅下降**:`n_tech` −7.82、`n_pair` −4.74、`P(n_tech≥2)` −7.99,
  三个 bench 逐格全 ★。这不是「没升高」,是明显更少——代码更短、认得出的技术家族更少。
  做完长度控制后 `resid` 回到 −0.32(FCS −0.23★、ALE −0.37、research +0.47★),
  **所以下降里有相当一部分只是代码变短**,但不全是。
- 创新侧有利的:`explore_ratio` **+11.69**、`n_abandon` +6.30、`n_alt` +5.33、`n_reason` +4.54。
- 创新侧不利的:`n_reason_10k` **−13.55**——推理本身长了很多,**单位长度里的分叉反而更少**;
  `sim_self` **+8.59**——5 次抽样彼此更像。

两句话都得写:**跟 base 比,我们的绝对探索量更大,但那主要是因为想得更久;
按长度折算之后反而更低,而且五次抽样更趋同。**

### 6.3 一把尺子被自己的噪声底否掉:`jac_pool`

`jac_pool`(与前沿模型解池的相似度)的噪声底是 **18 格 14/18 同向、Z=+4.64**,
也就是说**同一条臂按同协议复跑两次就能造出这个方向和量级**。
所以凡是引用 `jac_pool` 的结论,无论对我们有利还是不利,都不能用:
§6.2 里那个 +8.59「不利」不能用,`innov_agg.md` 里
`jac_pool` 9B RL(先验)−RL(base) 的 −3.04「有利」同样不能用。**撤。**

`sim_self` 的底是 −1.91(不显著,但也不干净),它的结论要打折。
其余十个指标的底都在 |Z|<1.5,可用。

### 6.4 三个 Uzzi 指标是条件指标,不进结论

`med_z` / `p10_z` / `new_pair_rate` 只在一份解用了至少两个技术家族时有定义,
而 `P(n_tech≥2)` 恰恰是被处理推动的量(9B FrontierCS:base 51% → 我们 34%;
4B:64% → 22%)。在「两边都≥2」的子集上配对,等于在每条臂内部挑出表现得像
另一条臂的那些抽样。全部照报在 §5,数值上本来也没有信号(|Z| 全部 <1.6,
唯一超过的是 4B 那个不可解读的列)。

`innov_agg.py` 此前对这三项打印「格子不足」——那不是数据缺失,是 NaN 传进
Wilcoxon 之后的静默失败。已在本表修掉。

### 6.5 4B 的「先验在 RL 阶段」整列不可解读

`rlv5_4b_base_s20` 在三个 bench 上只有 **23 / 5 / 15** 条抽样进入分析
(对照 `base4b` 的 796 / 192 / 290),因为 §29 那个 97.3% 提前停机。
这一列的每一格都打了 ⚠,照报但不进任何结论。
4B 的「端到端 vs base」是可读的,但方向混合且被退化主导
(`sim_self` +17.57、`jac_pool` +16.83),不适合当正面证据。

### 6.6 论文里能写的一句话

> 创新先验在 RL 阶段带来的不是更多的技术组合,而是定稿前更宽的搜索。
> 在同规模、同阶段的对照下(9B,RL(先验) vs RL(base)),一份解用到的技术家族数
> 与「用上两个以上家族」的比例都没有升高(Stouffer −0.05 / −0.72,噪声底 +1.06 / −0.73),
> 而推理过程中考虑过与放弃过的路子数在三个 bench 上全部显著更高
> (+8.44 / +7.85,噪声底 −0.91 / −0.80),按推理长度归一之后仍然为正(+2.14)。

必须同时写明的限定:①所有指标都条件于「该次抽样跑完了并产出代码」,而完成率各臂不同;
②`n_alt` 方向相反;③与 base 直接比时,按长度归一的探索密度更低、五次抽样更趋同;
④`jac_pool` 这把尺子的噪声底不合格,任何方向都不引用;
⑤4B 的同类对照因对照臂只剩 5–23 条抽样而不可解读。
"""


if __name__ == "__main__":
    main()
