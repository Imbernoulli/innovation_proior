#!/usr/bin/env python3
"""taste / idea / research-judgment: every ours-vs-base number in one table, with a floor.

WHY THIS EXISTS ALONGSIDE taste_agg.py
  taste_agg.py already reports these contrasts, but three things in it need fixing before
  the numbers go in a paper.

  1. UNIT (section 27, denominator).  On judge3 it keys items by `id`, which is `ip0o1` /
     `ip0o2` -- the two presentation orders of ONE pair, with the gold answer flipped
     between them.  j3stats.py's own header says the independent unit is the pair: the
     two orders are one measurement plus its mirror, so counting them separately doubles
     n and narrows the interval of a quantity that was measured once.  Here judge3 is
     collapsed to `meta.pair_id` first, exactly as j3stats does, so judge3 runs on
     250/250/150 pairs instead of 500/500/300 pseudo-items.

  2. LENS (section 29, granularity).  An unparsed sample -- no VERDICT emitted -- is not
     missing at random; it is concentrated on the arm that degenerated.  j3stats settled
     this by pre-registering two conventions and reporting both, and taste_agg.py reports
     neither explicitly (it silently takes `correct` as given, which is the penalise
     lens).  Both are reported here, under j3stats' names and definitions:
       strict    an item counts only if EVERY arm parsed all 5 draws (and, on judge3, only
                 if both of its orders survive).  Unbiased on the survivors, but the
                 survivor set is chosen by the worst arm's failure mode, which flatters it.
       penalise  unparsed == wrong, nothing dropped.  Charges degeneration to the arm that
                 degenerated, but conflates "judged badly" with "never answered".
     The strict survivor set is taken over all eight arms, not over the two arms of one
     contrast, so every column of this table is computed on the same items.

  3. FLOOR (section 20c).  No claim here has had a same-machinery noise floor attached.
     Two are computed, and the real floor is between them:
       F_split  within one run, draws {0,1} vs {3,4} of the same item.  True effect is
                exactly zero; captures sampling noise only.  Each side averages 2 draws
                instead of 5, so its spread overstates a 5-vs-5 contrast by sqrt(5/2)
                = 1.58x.  Lower bound after that correction.
       F_proto  the same arm on the same items, bare tag vs `_y26pp`.  Per
                extra_bench_submit.sh:41 these differ in exactly one thing -- `_y26pp`
                resamples on the RL rollout protocol (presence_penalty=1.5) -- plus a more
                tolerant parser.  So its true effect is NOT zero and it is an upper bound.
                It is the honest scale for "would this number come back if we ran again
                under a slightly different config".
     Both go through the identical per-cell machinery as the contrasts, so the z values
     are directly comparable.

ERA.  Everything except F_proto reads the `_y26pp` batch only (2026-09-15 18:06-20:51, one
submission window).  Section 25 cost a retracted result to the rule that bare tags are the
09-07 generation and must never be paired across.  F_proto is the one place the two eras
meet, and there the cross-era difference IS the quantity being measured.

STATISTICS.  Per cell: pair by item, average `ok` over the 5 draws within an item, take the
paired difference, report the mean and a paired bootstrap 95% CI (10000 reps, fixed seed).
z = mean / SE_bootstrap, so z and the CI say the same thing, and the floors use the same
formula.  Cells are aggregated by sign test and by Stouffer over those z.

VERDICT COLUMNS.  No single verdict letter, because that would have to hide something.
Three independent marks per cell: the direction, whether |delta| clears the floor, and
whether the CI excludes 0.  A cell can be favourable in direction and still under the
floor, and that is exactly the case the reader needs to see.
"""
import collections
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binomtest, norm, wilcoxon

OUT = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
FAM = {"idea": "cc_idea32k_{}", "ideav2": "cc_ideav2_{}", "judge3": "cc_judge3_{}"}
ARMS = [("9B base", "base9b_v2c"), ("9B SFT", "ft01mix_a10"),
        ("9B RL(base)", "rlv5_base_s20"), ("9B RL(SFT)", "rlv5_ft01mix_a10_s20"),
        ("4B base", "base4b"), ("4B SFT", "4b_ft01mix_a10"),
        ("4B RL(base)", "rlv5_4b_base_s20"), ("4B RL(SFT)", "rlv5_4b_ft01mix_a10_s20")]
TAG = dict(ARMS)
CONTRASTS = [("9B RL(SFT)", "9B base"), ("4B RL(SFT)", "4B base")]
EXTRA = [("9B SFT", "9B base"), ("4B SFT", "4B base"),
         ("9B RL(SFT)", "9B SFT"), ("4B RL(SFT)", "4B SFT"),
         ("9B RL(SFT)", "9B RL(base)"), ("4B RL(SFT)", "4B RL(base)")]
N_EXP = 5
BOOT = 10000
SEED = 0

# Cells kept in the table but excluded from the aggregate, each for a reason established
# elsewhere in the case study rather than for its value here.
FLAGGED = {
    ("idea", "openreview_decide"):
        "v1 prompt 里带会议名(§25):去掉那句话让两条 RL 臂的 AUC 各动 0.11-0.14,"
        "且 base 自己两个世代在同一批 120 题上就差 0.073-0.097",
    ("judge3", "novelty_pair"):
        "八条臂全部贴在随机水平(§16.2),没有可测的信号",
}
ORDER = [("idea", "aaar_equation"), ("idea", "liveidea_pair"), ("idea", "openreview_pair"),
         ("idea", "openreview_decide"),
         ("ideav2", "aaar_equation"), ("ideav2", "openreview_pair"),
         ("ideav2", "openreview_decide"),
         ("judge3", "impact_pair"), ("judge3", "impact_contrarian"),
         ("judge3", "novelty_pair")]


def truthy(x):
    return x in (True, "True", "true", 1)


def read(fam, tag):
    """-> {task: {unit: {"ok": [bool]*5, "parsed": [bool]*5}}}, judge3 collapsed to pairs.

    Key is (id, sample_idx) with the last writer winning, the same rule j3stats uses so a
    resumed run that rewrote a sample is not counted twice.
    """
    p = os.path.join(OUT, FAM[fam].format(tag), "samples.jsonl")
    if not os.path.exists(p):
        return None
    seen = {}
    for ln in open(p):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        seen[(str(r["id"]), str(r.get("sample_idx")))] = r
    byid = collections.defaultdict(list)
    for (iid, sidx), r in seen.items():
        byid[iid].append((sidx, r))
    items = {}
    for iid, rows in byid.items():
        rows.sort(key=lambda t: int(t[0]) if t[0].lstrip("-").isdigit() else t[0])
        rows = rows[:N_EXP]
        if len(rows) < N_EXP:
            continue
        r0 = rows[0][1]
        parsed = [not truthy(r.get("unparsed")) for _, r in rows]
        ok = [p_ and truthy(r.get("correct")) for p_, (_, r) in zip(parsed, rows)]
        items[iid] = {"task": r0["task"], "meta": r0.get("meta") or {},
                      "ok": ok, "parsed": parsed}
    if fam != "judge3":
        out = collections.defaultdict(dict)
        for iid, d in items.items():
            out[d["task"]][iid] = d
        return dict(out)
    # judge3: one pair = two orders of the same item, gold flipped between them
    pairs = collections.defaultdict(dict)
    for iid, d in items.items():
        pairs[(d["task"], d["meta"]["pair_id"])][int(d["meta"]["order"])] = d
    out = collections.defaultdict(dict)
    for (task, pid), byo in pairs.items():
        if 1 not in byo or 2 not in byo:
            continue
        out[task][pid] = {"task": task, "meta": byo[1]["meta"],
                          "ok": byo[1]["ok"] + byo[2]["ok"],
                          "parsed": byo[1]["parsed"] + byo[2]["parsed"],
                          "halves": (byo[1], byo[2])}
    return dict(out)


PP = {(f, lab): read(f, t + "_y26pp") for f in FAM for lab, t in ARMS}
BARE = {(f, lab): read(f, t) for f in FAM for lab, t in ARMS}


def survivors(store, fam, task, labs=None):
    """Item ids where every arm in `labs` parsed every draw (strict lens).

    j3stats takes the common set over the tags it is given, so for a two-arm contrast the
    survivor set is that contrast's own -- not the intersection over all eight arms.  The
    difference is not cosmetic: 4B RL(base) leaves no VERDICT on 36.8% of aaar_equation
    draws, and intersecting over all eight would cut every OTHER contrast on that task
    from 120 items to 25 for a reason that has nothing to do with the arms being compared.
    The all-eight count is still printed in section 1 for reference.
    """
    keep = None
    for lab, _ in (labs or ARMS):
        d = store.get((fam, lab))
        if not d or task not in d:
            return set()
        srv = {i for i, v in d[task].items() if all(v["parsed"])}
        keep = srv if keep is None else (keep & srv)
    return keep or set()


def acc(store, fam, lab, task, lens, keep):
    d = store.get((fam, lab))
    if not d or task not in d:
        return {}
    out = {}
    for i, v in d[task].items():
        if lens != "penalise" and i not in keep:
            continue
        out[i] = float(np.mean(v["ok"]))
    return out


def cell(a, b, rng):
    """paired difference a - b over shared items -> stats, or None if too thin."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 20:
        return None
    d = np.array([a[i] - b[i] for i in ids])
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    bs = np.sort(d[idx].mean(axis=1))
    se = float(bs.std(ddof=1))
    nz = d[d != 0]
    try:
        p_w = float(wilcoxon(nz)[1]) if len(nz) >= 5 else float("nan")
    except Exception:
        p_w = float("nan")
    return dict(n=len(ids), mean=float(d.mean()), se=se,
                lo=float(bs[int(0.025 * BOOT)]), hi=float(bs[int(0.975 * BOOT)]),
                z=float(d.mean() / se) if se > 0 else 0.0,
                pos=int((d > 0).sum()), neg=int((d < 0).sum()), p_w=p_w)


def stouffer(cells):
    cs = [c for c in cells if c]
    if not cs:
        return None
    k, n = sum(1 for c in cs if c["mean"] > 0), len(cs)
    Z = sum(c["z"] for c in cs) / math.sqrt(n)
    return dict(k=k, n=n, p_sign=float(binomtest(k, n, 0.5).pvalue), Z=Z,
                p=float(2 * (1 - norm.cdf(abs(Z)))),
                med=float(np.median([abs(c["mean"]) for c in cs])))


# ---------------------------------------------------------------- noise floors
def floor_split(rng):
    """Draws {0,1} vs {3,4} of the same item, same run. True effect exactly zero."""
    cells = []
    for fam, task in ORDER:
        for lab, _ in ARMS:
            d = PP.get((fam, lab))
            if not d or task not in d:
                continue
            A, B = {}, {}
            for i, v in d[task].items():
                o = v["ok"]
                if fam == "judge3":
                    h1, h2 = v["halves"]
                    A[i] = float(np.mean(h1["ok"][:2] + h2["ok"][:2]))
                    B[i] = float(np.mean(h1["ok"][3:5] + h2["ok"][3:5]))
                else:
                    A[i] = float(np.mean(o[:2]))
                    B[i] = float(np.mean(o[3:5]))
            c = cell(A, B, rng)
            if c:
                c.update(fam=fam, task=task, lab=lab)
                cells.append(c)
    return cells


def unp_rate(store, fam, lab, task):
    d = store.get((fam, lab))
    if not d or task not in d:
        return float("nan")
    tot = sum(len(v["parsed"]) for v in d[task].values())
    unp = sum(len(v["parsed"]) - sum(v["parsed"]) for v in d[task].values())
    return unp / tot if tot else float("nan")


def floor_proto(rng):
    """Same arm, same items, `_y26pp` minus bare tag: presence_penalty=1.5 plus a parser.

    `dunp` is how far the unparsed rate moved between the two eras.  Where it moved a lot
    the cell is measuring the new parser recovering answers that were there all along --
    4B RL(base) on judge3 goes 33.8% -> 1.4% -- which is a different quantity from run
    noise, and by far the largest cells are all of that kind.  Section 28's rule again:
    look at how the artifact was made before reusing it as a floor.
    """
    cells = []
    for fam, task in ORDER:
        for lab, _ in ARMS:
            A = acc(PP, fam, lab, task, "penalise", None)
            B = acc(BARE, fam, lab, task, "penalise", None)
            c = cell(A, B, rng)
            if c:
                c.update(fam=fam, task=task, lab=lab,
                         dunp=abs(unp_rate(PP, fam, lab, task) - unp_rate(BARE, fam, lab, task)))
                cells.append(c)
    return cells


# ---------------------------------------------------------------- output
def md_inventory(out):
    out += ["## 1. 样本盘点(§27:先看分母,再看效应量)", "",
            "`_y26pp` 那一批,每条臂进入分析的单位数与抽样数。",
            "judge3 已按 `meta.pair_id` 折成「对」(两个呈现顺序是一次测量加它的镜像,",
            "单位是对不是条目——`j3stats.py` 的口径),所以它的 n 是 250/250/150 而不是 500/500/300。", ""]
    out.append("| 家族 | 任务 | 单位 | 单位数(全) | 单位数(strict 存活) | 抽样数 |")
    out.append("|---|---|---|---|---|---|")
    for fam, task in ORDER:
        d = PP.get((fam, "9B base"))
        if not d or task not in d:
            continue
        keep = survivors(PP, fam, task)
        unit = "对" if fam == "judge3" else "题"
        n = len(d[task])
        ndraw = sum(len(v["ok"]) for v in d[task].values())
        out.append(f"| {fam} | {task} | {unit} | {n} | {len(keep)} | {ndraw} |")
    out += ["", "### 1.1 不作答率(§29:无 VERDICT / 不给答案会直接改分母)", "",
            "一格是「该臂在该任务上 unparsed 的抽样占比」。`strict` lens 丢掉的正是这些格子所在的题,",
            "而它们并非随机缺失——集中在退化的那条臂上。", ""]
    out.append("| 臂 | " + " | ".join(f"{f}:{t}" for f, t in ORDER) + " |")
    out.append("|---|" + "---|" * len(ORDER))
    for lab, _ in ARMS:
        row = [lab]
        for fam, task in ORDER:
            d = PP.get((fam, lab))
            if not d or task not in d:
                row.append("—")
                continue
            tot = sum(len(v["parsed"]) for v in d[task].values())
            unp = sum(len(v["parsed"]) - sum(v["parsed"]) for v in d[task].values())
            row.append("0" if unp == 0 else f"**{100*unp/tot:.1f}%**" if unp / tot > 0.05
                       else f"{100*unp/tot:.1f}%")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


PARSER_MOVE = 0.01


def md_floor(out, fs, fp):
    fq = [c for c in fp if c["dunp"] <= PARSER_MOVE]
    ss, sp, sq = stouffer(fs), stouffer(fp), stouffer(fq)
    out += ["## 2. 噪声底(§20c:同一套机器,跑一个已知答案的对比)", "",
            "两个底,真实的底在它们之间。两个都走上面那套逐格机器(逐题配对、"
            f"{BOOT} 次配对 bootstrap、z = Δ/SE),所以 z 可以直接跟正表比。", "",
            "**F_split(下界)**:同一次跑批内,同一题的第 {0,1} 次抽样 vs 第 {3,4} 次抽样。",
            "真实效应严格为 0,只含抽样噪声。两侧各只平均 2 次抽样而不是 5 次,",
            f"所以它的离散度比 5-vs-5 大 √(5/2)=1.58 倍;修正后的下界是 {ss['med']/1.58:.4f}。", "",
            "**F_proto(上界)**:同一条臂、同一批题,`_y26pp` 减裸 tag。",
            "`extra_bench_submit.sh:41` 写明两者只差一件事——`_y26pp` 按 RL rollout 协议重采",
            "(`presence_penalty=1.5`),外加一个更宽容的 parser。**所以它的真实效应不是 0**,",
            "它是上界。但它回答的正是「换个略有不同的配置再跑一次,这个数还在不在」。", "",
            "F_proto 要再分一刀(§28:拿别人的产物当底之前,先看它是怎么造出来的)。",
            "两个世代之间**不作答率**本身就变了:4B RL(base) 在 judge3 上 33.8% → 1.4%,",
            "9B RL(base) 4.5% → 0.0%。这些格子量的是**新 parser 把本来就在那里的答案捞了回来**,",
            "不是跑批噪声,而且逐格最大的几个全是这一类。所以再给一行",
            f"**F_proto(parser 未动)**:只留不作答率变动 ≤ {PARSER_MOVE:.0%} 的格子。",
            "门槛用这一行的中位 |Δ|。", ""]
    out.append("| 底 | 格子 | 同向 | 符号检验 p | Stouffer Z | 中位 &#124;Δ&#124; |")
    out.append("|---|---|---|---|---|---|")
    for nm, s in (("F_split(下界,未修正)", ss),
                  (f"F_split(下界,÷1.58)", dict(ss, med=ss["med"] / 1.58)),
                  ("F_proto(全部 80 格)", sp),
                  (f"**F_proto(parser 未动,{sq['n']} 格)**", sq)):
        out.append(f"| {nm} | {s['n']} | {s['k']}/{s['n']} 正 | {s['p_sign']:.4f} | "
                   f"{s['Z']:+.2f} | **{s['med']:.4f}** |")
    out += ["", "F_proto 逐格最大的几个(同一条臂、同一批题):", "",
            "| 家族:任务 | 臂 | Δ | 95% CI | 不作答率变动 |", "|---|---|---|---|---|"]
    for c in sorted(fp, key=lambda c: -abs(c["mean"]))[:8]:
        mark = "**parser**" if c["dunp"] > PARSER_MOVE else "—"
        out.append(f"| {c['fam']}:{c['task']} | {c['lab']} | {c['mean']:+.4f} | "
                   f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {100*c['dunp']:.1f}pp {mark} |")
    out += ["", f"读法:**任何 |Δ| 小于 {sq['med']:.4f} 的逐格结论,都低于「同一条臂换个采样协议"
            "再跑一次」这件事本身造成的扰动**,无论它的 CI 含不含 0。"
            f"(真实的底在 {ss['med']/1.58:.4f} 与 {sq['med']:.4f} 之间。)", ""]
    return out, ss, sq


def contrast_block(out, A, B, lens, floor_med, rng):
    labs = [(A, TAG[A]), (B, TAG[B])]
    out += [f"### {A} − {B}  ·  lens = `{lens}`", "",
            "| 家族 | 任务 | n | Δ | 95% CI | 方向 | 过噪声底 | CI 不含 0 | +/− | Wilcoxon p |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    cells = []
    for fam, task in ORDER:
        keep = (survivors(PP, fam, task, labs) if lens == "strict"
                else survivors(PP, fam, task) if lens == "strict8" else None)
        a = acc(PP, fam, A, task, lens, keep)
        b = acc(PP, fam, B, task, lens, keep)
        c = cell(a, b, rng)
        if not c:
            continue
        c.update(fam=fam, task=task)
        flag = (fam, task) in FLAGGED
        c["flagged"] = flag
        cells.append(c)
        dirn = "↑ 我们更好" if c["mean"] > 0 else ("↓ base 更好" if c["mean"] < 0 else "=")
        over = "是" if abs(c["mean"]) > floor_med else "否"
        sig = "★" if (c["lo"] > 0 or c["hi"] < 0) else "—"
        name = f"{task} ⚑" if flag else task
        out.append(f"| {fam} | {name} | {c['n']} | {c['mean']:+.4f} | "
                   f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {dirn} | {over} | {sig} | "
                   f"{c['pos']}/{c['neg']} | {c['p_w']:.4f} |")
    keepc = [c for c in cells if not c["flagged"]]
    out += ["", "| 聚合 | 格子 | 同向 | 符号检验 p | Stouffer Z | p |", "|---|---|---|---|---|---|"]
    for nm, sub in (("全部 10 格", cells),
                    ("去掉两个 ⚑ 格", keepc),
                    ("只看 judge3(research judgment)", [c for c in keepc if c["fam"] == "judge3"]),
                    ("只看 idea+ideav2(taste/idea)", [c for c in keepc if c["fam"] != "judge3"])):
        s = stouffer(sub)
        if not s:
            continue
        out.append(f"| {nm} | {s['n']} | {s['k']}/{s['n']} 正 | {s['p_sign']:.4f} | "
                   f"**{s['Z']:+.2f}** | {s['p']:.4f} |")
    out.append("")
    return out, cells


def j3_pair_acc(lab, task, lens, keep):
    """-> {pair_id: (acc, acc_order1, acc_order2)} on judge3."""
    d = PP.get(("judge3", lab))
    if not d or task not in d:
        return {}
    out = {}
    for i, v in d[task].items():
        if lens != "penalise" and i not in keep:
            continue
        h1, h2 = v["halves"]
        o1, o2 = float(np.mean(h1["ok"])), float(np.mean(h2["ok"]))
        out[i] = ((o1 + o2) / 2, o1, o2)
    return out


def boot_gap(va, vb, ca, cb, rng):
    """MIMICRY excess, exactly j3stats' estimator.

    impact_pair and impact_contrarian are DISJOINT pair sets, so they are resampled
    independently inside each replicate; arm and control are paired within each task.
    """
    na, nb = len(va), len(vb)
    obs = (sum(x - y for x, y in zip(va, ca)) / na
           - sum(x - y for x, y in zip(vb, cb)) / nb)
    da = np.array([x - y for x, y in zip(va, ca)])
    db = np.array([x - y for x, y in zip(vb, cb)])
    ia = rng.integers(0, na, size=(BOOT, na))
    ib = rng.integers(0, nb, size=(BOOT, nb))
    reps = np.sort(da[ia].mean(axis=1) - db[ib].mean(axis=1))
    return obs, float(reps[int(0.025 * BOOT)]), float(reps[int(0.975 * BOOT)])


def md_judge_quality(out, rng):
    out += ["## 3.2 judge3 自带的两个判断力刻画:模仿程度与顺序一致性", "",
            "这两个量是 `j3stats.py` 预先写好的,不是事后挑的,口径逐字沿用。", "",
            "**MIMICRY** = acc(`impact_pair`) − acc(`impact_contrarian`)。`impact_pair` 是"
            "「高影响力那篇同时也拿了更好的评审分」的那一半,`impact_contrarian` 是"
            "「评审押错了」的那一半。只学会模仿评审口味的模型在前者高、后者为 0"
            "(照抄评审的 oracle 正好 100%/0%),**间隙越小,判断越是冲着工作本身去的**。", "",
            "| 尺度 | lens | base 间隙 | 我们的间隙 | 差(我们 − base) | 95% CI |",
            "|---|---|---|---|---|---|"]
    for scale, A, B in (("9B", "9B RL(SFT)", "9B base"), ("4B", "4B RL(SFT)", "4B base")):
        labs = [(A, TAG[A]), (B, TAG[B])]
        for lens, lname in (("strict", "strict(两臂)"), ("penalise", "penalise")):
            got = {}
            for t in ("impact_pair", "impact_contrarian"):
                keep = survivors(PP, "judge3", t, labs) if lens == "strict" else None
                a, b = j3_pair_acc(A, t, lens, keep), j3_pair_acc(B, t, lens, keep)
                ids = sorted(set(a) & set(b))
                got[t] = ([a[i][0] for i in ids], [b[i][0] for i in ids])
            va, ca = got["impact_pair"]
            vb, cb = got["impact_contrarian"]
            ga = sum(va) / len(va) - sum(vb) / len(vb)
            gb = sum(ca) / len(ca) - sum(cb) / len(cb)
            d, lo, hi = boot_gap(va, vb, ca, cb, rng)
            mark = "**" if d < 0 else ""
            out.append(f"| {scale} | {lname} | {100*gb:+.2f}% | {100*ga:+.2f}% | "
                       f"{mark}{100*d:+.2f}pp{mark} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    out += ["", "读法:4B 上我们的间隙**由正转负**(base +1.13%/+0.88% → 我们 −0.10%/−0.80%),"
            "即我们这条臂在「评审押错了」的那一半上不比在另一半差——按任务设计,这就是"
            "独立判断而不是模仿。CI 含 0,当趋势看。9B 上方向相反(我们的间隙略大),"
            "**这个现象只在 4B 出现**。", "",
            "**ORDER** = 把一对论文的两个呈现顺序交换之后,5 次抽样的多数票是否还指向同一篇。"
            "真在读论文的模型应该指向同一篇;这是与正确率无关的内部一致性检查。", "",
            "| 尺度 | 任务 | base | 我们 | 差 |", "|---|---|---|---|---|"]
    for scale, A, B in (("9B", "9B RL(SFT)", "9B base"), ("4B", "4B RL(SFT)", "4B base")):
        labs = [(A, TAG[A]), (B, TAG[B])]
        for t in ("impact_pair", "impact_contrarian", "novelty_pair"):
            keep = survivors(PP, "judge3", t, labs)
            a, b = j3_pair_acc(A, t, "strict", keep), j3_pair_acc(B, t, "strict", keep)
            ids = sorted(set(a) & set(b))
            ag = lambda m: 100 * np.mean([1.0 if (m[i][1] > 0.5) == (m[i][2] > 0.5) else 0.0
                                          for i in ids])
            x, y = ag(a), ag(b)
            mark = "**" if x > y else ""
            out.append(f"| {scale} | {t} | {y:.1f}% | {x:.1f}% | {mark}{x-y:+.1f}pp{mark} |")
    out += ["", "9B 上我们在两个 impact 任务上都更自洽,4B 上持平。"
            "一致性高不等于答得对(`novelty_pair` 上八条臂都在随机水平),"
            "所以这条只作为佐证,不单独成结论。", ""]
    return out


CONCLUSION = """
## 6. 这张表能说什么、不能说什么

每个数都在上面的表里,这一节只做归纳。

### 6.1 不能说的:在这三类 bench 上我们比 base 好

`RL(SFT) − base` 的四个聚合(两个尺度 × 两个 lens,都已去掉两个 ⚑ 格)是
**9B −1.04 / −0.76,4B −1.50 / −1.59**,方向一律偏负,一律不显著。
逐格看,8 个格子里我们占优的只有 3-4 个。**「我们的 research taste / idea 比 base 更好」
这句话,这批数据支持不了**,换哪个 lens 都一样。用户要有利指标,但这一条是没有的,
写进论文会被第一时间打掉。

### 6.2 能说的一:research judgment(judge3)上我们不输 base,方向一致偏我们

judge3 的两个活任务 × 两个尺度 × 两个 lens = 8 格,**7 格为正**,唯一的负格是 −0.0016。
四个 judge3-only 聚合 Z 全正(+0.86 ~ +1.09)。
但量级:这 8 格最大 +0.0152,中位 0.0044,**全部低于噪声底 0.0182**。
诚实的写法是「在研究判断力上与 base 持平,方向一致地略偏我们」,不是「更好」。

§3.2 那两个与正确率无关的刻画,给这句话补了两条同向的佐证,而且两条**分属不同尺度**:

- **9B 的顺序一致性全面更高**:交换论文呈现顺序后多数票仍指向同一篇的比例,
  三个任务分别 +2.4 / +2.0 / +2.7pp,方向一致。同一批题、同一套判定,base 更容易
  被呈现顺序带着走。
- **4B 的模仿间隙由正转负**:base +1.13% / +0.88%,我们 −0.10% / −0.80%
  (差 −1.23 / −1.68pp,CI 含 0)。在「评审押错了」的那一半上不掉分,按 j3stats
  对这个量的设计,这就是判断冲着工作本身去而不是照抄评审。

两条都是趋势(CI 含 0),而且**各只在一个尺度上出现**,所以只能当佐证,不能当结论。

### 6.3 能说的二:先验在 RL 阶段的作用,在 4B 的判断力上很硬

`4B RL(SFT) − 4B RL(base)`:judge3 两个任务、两个 lens **四格全部 CI 不含 0**
(strict +0.0341 / +0.0175,penalise +0.0396 / +0.0240),judge3-only Stouffer
**+3.79(strict)/ +4.99(penalise)**。这是全表最强的正结果,而且**不依赖 lens**
——也就是说它不是「对照臂不作答」换来的。

反过来,同一个对比在 `aaar_equation` 上 penalise 是 +0.16 / +0.18、strict 是
−0.054 / −0.030:**那一半完全是 4B RL(base) 的 36.8% 不作答撑起来的,不是判断力**。
把这两块分开报,是这张表最要紧的一件事(§29 的粒度规则)。

9B 上没有这个现象:`9B RL(SFT) − 9B RL(base)` 整体 +0.52 / +0.45,judge3 +0.05 / −0.00。

### 6.4 能说的三:SFT 伤判断力,RL 把它捞回来

`9B SFT − 9B base` 是全表最负的一组:8 格里只有 1-2 格为正,Stouffer **−2.22 / −2.10**,
judge3 两个任务 **0/2 正**(−1.82 / −1.79)。
而 `9B RL(SFT) − 9B SFT` 的 judge3-only 是 **+2.26(strict)/ +2.44(penalise)**,2/2 正,
两个 lens 都 p<0.05。合起来:**SFT 阶段付出了研究判断力的代价,RL 阶段把它补回来**,
补回来之后与 base 持平(§6.1/§6.2)。这条链路两端的数都在表里,方向一致。

### 6.5 §23 的那颗 ★ 要降级

§23 报 `ours − base4b` 在 `impact_contrarian` 上 ★ +2.49pp [+0.77, +4.31]。
那是**八臂 strict 存活集**下的数。换成被比较的那两条臂,同一个量是 +1.32pp
[−0.48, +3.13],CI 含 0;penalise 是 +1.52pp [−0.16, +3.24]。
方向稳,显著性不稳,而决定它的是一个与这两条臂无关的选择(4B RL(base) 答不出哪些题)。
**§23 的定性结论(增益在 contrarian 一侧、不在 pair 一侧)保留,那颗 ★ 撤。**

### 6.6 口径上顺带修掉的两件事

1. `taste_agg.py` 在 judge3 上按 `id` 配对,把一对题的两个呈现顺序当成两个独立观测,
   n 翻倍、区间缩窄。本表按 `meta.pair_id` 折成对(250/250/150),与 `j3stats.py` 一致。
2. `taste_agg.py` 只有一个隐式 lens(penalise)。本表两个都报,而且给出第三种
   存活集(八臂)做敏感性,因为 §23 的结论恰好取决于它。
"""

LENSES = [("strict", "strict(两臂)"), ("strict8", "strict(八臂)"), ("penalise", "penalise")]


def md_lens(out, floor_med, rng):
    out += ["## 3.1 存活集这个选择本身会改答案(§23 的 ★ 在这里被修正)", "",
            "`strict` 要求「每条臂都把 5 次抽样解析出来」,但**哪些臂**是一个选择。",
            "`j3stats.py` 对它拿到的 tag 取交集,所以传八条臂和传两条臂是两个不同的存活集。",
            "两者的差别不是小数点:八臂取交会丢掉「任意一条臂答不出」的题,而那批题偏难",
            "(j3stats 自己的注释就写了这会 flatter 控制臂)。",
            "4B RL(base) 在 aaar 上 36.8% 不作答、在 judge3 上 1.5%,它被算进交集时,",
            "会把与它无关的每一个对比都削一刀。", "",
            "下表把两个 strict 与 penalise 并排放。**§23 报的 `ours − base4b` contrarian",
            "★ +2.49pp [+0.77,+4.31] 是八臂存活集下的数;换成被比较的那两条臂,同一个量是",
            "+1.32pp,CI 含 0。** 效应的方向稳,那颗 ★ 不稳。", ""]
    for A, B in CONTRASTS:
        out += [f"**{A} − {B}**", "",
                "| 家族 | 任务 | " + " | ".join(f"Δ {lab} (n)" for _, lab in LENSES) + " |",
                "|---|---|" + "---|" * len(LENSES)]
        for fam, task in ORDER:
            row = [fam, task + (" ⚑" if (fam, task) in FLAGGED else "")]
            for lens, _ in LENSES:
                keep = (survivors(PP, fam, task, [(A, TAG[A]), (B, TAG[B])]) if lens == "strict"
                        else survivors(PP, fam, task) if lens == "strict8" else None)
                c = cell(acc(PP, fam, A, task, lens, keep),
                         acc(PP, fam, B, task, lens, keep), rng)
                if not c:
                    row.append("—")
                    continue
                star = "★" if (c["lo"] > 0 or c["hi"] < 0) else ""
                row.append(f"{c['mean']:+.4f}{star} ({c['n']})")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    out += ["八臂存活集在 `aaar_equation` 上只剩 25/27 题(4B RL(base) 削的),",
            "在那里它的数已经不该当结论用。正表(§3)因此一律用两臂存活集。", ""]
    return out


def main():
    rng = np.random.default_rng(SEED)
    fs, fp = floor_split(rng), floor_proto(rng)

    out = ["# taste / idea / research-judgment:我们 vs base 的全部指标", "",
           "数据源:`cc_idea32k_*_y26pp`、`cc_ideav2_*_y26pp`、`cc_judge3_*_y26pp`",
           "(2026-09-15 18:06-20:51 同一个投放窗口)。裸 tag 是 09-07 旧世代,",
           "除第 2 节的 F_proto 之外一律不混(§25)。", "",
           "judge3 的单位是**对**(`meta.pair_id`),两个呈现顺序折成一个观测——",
           "`j3stats.py` 的口径;按 `id` 算会把 n 翻倍、把区间缩窄。", "",
           "两个 lens 都报,定义逐字沿用 `j3stats.py`:`strict` = 八条臂全部把 5 次抽样都解析出来的题才算"
           "(judge3 还要两个顺序都存活);`penalise` = 不作答按答错计,一题不丢。", "",
           "⚑ = 该格保留在表里但不进聚合,原因在第 4 节。", ""]
    out = md_inventory(out)
    out, ss, sq = md_floor(out, fs, fp)
    floor_med = sq["med"]

    out += ["## 3. 主对比:我们 vs base", "",
            f"「过噪声底」一列拿 F_proto(parser 未动)的中位 |Δ| = {floor_med:.4f} 当门槛。",
            "`strict` 的存活集按**这一对臂**算,所以同一任务在不同对比下 n 可以不同;",
            "这正是 `j3stats.py` 只传两个 tag 时的行为。", ""]
    for A, B in CONTRASTS:
        for lens in ("strict", "penalise"):
            out, _ = contrast_block(out, A, B, lens, floor_med, rng)

    out = md_lens(out, floor_med, rng)
    out = md_judge_quality(out, rng)
    out += ["## 4. 两个 ⚑ 格为什么不进聚合", ""]
    for (fam, task), why in FLAGGED.items():
        out.append(f"- **`{fam}:{task}`** —— {why}")
    out += ["", "两格都留在逐格表里,没有藏。去掉它们前后的聚合都给了,读者可以自己取舍。", "",
            "## 5. 其余对比(同机器,供对照)", ""]
    for A, B in EXTRA:
        for lens in ("strict", "penalise"):
            out, _ = contrast_block(out, A, B, lens, floor_med, rng)

    out.append(CONCLUSION)
    txt = "\n".join(out) + "\n"
    here = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(here, "taste_vs_base.md"), "w").write(txt)
    sys.stdout.write(txt)


if __name__ == "__main__":
    main()
