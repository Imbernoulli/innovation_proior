# -*- coding: utf-8 -*-
"""MLS-Bench:**lora 线**写的方法,是不是题目给的 baselines 的拼接?

起因(用户 2026-09-18:「mls 对 baselines 的 recombination 呢」)。
`baseline_recomb.py` 的 BASE 里只有 ft01mix 那八条臂,**lora 线从来没进过这张表** ——
和 MLS 分数表当初的毛病一样(见 `mls_lora.py`)。这个脚本不改那一份,
只把同一套口径套到 12 条臂上(原八条 + lora 四条),好让 lora 能和它的对照直接比。

口径全部复用 `baseline_recomb.py` 的函数(词表、加行提取、Jaccard),
所以**这张表和 `baseline_recomb_p1.md` 是同一把尺子**,只有两处必须说清楚:

  1. `sim_resid` 是 sim 对 log(1+ntok) 做回归后的残差。回归池从 8 臂扩到 12 臂,
     拟合系数会动一点,所以本表的 sim_resid 与 `baseline_recomb_p1.md` 的**不逐位相同**。
     §1 把两套系数并排打出来,让这点差异是看得见的,不是静默的(§27)。
  2. 其余各列(n_base / n_hard / sim / sim_tmpl / novel / 动手率)与 8 臂表**逐位可对**,
     因为它们是逐格算的,不依赖池子。

方向约定:这套指标**低 = 好**(越不贴 baseline 越好),只有 `novel` 是高 = 好。
`sim_tmpl` 和 `ntok` 是对照项,不是结论。

★ §39 半成品防护:`mls21-<arm>_p1` 还在 squeue 里就整臂跳过。

用法:baseline_recomb_lora.py  → 打印并写 baseline_recomb_lora.md
"""
import io
import os
import sys

sys.argv = [sys.argv[0], "_p1"]          # 锁死协议 p1(铁律 k),别让外面的 argv 改它

import numpy as np
from scipy import stats

import baseline_recomb as BR

HERE = os.path.dirname(os.path.abspath(__file__))
SUF = "_p1"

OLD = [("base9b_v2c", "9B base"), ("ft01mix_a10", "9B SFT"),
       ("rlv5_base_s20", "9B RL(base)"), ("rlv5_ft01mix_a10_s20", "9B RL 我们"),
       ("base4b", "4B base"), ("4b_ft01mix_a10", "4B SFT"),
       ("rlv5_4b_base_s20", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20", "4B RL 我们")]
LORA = [("lo32nm_a10", "9B SFT lora"), ("rlv5_lo32nm_a10_s20", "9B RL lora"),
        ("4b_lo32nm_a10", "4B SFT lora"), ("rlv5_4b_lo32nm_a10_s20", "4B RL lora")]
BASE12 = OLD + LORA

BR.ARMS = [(b + SUF, lab) for b, lab in BASE12]
TAG = {lab: b + SUF for b, lab in BASE12}

# 只比 RL 之后的臂(铁律);预训练 base 作旁证,单独一行。
# 最后两行是**控制行,不是结论行**:如果 lora 相对预训练 base 贴基线更紧,而 RL(base)
# 相对预训练 base 也一样紧,那就是 RL 本身的效应,不是 lora 的。没有这两行,
# 「lora 更抄 baseline」这句话就无法证伪。
CONTR = [(TAG["9B RL lora"], TAG["9B RL(base)"], "9B RL lora − RL(base)"),
         (TAG["9B RL lora"], TAG["9B RL 我们"], "9B RL lora − RL 我们"),
         (TAG["9B RL lora"], TAG["9B base"], "9B RL lora − 预训练 base(旁证)"),
         (TAG["4B RL lora"], TAG["4B RL(base)"], "4B RL lora − RL(base)"),
         (TAG["4B RL lora"], TAG["4B RL 我们"], "4B RL lora − RL 我们"),
         (TAG["4B RL lora"], TAG["4B base"], "4B RL lora − 预训练 base(旁证)"),
         (TAG["9B RL(base)"], TAG["9B base"], "【控制】9B RL(base) − 预训练 base"),
         (TAG["4B RL(base)"], TAG["4B base"], "【控制】4B RL(base) − 预训练 base")]

METRICS = [("n_hard", "真 import/调用的 baseline 条数", "低"),
           ("n_base", "点到的 baseline 条数,含注释", "低"),
           ("sim", "与参考实现的最大 Jaccard", "低"),
           ("sim_excess", "sim − sim_tmpl:超出惯用法的那部分相似", "低"),
           ("sim_resid", "sim 对代码长度回归后的残差 ★这条才是干净的", "低"),
           ("novel", "不含 baseline 名的实义行占比", "高"),
           ("sim_tmpl", "与空脚手架模板的 Jaccard(对照项)", "—"),
           ("ntok", "写进去的不同标识符数(对照项)", "—")]

_BUF = []


def emit(s=""):
    print(s)
    _BUF.append(s)


def running():
    import subprocess
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set()
    return {ln.strip()[len("mls21-"):] for ln in out.splitlines()
            if ln.strip().startswith("mls21-")}


def fit(rows, arms, valid):
    """sim ~ a + b·log(1+ntok),只在「动手了且有 token」的格子上拟合。"""
    pool = [(a, t) for a, _ in arms for t in BR.TASKS
            if valid[(a, t)] and rows[(a, t)]["acted"] and rows[(a, t)]["ntok"] > 0]
    X = np.log1p([rows[k]["ntok"] for k in pool])
    Y = np.array([rows[k]["sim"] for k in pool])
    b, a0 = np.polyfit(X, Y, 1)
    r = stats.pearsonr(X, Y)
    return a0, b, len(pool), r[0], r[1]


def main():
    live = running()
    arms = [(a, lab) for a, lab in BR.ARMS if a not in live]
    skipped = [lab for a, lab in BR.ARMS if a in live]
    BR.ARMS = arms

    # main() 会打 §0 的每题 baseline 清单(和 baseline_recomb_p1.md §0 一模一样),不重复打。
    buf, sys.stdout = sys.stdout, io.StringIO()
    try:
        rows, vocab, hitc = BR.main()
    finally:
        sys.stdout = buf

    st = {a: BR.status_map(a) for a, _ in arms}
    valid = {(a, t): st[a].get(t) != "agent_failed" for a, _ in arms for t in BR.TASKS}

    emit("# MLS:方法是不是给定 baseline 的重组 —— **把 lora 线补进来**\n")
    emit("协议 `p1`,分母口径与 `baseline_recomb_p1.md` 同一把尺子。"
         "**低 = 好**(越不贴 baseline 越好),只有 `novel_frac` 高 = 好。\n")
    if skipped:
        emit(f"> ⚠ 跳过(作业还在跑):{', '.join(skipped)}\n")

    a12, b12, n12, r12, p12 = fit(rows, arms, valid)
    old = [(a, lab) for a, lab in arms if lab in dict(
        (l, b) for b, l in OLD)]
    a8, b8, n8, r8, p8 = fit(rows, old, valid)
    emit("## 1. 长度混淆的那条回归(sim_resid 的来源)\n")
    emit("Jaccard 的分母是并集 —— **写得少,分母小,sim 天然偏高**。所以 sim 必须对代码长度校正。\n")
    emit("| 回归池 | n 格 | 截距 | 斜率 | Pearson r | p |")
    emit("|---|---:|---:|---:|---:|---:|")
    emit(f"| 本表(12 臂) | {n12} | {a12:+.4f} | {b12:+.4f} | {r12:+.3f} | {p12:.2g} |")
    emit(f"| `baseline_recomb_p1.md`(原 8 臂) | {n8} | {a8:+.4f} | {b8:+.4f} | {r8:+.3f} | {p8:.2g} |")
    emit("\n两套系数几乎重合,但**不完全相同**,所以本表的 `sim_resid` 与 8 臂表不逐位相等;"
         "其余各列是逐格算的,两表逐位可对。\n")

    for k in rows:
        rows[k]["sim_resid"] = float("nan")
    for a, _ in arms:
        for t in BR.TASKS:
            r = rows[(a, t)]
            if valid[(a, t)] and r["acted"] and r["ntok"] > 0:
                r["sim_resid"] = r["sim"] - (a12 + b12 * np.log1p(r["ntok"]))

    emit("## 2. 逐臂\n")
    emit("| arm | 有效题 | 动手率 | n_base | n_hard | P(n_hard≥1) | P(n_hard≥2) | "
         "sim_max | sim_tmpl | sim_resid | novel_frac | 新增行 |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    per = {}
    for a, lab in arms:
        ts = [t for t in BR.TASKS if valid[(a, t)]]
        act = [rows[(a, t)] for t in ts if rows[(a, t)]["acted"]]
        per[a] = ts
        m = lambda k: (np.nanmean([x[k] for x in act]) if act else float("nan"))
        star = " **←**" if "lora" in lab else ""
        emit(f"| {lab}{star} | {len(ts)} | {len(act)/max(1,len(ts)):.2f} | "
             f"{m('n_base'):.2f} | {m('n_hard'):.2f} | "
             f"{np.mean([x['n_hard']>=1 for x in act]) if act else float('nan'):.2f} | "
             f"{np.mean([x['n_hard']>=2 for x in act]) if act else float('nan'):.2f} | "
             f"{m('sim'):.3f} | {m('sim_tmpl'):.3f} | {m('sim_resid'):+.3f} | "
             f"{m('novel'):.3f} | {m('n_add'):.0f} |")
    emit("\n`动手率` = 21 题里有 edit 动作的比例。**没动手 = 直接交默认脚手架,"
         "方法就是那条默认 baseline** —— 这是「重组」的最强形态。后面各列只在动手了的格子上算。\n")

    emit("## 3. 配对对照(同题配对,两臂都动手的题才进)\n")
    onegl = {}
    for key, desc, good in METRICS:
        emit(f"### {key} — {desc}({good} = 好)\n")
        emit("| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |")
        emit("|---|---:|---:|:---:|---:|---:|")
        for A, B, lab in CONTR:
            pa, pb = [], []
            for t in BR.TASKS:
                if not (valid.get((A, t)) and valid.get((B, t))):
                    continue
                ra, rb = rows[(A, t)], rows[(B, t)]
                if not (ra["acted"] and rb["acted"]):
                    continue
                va, vb = ra[key], rb[key]
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                pa.append(va); pb.append(vb)
            n = len(pa)
            if n < 3:
                emit(f"| {lab} | {n} | — | — | — | ⚠样本不足 |")
                continue
            d = np.array(pa) - np.array(pb)
            pos, neg = int((d > 0).sum()), int((d < 0).sum())
            sp = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
            try:
                wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except Exception:
                wp = float("nan")
            onegl[(key, lab)] = (d.mean(), n, wp, good)
            emit(f"| {lab} | {n} | {d.mean():+.3f} | {pos}/{neg} | {sp:.4f} | {wp:.4f} |")
        emit()

    emit("## 4. 最强形态:一行没改就提交(方法**就是**默认 baseline)\n")
    emit("| arm | 有效题 | 没动手题数 | 没动手的题 |")
    emit("|---|---:|---:|---|")
    for a, lab in arms:
        nz = [t for t in per[a] if not rows[(a, t)]["acted"]]
        emit(f"| {lab} | {len(per[a])} | {len(nz)} | {', '.join(nz) if nz else '—'} |")
    emit()

    emit("## 5. 一眼表:lora 在「不抄 baseline」上比对照好还是差\n")
    emit("每格 = Δ均值,并按该指标的方向翻译成 **好 / 差**。★ = Wilcoxon p < 0.05。\n")
    emit("| 指标 | " + " | ".join(l for _, _, l in CONTR) + " |")
    emit("|---|" + "---:|" * len(CONTR))
    for key, _, good in METRICS:
        if good == "—":
            continue
        cells = []
        for _, _, lab in CONTR:
            v = onegl.get((key, lab))
            if not v:
                cells.append("—"); continue
            d, n, wp, g = v
            better = (d < 0) if g == "低" else (d > 0)
            cells.append(f"{d:+.3f} {'好' if better else '差'}" + ("★" if wp < 0.05 else ""))
        emit(f"| {key} | " + " | ".join(cells) + " |")
    emit()

    emit("## 6. 读法\n")
    emit("**(a) 对 RL 之后的两条对照,lora 在任何一个重组指标上都没有可测差异。** "
         "12 个格子(2 尺寸 × 2 对照 × 6 指标 的前四列)没有一颗 ★。"
         "干净的那条 `sim_resid` 四格全是「好」方向(−0.004 ~ −0.008),但都远不显著。"
         "**结论是「打平」,不是「更好」** —— 想说 lora 更不抄 baseline,这份数据给不了。\n")
    emit("**(b) 相对预训练 base 贴基线更紧,是 RL 的效应,不是 lora 的。** "
         "9B RL lora − 预训练 base 的 `sim_resid` = +0.084 ★,看着像个坏消息;"
         "但控制行 9B RL(base) − 预训练 base = **+0.096 ★**,比 lora 还大。"
         "也就是说 RL 本身会让代码更贴题目给的参考实现,而**lora 是这两条 RL 臂里受影响较小的那条**。"
         "这和已记录的老结论一致(RL 后代码更贴基线、更短、不更新颖)。\n")
    emit("**(c) 最强形态那一列要分尺寸看。** 「一行没改就提交」= 方法直接就是默认 baseline:"
         "4B RL lora **2/21**,与 4B RL(base) 并列全场最好,优于 4B RL 我们(4/21);"
         "而 9B RL lora **6/21** 是三条 9B RL 臂里最差的(RL(base) 3、我们 2)。"
         "这一条对 9B 不利,照记。\n")


main()
with open(os.path.join(HERE, "baseline_recomb_lora.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
