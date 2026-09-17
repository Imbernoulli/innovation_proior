#!/usr/bin/env python3
"""The main results table with the full @5 family, recomputed on the current complete grids.

Why this exists: the original main report (experiments/EVAL_REPORT_v2_9B_zh.md section 3)
gave six numbers per arm x bench -- mean@5, 95% CI, best@5, worst@5, pass@5, 5/5>0 -- and
the paper digest's table had shrunk to mean@5 alone. Nothing was lost in the data; the
table just stopped carrying the columns. mean and best answer different questions
(average draw vs. can it be done at all) and the original report's own conclusions leaned
on best@5 and worst@5, so both belong in the paper.

Definitions are the original report's, verbatim:
  mean@5   expected score of one draw        = mean over problems of the draw mean
  best@5   the ceiling                       = mean over problems of the draw max
  worst@5  the floor                         = mean over problems of the draw min
  pass@5   coverage                          = share of problems with ANY draw scoring > 0
  5/5>0    reliability                       = share of problems where every draw scores > 0

One naming trap worth keeping straight: `stats2.py` also prints a column called pass@5,
but it defines pass as a FULL score (>= 100) on frontiercs and frontiercs_research, and
only uses "> 0" on alebench. That is a different quantity and it is much smaller (1-2%
against 30-60%). It is reported here as 满分@5, under its own name.

Rows come from dump2.load -- the paircommon dedupe rule, key (ground_truth, sample_idx),
error rows skipped, a later shard overwriting an earlier one -- so these numbers are
comparable with coverage_audit.md and report_tables.md rather than a private re-read.

Every statistic here weights problems equally: compute the draw statistic within a
problem, then average over problems. best@5, worst@5, pass@5 and 5/5>0 can only be
defined that way, so mean@5 follows them. This differs by a hair from the 4.468 the
digest carried for base9b_v2c on FrontierCS, which divided the surviving draws by the
nominal 860 and so imputed a score of 0 for the two cells the judge never scored
(problems 148 and 160, where the model did produce text and the judge timed out at
7200s). The three readings are 4.472 (per problem, used here), 4.479 (flat mean over the
858 scored draws) and 4.468 (missing imputed as 0). Only base9b_v2c is affected; every
other arm x bench has a full grid and all three agree.

Per section 27, every row prints the number of problems AND the number of draws behind it
before any statistic.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump2 import load, judge_meta  # noqa: E402

SLOW = 0.90   # 判题节点速度低于这个值就标 ⚠:11 条臂在 0.949-1.017,掉出来的那三条在 0.56-0.83

BOOT = 5000
SEED = 0
BENCHES = [("frontiercs", "FrontierCS"),
           ("alebench", "ALE-Bench"),
           ("frontiercs_research", "FCS-research")]
ARMS = [("base9b_v2c", "9B base"),
        ("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"),
        ("rlv5_ft01mix_a10_s20", "9B RL(SFT)"),
        ("base4b", "4B base"),
        ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"),
        ("rlv5_4b_ft01mix_a10_s20", "4B RL(SFT)")]
OTHER = [("ft03nm_a20", "9B SFT ft03nm"), ("lo32nm_a10", "9B SFT lo32nm"),
         ("rlv5_ft03nm_a20_s20", "9B RL ft03nm"), ("rlv5_lo32nm_a10_s20", "9B RL lo32nm"),
         ("4b_lo32nm_a10", "4B SFT lo32nm"), ("rlv5_4b_lo32nm_a10_s20", "4B RL lo32nm")]
FULL = 100 - 1e-9


def per_problem(arm, bench):
    """-> {problem: [scores]}; drops nothing except rows that have no score."""
    ok, _ = load(arm, bench)
    by = {}
    for (gt, _idx), v in ok.items():
        by.setdefault(gt, []).append(v["score"])
    return by


def stats(by):
    ps = sorted(by)
    if not ps:
        return None
    mean = np.array([float(np.mean(by[p])) for p in ps])
    best = np.array([float(max(by[p])) for p in ps])
    worst = np.array([float(min(by[p])) for p in ps])
    passed = np.array([1.0 if any(s > 0 for s in by[p]) else 0.0 for p in ps])
    allpos = np.array([1.0 if all(s > 0 for s in by[p]) else 0.0 for p in ps])
    solved = np.array([1.0 if any(s >= FULL for s in by[p]) else 0.0 for p in ps])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ps), size=(BOOT, len(ps)))
    bs = np.sort(mean[idx].mean(axis=1))
    return dict(n_prob=len(ps), n_draw=sum(len(by[p]) for p in ps),
                mean=mean.mean(), lo=float(bs[int(0.025 * BOOT)]), hi=float(bs[int(0.975 * BOOT)]),
                best=best.mean(), worst=worst.mean(),
                pass_=100 * passed.mean(), allpos=100 * allpos.mean(), solve=100 * solved.mean())


def fmt(bench, x):
    return f"{x:.1f}" if bench == "alebench" else f"{x:.3f}"


def speeds(arm, bench):
    """这条臂在这个 bench 上的判题节点速度因子。

    `gojudge_shim_v2._calibrate_node_speed` 把节点速度写进 judge_node_meta.json,
    但**只记录、不修正** —— NODE_CALIB 从来没有拿去缩放时限。FrontierCS 与
    ALE-Bench 都是按运行时间给分的,所以换一台慢节点判题,分数会被系统性压低,
    而且这件事在 samples.jsonl 里看不出来。凡是跨臂比分,先比这一列。
    """
    v = [m["speed"] for m in judge_meta(arm, bench) if m.get("speed")]
    parts = sorted({m["partition"] for m in judge_meta(arm, bench) if m.get("partition")})
    return v, parts


def emit(title, arms, out):
    for b, zh in BENCHES:
        out.append(f"### {zh}\n")
        out.append("| 臂 | n题 | n抽样 | mean@5 | 95% CI | best@5 | worst@5 | pass@5 | 5/5>0 | 满分@5 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for a, lab in arms:
            sp, _ = speeds(a, b)
            if sp and min(sp) < SLOW:
                lab = f"⚠ {lab}"
            s = stats(per_problem(a, b))
            if not s:
                out.append(f"| {lab} `{a}` | — | — | — | — | — | — | — | — | — |")
                continue
            out.append(f"| {lab} `{a}` | {s['n_prob']} | {s['n_draw']} | {fmt(b, s['mean'])} | "
                       f"[{fmt(b, s['lo'])}, {fmt(b, s['hi'])}] | {fmt(b, s['best'])} | {fmt(b, s['worst'])} | "
                       f"{s['pass_']:.1f}% | {s['allpos']:.1f}% | {s['solve']:.1f}% |")
        out.append("")


def main():
    out = ["# 主结果表:@5 全家(当前完整网格)", "",
           "定义见 `EVAL_REPORT_v2_9B_zh.md` §2.3,逐字沿用。",
           "`满分@5` 是 `stats2.py` 里那个也叫 pass@5 的量(FCS/research 上要求满分),",
           "换了名字并入这里,避免两份报告同名不同义。",
           f"CI 为逐题 bootstrap {BOOT} 次(seed={SEED})。",
           "",
           "**口径**:所有统计量一律逐题等权——先在题内对 5 次抽样求统计量,再对题求平均。",
           "`base9b_v2c` 的 FrontierCS 上此前论文稿写的是 4.468,那是把两个判题从未给分的格子",
           "(题 148、160,7200s 判题超时,正文是有的)当 0 计入分母 860;逐题等权是 4.472,",
           "只对已判的 858 行拉平是 4.479。仅此一臂受影响,其余格子网格全满、三种算法一致。", ""]
    out.append("## 8 条主臂\n")
    emit("main", ARMS, out)
    out.append("> **⚠ 标记 = 这一格的判题节点明显慢于其余臂**,见文末「判题节点速度核验」。"
               "FrontierCS 与 ALE-Bench 按运行时间给分,慢节点会系统性压低分数,"
               "而 speedFactor 只被记录、从不用来修正时限,所以带 ⚠ 的行**不能和别的臂直接比**。\n")
    out.append("## 其余 6 条臂(ft03nm / lo32nm 线,供 §18 选臂用)\n")
    emit("other", OTHER, out)
    out.append("## 判题节点速度核验(speedFactor:1.0 = ailab EPYC 基准)\n")
    out.append("`gojudge_shim_v2.py` 的 `_calibrate_node_speed()` 把节点速度写进每个 shard 的")
    out.append("`judge_node_meta.json`,但 `NODE_CALIB` **只落盘、从不拿去缩放时限**。")
    out.append("FrontierCS 与 ALE-Bench 共用同一批 `thinking_32k_both_vllm` 的 shard,")
    out.append("所以这两个 bench 的速度是同一个值;FCS-research 是单独的目录。\n")
    out.append("| 臂 | FrontierCS + ALE-Bench | 分区 | FCS-research | 分区 |")
    out.append("|---|---|---|---|---|")
    for a, lab in ARMS + OTHER:
        cells = []
        for b in ("frontiercs", "frontiercs_research"):
            sp, parts = speeds(a, b)
            rng = "—" if not sp else (f"{min(sp):.3f}" if min(sp) == max(sp)
                                      else f"{min(sp):.3f} – {max(sp):.3f}")
            if sp and min(sp) < SLOW:
                rng = f"**⚠ {rng}**"
            cells += [rng, "/".join(parts) or "—"]
        out.append(f"| {lab} `{a}` | " + " | ".join(cells) + " |")
    out.append("")
    out.append("**结论**:FCS-research 上 14 条臂全部落在 ailab 0.96–1.02,**这一层是干净的**")
    out.append("(而创新性/探索性的主表就是 FCS-research)。FrontierCS + ALE 上,")
    out.append("`ft03nm_a20`(0.82)、`lo32nm_a10`(0.79)、`rlv5_ft03nm_a20_s20`(**0.56**)")
    out.append("三条臂的判题跑在 cpu / gpu-ee 分区的慢节点上,与其余 11 条臂的 0.95–1.02 不可比。")
    out.append("要用它们的 FrontierCS / ALE 数,得先在同速节点上重判。\n")
    txt = "\n".join(out) + "\n"
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_table.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
