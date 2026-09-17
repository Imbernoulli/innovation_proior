# -*- coding: utf-8 -*-
"""MLS-21(p1,分母 21):把 ft03nm / lora 两条线也放进来。

起因(用户 2026-09-17:「你就给我 lora 这个给全吧」「不全的修一下」)。
`mls_align_score.py` / `mls_filestate.py` 的 ARMS 都只写了 ft01mix 线的八条臂,
所以 lora 线明明有 p1 的落盘数据,却从来没有出现在任何一张 MLS 表里。
这个脚本不改那两份(它们各自是 al1↔p1 的 A/B 与 file-state 口径,换臂会改变它们的存活集),
只补一张 as-run / p1 / 分母 21 的逐臂表。

口径三条,与 §33 一致:
  1. 分数取 summary.json 每题的 `score`,**不是 `mean_score`**(后者分母是 n_scored)。
  2. **分母恒为 21**。真 0 记 0;题目缺失也记 0 并单列出来 —— 不能靠少算题把均分做高。
  3. 合并 `<tag>-fix` 补跑,同名题以 -fix 为准,与 year_grid.py / mls_audit21.py 一致。
"""
import json, os, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
N = 21
ARMS = [("9B base", "base9b_v2c"), ("9B SFT", "ft01mix_a10"),
        ("9B SFT ft03nm", "ft03nm_a20"), ("9B SFT lora", "lo32nm_a10"),
        ("9B RL(base)", "rlv5_base_s20"), ("9B RL 我们", "rlv5_ft01mix_a10_s20"),
        ("9B RL ft03nm", "rlv5_ft03nm_a20_s20"), ("9B RL lora", "rlv5_lo32nm_a10_s20"),
        ("4B base", "base4b"), ("4B SFT", "4b_ft01mix_a10"),
        ("4B SFT lora", "4b_lo32nm_a10"),
        ("4B RL(base)", "rlv5_4b_base_s20"), ("4B RL 我们", "rlv5_4b_ft01mix_a10_s20"),
        ("4B RL lora", "rlv5_4b_lo32nm_a10_s20")]


def as_run(tag):
    out = {}
    for d in (f"{D}/outputs/cc_mls21_{tag}", f"{D}/outputs/cc_mls21_{tag}-fix"):
        p = f"{d}/summary.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        ts = j.get("tasks", j)
        ts = ts if isinstance(ts, list) else list(ts.values())
        for t in ts:
            out[t["task"]] = t.get("score")
    return out


B = []
def emit(s=""):
    print(s); B.append(s)


def main():
    suf = sys.argv[1] if len(sys.argv) > 1 else "p1"
    emit(f"# MLS-21(as-run,协议 `{suf}`,分母恒 {N})—— 含 ft03nm / lora 两条线\n")
    emit("`跑出题` = summary.json 里有这道题;缺的题按 **0** 计入分母,不缩分母。\n")
    emit("| 臂 | 跑出题/21 | 非零题 | **均分(/21)** | 只按跑出题算 |")
    emit("|---|---:|---:|---:|---:|")
    got = {}
    for lbl, a in ARMS:
        s = as_run(f"{a}_{suf}")
        if not s:
            emit(f"| {lbl} `{a}` | **没有 `{suf}` 跑** | — | — | — |")
            continue
        v = [x for x in s.values() if x is not None]
        tot = sum(v)
        got[a] = tot / N
        emit(f"| {lbl} `{a}` | {len(s)} | {sum(1 for x in v if x > 0)} | "
             f"**{tot/N:.4f}** | {tot/max(1,len(v)):.4f} |")
    emit()
    emit("## 对照(只看 RL 之后)\n")
    emit("| 对照 | Δ 均分(/21) |")
    emit("|---|---:|")
    for lo, hi, lbl in [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
                        ("rlv5_base_s20", "rlv5_lo32nm_a10_s20", "9B lora − RL(base)"),
                        ("rlv5_base_s20", "rlv5_ft03nm_a20_s20", "9B ft03nm − RL(base)"),
                        ("rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20", "9B lora − 我们"),
                        ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)"),
                        ("rlv5_4b_base_s20", "rlv5_4b_lo32nm_a10_s20", "4B lora − RL(base)"),
                        ("rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20", "4B lora − 我们")]:
        if lo in got and hi in got:
            emit(f"| {lbl} | {got[hi]-got[lo]:+.4f} |")
        else:
            emit(f"| {lbl} | 缺 `{hi if hi not in got else lo}` 的 {suf} 跑 |")
    emit()


main()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mls_lora.md"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(B) + "\n")
