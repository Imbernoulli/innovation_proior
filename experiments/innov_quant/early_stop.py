#!/usr/bin/env python3
"""Split "never closed </think>" into its two causes: ran out of budget, or stopped early.

Section 22 reported that rlv5_4b_base_s20 closes </think> on only 2.1% of its FrontierCS
draws, against 63.7% for the arm that had the prior. Stated that way it reads as the same
failure every RL arm has -- generations that run to the 32768 cap -- only worse. It is
not the same failure.

A draw with no </think> is one of two things:
  cap    completion_tokens >= 32700: the model was still thinking when the budget ran out
  early  below the cap: the server stopped generating, i.e. the model emitted EOS while
         still inside its thinking block, and the text ends mid-sentence

Every other arm is essentially all `cap` (0.0-1.5% early). rlv5_4b_base_s20 is essentially
all `early`, with ZERO draws at the cap, at a median of roughly 1300 tokens. It does not
run out of room to think; it stops talking. That is the failure the prior prevents, and it
is a categorical difference, not a matter of degree.

Rows come from dump2.load so the 9B research shard duplicates are deduped (key is
(ground_truth, sample_idx), later shard wins) -- reading the jsonl directly counts them
twice and inflates the research denominators.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump2 import load, TRUNC_TOK  # noqa: E402

BENCHES = [("frontiercs", "FrontierCS"), ("alebench", "ALE"), ("frontiercs_research", "FCS-research")]
ARMS = [("base9b_v2c", "9B base"), ("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"), ("rlv5_ft01mix_a10_s20", "9B RL(SFT)"),
        ("base4b", "4B base"), ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20", "4B RL(SFT)")]


def row(arm, bench):
    ok, _ = load(arm, bench)
    n = len(ok)
    if not n:
        return None
    inc = [v for v in ok.values() if "</think>" not in (v["text"] or "")]
    ct = np.array([v["completion_tokens"] or 0 for v in inc], float)
    cap = int((ct >= TRUNC_TOK).sum()) if len(ct) else 0
    early = len(inc) - cap
    med = float(np.median(ct[ct < TRUNC_TOK])) if early else None
    return dict(n=n, inc=len(inc), cap=cap, early=early,
                p_early=100 * early / n, med=med)


def main():
    out = ["# 「没闭合 `</think>`」拆成两种原因", "",
           "`cap` = `completion_tokens >= %d`,想完还没完就没预算了;" % TRUNC_TOK,
           "`early` = 没到上限就停了,即模型在思考块内部吐了结束符,正文断在句子中间。", "",
           "去重走 `dump2.load`(键 =(ground_truth, sample_idx),后出现的覆盖先出现的),",
           "否则 9B 的 research 分片重复会把分母撑大。", ""]
    for b, zh in BENCHES:
        out.append(f"### {zh}\n")
        out.append("| 臂 | n | 未闭合 | 撞上限 | 早停 | 早停占全体 | 早停长度中位 |")
        out.append("|---|---|---|---|---|---|---|")
        for a, lab in ARMS:
            r = row(a, b)
            if not r:
                out.append(f"| {lab} | — | — | — | — | — | — |")
                continue
            med = "—" if r["med"] is None else f"{r['med']:.0f}"
            bold = "**" if r["p_early"] > 50 else ""
            out.append(f"| {lab} `{a}` | {r['n']} | {r['inc']} | {r['cap']} | {bold}{r['early']}{bold} | "
                       f"{bold}{r['p_early']:.1f}%{bold} | {med} |")
        out.append("")
    txt = "\n".join(out) + "\n"
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "early_stop.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
