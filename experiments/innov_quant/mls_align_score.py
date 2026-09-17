# -*- coding: utf-8 -*-
"""MLS 采样对齐 A/B —— **分数**。

上一版 `mls_align_ab.py` 报了 edit 动作数和总步数。那两列不是结果:动作多不等于做得好,
「高=好」这个标注本身就没依据。保留「无 action 停」是因为 §33 要拆「没动手 vs 做得差」,
一个指标就够了。这个脚本只看真正要回答的事:

  Q1 对齐把每条臂的分数抬了还是压了?(逐题配对)
  Q2 **对齐之后,ours − base 变了吗?** —— 主表结论会不会因为协议修复而翻。

口径:
  - 分数取 summary.json 每题的 `score`。**不要用 `mean_score`,它的分母是 `n_scored`**。
  - 逐题配对;互斥剔除:agent 没被问过 / 撞墙钟 / serve 死(≥3 APIConnectionError)。
  - Q2 的存活集是「四条臂在该协议下都有分」的公共题(§31.5:存活集是个选择,写明白)。
"""
import os, re, json, glob
from scipy import stats

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ARMS = [("base9b_v2c", "9B base"), ("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"), ("rlv5_ft01mix_a10_s20", "9B RL(先验)"),
        ("base4b", "4B base"), ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20", "4B RL(先验)")]
G9 = [a for a, _ in ARMS[:4]]
G4 = [a for a, _ in ARMS[4:]]


def load(arm, suf):
    """-> (是否跑完, {task: score or None})。None = 该题无效,不能进配对。"""
    p = f"{D}/outputs/cc_mls21_{arm}_{suf}/summary.json"
    if not os.path.exists(p):
        return False, {}
    d = json.load(open(p))
    ts = d.get("tasks", d)
    if isinstance(ts, dict):
        ts = list(ts.values())
    out = {}
    for t in ts:
        n = t.get("task")
        st = t.get("status") or ""
        if "agent_failed" in st or "timeout" in st:
            out[n] = None
            continue
        lg = t.get("log") or f"{D}/outputs/cc_mls21_{arm}_{suf}/task_logs/{n}.log"
        try:
            s = ANSI.sub("", open(lg, encoding="utf-8", errors="replace").read())
            if len(re.findall("APIConnectionError", s)) >= 3:
                out[n] = None
                continue
        except OSError:
            pass
        sc = t.get("score")
        out[n] = float(sc) if sc is not None else None
    return len(ts) >= 21, out


def test(d):
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
    sp = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
    try:
        wp = stats.wilcoxon(d).pvalue if any(x != 0 for x in d) else 1.0
    except Exception:
        wp = float("nan")
    return pos, neg, sp, wp


def main():
    AL = {a: load(a, "al1") for a, _ in ARMS}
    P1 = {a: load(a, "p1") for a, _ in ARMS}

    print("# MLS 采样对齐 A/B —— 分数\n")
    print("分数取 summary.json 每题的 `score`(**不是 `mean_score`,它的分母是 `n_scored`**)。\n")
    print("## 1. 对齐本身把分数抬了吗(逐臂,同题配对)\n")
    print("| arm | 作业 | 配对 n | al1 均分 | p1 均分 | Δ | +/− | 符号 p | Wilcoxon p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, lab in ARMS:
        ran_a, sa = AL[arm]; ran_p, sp_ = P1[arm]
        ts = [t for t in sa if sa.get(t) is not None and sp_.get(t) is not None]
        if len(ts) < 3:
            print(f"| {lab} | {'已结束' if ran_a else '⚠在跑'} | {len(ts)} | — | — | — | — | — | — |")
            continue
        a = [sa[t] for t in ts]; b = [sp_[t] for t in ts]
        d = [x - y for x, y in zip(a, b)]
        pos, neg, s, w = test(d)
        tag = "" if ran_a else " ⚠在跑"
        print(f"| {lab}{tag} | {'已结束' if ran_a else '未结束'} | {len(ts)} | {sum(a)/len(a):.4f} | "
              f"{sum(b)/len(b):.4f} | {sum(d)/len(d):+.4f} | {pos}/{neg} | {s:.4f} | {w:.4f} |")

    print("\n## 2. ★真正的问题:对齐之后 ours − base 变了吗\n")
    CONTR = [("rlv5_ft01mix_a10_s20", "base9b_v2c", "9B RL(先验) − base"),
             ("rlv5_ft01mix_a10_s20", "rlv5_base_s20", "9B RL(先验) − RL(base)"),
             ("rlv5_ft01mix_a10_s20", "ft01mix_a10", "9B RL(先验) − SFT"),
             ("rlv5_4b_ft01mix_a10_s20", "base4b", "4B RL(先验) − base"),
             ("rlv5_4b_ft01mix_a10_s20", "rlv5_4b_base_s20", "4B RL(先验) − RL(base)"),
             ("rlv5_4b_ft01mix_a10_s20", "4b_ft01mix_a10", "4B RL(先验) − SFT")]
    for suf, S in (("p1", P1), ("al1", AL)):
        print(f"\n### 协议 = `{suf}`\n")
        print("| 对照 | 两臂公共题 n | ours | base | Δ | +/− | 符号 p | Wilcoxon p |")
        print("|---|---|---|---|---|---|---|---|")
        for A, B, lab in CONTR:
            _, sa = S[A]; _, sb = S[B]
            ts = [t for t in sa if sa.get(t) is not None and sb.get(t) is not None]
            if len(ts) < 3:
                print(f"| {lab} | {len(ts)} | — | — | — | — | — | ⚠不足 |")
                continue
            a = [sa[t] for t in ts]; b = [sb[t] for t in ts]
            d = [x - y for x, y in zip(a, b)]
            pos, neg, s, w = test(d)
            print(f"| {lab} | {len(ts)} | {sum(a)/len(a):.4f} | {sum(b)/len(b):.4f} | "
                  f"{sum(d)/len(d):+.4f} | {pos}/{neg} | {s:.4f} | {w:.4f} |")
    print("\n存活集(§31.5):第 2 节每一行只在**该对照的两条臂**上取交,不是八臂取交;n 已逐行列出。")


if __name__ == "__main__":
    main()
