# -*- coding: utf-8 -*-
"""MLS 采样对齐 A/B:`_al1`(对齐)vs `_p1`(未对齐)。

§35 查出 MLS 从来没按评测采样协议跑过——那条请求路径一个采样参数都不发,
所以每一个 MLS 数都是在 serve 默认值下产生的。`_al1` 批把协议补上
(temp=1.0 top_p=0.95 top_k=20 min_p=0.0 pp=1.5 rep=1.0),其余 env 逐项与 p1 相同
(已从运行进程 /proc/<pid>/environ 直读确认,见 mls21_aligned_submit.sh 头部)。

要回答的问题:§33 把 MLS 的低分主要归因于「模型不吐 action」(419 格里 143 格
score=0 且 settings==[],其中 65% 以 `No action returned after 3 attempts` 收尾),
并猜这是 pp=1.5 本该压住的失控重复。**pp 补上之后,那批格子救回来了吗?**

口径:
  - 逐题配对,只取两边都写出 `[done]` 的题(§27:先打印 n 再看效应量)
  - 任一侧 >=3 个 APIConnectionError 的臂级格子剔除(serve 死了,基础设施无效)
  - 每题三个互斥观测:无action停 / edit 动作数 / 总步数(§29:别把失败糊成一个数)
  - 符号检验 + Wilcoxon;臂没跑完就标 ⚠部分,不进结论
"""
import os, re, sys, glob
from scipy import stats

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"^Step\s+(\d+)\s+(\w+)\s*$", re.M)
ARMS = [("base9b_v2c", "9B base"), ("ft01mix_a10", "9B SFT"),
        ("rlv5_base_s20", "9B RL(base)"), ("rlv5_ft01mix_a10_s20", "9B RL(先验)"),
        ("base4b", "4B base"), ("4b_ft01mix_a10", "4B SFT"),
        ("rlv5_4b_base_s20", "4B RL(base)"), ("rlv5_4b_ft01mix_a10_s20", "4B RL(先验)")]
METRICS = [("noact", "无action停(低=好)"), ("nedit", "edit 动作数(高=好)"),
           ("nstep", "总步数(高=好)")]


def feats(p):
    s = ANSI.sub("", open(p, encoding="utf-8", errors="replace").read())
    steps = STEP.findall(s)
    return dict(noact=int("No action returned after 3 attempts" in s),
                conn=len(re.findall("APIConnectionError", s)),
                nedit=sum(1 for _, k in steps if k == "edit"),
                nstep=len(steps), done=int("[done]" in s))


def pair(arm):
    A = f"{D}/outputs/cc_mls21_{arm}_al1/task_logs"
    P = f"{D}/outputs/cc_mls21_{arm}_p1/task_logs"
    rows, na, np_ = [], 0, 0
    for f in sorted(glob.glob(A + "/*.log")):
        t = os.path.basename(f)[:-4]
        g = f"{P}/{t}.log"
        if not os.path.exists(g):
            continue
        a, b = feats(f), feats(g)
        na += a["done"]; np_ += b["done"]
        if a["done"] and b["done"]:
            rows.append((t, a, b))
    dead = sum(a["conn"] >= 3 or b["conn"] >= 3 for _, a, b in rows)
    rows = [r for r in rows if not (r[1]["conn"] >= 3 or r[2]["conn"] >= 3)]
    return rows, na, np_, dead


def main():
    print("# MLS 采样对齐 A/B:`_al1`(协议对齐)− `_p1`(未对齐)\n")
    print("两批除采样外 env 逐项相同(已从 /proc/<pid>/environ 直读确认)。\n")
    print("## 0. 每条臂进入分析的题数(§27:先看分母)\n")
    print("| arm | al1 完成 | p1 完成 | 配对 | serve死格剔除 | 状态 |")
    print("|---|---|---|---|---|---|")
    keep = {}
    for arm, lab in ARMS:
        rows, na, np_, dead = pair(arm)
        st = "完整" if (na >= 21 and np_ >= 21) else f"⚠部分(al1 {na}/21)"
        print(f"| {lab} | {na} | {np_} | {len(rows)} | {dead} | {st} |")
        keep[arm] = (rows, st.startswith("完整"))
    for key, lab in METRICS:
        print(f"\n## {lab}\n")
        print("| arm | n | al1 | p1 | Δ | +/− | 符号 p | Wilcoxon p | |")
        print("|---|---|---|---|---|---|---|---|---|")
        for arm, albl in ARMS:
            rows, full = keep[arm]
            if len(rows) < 3:
                print(f"| {albl} | {len(rows)} | — | — | — | — | — | — | ⚠样本不足 |")
                continue
            da = [a[key] for _, a, b in rows]; db = [b[key] for _, a, b in rows]
            d = [x - y for x, y in zip(da, db)]
            pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
            sp = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
            try:
                wp = stats.wilcoxon(d).pvalue if any(d) else 1.0
            except Exception:
                wp = float("nan")
            flag = "" if full else "⚠部分,不进结论"
            print(f"| {albl} | {len(rows)} | {sum(da)/len(da):.2f} | {sum(db)/len(db):.2f} | "
                  f"{sum(d)/len(d):+.2f} | {pos}/{neg} | {sp:.4f} | {wp:.4f} | {flag} |")


if __name__ == "__main__":
    main()
