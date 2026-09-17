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
# 2026-09-16 用户指出:动作数不是结果,「动作多=好」没依据。分数在 mls_align_score.py。
# 这里只留「无 action 停」,因为 §33 要拆的是「没动手 vs 做得差」,那一个指标就够了。
# nedit/nstep 保留为诊断项,明确标注「非结果指标」—— 事实上 9B base 在 al1 下动作数
# 显著上升而分数显著下降(−0.0707,Wilcoxon 0.0357),正好说明这两列不能当好坏读。
METRICS = [("noact", "无action停(低=好)"),
           ("nedit", "edit 动作数(诊断项,非结果指标)"),
           ("nstep", "总步数(诊断项,非结果指标)")]


def feats(p):
    s = ANSI.sub("", open(p, encoding="utf-8", errors="replace").read())
    steps = STEP.findall(s)
    return dict(noact=int("No action returned after 3 attempts" in s),
                conn=len(re.findall("APIConnectionError", s)),
                nedit=sum(1 for _, k in steps if k == "edit"),
                nstep=len(steps), done=int("[done]" in s))


def status(arm, suf):
    """summary.json 的 per-task status。这才是「这一格能不能用」的判据:
    `[done]` 这行 agent_failed / timeout+scored 的题根本不写,拿它当完成判据会把
    已结束的作业误判成还在跑(2026-09-16 21:14 踩到)。"""
    import json
    p = f"{D}/outputs/cc_mls21_{arm}_{suf}/summary.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ts = d.get("tasks", d)
    if isinstance(ts, dict):
        ts = list(ts.values())
    # 合并补跑:同名题以 <tag>-fix 里的为准,口径与 year_grid.py / mls_audit21.py 一致。
    fp = f"{D}/outputs/cc_mls21_{arm}_{suf}-fix/summary.json"
    if os.path.exists(fp):
        fd = json.load(open(fp)); fts = fd.get("tasks", fd)
        fts = list(fts.values()) if isinstance(fts, dict) else fts
        byn = {t.get("task"): t for t in ts}
        byn.update({t.get("task"): t for t in fts})
        ts = list(byn.values())
    out = {}
    for t in ts:
        n = t.get("task") or t.get("task_name") or t.get("name")
        if n:
            out[n] = t.get("status")
    return out


def pair(arm):
    A = f"{D}/outputs/cc_mls21_{arm}_al1/task_logs"
    P = f"{D}/outputs/cc_mls21_{arm}_p1/task_logs"
    sa, sp = status(arm, "al1"), status(arm, "p1")
    ran = sa is not None and sp is not None and len(sa) >= 21 and len(sp) >= 21
    rows, drop = [], {"agent_failed": 0, "timeout": 0, "dead_serve": 0, "no_log": 0}
    for f in sorted(glob.glob(A + "/*.log")):
        t = os.path.basename(f)[:-4]
        g = f"{P}/{t}.log"
        if not os.path.exists(g):
            drop["no_log"] += 1; continue
        st_a = (sa or {}).get(t, ""); st_b = (sp or {}).get(t, "")
        # 互斥剔除,按优先级:agent 根本没被问过 > 撞墙钟 > serve 死
        if "agent_failed" in (st_a or "") or "agent_failed" in (st_b or ""):
            drop["agent_failed"] += 1; continue
        if "timeout" in (st_a or "") or "timeout" in (st_b or ""):
            drop["timeout"] += 1; continue
        a, b = feats(f), feats(g)
        if a["conn"] >= 3 or b["conn"] >= 3:
            drop["dead_serve"] += 1; continue
        rows.append((t, a, b))
    return rows, ran, drop


def main():
    print("# MLS 采样对齐 A/B:`_al1`(协议对齐)− `_p1`(未对齐)\n")
    print("两批除采样外 env 逐项相同(已从 /proc/<pid>/environ 直读确认)。\n")
    print("## 0. 每条臂进入分析的题数(§27:先看分母)\n")
    print("| arm | 作业 | 可配对 n | 剔:agent没被问 | 剔:撞墙钟 | 剔:serve死 | 状态 |")
    print("|---|---|---|---|---|---|---|")
    keep = {}
    for arm, lab in ARMS:
        rows, ran, drop = pair(arm)
        st = "完整" if ran else "⚠还在跑"
        print(f"| {lab} | {'已结束' if ran else '未结束'} | {len(rows)} | {drop['agent_failed']} | "
              f"{drop['timeout']} | {drop['dead_serve']} | {st} |")
        keep[arm] = (rows, ran)
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
