"""Research idea judgment(idea V1)——只看 RL:我们的 RL vs baseline 的 RL。

用户 2026-09-17 定的口径:
  - **只用 idea V1**(`cc_idea32k_*_y26pp`)。V2 是另一版提示词,两版不混。
  - **只放 RL 两条臂**(9B/4B 各一对):RL 是我们最后 shape 出来的 final model,
    base/SFT 是中间态,不进这张表。
  - **原始数值要给**,不能只给 Δ。下面第 1 节就是每条臂的绝对正确率。

机器全部复用 `taste_vs_base.py`(同一套 read / survivors / acc / cell),
不另写一份 —— 两份实现迟早会对不上同一个数(第 27 号)。
strict / penalise 两个 lens 都报,定义逐字沿用 `j3stats.py`:
  strict   = 这一格里**参与对照的两条臂**都把 5 抽全解析出来的题才算;
  penalise = 没作答按答错记,一题不丢。
strict 的存活集只对本对照的两条臂取交 —— 八臂取交会让一条臂的失败把
别的对照的题数从 120 砍到 25(第 6 号)。

`openreview_decide` 打 ⚑:V1 的提示词里带会议名(§25),去掉那句话两条 RL 臂的
AUC 各动 0.11-0.14。留在表里,但聚合给「含/不含」两版。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taste_vs_base as T

TASKS = ["aaar_equation", "liveidea_pair", "openreview_pair", "openreview_decide"]
FLAG = {"openreview_decide"}
PAIRS = [("9B RL(SFT)", "9B RL(base)", "9B 我们 − 9B RL(base)"),
         ("4B RL(SFT)", "4B RL(base)", "4B 我们 − 4B RL(base)")]
SHOW = [("9B RL(base)", "9B RL(base)"), ("9B RL(SFT)", "9B 我们"),
        ("4B RL(base)", "4B RL(base)"), ("4B RL(SFT)", "4B 我们")]

_BUF = []


def emit(s=""):
    sys.stdout.write(s + "\n")
    _BUF.append(s)


def main():
    rng = np.random.default_rng(T.SEED)
    emit("# Research idea judgment(idea V1):我们的 RL vs baseline 的 RL\n")
    emit("数据源 `outputs/cc_idea32k_<arm>_y26pp/samples.jsonl`,每题 5 抽,"
         "单位=题。⚑ 见文件头。\n")

    emit("## 1. 原始数值:每条臂的绝对正确率(题内 5 抽取均值,再对题取平均)\n")
    for lens in ("strict", "penalise"):
        emit(f"### lens = {lens}\n")
        if lens == "strict":
            emit("> 这一节的 strict 存活集对**四条臂**取交(四列同题同分母,列间才可比),"
                 "所以同一条臂的 strict 值会和第 2 节不一样 —— 第 2 节按对照的**两条臂**"
                 "取交,是 `j3stats.py` 只传两个 tag 时的行为。两处都不是笔误,分母不同而已。\n")
        emit("| 任务 | n | " + " | ".join(n for _, n in SHOW) + " |")
        emit("|---|---:|" + "---:|" * len(SHOW))
        for task in TASKS:
            # strict 存活集按「本表这四条臂」取交,四条臂同题同分母,列间可比。
            keep = T.survivors(T.PP, "idea", task, labs=[(l, T.TAG[l]) for l, _ in SHOW])
            accs = {l: T.acc(T.PP, "idea", l, task, lens, keep) for l, _ in SHOW}
            ids = set.intersection(*[set(a) for a in accs.values()]) if accs else set()
            if not ids:
                continue
            cells = [f"{np.mean([accs[l][i] for i in ids]):.4f}" for l, _ in SHOW]
            mark = " ⚑" if task in FLAG else ""
            emit(f"| `{task}`{mark} | {len(ids)} | " + " | ".join(cells) + " |")
        emit()

    emit("## 2. 配对差(Δ = 我们 − baseline 的 RL,同题配对)\n")
    agg = {}
    for A, B, lbl in PAIRS:
        for lens in ("strict", "penalise"):
            emit(f"### {lbl} · lens = {lens}\n")
            emit("| 任务 | n | 我们 | RL(base) | Δ | 95% CI | +/− | z | Wilcoxon p |")
            emit("|---|---:|---:|---:|---:|---|---:|---:|---:|")
            cs = []
            for task in TASKS:
                keep = T.survivors(T.PP, "idea", task, labs=[(A, T.TAG[A]), (B, T.TAG[B])])
                a = T.acc(T.PP, "idea", A, task, lens, keep)
                b = T.acc(T.PP, "idea", B, task, lens, keep)
                c = T.cell(a, b, rng)
                if not c:
                    continue
                ids = sorted(set(a) & set(b))
                ma, mb = np.mean([a[i] for i in ids]), np.mean([b[i] for i in ids])
                mark = " ⚑" if task in FLAG else ""
                emit(f"| `{task}`{mark} | {c['n']} | {ma:.4f} | {mb:.4f} | {c['mean']:+.4f} | "
                     f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['pos']}/{c['neg']} | "
                     f"{c['z']:+.2f} | {c['p_w']:.4f} |")
                cs.append((task, c))
            emit()
            agg[(lbl, lens)] = cs

    emit("## 3. 聚合(Stouffer / 符号检验,含 ⚑ 与不含 ⚑ 两版)\n")
    emit("| 对照 | lens | 口径 | 格子 | 方向 | 符号 p | Stouffer Z | p |")
    emit("|---|---|---|---:|---:|---:|---:|---:|")
    for (lbl, lens), cs in agg.items():
        for keepflag, nm in ((True, "含 ⚑"), (False, "不含 ⚑")):
            sub = [c for t, c in cs if keepflag or t not in FLAG]
            s = T.stouffer(sub)
            if not s:
                continue
            emit(f"| {lbl} | {lens} | {nm} | {s['n']} | {s['k']}/{s['n']} 正 | "
                 f"{s['p_sign']:.4f} | {s['Z']:+.2f} | {s['p']:.4f} |")
    emit()


main()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "idea_v1_rl.md"),
          "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
