#!/usr/bin/env python3
"""创新性 vs 组合性 —— **lora 视角**(用户 2026-09-18:「我现在只看 lora 的了」)。

`innov_vs_recomb.py` 的 ARMS 只有 ft01mix 那八条,lora 线从来没进过这张表;
但底层的 `metrics.csv` / `recomb_metrics.csv` / `explore_metrics.csv` **本来就有 lora 的每一抽**。
所以这个脚本不重跑任何模型、不重算任何指标,只是把同一套机器换一组对照重放一遍。
口径、bootstrap 次数、种子、聚合方式全部复用 `innov_vs_recomb.py` 的函数,
**所以两张表可以逐格对照**(§27)。

要回答的问题和原表一样,只是主角换成 lora:

  组合侧(combination)  一份解拼了几个**技术家族**(n_tech / n_pair / P(n_tech≥2) /
      长度归一后的密度与残差)。想说「我们更创新」,这一侧就**不能**升高 ——
      零件更多不是创新,是拼装。
  创新侧(innovation)  定稿前搜得宽不宽(n_reason / n_abandon / explore_ratio / n_alt),
      代码跟前沿解池像不像(jac_pool),五次抽样彼此像不像(sim_self)。

两条必须带着的限定,原表已经写死,这里照搬:
  1. 所有指标都条件于「这一抽跑完了且产出了代码」,而完成率是各臂不同的(§27)。
     `rlv5_4b_base_s20` 过这道线的抽样极少,**所以 4B 对 RL(base) 的整列不可解读**,
     表里标 ⚠ 并且不进任何结论。干净的 4B 对照是 `4B RL lora − 4B RL 我们`。
  2. `med_z` / `p10_z` / `new_pair_rate` 在「技术家族数 < 2」时**根本没有定义**,
     而 P(n_tech≥2) 恰恰是处理会动的量 —— 配对就等于在每条臂内部挑出「表现得像对方」
     的那些抽样。单列在 §5,不进结论。

噪声底(§20c):六对 `_y2026` 同协议复跑走同一套机器,**其中四对就是 lora 臂本身**。
底是逐指标的,不是一个数;某个指标的底自己就显著,那个指标无论对照说什么都不引用。

用法:innov_lora.py  → 打印并写 innov_lora.md
"""
import math
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import innov_vs_recomb as IV          # 只 import,不触发它的 main()

BENCHES = IV.BENCHES
ARMS = [("9B base", "base9b_v2c"), ("9B SFT lora", "lo32nm_a10"),
        ("9B RL(base)", "rlv5_base_s20"), ("9B RL 我们", "rlv5_ft01mix_a10_s20"),
        ("9B RL lora", "rlv5_lo32nm_a10_s20"),
        ("4B base", "base4b"), ("4B SFT lora", "4b_lo32nm_a10"),
        ("4B RL(base)", "rlv5_4b_base_s20"), ("4B RL 我们", "rlv5_4b_ft01mix_a10_s20"),
        ("4B RL lora", "rlv5_4b_lo32nm_a10_s20")]
TAG = dict(ARMS)
# 只比 RL 之后的臂(铁律);SFT lora vs 预训练 base 作旁证。
CONTRASTS = [("9B RL lora", "9B RL(base)", "lora 先验在 RL 阶段"),
             ("9B RL lora", "9B RL 我们", "lora vs 我们自己的先验"),
             ("4B RL lora", "4B RL(base)", "lora 先验在 RL 阶段"),
             ("4B RL lora", "4B RL 我们", "lora vs 我们自己的先验"),
             ("9B SFT lora", "9B base", "只做 SFT(旁证)"),
             ("4B SFT lora", "4B base", "只做 SFT(旁证)")]

_BUF = []


def emit(s=""):
    print(s)
    _BUF.append(s)


def main():
    rng = np.random.default_rng(IV.SEED)
    emit("# 创新性 vs 组合性 —— lora 视角\n")
    emit("机器与 `innov_vs_recomb.md` **完全相同**(同 bootstrap 次数、同种子、同聚合),"
         "只换了对照组,所以两张表逐格可对。\n")
    emit("| 侧 | 想证明的事 | 该往哪个方向动 |")
    emit("|---|---|---|")
    emit("| **组合侧** | 「更创新」**不是**「零件更多」 | 最好**不动**(升高会毁掉论点) |")
    emit("| **创新侧** | 定稿前搜索更宽、产出更不像解池 | n_reason/n_abandon 类**升高**;"
         "jac_pool / sim_self **降低** |")
    emit()

    emit("## 1. 样本盘点(§27:先看分母)\n")
    emit("每格 = 过了「跑完且有代码」这道线的抽样数。**完成率是各臂不同的,"
         "所以分母本身就是被处理动过的量。**\n")
    emit("| 臂 | " + " | ".join(n for _, n in BENCHES) + " |")
    emit("|---|" + "---:|" * len(BENCHES))
    thin_arms = set()
    for lab, tag in ARMS:
        cells = []
        for b, _ in BENCHES:
            g = IV.BY.get((b, tag))
            n = 0 if g is None else int(g["n_tech"].notna().sum())
            if n < IV.THIN:
                thin_arms.add(lab)
            cells.append(f"**⚠{n}**" if n < IV.THIN else str(n))
        emit(f"| {lab} | " + " | ".join(cells) + " |")
    emit(f"\n⚠ = 少于 {IV.THIN} 抽,这一格读不出东西。"
         + (f"**{', '.join(sorted(thin_arms))} 触线** —— 凡是以它为对照的列整列不可解读。\n"
            if thin_arms else "\n"))

    emit("## 2. 噪声底(§20c):同一条臂跑两次能差出多少\n")
    emit("六对 `_y2026` 同协议复跑,**其中四对就是 lora 臂本身**"
         "(`lo32nm_a10`、`rlv5_lo32nm_a10_s20`、`4b_lo32nm_a10`、`rlv5_4b_lo32nm_a10_s20`)。"
         "底是**逐指标**的:某个指标的底自己就显著,那把尺子**任何方向都不引用**。\n")
    fl = {}
    emit("| 指标 | 底的 Stouffer Z | p | 合格? |")
    emit("|---|---:|---:|:---:|")
    for group in (IV.COMBI, IV.INNOV, IV.COND):
        for f, lab, _ in group:
            cs = [IV.cell(b, A, B, f, rng) for b, _ in BENCHES for A, B in IV.REP]
            st = IV.stouffer(cs)
            fl[f] = st
            if not st:
                emit(f"| `{f}` {lab} | — | — | — |")
                continue
            ok = "✅" if st["p"] >= 0.05 else "❌ **不合格**"
            emit(f"| `{f}` {lab} | {st['Z']:+.2f} | {st['p']:.4f} | {ok} |")
    emit()

    def block(title, fields, note=""):
        emit(f"## {title}\n")
        if note:
            emit(note + "\n")
        emit("| 指标 | 方向 | 底 | " + " | ".join(f"{a}<br>− {b}" for a, b, _ in CONTRASTS) + " |")
        emit("|---|:---:|:---:|" + "---|" * len(CONTRASTS))
        summ = {}
        for f, lab, good in fields:
            st = fl.get(f)
            bad = st and st["p"] < 0.05
            row = [f"`{f}` {lab}", {"flat": "不动", "higher": "高↑", "lower": "低↓"}[good],
                   "❌" if bad else "✅"]
            for A, B, _ in CONTRASTS:
                cs = [IV.cell(b, TAG[A], TAG[B], f, rng) for b, _ in BENCHES]
                st2 = IV.stouffer(cs)
                if not st2:
                    row.append("—"); continue
                thin = A in thin_arms or B in thin_arms
                mark = "⚠" if thin else ("★" if st2["p"] < 0.05 else "")
                row.append(f"{st2['Z']:+.2f}{mark}<br><sub>{st2['k']}/{st2['n']} 正</sub>")
                summ[(f, A, B)] = (st2, thin, bad)
            emit("| " + " | ".join(row) + " |")
        emit("\n每格 = 三个 bench 的 Stouffer Z + 正向格数。"
             "★ = p<0.05;⚠ = 该对照含分母过薄的臂,**不可解读**;"
             "底那一列 ❌ = 这把尺子的噪声底自己就显著,整行不引用。\n")
        return summ

    c_sum = block("3. 组合侧:lora 有没有「只是拼得更多」", IV.COMBI,
                  "**这一侧最好是平的。** 如果 lora 的 n_tech / P(n_tech≥2) 显著升高,"
                  "那「更创新」就讲不通了 —— 那只是把更多现成零件塞进一份解。")
    i_sum = block("4. 创新侧:定稿前搜得宽不宽、产出像不像解池", IV.INNOV)
    block("5. 条件指标(单列,不进结论)", IV.COND,
          "`med_z` / `p10_z` / `new_pair_rate` 在「技术家族数 < 2」时**没有定义**,"
          "而 P(n_tech≥2) 正是处理会动的量。配对等于在每条臂内部挑出「表现得像对方」的抽样,"
          "**这是被处理污染过的条件**。报出来是因为「因为不好看就丢掉一把尺子」本身也是偏倚,"
          "但它不进任何结论。")

    emit("## 6. 读法\n")
    emit("**(a) 先划掉不可读的。** 4B 对 `RL(base)` 的整列带 ⚠ —— 那条臂过「跑完且有代码」"
         "这道线的抽样只有 5/15/23 抽,不是它表现差,是它根本没交卷。"
         "干净的 4B 对照只有 `4B RL lora − 4B RL 我们`。"
         "`jac_pool` 整行不引用(噪声底 +4.64,自己就显著)。\n")
    emit("**(b) 9B lora vs RL(base):创新侧三把尺子全线拿下,但代价写在同一列里。**\n")
    emit("| | 指标 | Z | 方向 |")
    emit("|---|---|---:|---|")
    emit("| ✅ | `n_reason` 考虑过的路子 | **+8.48 ★**(3/3 bench) | 支持 |")
    emit("| ✅ | `n_abandon` 放弃过的路子 | **+7.71 ★**(3/3) | 支持 |")
    emit("| ✅ | `explore_ratio` 放弃/考虑 | **+4.21 ★**(3/3) | 支持 |")
    emit("| ⚠ | `n_reason_10k` 按长度归一 | −0.01 | **宽度优势按长度归一后消失** |")
    emit("| ⚠ | `n_alt` 备选方案 | **−3.28 ★** | 反向 |")
    emit("| ⚠ | `sim_self` 五抽自相似度 | **+2.01 ★** | 反向(五次更雷同) |")
    emit("| ❌ | `resid` n_tech 对长度的残差 | **+2.37 ★**(3/3) | **组合侧被打破** |")
    emit()
    emit("最后一行是这张表里对我们最不利的一格,得说清楚:原始的 `n_tech`(+0.84)、"
         "`n_pair`(+0.98)、`P(n_tech≥2)`(+1.31)、密度(+1.14)**都不显著**,看上去组合侧是平的;"
         "但把 n_tech 对 log(1+LOC) 回归之后,残差 **+2.37 ★ 且三个 bench 全正** —— "
         "**同样长度下 lora 确实塞进了更多技术家族**。所以「更创新而不是更会拼」这句话,"
         "在 9B/RL(base) 这一列上**只成立到「原始计数不升高」为止**,长度校正之后不成立。"
         "同理 `n_reason` 的 +8.48 也要配着 `n_reason_10k` 的 −0.01 一起读:"
         "**lora 想得更久,多出来的路子大体是多出来的长度换的。**\n")
    emit("**(c) 9B lora vs 我们自己的先验:搜索宽度打平,组合侧反而升高。** "
         "`n_reason` +1.81、`n_abandon` +0.89 都不显著,而 `P(n_tech≥2)` **+2.09 ★(3/3)**。"
         "对上自家先验,lora 没有更宽的搜索,只有更多的家族组合。\n")
    emit("**(d) 4B lora vs 我们自己的先验:原始宽度显著更低,长度归一后反超。** "
         "`n_reason` **−7.59 ★**、`n_abandon` **−7.90 ★**、`explore_ratio` **−6.18 ★**,"
         "但 `n_reason_10k` **+2.03 ★(3/3)**。两者不矛盾:4B lora 的推理明显更短,"
         "**单位长度上搜得更宽,总量上搜得更少**。要引用哪一个,取决于你想说的是「搜索效率」"
         "还是「搜索总量」—— 两个都得摆出来。\n")
    emit("**(e) SFT lora vs 预训练 base(旁证):组合侧是往下走的。** "
         "`P(n_tech≥2)` 9B **−2.71 ★**、4B **−2.06 ★**,两边都 0/3 正 —— "
         "SFT 阶段的 lora 先验让解**更少**地堆家族。这一格反而是「创新 ≠ 拼装」最干净的支持,"
         "只是它在 SFT 阶段、不在 RL 之后。\n")


main()
with open(os.path.join(HERE, "innov_lora.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
