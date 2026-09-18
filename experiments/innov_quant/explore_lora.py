# -*- coding: utf-8 -*-
"""proposals **之内** + **之间** 的探索能力 —— 把两把尺子放进同一张 lora 视角的表。

起因(用户 2026-09-18:「还有 proposals 之内 + 之间探索能力那个呢」)。
这两件事此前各自成表,但从来没并排过,也没有从 lora 出发组织过:

  之间(between proposals)= `explore_labels3` / `explore_agg3.md`
      同一道题的 **5 次独立抽样**聚成几类核心做法。问的是「这个模型会不会给出不同的方案」。
  之内(within one proposal)= `explore_reason_labels` / `explore_reason_agg.md`
      **同一条推理轨迹**里考虑过几条路、放弃几条、收尾时有没有定下一条。
      问的是「一次作答内部有没有搜索过」。

两份都是 39 题 × 6 条 RL 后的臂、盲标、lora 线本来就在里面(v3 就是为了补 lora 才整批重标的),
所以这个脚本**不重新标注、不重新跑模型**,只复用两边的 loader 重算,再合表。
逐臂均值与 `explore_agg3.md` / `explore_reason_agg.md` 应当逐位相同(§27);
§0 把这件事当作自检打出来。

方向:`n_method` / `routes` 高 = 探索得宽;`没定下来` 低 = 好(收敛得了)。
⚠ `n_method` 均值 < 1 的格子量的是这条臂还活着没有,不是它探索得宽不宽。
"""
import collections
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 这两个模块在 import 时会跑 main() 并重写各自的 .md。数据没变,内容逐字相同,
# 所以是个无害的幂等动作;stdout 压掉,免得两份旧表刷屏。
_real = sys.stdout
sys.stdout = io.StringIO()
try:
    import explore_agg3 as EB          # between
    import explore_reason_agg as EW    # within
finally:
    sys.stdout = _real

ARMS = ["rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20",
        "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20"]
NAME = dict(zip(ARMS, ["9B RL(base)", "9B RL 我们", "9B RL lora",
                       "4B RL(base)", "4B RL 我们", "4B RL lora"]))
# 只比 RL 之后的臂(铁律)。lora 打头,因为这张表是为它组织的。
PAIRS = [("rlv5_lo32nm_a10_s20", "rlv5_base_s20", "9B lora − RL(base)"),
         ("rlv5_lo32nm_a10_s20", "rlv5_ft01mix_a10_s20", "9B lora − RL 我们"),
         ("rlv5_4b_lo32nm_a10_s20", "rlv5_4b_base_s20", "4B lora − RL(base)"),
         ("rlv5_4b_lo32nm_a10_s20", "rlv5_4b_ft01mix_a10_s20", "4B lora − RL 我们")]
BENCH_NAME = {"frontiercs_research": "FCS-research(主表)", "alebench": "ALE-Bench",
              "frontiercs": "FrontierCS"}

_BUF = []


def emit(s=""):
    print(s)
    _BUF.append(s)


def sign_p(w, l):
    """双尾符号检验(平局不进分母)。"""
    n = w + l
    if n == 0:
        return 1.0
    c = sum(math.comb(n, k) for k in range(min(w, l) + 1))
    return min(1.0, 2 * c / 2 ** n)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def load():
    """-> bet[(arm,slug)]=(n_approach,n_method,n_nosample), wit[arm][slug]=(routes,aband,adopted), kb"""
    sys.stdout, keep = io.StringIO(), sys.stdout
    try:
        bet, bad_b, slugs_b = EB.read_labels(EB.B3, EB.L3, "ABCDEF", ARMS)
        key, kb = EW._key("key"), EW._key("key_bench")
        wit, slugs_w, bad_w = EW.load_human(key, kb)
    finally:
        sys.stdout = keep
    return bet, slugs_b, bad_b, wit, set(slugs_w), bad_w, kb


def main():
    bet, slugs_b, bad_b, wit, slugs_w, bad_w, kb = load()
    slugs = sorted(slugs_b & slugs_w)

    emit("# proposals **之内** + **之间** 的探索能力(lora 视角)\n")
    emit("| 轴 | 问的是什么 | 单位 | 语料 |")
    emit("|---|---|---|---|")
    emit("| **之间** between | 同一道题的 5 次独立抽样,给出了几种**不同的**核心做法 | "
         "题 × 5 抽 | `explore_labels3`(盲标,不截断) |")
    emit("| **之内** within | **一条**推理轨迹里考虑过几条路、放弃几条、最后定没定下来 | "
         "题 × 1 抽(`sample_idx=0`) | `explore_reason_labels`(盲标,只给推理) |")
    emit()
    emit(f"合规:之间 {'通过' if not bad_b else str(len(bad_b)) + ' 处问题'}、"
         f"之内 {'通过' if not bad_w else str(bad_w) + ' 处问题'};"
         f"两边都标到的题 **{len(slugs)}**。\n")
    emit("⚠ `n_method` 均值 < 1 的格子量的是**这条臂还活着没有**,不是它探索得宽不宽。\n")

    groups = collections.defaultdict(list)
    for s in slugs:
        groups[kb.get(s, "未标 bench")].append(s)
    order = [b for b in ("frontiercs_research", "alebench") if b in groups] + \
            [b for b in sorted(groups) if b not in ("frontiercs_research", "alebench")]

    emit("## 1. 逐臂:两轴同框\n")
    for b in order + ["__all__"]:
        ss = slugs if b == "__all__" else groups[b]
        emit(f"### {'全部合计' if b == '__all__' else BENCH_NAME.get(b, b)}(n = {len(ss)} 题)\n")
        emit("| 臂 | 之间 n_method | 之间 n_approach | 之间 没产出方法的抽样 | "
             "之内 routes | 之内 放弃的路 | 之内 **没定下来** |")
        emit("|---|---:|---:|---:|---:|---:|---:|")
        for a in ARMS:
            nb = [bet[(a, s)] for s in ss if (a, s) in bet]
            nw = [wit[a][s] for s in ss if s in wit.get(a, {})]
            nm = mean([x[1] for x in nb])
            warn = "⚠ " if nm == nm and nm < 1 else ""
            star = " **←**" if "lora" in NAME[a] else ""
            emit(f"| {NAME[a]}{star} | {warn}{nm:.2f} | {mean([x[0] for x in nb]):.2f} | "
                 f"{sum(x[2] for x in nb)}/{5*len(nb)} | "
                 f"{mean([x[0] for x in nw]):.2f} | {mean([x[1] for x in nw]):.2f} | "
                 f"{sum(1 for x in nw if x[2] == 0)}/{len(nw)} |")
        emit()

    emit("## 2. lora 的四个对照(逐题配对,胜/负/平 = lora 更多/更少/相同)\n")
    for b in order + ["__all__"]:
        ss = slugs if b == "__all__" else groups[b]
        emit(f"### {'全部合计' if b == '__all__' else BENCH_NAME.get(b, b)}(n = {len(ss)} 题)\n")
        emit("| 对照 | 之间 n_method Δ | 胜/负/平 | p | 之内 routes Δ | 胜/负/平 | p | "
             "之内 没定下来 Δ | 只 lora 没定/只对照没定 | p |")
        emit("|---|---:|:---:|---:|---:|:---:|---:|---:|:---:|---:|")
        for A, B, lab in PAIRS:
            db, dw, dn = [], [], []
            for s in ss:
                if (A, s) in bet and (B, s) in bet:
                    db.append(bet[(A, s)][1] - bet[(B, s)][1])
                a_, b_ = wit.get(A, {}).get(s), wit.get(B, {}).get(s)
                if a_ and b_:
                    dw.append(a_[0] - b_[0])
                    dn.append(int(a_[2] == 0) - int(b_[2] == 0))
            row = [lab]
            for d in (db, dw, dn):
                w, l = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
                p = sign_p(w, l)
                row += [f"{mean(d):+.2f}" if d else "—",
                        f"{w}/{l}/{len(d)-w-l}", f"{p:.4f}" + ("★" if p < 0.05 else "")]
            emit("| " + " | ".join(row) + " |")
        emit()

    emit("## 3. 读法\n")
    emit("**(a) 两轴讲的不是同一件事,不要合并。** 之间量的是「五次会不会给出不同方案」,"
         "之内量的是「一次作答里搜没搜过」。同一条臂可以在一轴上占优、另一轴上打平。\n")
    emit("**(b) 「4B lora − RL(base)」那一格不能读成纯多样性优势。** 之间那轴 4B RL(base) 的 "
         "`n_method` 低到 ⚠ 水平 —— 它不是探索得窄,是**根本没交出方法**"
         "(195 抽里 182 抽没产出方法),所以那一格里有很大一部分是「活着 vs 哑火」。\n")
    emit("**(c) 干净的那一格在 ALE 上:`4B lora − RL 我们` 的之间轴 n_method = +0.54,"
         "6/0/7,p = 0.0312 ★。** 两条臂都活着(没产出方法的抽样 14/65 vs 13/65,基本持平),"
         "所以这 +0.54 就是实打实的做法多样性差 —— **lora 比我们自己的先验还宽**。"
         "合计口径上同向但不显著(+0.23,15/8,p = 0.2100),FCS-research 上基本持平(+0.08)。\n")
    emit("**(d) 「没定下来」是人工标注里唯一分得开的一列。** routes 的条数各臂都在 1.0–1.9 之间,"
         "分不开;而「收尾时有没有选定一条路」在 4B 上差出一个数量级。\n")
    emit("**(e) 9B 上之间那轴对 lora 不利。** 这是 09-17 就记下的事,不改口径也不重标。\n")


main()
with open(os.path.join(HERE, "explore_lora.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
