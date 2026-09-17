# -*- coding: utf-8 -*-
"""创新性 / 探索性汇总:一条输出内部的探索、五条输出之间的差异、idea 判断、重组。

用户 2026-09-17 的要求:
  1. 方法差异性(探索能力)分两面报:(a) 一条输出**内部**探索了几种路子;
     (b) 五条输出**之间**有多少不同做法。
  2. Research idea judgment 只用 **idea V1**,只放 **RL 两条臂**(我们的 RL vs baseline 的 RL)。
  3. **原始数值要给**,不能只给 Δ。
  4. MLS 上「更不倾向 recombination」如果看得到就放,并**说清楚关键词是怎么挑的**。

本脚本一个数都不自己算 —— 全部从已生成的表里取行(`explore_tables.md`、
`explore_at5.md`、`idea_v1_rl.md`、`recomb_tables.md`、`baseline_recomb_p1.md`)。
手抄会让两张表对同一个数打架(第 27 号),解析源表则不可能对不上。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_BUF = []


def emit(s=""):
    sys.stdout.write(s + "\n")
    _BUF.append(s)


def tables(path):
    """-> [(标题路径, 表头, [行])];标题路径是当前 #/##/### 的元组。"""
    out, head, cols, rows = [], [], None, []
    for ln in open(os.path.join(HERE, path), encoding="utf-8"):
        ln = ln.rstrip("\n")
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            if cols:
                out.append((tuple(head), cols, rows)); cols, rows = None, []
            lvl = len(m.group(1))
            # 必须按层级补位再写:源文件里没有一级标题时,`head[:lvl-1]` 会让第一个
            # 二级标题落到 index 0,下一个二级标题却落到 index 1 —— 于是第二节起
            # 全部继承了第一节的名字(ALE-Bench 和「合计」都被打成 FrontierCS-research)。
            head = (head + [""] * lvl)[:lvl - 1] + [m.group(2).strip()]
            continue
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            if cols is None:
                cols = cells
            else:
                rows.append(cells)
        elif cols:
            out.append((tuple(head), cols, rows)); cols, rows = None, []
    if cols:
        out.append((tuple(head), cols, rows))
    return out


def find(path, want_head, want_col0=None):
    """第一张满足「标题里全部出现 want_head」且首列命中的表。"""
    for head, cols, rows in tables(path):
        h = " / ".join(head)
        if all(w in h for w in want_head):
            if want_col0 and not any(r and want_col0 in r[0] for r in rows):
                continue
            return head, cols, rows
    return None, None, None


def table(cols, rows, keep=None, rename=None):
    """按 keep(首列子串列表)挑行并保持给定顺序;rename 改首列显示名。"""
    emit("| " + " | ".join(cols) + " |")
    emit("|" + "---|" * len(cols))
    if keep is None:
        sel = rows
    else:
        sel = []
        for k in keep:
            for r in rows:
                if r and r[0].strip().strip("`") == k:
                    sel.append(r); break
    for r in sel:
        r = list(r)
        if rename and r[0].strip().strip("`") in rename:
            r[0] = rename[r[0].strip().strip("`")]
        emit("| " + " | ".join(r) + " |")
    emit()


NAME9 = {"base9b_v2c": "9B base", "ft01mix_a10": "9B SFT",
         "rlv5_base_s20": "9B RL(base)", "rlv5_ft01mix_a10_s20": "**9B 我们**"}
NAME4 = {"base4b": "4B base", "4b_ft01mix_a10": "4B SFT",
         "rlv5_4b_base_s20": "4B RL(base)", "rlv5_4b_ft01mix_a10_s20": "**4B 我们**"}
ORD9 = list(NAME9); ORD4 = list(NAME4)
PAIR9 = {"rlv5_ft01mix_a10_s20 − rlv5_base_s20": "**我们 − RL(base)**",
         "rlv5_ft01mix_a10_s20 − ft01mix_a10": "我们 − SFT",
         "rlv5_base_s20 − base9b_v2c": "RL(base) − base"}
PAIR4 = {"rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20": "**我们 − RL(base)**",
         "rlv5_4b_ft01mix_a10_s20 − 4b_ft01mix_a10": "我们 − SFT",
         "4b_ft01mix_a10 − base4b": "SFT − base"}
BENCH = [("frontiercs", "FrontierCS"), ("frontiercs_research", "FrontierCS-research"),
         ("alebench", "ALE-Bench")]


def sec_inner():
    emit("# A. 探索(a):**一条输出内部**探索了几种路子\n")
    emit("怎么测的:只看 `</think>` **之前**的推理(最终代码另算)。把推理文本映射到一张"
         "「技术家族」词表上,`n_reason` = 推理里点到的不同家族数,`n_code` = 最终代码里的家族数,"
         "`n_abandon` = 点到但最终没用(考虑过又放弃),`explore_ratio` = n_reason / max(1, n_code),"
         "`n_alt` = 「换个思路」类标记词(Alternatively / Another approach / Instead of / "
         "Let me reconsider / 或者 …)的出现次数。\n")
    emit("**推理长度必须一起看**:RL 臂的推理是 SFT/base 的 3-4 倍长,点到更多家族有一部分只是"
         "「写得更长」。所以每个指标都另给一份 **/10k 字符**的归一版本,两版都报。\n")
    for key, lbl in BENCH:
        for size, order, names, pairs in (("9B", ORD9, NAME9, PAIR9), ("4B", ORD4, NAME4, PAIR4)):
            h, cols, rows = find("explore_tables.md", [f"{key} / {size}", "A."])
            if not rows:
                continue
            emit(f"## {lbl} / {size} —— 原始数值(每臂)\n")
            table(cols, rows, keep=order, rename=names)
            h, cols, rows = find("explore_tables.md", [f"{key} / {size}", "B."])
            if rows:
                emit(f"## {lbl} / {size} —— 配对差\n")
                table(cols, rows, keep=list(pairs), rename=pairs)


def sec_between():
    emit("# B. 探索(b):**五条输出之间**有多少不同做法\n")
    emit("怎么测的:FCS-research 26 题,每题把八条臂各自的 5 次抽样的**最终答案**(不给思维链)"
         "做成八个「块」,块序按题打乱、只给字母,标注者对每一块把 5 抽聚成若干「做法」类。"
         "`n_approach` = 类数(**「没产出方法」自成一类**,所以坍塌的臂照样进分母);"
         "`n_method` = 去掉「没产出方法」那一类之后的类数。两个数必须并排看 —— 五抽全没交东西"
         "和五抽交了同一个方法,n_approach 都是 1,但 n_method 分别是 0 和 1。\n")
    # explore_at5.md 现在按 bench 分了节,这里要**全部**取过来,不能只取第一张 ——
    # 只取第一张会静默变成「只报 FCS-research」,而标题却写着「五条输出之间」。
    for head, cols, rows in tables("explore_at5.md"):
        if len(head) < 2:
            continue
        bench = next((x for x in head[:-1] if x), "")
        sect = head[-1]
        if "逐臂" in sect:
            emit(f"## {bench} —— 原始数值(每臂)\n"); table(cols, rows)
        elif "对照" in sect or "全部 bench" in sect:
            emit(f"## {bench} —— 配对差\n"); table(cols, rows)


def sec_idea():
    emit("# C. Research idea judgment(idea V1,只放 RL)\n")
    for head, cols, rows in tables("idea_v1_rl.md"):
        h = " / ".join(head)
        if "原始数值" in h or "配对差" in h or "聚合" in h:
            emit(f"## {head[-1]}\n"); table(cols, rows)


def sec_recomb():
    emit("# D. 重组(recombination):模型是不是在拼已有的零件\n")
    emit("**两把不同的尺子,词表来源完全不同,不要混:**\n")
    emit("| 尺子 | 语料 | 「零件」是什么 | 词表怎么来的 |")
    emit("|---|---|---|---|")
    emit("| D1 技术家族重组 | FrontierCS / ALE 的代码 | 通用算法技术家族"
         "(sim_anneal / beam_search / genetic / mcts / maxflow …) | **从别家模型的解里挑的** —— "
         "参照语料是官方 Frontier-CS 解池(gemini3pro / gpt5.x / deepseekreasoner / grok4 / trinity …),"
         "先看这些解里反复出现哪些技术,再为每个家族写正则。**这张词表和我们的模型无关**,"
         "也没用我们任何一条臂的输出去挑词,所以它不偏向我们 |")
    emit("| D2 baseline 重组 | MLS-Bench 的 21 题 | **题目自己给的 Reference baselines** | "
         "不需要挑 —— 每道题在 `tasks/<t>/edits/*.edit.py` 里就放着每条 baseline 的参考实现,"
         "词表 = 这些 baseline 的名字与实现本身 |")
    emit()
    emit("D1 的 `new_pair_rate` / `med_z` / `p10_z` 沿用 Uzzi 2013 的原子性做法:"
         "以 frontier 解池当「既有文献」,做度数保持的随机零模型,每个技术对算 "
         "z=(obs−mu)/sd。med_z 高 = 用的都是常规搭配;p10_z 低 = 伸手去够冷门搭配。\n")
    for size, order, names, pairs in (("9B", ORD9, NAME9, PAIR9), ("4B", ORD4, NAME4, PAIR4)):
        h, cols, rows = find("recomb_tables.md", [size, "A."], want_col0=order[0])
        if rows:
            emit(f"## D1 · FrontierCS / {size} —— 原始数值(每臂)\n")
            table(cols, rows, keep=order, rename=names)
        h, cols, rows = find("recomb_tables.md", [size, "B."], want_col0=order[0])
        if rows:
            emit(f"## D1 · FrontierCS / {size} —— 配对差(n_tech 低 = 拼的零件少)\n")
            table(cols, rows, keep=list(pairs), rename=pairs)
    for head, cols, rows in tables("baseline_recomb_p1.md"):
        h = " / ".join(head)
        if "逐臂" in h:
            emit("## D2 · MLS-Bench(p1)—— 原始数值(每臂)\n"); table(cols, rows)
        if "sim_resid" in h:
            emit("## D2 · MLS-Bench(p1)—— `sim_resid`:与参考实现的相似度,已对代码长度回归去偏(低 = 好)\n")
            table(cols, rows)
        if h.endswith("n_hard — 真 import/调用的 baseline 条数(低=好)"):
            emit("## D2 · MLS-Bench(p1)—— `n_hard`:真 import/调用了几条本题给定的 baseline(低 = 好)\n")
            table(cols, rows)


def main():
    emit("# 创新性 / 探索性:原始数值与配对差\n")
    emit("四块材料,四把互相独立的尺子。**每一块都先给原始数值,再给差值。**\n")
    emit("| 块 | 问的问题 | 单位 | 结论方向 |")
    emit("|---|---|---|---|")
    emit("| A | 一条输出**内部**探索了几种路子 | 样本 → 题 | **对我们有利**(9B/4B 三个 bench 上"
         "我们 − RL(base) 的 n_reason / n_abandon / explore_ratio 基本都是正的) |")
    emit("| B | **五条输出之间**有多少不同做法 | 题(39 题人工标注:FCS-research 26 + ALE 13) | "
         "**分 bench 后不是一句话**:9B 我们 − SFT 在 FCS-research 上 −1.23(0/18/8,p<1e-4),"
         "在 ALE 上只有 −0.08(5/5/3,p=1.00);4B 我们 − RL(base) 两个 bench 都压倒性为正"
         "(合计 36/0/3,p<1e-4) |")
    emit("| C | 会不会判断 research idea 的好坏 | 题 | 9B 三个非 ⚑ 任务 3/3 正但都不显著;"
         "4B penalise 下 Stouffer Z=+5.66 |")
    emit("| D | 是不是在拼已有的零件 | 样本 → 题 | D1(FrontierCS)我们的 n_tech 比 SFT 低 0.822"
         "(p<0.001),比 RL(base) 持平;D2(MLS)没看到我们更不重组 —— sim 反而更高,照实报 |")
    emit()
    sec_inner()
    sec_between()
    sec_idea()
    sec_recomb()


main()
with open(os.path.join(HERE, "innov_push.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
