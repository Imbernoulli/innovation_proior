"""@5 探索度的汇总:合规检查 + 解盲 + 两个指标,**按 benchmark 分层**。

n_approach = 5 抽里的做法数,「没产出方法」那一类也算一种(坍塌的臂照样进分母)。
n_method   = 去掉「没产出方法」后的做法数。

两个数必须并排报:一个臂五抽全没交东西,和五抽交了同一个方法,n_approach 都是 1,
但前者 n_method=0、后者 =1。试点里 4B RL(base) 和 4B RL(先验) 正好撞上这一对。

「没产出方法」优先认 cluster 里的 `no_method` 布尔;没有这个字段的(第一批标注还没
要求)退回按 desc 里是否含「没产出方法」判断。

分层的理由(用户 2026-09-17):抽题要横跨多个 benchmark、且必须含 research。
三个 bench 的题型差得很远(ALE 是长时优化、FCS 是竞赛题、FCS-research 是研究题),
把它们混成一张表,均值就只是「各 bench 各抽了几题」的函数 —— 换一下配比结论就变。
所以每个 bench 各出一份,最后才给合计,并且合计只用逐题配对(配对天然按题分层)。

用法:explore_agg.py [工作目录](默认脚本所在目录)。工作目录下要有
explore_blind/key.json(或 explore_labels/_key.json)与 explore_labels/。
"""
import collections
import json
import math
import os
import sys

S = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
BLIND = f"{S}/explore_blind"
LAB = f"{S}/explore_labels"
NO = "没产出方法"

ARMS = ["base9b_v2c", "ft01mix_a10", "rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "base4b", "4b_ft01mix_a10", "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
NAME = dict(zip(ARMS, ["9B base", "9B SFT", "9B RL(base)", "9B RL(先验)",
                       "4B base", "4B SFT", "4B RL(base)", "4B RL(先验)"]))
PAIRS = [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
         ("ft01mix_a10", "rlv5_ft01mix_a10_s20", "9B 我们 − SFT"),
         ("base9b_v2c", "rlv5_ft01mix_a10_s20", "9B 我们 − base"),
         ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)"),
         ("4b_ft01mix_a10", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − SFT"),
         ("base4b", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − base")]
BENCH_ORDER = ["frontiercs_research", "frontiercs", "alebench"]
BENCH_NAME = {"frontiercs_research": "FrontierCS-research", "frontiercs": "FrontierCS",
              "alebench": "ALE-Bench"}

_BUF = []


def emit(s=""):
    sys.stdout.write(s + "\n")
    _BUF.append(s)


def sign_p(w, l):
    """双边符号检验(平局剔除)。格子少,用精确二项,不用正态近似。"""
    n = w + l
    if n == 0:
        return float("nan")
    c = [math.comb(n, k) for k in range(n + 1)]
    tot = float(sum(c))
    k = min(w, l)
    return min(1.0, 2 * sum(c[:k + 1]) / tot)


def is_no(c):
    if "no_method" in c:
        return bool(c["no_method"])
    return NO in c.get("desc", "")


def load_key():
    for p in (f"{BLIND}/key.json", f"{LAB}/_key.json"):
        if os.path.exists(p):
            return json.load(open(p))
    raise SystemExit(f"找不到 key.json({BLIND} / {LAB})")


def load_bench():
    for p in (f"{BLIND}/key_bench.json", f"{LAB}/_key_bench.json"):
        if os.path.exists(p):
            return json.load(open(p))
    return {}


def per_arm(rec, slugs, title):
    emit(f"### {title}(n = {len(slugs)} 题)\n")
    emit("| 臂 | n_approach 均值 | n_method 均值 | 没产出方法的抽样 |")
    emit("|---|---:|---:|---:|")
    for a in ARMS:
        v = [rec[a][s] for s in slugs if s in rec[a]]
        if not v:
            continue
        emit(f"| {NAME[a]} | {sum(x[0] for x in v)/len(v):.2f} | "
             f"{sum(x[1] for x in v)/len(v):.2f} | {sum(x[2] for x in v)}/{5*len(v)} |")
    emit()


def per_pair(rec, slugs, title):
    emit(f"### {title}——逐题配对(胜/负/平 = 我们的 n_method 更高/更低/相同)\n")
    emit("| 对照 | n_method 胜 | 负 | 平 | 均差 | 符号检验 p | n_approach 均差 |")
    emit("|---|---:|---:|---:|---:|---:|---:|")
    for lo, hi, lbl in PAIRS:
        w = l = t = 0
        dm, da = [], []
        for s in slugs:
            if s not in rec[lo] or s not in rec[hi]:
                continue
            a0, m0, _ = rec[lo][s]
            a1, m1, _ = rec[hi][s]
            dm.append(m1 - m0); da.append(a1 - a0)
            w += m1 > m0; l += m1 < m0; t += m1 == m0
        if not dm:
            continue
        emit(f"| {lbl} | {w} | {l} | {t} | {sum(dm)/len(dm):+.2f} | "
             f"{sign_p(w, l):.4f} | {sum(da)/len(da):+.2f} |")
    emit()


def main():
    key = load_key()
    kb = load_bench()
    bad = 0
    rec = collections.defaultdict(dict)          # arm -> slug -> (n_app, n_met, n_miss)
    slugs = []
    for f in sorted(os.listdir(LAB)):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        slug = f[:-5]
        if slug not in key:
            emit(f"[bad] {slug} 不在 key.json"); bad += 1; continue
        d = json.load(open(f"{LAB}/{f}"))
        if sorted(d) != list("ABCDEFGH"):
            emit(f"[bad] {slug} 块不齐 {sorted(d)}"); bad += 1; continue
        okslug = True
        for li, v in d.items():
            ss = sorted(x for c in v["clusters"] for x in c["samples"])
            if ss != [1, 2, 3, 4, 5]:
                emit(f"[bad] {slug}/{li} samples={ss}"); bad += 1; okslug = False
            if v["n"] != len(v["clusters"]) and ss == [1, 2, 3, 4, 5]:
                # n 是冗余字段,真身是 clusters 的个数。分块本身合法(1-5 不重不漏)时
                # 只是标注者数错了个数,按 len(clusters) 纠正;分块不合法才作废这题。
                emit(f"[warn] {slug}/{li} n={v['n']} 但 clusters={len(v['clusters'])},按 clusters 纠正")
                v["n"] = len(v["clusters"])
            elif v["n"] != len(v["clusters"]):
                emit(f"[bad] {slug}/{li} n={v['n']} clusters={len(v['clusters'])}"); bad += 1; okslug = False
        if not okslug:
            continue
        slugs.append(slug)
        for li, v in d.items():
            arm = key[slug][li]
            nm = [c for c in v["clusters"] if not is_no(c)]
            miss = sum(len(c["samples"]) for c in v["clusters"] if is_no(c))
            rec[arm][slug] = (v["n"], len(nm), miss)

    groups = collections.defaultdict(list)
    for s in slugs:
        groups[kb.get(s, "未标 bench")].append(s)
    emit(f"合规:{'通过' if bad == 0 else str(bad) + ' 处问题'} | 已标注 {len(slugs)} 题 | "
         + "、".join(f"{BENCH_NAME.get(b, b)} {len(v)}" for b, v in
                     sorted(groups.items(), key=lambda kv: BENCH_ORDER.index(kv[0])
                            if kv[0] in BENCH_ORDER else 99)) + "\n")

    for b in BENCH_ORDER + [g for g in groups if g not in BENCH_ORDER]:
        if not groups.get(b):
            continue
        emit(f"## {BENCH_NAME.get(b, b)}\n")
        per_arm(rec, groups[b], "逐臂")
        per_pair(rec, groups[b], "对照")

    emit("## 合计(三个 bench 合起来)\n")
    emit("> 逐臂均值不给合计 —— 它只是各 bench 抽题配比的函数。配对差天然按题分层,"
         "合起来仍然有意义,所以只给配对。\n")
    per_pair(rec, slugs, "全部 bench")

    emit("## 逐题明细(n_approach / n_method)\n")
    for b in BENCH_ORDER + [g for g in groups if g not in BENCH_ORDER]:
        ss = sorted(groups.get(b, []))
        if not ss:
            continue
        emit(f"### {BENCH_NAME.get(b, b)}\n")
        emit("| 臂 | " + " | ".join(s[:24] for s in ss) + " |")
        emit("|---|" + "---:|" * len(ss))
        for a in ARMS:
            cells = []
            for s in ss:
                v = rec[a].get(s)
                cells.append(f"{v[0]}/{v[1]}" if v else "-")
            emit(f"| {NAME[a]} | " + " | ".join(cells) + " |")
        emit()

    json.dump({a: {s: rec[a][s] for s in rec[a]} for a in ARMS},
              open(f"{S}/explore_agg.json", "w"), ensure_ascii=False, indent=1)


HERE = os.path.dirname(os.path.abspath(__file__))
main()
with open(os.path.join(HERE, "explore_at5.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
