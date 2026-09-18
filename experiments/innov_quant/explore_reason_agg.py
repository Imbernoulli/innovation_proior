"""「一条输出之内考虑过几条路」的人工标注汇总 —— 并且**拿它去验那张正则表**。

这份存在的理由(用户 2026-09-17:「Routes considered per proposal 那些也要标注哈」):
`explore.py` 的 `n_reason` / `n_abandon` 不是「模型考虑过几条路」,而是
`recomb.PAT` 这张关键词正则表在推理里**命中了几个技术家族**。两件事只是名字像。
`explore_reason_corpus.py` 已经把同样 39 题 × 同样 6 条 RL 臂 × **同一条抽样
(`sample_idx=0`)** 的推理全文做成了盲盒,人工标注落在 `explore_reason_labels/`。

所以这里做三件事:
  §0 **覆盖率** —— 两把尺子各自有几个格子。正则那侧是在 `explore.py` 里算的,
     那个循环 `if i < 0: continue`(没有 `</think>` 就跳过)、`if not code.strip(): continue`
     (没有最终代码就跳过)。**哑火的臂会被这两道闸门整臂删掉**,于是正则表上
     根本看不见它;人工标注 39 题一题不少。先把这张表打出来,再看别的。
  §1 人工口径的逐臂逐 bench 均值。
  §2 人工口径的逐题配对对照(六个组合)。
  §3 **尺子核验**:在两侧都有的格子上,逐臂并排、逐格相关、以及六个对照的方向一致性。

口径同 v3:主表是 FCS-research,FrontierCS / ALE 只作同向旁证。
"""
import collections
import csv
import json
import math
import os
import re
import sys

S = os.path.dirname(os.path.abspath(__file__))
LAB = f"{S}/explore_reason_labels"
BLIND = "/scratch/gpfs/CHIJ/ziran/.tmp/claude-374317/-scratch-gpfs-CHIJ-bohan-1-innovation-proior/20154e8e-f2e8-4272-a550-4f0c059fe5b0/scratchpad/explore_reason"
METRICS = "/scratch/gpfs/CHIJ/ziran/.tmp/claude-374317/-scratch-gpfs-CHIJ-bohan-1-innovation-proior/20154e8e-f2e8-4272-a550-4f0c059fe5b0/scratchpad/case/innov/explore_metrics.csv"

ARMS = ["rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20",
        "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20"]
NAME = dict(zip(ARMS, ["9B RL(base)", "9B 我们", "9B lora",
                       "4B RL(base)", "4B 我们", "4B lora"]))
PAIRS = [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
         ("rlv5_base_s20", "rlv5_lo32nm_a10_s20", "9B lora − RL(base)"),
         ("rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20", "9B lora − 我们"),
         ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)"),
         ("rlv5_4b_base_s20", "rlv5_4b_lo32nm_a10_s20", "4B lora − RL(base)"),
         ("rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20", "4B lora − 我们")]
BENCH_ORDER = ["frontiercs_research", "frontiercs", "alebench"]
BENCH_NAME = {"frontiercs_research": "FCS-research(主表)", "frontiercs": "FrontierCS",
              "alebench": "ALE-Bench"}
PREFIX = {"frontiercs_research": "", "frontiercs": "fcs_", "alebench": "ale_"}

_BUF = []


def emit(s=""):
    sys.stdout.write(s + "\n")
    _BUF.append(s)


def sign_p(w, l):
    n = w + l
    if n == 0:
        return float("nan")
    c = [math.comb(n, k) for k in range(n + 1)]
    return min(1.0, 2 * sum(c[:min(w, l) + 1]) / float(sum(c)))


def _key(name):
    """盲盒语料在 scratchpad(8MB,不进仓库),key 另存一份进 labels 目录 —— 同 v1/v2/v3 惯例。"""
    for p in (f"{LAB}/_{name}.json", f"{BLIND}/{name}.json"):
        if os.path.exists(p):
            return json.load(open(p))
    raise SystemExit(f"找不到 {name}.json({LAB} / {BLIND})")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def load_human(key, kb):
    """-> hum[arm][slug] = (n_routes, n_abandoned, n_adopted),slugs,bad"""
    hum = collections.defaultdict(dict)
    slugs, bad = [], 0
    for f in sorted(os.listdir(LAB)):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        slug = f[:-5]
        if slug not in key:
            emit(f"[bad] {slug} 不在 key.json"); bad += 1; continue
        d = json.load(open(f"{LAB}/{f}"))
        if sorted(d) != list("ABCDEF"):
            emit(f"[bad] {slug} 块不齐 {sorted(d)}"); bad += 1; continue
        ok = True
        for li, v in d.items():
            rts = v.get("routes", [])
            ad = sum(1 for r in rts if r.get("adopted"))
            if v.get("n_routes") != len(rts):
                emit(f"[warn] {slug}/{li} n_routes={v.get('n_routes')} 但 routes={len(rts)},按 routes 纠正")
                v["n_routes"] = len(rts)
            nab = sum(1 for r in rts if not r.get("adopted"))
            if v.get("n_abandoned") != nab:
                emit(f"[warn] {slug}/{li} n_abandoned={v.get('n_abandoned')} 但实际 {nab},按 routes 纠正")
                v["n_abandoned"] = nab
            if ad > 1:
                emit(f"[bad] {slug}/{li} adopted={ad} 条(最多 1)"); bad += 1; ok = False
        if not ok:
            continue
        slugs.append(slug)
        for li, v in d.items():
            rts = v.get("routes", [])
            hum[key[slug][li]][slug] = (v["n_routes"], v["n_abandoned"],
                                        sum(1 for r in rts if r.get("adopted")))
    return hum, slugs, bad


def load_regex(kb):
    """正则侧,只取 sample_idx==0 -> re_[arm][slug] = (n_reason, n_abandon)"""
    rex = collections.defaultdict(dict)
    if not os.path.exists(METRICS):
        emit(f"> ⚠ 找不到 `{METRICS}`,§0/§3 的正则侧留空。\n")
        return rex
    for r in csv.DictReader(open(METRICS)):
        if r["sample_idx"] != "0" or r["arm"] not in ARMS:
            continue
        b = r["bench"]
        if b not in PREFIX:
            continue
        slug = PREFIX[b] + re.sub(r"[^A-Za-z0-9_.-]+", "_", r["problem"])
        if kb.get(slug) != b:
            continue
        rex[r["arm"]][slug] = (int(r["n_reason"]), int(r["n_abandon"]))
    return rex


def per_arm(hum, slugs, title):
    emit(f"### {title}(n = {len(slugs)} 题)\n")
    emit("| 臂 | routes 均值 | 放弃的路均值 | 没定下来的抽样 |")
    emit("|---|---:|---:|---:|")
    for a in ARMS:
        v = [hum[a][s] for s in slugs if s in hum[a]]
        if not v:
            continue
        emit(f"| {NAME[a]} | {sum(x[0] for x in v)/len(v):.2f} | "
             f"{sum(x[1] for x in v)/len(v):.2f} | {sum(1 for x in v if x[2] == 0)}/{len(v)} |")
    emit()


def per_pair(hum, slugs, title):
    emit(f"### {title}——逐题配对(胜/负/平 = 后者的 routes 更多/更少/相同)\n")
    emit("| 对照 | routes 均差 | 胜/负/平 | 符号 p | 放弃的路均差 |")
    emit("|---|---:|:---:|---:|---:|")
    for lo, hi, lbl in PAIRS:
        w = l = t = 0
        dr, da = [], []
        for s in slugs:
            if s not in hum[lo] or s not in hum[hi]:
                continue
            r0, a0, _ = hum[lo][s]
            r1, a1, _ = hum[hi][s]
            dr.append(r1 - r0); da.append(a1 - a0)
            w += r1 > r0; l += r1 < r0; t += r1 == r0
        if not dr:
            continue
        emit(f"| {lbl} | {sum(dr)/len(dr):+.2f} | {w}/{l}/{t} | {sign_p(w, l):.4f} | "
             f"{sum(da)/len(da):+.2f} |")
    emit()


def main():
    key, kb = _key("key"), _key("key_bench")
    hum, slugs, bad = load_human(key, kb)
    rex = load_regex(kb)
    groups = collections.defaultdict(list)
    for s in slugs:
        groups[kb.get(s, "未标 bench")].append(s)

    emit("# 一条输出之内考虑过几条路 —— 人工标注,兼查那张正则表\n")
    emit("同 39 题 × 6 条 RL 臂 × **同一条抽样(`sample_idx=0`)**,只给推理不给最终代码。")
    emit("协议 `EXPLORE_REASON_PROTOCOL.md`;语料 `explore_reason_corpus.py`。\n")
    emit(f"合规:{'通过' if bad == 0 else str(bad) + ' 处问题'} | 已标注 {len(slugs)} 题 | "
         + "、".join(f"{BENCH_NAME.get(b, b).split('(')[0]} {len(v)}" for b, v in
                     sorted(groups.items(), key=lambda kv: BENCH_ORDER.index(kv[0])
                            if kv[0] in BENCH_ORDER else 99)) + "\n")

    emit("## 0. 覆盖率:两把尺子各自看得见几个格子\n")
    emit("`explore.py` 里那个循环有两道闸门:没有 `</think>` 就 `continue`、"
         "最终代码为空也 `continue`。**哑火的臂会被整臂删掉**,正则表上就看不见它了。"
         "人工标注没有这两道闸门 —— 推理是空的就记 0 条路,照样进分母。\n")
    emit("| 臂 | 人工有的格子 | 正则有的格子(si=0) | 正则丢掉的 |")
    emit("|---|---:|---:|---:|")
    for a in ARMS:
        h = sum(1 for s in slugs if s in hum[a])
        r = sum(1 for s in slugs if s in rex.get(a, {}))
        emit(f"| {NAME[a]} | {h} | {r} | **{h - r}** |")
    emit()

    emit("## 1. 逐臂(人工口径)\n")
    for b in BENCH_ORDER:
        if groups.get(b):
            per_arm(hum, groups[b], BENCH_NAME[b])

    emit("## 2. 对照:逐题配对(人工口径)\n")
    for b in BENCH_ORDER:
        if groups.get(b):
            per_pair(hum, groups[b], BENCH_NAME[b])
    per_pair(hum, slugs, "全部 bench 合计")

    emit("## 3. 尺子核验:正则表量到的是不是同一回事\n")
    both = [(a, s) for a in ARMS for s in slugs if s in hum[a] and s in rex.get(a, {})]
    emit(f"两侧都有的格子:**{len(both)}**(共 {len(ARMS)} 臂 × {len(slugs)} 题 = "
         f"{len(ARMS) * len(slugs)} 个)\n")
    if both:
        hx = [hum[a][s][0] for a, s in both]
        rx = [rex[a][s][0] for a, s in both]
        ha = [hum[a][s][1] for a, s in both]
        ra = [rex[a][s][1] for a, s in both]
        emit("| 量 | Pearson r | Spearman ρ |")
        emit("|---|---:|---:|")
        emit(f"| 路线数(人工 routes vs 正则 n_reason) | {pearson(hx, rx):+.3f} | "
             f"{pearson(rank(hx), rank(rx)):+.3f} |")
        emit(f"| 放弃数(人工 n_abandoned vs 正则 n_abandon) | {pearson(ha, ra):+.3f} | "
             f"{pearson(rank(ha), rank(ra)):+.3f} |")
        emit()
        emit("### 逐臂并排(只在两侧都有的格子上算)\n")
        emit("| 臂 | n格 | 人工 routes | 正则 n_reason | 人工 放弃 | 正则 n_abandon |")
        emit("|---|---:|---:|---:|---:|---:|")
        for a in ARMS:
            cs = [s for s in slugs if s in hum[a] and s in rex.get(a, {})]
            if not cs:
                emit(f"| {NAME[a]} | 0 | — | — | — | — |")
                continue
            emit(f"| {NAME[a]} | {len(cs)} | {sum(hum[a][s][0] for s in cs)/len(cs):.2f} | "
                 f"{sum(rex[a][s][0] for s in cs)/len(cs):.2f} | "
                 f"{sum(hum[a][s][1] for s in cs)/len(cs):.2f} | "
                 f"{sum(rex[a][s][1] for s in cs)/len(cs):.2f} |")
        emit()
        emit("### 六个对照的方向一致吗(只在两侧都有的格子上配对)\n")
        emit("| 对照 | 人工 Δroutes | 正则 Δn_reason | 同向 |")
        emit("|---|---:|---:|:---:|")
        for lo, hi, lbl in PAIRS:
            cs = [s for s in slugs
                  if s in hum[lo] and s in hum[hi] and s in rex.get(lo, {}) and s in rex.get(hi, {})]
            if not cs:
                emit(f"| {lbl} | — | — | 没有可配对的格子 |")
                continue
            dh = sum(hum[hi][s][0] - hum[lo][s][0] for s in cs) / len(cs)
            dr = sum(rex[hi][s][0] - rex[lo][s][0] for s in cs) / len(cs)
            same = "✓" if (dh > 0) == (dr > 0) or (dh == 0 and dr == 0) else "**✗**"
            emit(f"| {lbl}(n={len(cs)}) | {dh:+.2f} | {dr:+.2f} | {same} |")
        emit()

    emit("## 4. 逐题明细(routes / 放弃)\n")
    for b in BENCH_ORDER:
        ss = sorted(groups.get(b, []))
        if not ss:
            continue
        emit(f"### {BENCH_NAME[b]}\n")
        emit("| 臂 | " + " | ".join(s[:22] for s in ss) + " |")
        emit("|---|" + "---:|" * len(ss))
        for a in ARMS:
            emit(f"| {NAME[a]} | " + " | ".join(
                f"{hum[a][s][0]}/{hum[a][s][1]}" if s in hum[a] else "-" for s in ss) + " |")
        emit()

    json.dump({a: hum[a] for a in ARMS}, open(f"{S}/explore_reason_agg.json", "w"),
              ensure_ascii=False, indent=1)


main()
with open(os.path.join(S, "explore_reason_agg.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(_BUF) + "\n")
