"""@5 标注统计的审计:语料本身会不会造出结论。

只覆盖 **RL 之后的四条臂**(用户裁决:主表与这两份案例统计都只看 RL 之后),
并且 **FCS-research 与 ALE-Bench 分开报** —— 两个 bench 的答案长度差一个量级,
合起来看会把两件事搅成一件。

查出来的核心事实(§2):v1 语料把每条最终答案截到 4000 字符,而**标注者看到多少,
直接决定他判不判「没产出方法」** ——

    看到完整答案     判无方法  1.8%
    只看到 50-100%           8.7%
    只看到 25-50%           27.3%
    **只看到 <25%**       **58.1%**

这不是「截断会略微影响判断」,是这道闸门自己在生产「没产出方法」。而且它砸得不匀:
ALE 上我们 9B 有 7 个「判无方法」的格子可见比例 <50%,对手只有 1 个。所以 v1 的
n_method 差值里混着一截纯粹的语料假象,**必须重标**(见 `explore_corpus2.py`:
v2 语料一个字不截、只放四条 RL 臂、题目集合与 v1 完全相同)。

本文件保留 v1 的审计结论,作为「为什么要有 v2」的证据链,不是最终结果。
"""
import glob, json, math, os, re, statistics, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
NO = "没产出方法"
TRUNC = 4000                      # v1 explore_corpus.py 写语料时每条 fin[:4000]

# (data_source, 输出目录后缀, slug 前缀, 显示名)
BENCHES = [("frontiercs_research", "research_thinking_32k_vllm", "", "FCS-research"),
           ("alebench", "thinking_32k_both_vllm", "ale_", "ALE-Bench"),
           ("frontiercs", "thinking_32k_both_vllm", "fcs_", "FrontierCS")]

ARMS = ["rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
NAME = dict(zip(ARMS, ["9B RL(base)", "9B 我们", "4B RL(base)", "4B 我们"]))
PAIRS = [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
         ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)")]


def slugof(p):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", p)


def load():
    fin = {a: {} for a in ARMS}
    sc = {a: {} for a in ARMS}
    prob_of = {}
    for bench, sub, pre, _ in BENCHES:
        for a in ARMS:
            for f in sorted(glob.glob(f"{D}/cc_eval_{a}_{sub}/shard_*/samples.jsonl")):
                for ln in open(f):
                    try:
                        r = json.loads(ln)
                    except Exception:
                        continue
                    if r.get("data_source") != bench or r.get("error"):
                        continue
                    m = r.get("metrics") or {}
                    v = m.get("score", r.get("score"))
                    if v is None:
                        continue
                    p = str(r["ground_truth"])
                    prob_of[pre + slugof(p)] = (bench, p)
                    t = r.get("text") or ""
                    i = t.rfind("</think>")
                    k = (bench, p, int(r.get("sample_idx", -1)))
                    fin[a][k] = t[i + 8:].strip() if i >= 0 else ""
                    sc[a][k] = float(v)
    return fin, sc, prob_of


def is_no(c):
    return bool(c["no_method"]) if "no_method" in c else NO in c.get("desc", "")


def sign_p(w, l):
    n = w + l
    if n == 0:
        return float("nan")
    c = [math.comb(n, k) for k in range(n + 1)]
    return min(1.0, 2 * sum(c[:min(w, l) + 1]) / float(sum(c)))


def main():
    fin, sc, prob_of = load()
    key = json.load(open(f"{S}/explore_blind/key.json"))
    rec, cells, bench_of = {}, [], {}
    for f in sorted(os.listdir(f"{S}/explore_labels")):
        if not f.endswith(".json"):
            continue
        slug = f[:-5]
        if slug not in key or slug not in prob_of:
            continue
        bench, p = prob_of[slug]
        bench_of[slug] = bench
        lab = json.load(open(f"{S}/explore_labels/{f}"))
        for li, v in lab.items():
            a = key[slug][li]
            if a not in NAME:          # 只看 RL 之后的四条臂
                continue
            L = [len(fin[a].get((bench, p, si), "")) for si in range(5)]
            rec[(a, slug)] = dict(live=sum(1 for x in L if x > 0),
                                  trunc=sum(1 for x in L if x > TRUNC),
                                  nm=len([c for c in v["clusters"] if not is_no(c)]))
            for c in v["clusters"]:
                for si in c["samples"]:
                    n = L[si - 1]
                    if n == 0:
                        continue       # 没交东西,不是语料的事
                    cells.append(dict(arm=a, bench=bench, slug=slug, si=si, L=n,
                                      vis=min(1.0, TRUNC / n), no=is_no(c),
                                      sc=sc[a].get((bench, p, si - 1))))
    allslugs = sorted({s for (_, s) in rec})
    have = [(b, nm) for b, _, _, nm in BENCHES
            if any(bench_of.get(s) == b for s in allslugs)]

    print("# @5 标注统计的审计:语料会不会自己造出结论\n")
    print("只覆盖 **RL 之后的四条臂**;两个 bench 分开报。覆盖:"
          + "、".join(f"{nm} {sum(1 for s in allslugs if bench_of.get(s)==b)} 题"
                     for b, nm in have) + "。\n")

    print(f"## 1. v1 语料把每条最终答案截到 {TRUNC} 字符 —— 关键不是「截没截」,是「露出多少」\n")
    print("`可见比例` = min(1, 4000 / 答案长度)。只看有最终答案的格子。\n")
    print("| bench | 臂 | 活格 | 超 4000 字 | 可见比例 <50% 的格子 | 最差可见比例 | 长度中位 | 长度 p90 | 长度最大 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for bench, disp in have:
        for a in ARMS:
            g = [c for c in cells if c["arm"] == a and c["bench"] == bench]
            if not g:
                continue
            Ls = sorted(c["L"] for c in g)
            print(f"| {disp} | {NAME[a]} | {len(g)} | {sum(1 for c in g if c['L']>TRUNC)} | "
                  f"{sum(1 for c in g if c['vis']<0.5)} | {min(c['vis'] for c in g):.3f} | "
                  f"{Ls[len(Ls)//2]} | {Ls[int(len(Ls)*0.9)]} | {Ls[-1]} |")
    print("\n> 最长的一条答案有 10.8 万字,标注者看到的是它的 **3.7%**。"
          "「超不超 4000 字」这个二值统计(9B 两臂 20.2% vs 22.0%,看着很平)完全掩盖了这件事。\n")

    print("## 2. ★ 这道闸门自己在生产「没产出方法」\n")
    print("把所有活格按标注者看到的比例分层,看每层有多大比例被判「没产出方法」。\n")
    print("| 标注者看到的比例 | 活格数 | 被判「没产出方法」 | 占比 |")
    print("|---|---:|---:|---:|")
    for lo, hi, lbl in [(0, .25, "< 25%"), (.25, .5, "25 – 50%"),
                        (.5, 1.0, "50 – 100%"), (1.0, 1.01, "**看到完整答案**")]:
        g = [c for c in cells if lo <= c["vis"] < hi]
        if g:
            print(f"| {lbl} | {len(g)} | {sum(1 for c in g if c['no'])} | "
                  f"**{sum(1 for c in g if c['no'])/len(g):.1%}** |")
    print("\n> 完整可见时判无方法的只有 1.8%;只看到不足四分之一时是 **58.1%**,差 32 倍。"
          "这不是「截断略有影响」,是闸门在造数据。\n")
    print("| bench | 臂 | 判无方法的活格 | 其中可见比例 <50% |")
    print("|---|---|---:|---:|")
    for bench, disp in have:
        for a in ARMS:
            g = [c for c in cells if c["arm"] == a and c["bench"] == bench and c["no"]]
            if g:
                print(f"| {disp} | {NAME[a]} | {len(g)} | {sum(1 for c in g if c['vis']<0.5)} |")
    print("\n> 砸得不匀:ALE 上我们 9B 的 8 个「判无方法」里 7 个是只看到不足一半,"
          "对手 2 个里只有 1 个。v1 那个 `9B 我们 − RL(base) = −0.23` 里面**混着这一截假象**。\n")
    print("> **处理方式:重标。** `explore_corpus2.py` 生成的 v2 语料一个字不截"
          "(全量 2.6MB / 39 题),只放四条 RL 臂,题目集合与 v1 完全相同因而可配对;"
          "协议见 `EXPLORE_LABEL_PROTOCOL2.md`,新增一句「长 ≠ 没方法」。\n")

    print("## 3. 抽的题有没有代表性(题均分:标注子集 vs 该 bench 全集)\n")
    print("| bench | 臂 | 标注子集 | 全集 | 差 |")
    print("|---|---|---:|---:|---:|")
    for bench, disp in have:
        ps = {prob_of[s][1] for s in allslugs if bench_of.get(s) == bench}
        for a in ARMS:
            sub = [v for (b, p, _), v in sc[a].items() if b == bench and p in ps]
            allv = [v for (b, _, _), v in sc[a].items() if b == bench]
            if not sub or not allv:
                continue
            print(f"| {disp} | {NAME[a]} | {statistics.mean(sub):.2f} | "
                  f"{statistics.mean(allv):.2f} | {statistics.mean(sub)-statistics.mean(allv):+.2f} |")
    print("\n> 抽题是固定种子打乱后取前 N,不是挑的。\n")

    print("## 4. 分层重算:把截断这个混杂去掉之后,对照还剩多少\n")
    for bench, disp in have:
        slugs = [s for s in allslugs if bench_of.get(s) == bench]
        print(f"### {disp}\n")
        print("| 层 | 对照 | 可配对题 | Δ n_method | 胜/负/平 | 符号 p |")
        print("|---|---|---:|---:|:---:|---:|")
        for title, keep in [
                ("原样(全部题)", lambda A, B: True),
                ("两臂都 5 抽全活", lambda A, B: A["live"] == 5 and B["live"] == 5),
                ("**两臂都零截断**", lambda A, B: A["live"] == 5 and B["live"] == 5
                                              and A["trunc"] == 0 and B["trunc"] == 0),
                ("两臂截断格数相等", lambda A, B: A["live"] == 5 and B["live"] == 5
                                              and A["trunc"] == B["trunc"])]:
            for lo, hi, lbl in PAIRS:
                d, w, l, t = [], 0, 0, 0
                for s in slugs:
                    A, B = rec.get((lo, s)), rec.get((hi, s))
                    if not A or not B or not keep(A, B):
                        continue
                    d.append(B["nm"] - A["nm"])
                    w += B["nm"] > A["nm"]; l += B["nm"] < A["nm"]; t += B["nm"] == A["nm"]
                if len(d) < 3:
                    print(f"| {title} | {lbl} | {len(d)} | ⚠不足 | — | — |")
                    continue
                print(f"| {title} | {lbl} | {len(d)} | {statistics.mean(d):+.2f} | "
                      f"{w}/{l}/{t} | {sign_p(w,l):.4f} |")
        print()
    print("> 9B 那一对:research 从 −0.27 收到 −0.11 ~ −0.14,ALE 的零截断层只剩 3 题、不可读。"
          "两边 n 都掉到个位数 —— 这正是必须重标而不是继续分层的原因:"
          "分层是把有问题的格子扔掉,重标是把它们修好。\n")

    print("## 5. 「交了最终块但标注判无方法」逐例\n")
    nom = [c for c in cells if c["no"]]
    print(f"共 **{len(nom)}** 例。其中超 4000 字 **{sum(1 for c in nom if c['L']>TRUNC)}**;"
          f"可见比例 <50% **{sum(1 for c in nom if c['vis']<0.5)}**;"
          f"score 恰为 0 **{sum(1 for c in nom if c['sc'] == 0)}**。\n")
    print("| bench | 臂 | 题 | 抽样 | 答案长度 | 标注者看到 | score |")
    print("|---|---|---|---:|---:|---:|---:|")
    for c in sorted(nom, key=lambda c: (c["bench"], c["arm"], c["slug"], c["si"])):
        disp = dict((b, nm) for b, _, _, nm in BENCHES)[c["bench"]]
        print(f"| {disp} | {NAME[c['arm']]} | `{c['slug'][:34]}` | {c['si']} | {c['L']} | "
              f"{c['vis']:.0%} | {c['sc']:.1f} |")
    print("\n> 这些格子里大多数**根本没被完整看过**。分数侧不受影响(几乎全是 0 分),"
          "但 n_method 的绝对值和差值都受影响 —— v2 重标就是为了把这一列修掉。\n")


if __name__ == "__main__":
    main()
