"""@5 标注统计的审计:语料本身会不会造出结论。

三个问题,一个一个查:
  1 语料把每条最终答案截到 4000 字符。各臂被截的比例差 8 倍(base/SFT 40-50%,
    我们 6-22%),标注者看到的东西本来就不一样多。
  2 截断和 n_method 正相关(合并 r=+0.29)。那我们的劣势会不会是截断造的?
  3 「交了最终块但标注判无方法」那 25 例到底是什么。

结论写在各节末尾。要点:**劣势不是截断造的** —— 在标注者看到完整答案的那一层里
差距反而更大;但 25 例里 21 例确实是截断的副作用,该说清楚。
"""
import glob, json, math, os, re, statistics, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
SUB, BENCH, NO = "research_thinking_32k_vllm", "frontiercs_research", "没产出方法"
TRUNC = 4000                      # explore_corpus.py 写语料时每条 fin[:4000]

ARMS = ["base9b_v2c", "ft01mix_a10", "rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "base4b", "4b_ft01mix_a10", "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
NAME = dict(zip(ARMS, ["9B base", "9B SFT", "9B RL(base)", "9B 我们",
                       "4B base", "4B SFT", "4B RL(base)", "4B 我们"]))
# 用户裁决 2026-09-17:**只看 RL 之后,不看 RL 之前**。RL 是最终 shape 出来的模型,
# 所以主对照只有「我们的 RL − baseline 的 RL」。SFT / base 那些对照降为附录,
# 留着是为了不丢历史,不进主表、不进论文正文。
PAIRS = [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
         ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)")]
PAIRS_PRE = [("ft01mix_a10", "rlv5_ft01mix_a10_s20", "9B 我们 − SFT"),
             ("base9b_v2c", "rlv5_ft01mix_a10_s20", "9B 我们 − base"),
             ("4b_ft01mix_a10", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − SFT"),
             ("base4b", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − base")]


def slugof(p):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", p)


def load():
    fin, sc, prob_of = {}, {}, {}
    for a in ARMS:
        d, s2 = {}, {}
        for f in sorted(glob.glob(f"{D}/cc_eval_{a}_{SUB}/shard_*/samples.jsonl")):
            for ln in open(f):
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if r.get("data_source") != BENCH or r.get("error"):
                    continue
                m = r.get("metrics") or {}
                v = m.get("score", r.get("score"))
                if v is None:
                    continue
                p = str(r["ground_truth"])
                prob_of[slugof(p)] = p
                t = r.get("text") or ""
                i = t.rfind("</think>")
                k = (p, int(r.get("sample_idx", -1)))
                d[k] = t[i + 8:].strip() if i >= 0 else ""
                s2[k] = float(v)
        fin[a], sc[a] = d, s2
    return fin, sc, prob_of


def is_no(c):
    return bool(c["no_method"]) if "no_method" in c else NO in c.get("desc", "")


def sign_p(w, l):
    n = w + l
    if n == 0:
        return float("nan")
    c = [math.comb(n, k) for k in range(n + 1)]
    return min(1.0, 2 * sum(c[:min(w, l) + 1]) / float(sum(c)))


def pear(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main():
    fin, sc, prob_of = load()
    key = json.load(open(f"{S}/explore_blind/key.json"))
    rec, nom = {}, []
    for f in sorted(os.listdir(f"{S}/explore_labels")):
        if not f.endswith(".json"):
            continue
        slug = f[:-5]
        if slug not in key or slug not in prob_of:
            continue
        p = prob_of[slug]
        lab = json.load(open(f"{S}/explore_labels/{f}"))
        for li, v in lab.items():
            a = key[slug][li]
            L = [len(fin[a].get((p, si), "")) for si in range(5)]
            rec[(a, slug)] = dict(live=sum(1 for x in L if x > 0),
                                  trunc=sum(1 for x in L if x > TRUNC),
                                  nm=len([c for c in v["clusters"] if not is_no(c)]),
                                  med=statistics.median([x for x in L if x > 0] or [0]))
            for c in v["clusters"]:
                if not is_no(c):
                    continue
                for si in c["samples"]:
                    t = fin[a].get((p, si - 1), "")
                    if t:
                        nom.append((a, len(t), sc[a].get((p, si - 1))))
    slugs = sorted({s for (_, s) in rec})
    probs = [prob_of[s] for s in slugs]
    allp = sorted({p for a in ARMS for (p, _) in fin[a]})
    print("# @5 标注统计的审计:语料会不会自己造出结论\n")
    print(f"FrontierCS-research,已标注 {len(slugs)} 题 / 全集 {len(allp)} 题。\n")

    print(f"## 1. 语料把每条最终答案截到 {TRUNC} 字符 —— 各臂被截的比例差 8 倍\n")
    print("| 臂 | 有最终答案的格子 | 超 4000 字(只看到前 4000) | 占比 | 中位长 | p90 |")
    print("|---|---:|---:|---:|---:|---:|")
    for a in ARMS:
        L = sorted(len(fin[a].get((p, si), "")) for p in probs for si in range(5)
                   if fin[a].get((p, si)))
        if not L:
            continue
        over = sum(1 for x in L if x > TRUNC)
        print(f"| {NAME[a]} | {len(L)} | {over} | {over/len(L):.1%} | "
              f"{L[len(L)//2]} | {L[int(len(L)*0.9)]} |")
    print("\n> 标注者看到的**不是同一种东西**:base/SFT 有四到五成的答案只露出前 4000 字,"
          "我们只有 6-22%。这条必须先说,再谈任何差值。\n")

    print("## 2. 抽的 26 题代表性:标注子集 vs 全 64 题(题均分)\n")
    print("| 臂 | 标注 26 题 | 全 64 题 | 差 |")
    print("|---|---:|---:|---:|")
    for a in ARMS:
        def m(ps):
            v = [sc[a][(p, si)] for p in ps for si in range(5) if (p, si) in sc[a]]
            return statistics.mean(v) if v else float("nan")
        x, y = m(probs), m(allp)
        print(f"| {NAME[a]} | {x:.2f} | {y:.2f} | {x-y:+.2f} |")
    print("\n> 抽题是固定种子打乱后取前 26,不是挑的。逐臂偏差 −1.7 到 +1.6 分,"
          "对我们略偏有利(4B 我们 +1.55),记下来。\n")

    print("## 3. 截断和 n_method 是正相关 —— 那劣势会不会是截断造的\n")
    allr = [x for x in rec.values() if x["live"] > 0]
    full = [x for x in allr if x["live"] == 5]
    print(f"合并八臂(n={len(allr)}):r(块内截断数, n_method) = "
          f"**{pear([x['trunc'] for x in allr], [x['nm'] for x in allr]):+.3f}**;"
          f" r(活抽数, n_method) = **{pear([x['live'] for x in allr], [x['nm'] for x in allr]):+.3f}**。\n")
    print(f"只取 5 抽全活的格子(n={len(full)},把死活这个混杂去掉):"
          f"r(截断数, n_method) = {pear([x['trunc'] for x in full], [x['nm'] for x in full]):+.3f}。\n")
    print("所以要分层重算。**我们的臂截断最少,如果截断会抬高 n_method,"
          "那是它抬高了对手、不是压低了我们** —— 下面三层给出答案。\n")
    print("> 每层前两行是**主对照(RL 之后)**,其余是 RL 之前的附录。"
          "4B 我们 − RL(base) 在这里恒为 ⚠不足 —— 对手 5 抽全活的题几乎没有,"
          "这正是 `explore_why_nomethod.md` 那条结论,不是分层没做出来。\n")
    for title, keep in [
            ("3.1 基线:两臂都 5 抽全活", lambda A, B: A["live"] == 5 and B["live"] == 5),
            ("3.2 **两臂都零截断**(标注者看到的是完整答案)",
             lambda A, B: A["live"] == 5 and B["live"] == 5 and A["trunc"] == 0 and B["trunc"] == 0),
            ("3.3 两臂**截断格数相等**(截断量已配平)",
             lambda A, B: A["live"] == 5 and B["live"] == 5 and A["trunc"] == B["trunc"])]:
        print(f"### {title}\n")
        print("| 对照 | 可配对题 | Δ n_method | 胜/负/平 | 符号 p |")
        print("|---|---:|---:|:---:|---:|")
        for lo, hi, lbl in PAIRS + PAIRS_PRE:
            d, w, l, t = [], 0, 0, 0
            for s in slugs:
                A, B = rec.get((lo, s)), rec.get((hi, s))
                if not A or not B or not keep(A, B):
                    continue
                d.append(B["nm"] - A["nm"])
                w += B["nm"] > A["nm"]; l += B["nm"] < A["nm"]; t += B["nm"] == A["nm"]
            if len(d) < 3:
                print(f"| {lbl} | {len(d)} | ⚠不足 | — | — |")
                continue
            print(f"| {lbl} | {len(d)} | {statistics.mean(d):+.2f} | {w}/{l}/{t} | {sign_p(w,l):.4f} |")
        print()
    print("> **答案:不是截断造的。** 在两臂都零截断那一层,我们的劣势**更大**"
          "(9B −SFT −1.21→−1.29,4B −SFT −2.06→−2.50,4B −base −2.24→−3.00),"
          "只是 n 掉到 6-8 所以 p 变弱。9B 我们 − RL(base) 三层都在 −0.1 到 −0.3,是真打平。\n")

    print("## 4. 「交了最终块但标注判无方法」那 25 例\n")
    over = sum(1 for _, L, _ in nom if L > TRUNC)
    zero = sum(1 for _, _, s in nom if s == 0)
    print(f"共 **{len(nom)}** 例。其中最终块超 4000 字(标注者只看到 import + kernel 开头)的 "
          f"**{over}/{len(nom)}**;score 恰为 0 的 **{zero}/{len(nom)}**。\n")
    print("| 臂 | 例数 | 其中超 4000 字 |")
    print("|---|---:|---:|")
    for a in ARMS:
        g = [x for x in nom if x[0] == a]
        if g:
            print(f"| {NAME[a]} | {len(g)} | {sum(1 for x in g if x[1] > TRUNC)} |")
    print("\n> 这一类**大部分是语料截断的副作用**:长代码转储被切在 4000 字,"
          "露出来的只有 import 和 kernel 序言,标注者自然判不出做法。"
          "但它们**全部 score=0**,所以对分数侧的结论没有影响,"
          "只影响 n_method 的绝对值 —— 而且它砸在我们 9B 那条臂上最多(8 例),"
          "方向同样是压低我们。\n")


if __name__ == "__main__":
    main()
