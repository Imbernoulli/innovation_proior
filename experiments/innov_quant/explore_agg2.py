"""@5 做法多样性 · **v2**(不截断语料)的汇总,并与 v1 逐题配对比较。

v1 的问题写在 `explore_label_audit.md` §2:4000 字的语料闸门自己在生产
「没产出方法」(完整可见 1.8% → 只看到 <25% 时 58.1%),而且砸得不匀。
v2 重做的只有语料和标注,**模型输出、题目集合、配对方式一个没变**,
所以 v1→v2 的差就是那道闸门的影响。

口径与 v1 相同:
  n_approach = 5 抽里的做法数,「没产出方法」也算一类(坍塌的臂照样进分母)。
  n_method   = 去掉「没产出方法」后的做法数。
"""
import glob, json, math, os, re, statistics, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
B2, L2 = f"{S}/explore_blind2", f"{S}/explore_labels2"
B1, L1 = f"{S}/explore_blind", f"{S}/explore_labels"
NO = "没产出方法"
CAP = 32768

BENCHES = [("frontiercs_research", "research_thinking_32k_vllm", "", "FCS-research"),
           ("alebench", "thinking_32k_both_vllm", "ale_", "ALE-Bench"),
           ("frontiercs", "thinking_32k_both_vllm", "fcs_", "FrontierCS")]
ARMS = ["rlv5_base_s20", "rlv5_ft01mix_a10_s20",
        "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
NAME = dict(zip(ARMS, ["9B RL(base)", "9B 我们", "4B RL(base)", "4B 我们"]))
PAIRS = [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
         ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)")]


def sign_p(w, l):
    n = w + l
    if n == 0:
        return float("nan")
    c = [math.comb(n, k) for k in range(n + 1)]
    return min(1.0, 2 * sum(c[:min(w, l) + 1]) / float(sum(c)))


def slugof(p):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", p)


def is_no(c):
    return bool(c["no_method"]) if "no_method" in c else NO in c.get("desc", "")


def load_live():
    """(arm, bench, prob, si) -> (有最终答案?, completion_tokens);同时收分数"""
    out = {a: {} for a in ARMS}
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
                    if m.get("score", r.get("score")) is None:
                        continue
                    p = str(r["ground_truth"])
                    prob_of[pre + slugof(p)] = (bench, p)
                    t = r.get("text") or ""
                    i = t.rfind("</think>")
                    k = (bench, p, int(r.get("sample_idx", -1)))
                    out[a][k] = (bool(i >= 0 and t[i + 8:].strip()), r.get("completion_tokens"))
                    sc[a][k] = float(m.get("score", r.get("score")))
    return out, sc, prob_of


def read_labels(blind, lab, letters):
    """-> (rec[(arm,slug)] = (n_app, n_met, nomiss), 合规问题列表, 已读 slug 集)"""
    key = json.load(open(f"{blind}/key.json"))
    rec, bad, slugs = {}, [], set()
    for f in sorted(os.listdir(lab)):
        if not f.endswith(".json"):
            continue
        slug = f[:-5]
        if slug not in key:
            bad.append(f"{slug}:不在 key.json"); continue
        try:
            d = json.load(open(f"{lab}/{f}"))
        except Exception as e:
            bad.append(f"{slug}:JSON 坏了 {e!r}"); continue
        if sorted(d) != list(letters):
            bad.append(f"{slug}:块 {sorted(d)}"); continue
        ok = True
        for li, v in d.items():
            ss = sorted(x for c in v["clusters"] for x in c["samples"])
            if ss != [1, 2, 3, 4, 5]:
                bad.append(f"{slug}/{li}:samples={ss}"); ok = False
            if sum(1 for c in v["clusters"] if is_no(c)) > 1:
                bad.append(f"{slug}/{li}:{sum(1 for c in v['clusters'] if is_no(c))} 个 no_method 类"); ok = False
            if v.get("n") != len(v["clusters"]):
                v["n"] = len(v["clusters"])      # n 是冗余字段,以 clusters 为准
        if not ok:
            continue
        slugs.add(slug)
        for li, v in d.items():
            arm = key[slug][li]
            if arm not in NAME:
                continue
            nm = [c for c in v["clusters"] if not is_no(c)]
            rec[(arm, slug)] = (len(v["clusters"]), len(nm),
                                sum(len(c["samples"]) for c in v["clusters"] if is_no(c)))
    return rec, bad, slugs


def block(rec, slugs, live, prob_of, title):
    kb = json.load(open(f"{B2}/key_bench.json"))
    out = []
    for bench, _, _, disp in BENCHES:
        ss = sorted(s for s in slugs if kb.get(s) == bench)
        if not ss:
            continue
        out.append((bench, disp, ss))
    print(f"## {title}\n")
    for bench, disp, ss in out:
        print(f"### {disp}({len(ss)} 题)\n")
        print("| 臂 | n_approach | n_method | 没产出方法的抽样 | 有最终答案的抽样 |")
        print("|---|---:|---:|---:|---:|")
        for a in ARMS:
            v = [rec[(a, s)] for s in ss if (a, s) in rec]
            if not v:
                continue
            lv = sum(1 for s in ss for si in range(5)
                     if (live[a].get((bench, prob_of[s][1], si)) or (False, None))[0])
            print(f"| {NAME[a]} | {sum(x[0] for x in v)/len(v):.2f} | {sum(x[1] for x in v)/len(v):.2f} | "
                  f"{sum(x[2] for x in v)}/{5*len(v)} | {lv}/{5*len(v)} |")
        print("\n| 对照 | Δ n_method | 胜/负/平 | 符号 p | Δ n_approach |")
        print("|---|---:|:---:|---:|---:|")
        for lo, hi, lbl in PAIRS:
            dm, da, w, l, t = [], [], 0, 0, 0
            for s in ss:
                if (lo, s) not in rec or (hi, s) not in rec:
                    continue
                a0, m0, _ = rec[(lo, s)]; a1, m1, _ = rec[(hi, s)]
                dm.append(m1 - m0); da.append(a1 - a0)
                w += m1 > m0; l += m1 < m0; t += m1 == m0
            if not dm:
                continue
            print(f"| {lbl} | {statistics.mean(dm):+.2f} | {w}/{l}/{t} | {sign_p(w,l):.4f} | "
                  f"{statistics.mean(da):+.2f} |")
        print()
    return out


def main():
    live, sc, prob_of = load_live()
    r2, bad2, s2 = read_labels(B2, L2, "ABCD")
    r1, bad1, s1 = read_labels(B1, L1, "ABCDEFGH")
    kb = json.load(open(f"{B2}/key_bench.json"))

    print("# @5 做法多样性 v2:语料不截断之后\n")
    print(f"v2 标注 {len(s2)} 题;v1 标注 {len(s1)} 题;两版都有的 **{len(s1 & s2)}** 题(下面全部按这个交集配对)。")
    print("模型输出、题目、配对方式与 v1 完全相同,变的只有「标注者看到多少」。\n")
    if bad2:
        print("合规问题(v2):\n" + "\n".join(f"- {x}" for x in bad2) + "\n")
    else:
        print("v2 合规检查:**通过**(每块 5 抽不重不漏、至多一个 no_method 类)。\n")

    both = s1 & s2
    block(r2, both, live, prob_of, "1. v2(不截断)")
    block(r1, both, live, prob_of, "2. v1(截到 4000 字)——留作对照")

    print("## 3. ★ v1 → v2:重标把什么改掉了\n")
    print("| bench | 臂 | v1 n_method | v2 n_method | Δ | v1「没方法」抽样 | v2「没方法」抽样 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for bench, _, _, disp in BENCHES:
        ss = sorted(s for s in both if kb.get(s) == bench)
        if not ss:
            continue
        for a in ARMS:
            v1 = [r1[(a, s)] for s in ss if (a, s) in r1]
            v2 = [r2[(a, s)] for s in ss if (a, s) in r2]
            if not v1 or not v2:
                continue
            m1 = sum(x[1] for x in v1) / len(v1); m2 = sum(x[1] for x in v2) / len(v2)
            print(f"| {disp} | {NAME[a]} | {m1:.2f} | {m2:.2f} | {m2-m1:+.2f} | "
                  f"{sum(x[2] for x in v1)} | {sum(x[2] for x in v2)} |")

    print("\n## 4. ★ 主对照在两版语料下的值\n")
    print("| bench | 对照 | v1 Δ | v1 符号 p | v2 Δ | v2 符号 p |")
    print("|---|---|---:|---:|---:|---:|")
    for bench, _, _, disp in BENCHES:
        ss = sorted(s for s in both if kb.get(s) == bench)
        if not ss:
            continue
        for lo, hi, lbl in PAIRS:
            row = [disp, lbl]
            for r in (r1, r2):
                d, w, l = [], 0, 0
                for s in ss:
                    if (lo, s) not in r or (hi, s) not in r:
                        continue
                    m0, m1 = r[(lo, s)][1], r[(hi, s)][1]
                    d.append(m1 - m0); w += m1 > m0; l += m1 < m0
                row += [f"{statistics.mean(d):+.2f}" if d else "—",
                        f"{sign_p(w,l):.4f}" if d else "—"]
            print("| " + " | ".join(row) + " |")

    print("\n## 5. 少数的那几种做法,换到分了吗\n")
    print("`n_method` 少不一定是坏事:五抽都收敛到同一个**更好**的做法,"
          "和五抽都收敛到同一个烂做法,是两件事。逐题把 Δn_method 和 Δ题均分交叉制表。\n")
    print("| bench | 对照 | 题 | Δ n_method | Δ 题均分 | 对手题均分 | 我们题均分 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    rows = {}
    for bench, _, _, disp in BENCHES:
        ss = sorted(s for s in both if kb.get(s) == bench)
        if not ss:
            continue
        for lo, hi, lbl in PAIRS:
            rr = []
            for s in ss:
                if (lo, s) not in r2 or (hi, s) not in r2:
                    continue
                b, p = prob_of[s]
                m0 = [sc[lo][(b, p, i)] for i in range(5) if (b, p, i) in sc[lo]]
                m1 = [sc[hi][(b, p, i)] for i in range(5) if (b, p, i) in sc[hi]]
                if not m0 or not m1:
                    continue
                rr.append((r2[(hi, s)][1] - r2[(lo, s)][1],
                           statistics.mean(m1) - statistics.mean(m0),
                           statistics.mean(m0), statistics.mean(m1)))
            if not rr:
                continue
            rows[(bench, lbl)] = rr
            print(f"| {disp} | {lbl} | {len(rr)} | {statistics.mean(x[0] for x in rr):+.2f} | "
                  f"{statistics.mean(x[1] for x in rr):+.1f} | {statistics.mean(x[2] for x in rr):.1f} | "
                  f"{statistics.mean(x[3] for x in rr):.1f} |")
    print("\n**只看「我们做法更少」的那些题**(这是唯一对我们不利的格子,单独拎出来看代价):\n")
    print("| bench | 对照 | 做法更少的题 | 其中我们分更高 | 这些题上的 Δ 题均分 |")
    print("|---|---|---:|---:|---:|")
    for (bench, lbl), rr in rows.items():
        disp = dict((b, nm) for b, _, _, nm in BENCHES)[bench]
        few = [x for x in rr if x[0] < 0]
        if not few:
            print(f"| {disp} | {lbl} | 0 | — | — |")
            continue
        print(f"| {disp} | {lbl} | {len(few)} | {sum(1 for x in few if x[1] > 0)} | "
              f"{statistics.mean(x[1] for x in few):+.1f} |")
    print()

    print("## 6. 逐题明细 v1 → v2(n_method)\n")
    for bench, _, _, disp in BENCHES:
        ss = sorted(s for s in both if kb.get(s) == bench)
        if not ss:
            continue
        print(f"### {disp}\n")
        print("| 题 | " + " | ".join(NAME[a] for a in ARMS) + " |")
        print("|---|" + "---:|" * len(ARMS))
        for s in ss:
            cells = []
            for a in ARMS:
                x1 = r1.get((a, s)); x2 = r2.get((a, s))
                cells.append("—" if not x1 or not x2 else
                             (f"{x1[1]}" if x1[1] == x2[1] else f"**{x1[1]}→{x2[1]}**"))
            print(f"| `{s[:40]}` | " + " | ".join(cells) + " |")
        print()

    json.dump({"v2": {f"{a}|{s}": r2[(a, s)] for (a, s) in r2},
               "v1": {f"{a}|{s}": r1[(a, s)] for (a, s) in r1}},
              open(f"{S}/explore_agg2.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
