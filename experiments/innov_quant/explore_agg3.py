"""@5 做法多样性 · **v3**(不截断语料,RL 之后六条臂)的汇总。

v3 = v2 的四块 + lora 线的两条 RL 臂(`rlv5_lo32nm_a10_s20` / `rlv5_4b_lo32nm_a10_s20`)。
题目集合、模型输出、配对方式与 v2 **完全相同**,只多了两块并重排了块序。

为什么整批重标而不是给 lora 单独另标一份:每块的聚类阈值是标注者当场定的,
分两批就是两把尺子。§4 专门查这件事 —— 四条老臂在 v2 和 v3 下的 n_method 应该稳,
如果不稳,说明「多给两块」本身改变了标注尺度,那 v3 的跨臂比较就要先打问号。

口径:
  n_approach = 5 抽里的做法数,「没产出方法」也算一类(坍塌的臂照样进分母)。
  n_method   = 去掉「没产出方法」后的做法数。
  n_method 均值 < 1 的格子是**死活读数不是多样性读数**,表里打 ⚠。
"""
import glob, json, math, os, re, statistics, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
B3, L3 = f"{S}/explore_blind3", f"{S}/explore_labels3"
B2, L2 = f"{S}/explore_blind2", f"{S}/explore_labels2"
NO = "没产出方法"

BENCHES = [("frontiercs_research", "research_thinking_32k_vllm", "", "FCS-research"),
           ("alebench", "thinking_32k_both_vllm", "ale_", "ALE-Bench"),
           ("frontiercs", "thinking_32k_both_vllm", "fcs_", "FrontierCS")]
ARMS = ["rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20",
        "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20"]
NAME = dict(zip(ARMS, ["9B RL(base)", "9B 我们", "9B lora",
                       "4B RL(base)", "4B 我们", "4B lora"]))
V2ARMS = ["rlv5_base_s20", "rlv5_ft01mix_a10_s20",
          "rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20"]
PAIRS = [("rlv5_base_s20", "rlv5_ft01mix_a10_s20", "9B 我们 − RL(base)"),
         ("rlv5_base_s20", "rlv5_lo32nm_a10_s20", "9B lora − RL(base)"),
         ("rlv5_ft01mix_a10_s20", "rlv5_lo32nm_a10_s20", "9B lora − 我们"),
         ("rlv5_4b_base_s20", "rlv5_4b_ft01mix_a10_s20", "4B 我们 − RL(base)"),
         ("rlv5_4b_base_s20", "rlv5_4b_lo32nm_a10_s20", "4B lora − RL(base)"),
         ("rlv5_4b_ft01mix_a10_s20", "rlv5_4b_lo32nm_a10_s20", "4B lora − 我们")]

BUF = []
def emit(s=""):
    print(s); BUF.append(s)


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
                    out[a][k] = bool(i >= 0 and t[i + 8:].strip())
                    sc[a][k] = float(m.get("score", r.get("score")))
    return out, sc, prob_of


def _key(blind, lab, name):
    """盲盒语料放在 scratchpad(几 MB,不进仓库),key 复制一份进 labels 目录 —— 与 v1/v2 同惯例。"""
    for p in (f"{lab}/_{name}.json", f"{blind}/{name}.json"):
        if os.path.exists(p):
            return json.load(open(p))
    raise SystemExit(f"找不到 {name}.json({lab} / {blind})")


def read_labels(blind, lab, letters, arms):
    key = _key(blind, lab, "key")
    rec, bad, slugs = {}, [], set()
    for f in sorted(os.listdir(lab)):
        if not f.endswith(".json") or f.startswith("_"):
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
                bad.append(f"{slug}/{li}:多个 no_method 类"); ok = False
            if v.get("n") != len(v["clusters"]):
                v["n"] = len(v["clusters"])
        if not ok:
            continue
        slugs.add(slug)
        for li, v in d.items():
            arm = key[slug][li]
            if arm not in arms:
                continue
            nm = [c for c in v["clusters"] if not is_no(c)]
            rec[(arm, slug)] = (len(v["clusters"]), len(nm),
                                sum(len(c["samples"]) for c in v["clusters"] if is_no(c)))
    return rec, bad, slugs


def main():
    live, sc, prob_of = load_live()
    rec3, bad3, slugs3 = read_labels(B3, L3, "ABCDEF", set(ARMS))
    kb = _key(B3, L3, "key_bench")

    emit("# @5 做法多样性 · v3(不截断 / RL 之后六条臂,含 lora)\n")
    emit(f"合规:{'通过' if not bad3 else str(len(bad3)) + ' 处问题'} | 已标注 {len(slugs3)} 题\n")
    for b in bad3[:20]:
        emit(f"- [bad] {b}")
    if bad3:
        emit()

    emit("## 1. 逐臂(按 bench 分层)\n")
    emit("`live/5` = 平均每题有几抽真交了最终答案。**n_method < 1 的格子打 ⚠** —— "
         "那一行量的是这条臂还活着没有,不是它探索得宽不宽。\n")
    for bench, _, _, disp in BENCHES:
        ss = sorted(s for s in slugs3 if kb.get(s) == bench)
        if not ss:
            continue
        emit(f"### {disp}({len(ss)} 题)\n")
        emit("| 臂 | n_approach | n_method | live/5 | 没产出方法的抽样 |")
        emit("|---|---:|---:|---:|---:|")
        for a in ARMS:
            v = [rec3[(a, s)] for s in ss if (a, s) in rec3]
            if not v:
                continue
            nmet = sum(x[1] for x in v) / len(v)
            lv = sum(1 for s in ss for si in range(5)
                     if live[a].get((kb[s],) + (prob_of[s][1], si), False)) / len(ss)
            tag = "⚠ " if nmet < 1 else ""
            emit(f"| {tag}{NAME[a]} | {sum(x[0] for x in v)/len(v):.2f} | {nmet:.2f} | "
                 f"{lv:.2f} | {sum(x[2] for x in v)}/{5*len(v)} |")
        emit()

    emit("## 2. 对照:逐题配对(胜/负/平 = 后者的 n_method 更高/更低/相同)\n")
    for bench, _, _, disp in BENCHES:
        ss = sorted(s for s in slugs3 if kb.get(s) == bench)
        if not ss:
            continue
        emit(f"### {disp}({len(ss)} 题)\n")
        emit("| 对照 | n_method 均差 | 胜/负/平 | 符号 p | n_approach 均差 |")
        emit("|---|---:|:---:|---:|---:|")
        for lo, hi, lbl in PAIRS:
            dm, da, w, l, t = [], [], 0, 0, 0
            for s in ss:
                if (lo, s) not in rec3 or (hi, s) not in rec3:
                    continue
                a0, m0, _ = rec3[(lo, s)]; a1, m1, _ = rec3[(hi, s)]
                dm.append(m1 - m0); da.append(a1 - a0)
                w += m1 > m0; l += m1 < m0; t += m1 == m0
            if not dm:
                continue
            emit(f"| {lbl} | {statistics.mean(dm):+.2f} | {w}/{l}/{t} | "
                 f"{sign_p(w,l):.4f} | {statistics.mean(da):+.2f} |")
        emit()

    emit("## 3. 尺子核验:四条老臂在 v2 与 v3 下对不对得上\n")
    emit("多给标注者两块会不会改变他的聚类阈值。**对不上就说明 v3 的跨臂比较要先打问号。**\n")
    rec2, bad2, slugs2 = read_labels(B2, L2, "ABCD", set(V2ARMS))
    both = sorted(slugs2 & slugs3)
    emit(f"两版都标了的题:{len(both)}\n")
    emit("| 臂 | v2 n_method | v3 n_method | Δ | 逐题一致 |")
    emit("|---|---:|---:|---:|---:|")
    for a in V2ARMS:
        p = [(rec2[(a, s)][1], rec3[(a, s)][1]) for s in both
             if (a, s) in rec2 and (a, s) in rec3]
        if not p:
            continue
        same = sum(1 for x, y in p if x == y)
        emit(f"| {NAME[a]} | {statistics.mean(x for x, _ in p):.2f} | "
             f"{statistics.mean(y for _, y in p):.2f} | "
             f"{statistics.mean(y - x for x, y in p):+.2f} | {same}/{len(p)} |")
    emit()

    emit("## 4. 少数的那几种做法,换到分了吗\n")
    emit("每题按 n_method 把两条臂分成「谁的做法多」,再看题均分谁高。\n")
    for bench, _, _, disp in BENCHES:
        ss = sorted(s for s in slugs3 if kb.get(s) == bench)
        if not ss:
            continue
        emit(f"### {disp}\n")
        emit("| 对照 | 后者做法更多的题 | 那些题上后者−前者题均分 | 后者做法更少的题 | 那些题上后者−前者题均分 |")
        emit("|---|---:|---:|---:|---:|")
        for lo, hi, lbl in PAIRS:
            def mscore(a, s):
                b, p = prob_of[s]
                v = [sc[a][(b, p, si)] for si in range(5) if (b, p, si) in sc[a]]
                return statistics.mean(v) if v else None
            more, less = [], []
            for s in ss:
                if (lo, s) not in rec3 or (hi, s) not in rec3:
                    continue
                x, y = mscore(lo, s), mscore(hi, s)
                if x is None or y is None:
                    continue
                d = rec3[(hi, s)][1] - rec3[(lo, s)][1]
                (more if d > 0 else less if d < 0 else []).append(y - x)
            f = lambda v: f"{statistics.mean(v):+.2f}" if v else "—"
            emit(f"| {lbl} | {len(more)} | {f(more)} | {len(less)} | {f(less)} |")
        emit()


main()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_agg3.md"),
          "w", encoding="utf-8") as fh:
    fh.write("\n".join(BUF) + "\n")
