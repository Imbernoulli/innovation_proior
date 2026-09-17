"""@5 方法多样性:把「没产出方法」按原因拆开,并给一版只看活着的抽样的对照。

起因(两次):
  ① 4B RL(base) 在 FCS-research 上 n_method 均值只有 0.23 —— 一个「平均不到一种
     方法」的数不该被当成多样性读。
  ② 在 ALE-Bench 上同一条臂只有 0.08,而且我们的 9B 在 research 与 ALE 上都比
     baseline 的 RL 低一点。

逐格查下来,「没产出方法」其实是三种完全不同的事:

  早停哑火   模型吐了约 2000 token 就停,`</think>` 都没写完。这是模型真的垮了。
  撞 32k 上限 模型还在推理,预算用完了。这是**评测预算**,不是模型。
  交了但没方法 交出了最终块,标注者判定里面没有可辨认的做法。这是标注判断,不是故障。

把三者混成一个 n_method,量到的主要是「这条臂还活着吗」,不是「它探索得宽不宽」。
两个 bench 的主因**不一样**:research 上 4B 那条臂是早停哑火,ALE 上所有臂的死格
几乎全是撞 32k 预算 —— 所以 ALE 的 n_method 是被预算压的,两条 9B 臂被压得不一样多。
"""
import glob, json, math, os, re, statistics, sys

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
CAP = 32768
NO = "没产出方法"

# (data_source, 输出目录后缀, slug 前缀, 显示名)。前缀必须与 explore_corpus.py 一致 ——
# FCS-research 是裸名(那 26 题先标完的),后加的两个 bench 才有 `fcs_` / `ale_`。
BENCHES = [("frontiercs_research", "research_thinking_32k_vllm", "", "FCS-research"),
           ("alebench", "thinking_32k_both_vllm", "ale_", "ALE-Bench"),
           ("frontiercs", "thinking_32k_both_vllm", "fcs_", "FrontierCS")]

# 用户裁决 2026-09-17:**只看 RL 之后,不看 RL 之前**。RL 是最终 shape 出来的模型。
# 这份报告**整份**只覆盖 RL 之后的四条臂 —— 不是把 SFT/base 放附录,是根本不进这张表。
# 底层标注与样本一个没动,`explore_blind/` 与 `explore_labels/` 仍是八臂全量;
# 想看 RL 之前的,把下面两行换回八臂即可。
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


def load():
    """(arm, (bench, prob, sample_idx)) -> (最终块非空?, completion_tokens)。"""
    out, prob_of = {a: {} for a in ARMS}, {}
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
                    out[a][(bench, p, int(r.get("sample_idx", -1)))] = (
                        bool(i >= 0 and t[i + 8:].strip()), r.get("completion_tokens"))
    return out, prob_of


def is_no(c):
    return bool(c["no_method"]) if "no_method" in c else NO in c.get("desc", "")


def main():
    data, prob_of = load()
    key = json.load(open(f"{S}/explore_blind/key.json"))
    rec = {}                    # (arm, slug) -> dict
    bench_of = {}               # slug -> bench
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
            arm = key[slug][li]
            if arm not in NAME:        # 只看 RL 之后的四条臂,其余块整块跳过
                continue
            g = lambda si: data[arm].get((bench, p, si))
            live = {si + 1 for si in range(5) if (g(si) or (False, None))[0]}
            # 死掉的抽样按原因拆:撞上限 vs 早停。没有这条记录的当早停。
            cap = {si + 1 for si in range(5)
                   if not (g(si) or (False, None))[0]
                   and ((g(si) or (False, 0))[1] or 0) >= CAP}
            dead = {1, 2, 3, 4, 5} - live
            # 只看活着的抽样时:去掉 no_method 类,再把每个类里死掉的抽样剔掉,空类不算
            live_cl = [c for c in v["clusters"]
                       if not is_no(c) and (set(c["samples"]) & live)]
            rec[(arm, slug)] = dict(
                n_app=v["n"],
                n_met=len([c for c in v["clusters"] if not is_no(c)]),
                dead=len(dead), cap=len(cap), stall=len(dead - cap),
                nomet_live=sum(len(set(c["samples"]) & live)
                               for c in v["clusters"] if is_no(c)),
                live=len(live), met_live=len(live_cl))

    allslugs = sorted({s for (_, s) in rec})
    print("# @5 方法多样性:「没产出方法」到底是什么\n")
    print("只覆盖 **RL 之后的四条臂**。逐 bench 分开报 —— 两个 bench 的死因不一样,"
          "合起来看会把两件事搅成一件。\n")
    have = [(b, nm) for b, _, _, nm in BENCHES if any(bench_of.get(s) == b for s in allslugs)]
    print("标注覆盖:" + "、".join(
        f"{nm} {sum(1 for s in allslugs if bench_of.get(s)==b)} 题" for b, nm in have)
        + "(每题 4 臂 × 5 抽)。\n")

    for bench, disp in have:
        slugs = [s for s in allslugs if bench_of.get(s) == bench]
        print(f"---\n\n# {disp}({len(slugs)} 题)\n")

        print("## 1. 「没产出方法」的三种来源(格子数 = 题×抽样)\n")
        print("| 臂 | 抽样 | 有最终答案 | 无最终答案 | 其中**撞 32k 预算** | 其中**早停哑火** | 交了但标注判定无方法 |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for a in ARMS:
            v = [rec[(a, s)] for s in slugs if (a, s) in rec]
            if not v:
                continue
            tot = 5 * len(v)
            dead = sum(x["dead"] for x in v)
            print(f"| {NAME[a]} | {tot} | {tot-dead} | {dead} | {sum(x['cap'] for x in v)} | "
                  f"{sum(x['stall'] for x in v)} | {sum(x['nomet_live'] for x in v)} |")

        print("\n## 2. 两种口径的 n_method 并排\n")
        print("`n_method` = 原口径(分母是全部 5 抽,死掉的抽样拉低它);")
        print("`n_method|live` = 只数至少有一个活抽样的做法类;`live/5` = 平均每题有几抽真交了东西。\n")
        print("| 臂 | n_approach | n_method | **live/5** | **n_method\\|live** |")
        print("|---|---:|---:|---:|---:|")
        for a in ARMS:
            v = [rec[(a, s)] for s in slugs if (a, s) in rec]
            if not v:
                continue
            n = len(v)
            print(f"| {NAME[a]} | {sum(x['n_app'] for x in v)/n:.2f} | {sum(x['n_met'] for x in v)/n:.2f} | "
                  f"{sum(x['live'] for x in v)/n:.2f} | {sum(x['met_live'] for x in v)/n:.2f} |")

        print("\n## 3. 主对照:RL 之后(我们的 RL − baseline 的 RL)\n")
        print("末列 `两臂都≥3活抽` 是可读性闸门 —— 低于它,这一行量的是死活不是多样性。\n")
        print("| 对照 | n_method Δ | 胜/负/平 | 符号 p | n_method\\|live Δ | 胜/负/平 | 符号 p | 两臂都≥3活抽 |")
        print("|---|---:|:---:|---:|---:|:---:|---:|:---:|")
        for lo, hi, lbl in PAIRS:
            d1, d2 = [], []
            w1 = l1 = t1 = w2 = l2 = t2 = ok3 = 0
            for s in slugs:
                if (lo, s) not in rec or (hi, s) not in rec:
                    continue
                A, B = rec[(lo, s)], rec[(hi, s)]
                d1.append(B["n_met"] - A["n_met"])
                w1 += B["n_met"] > A["n_met"]; l1 += B["n_met"] < A["n_met"]; t1 += B["n_met"] == A["n_met"]
                if A["live"] >= 3 and B["live"] >= 3:
                    ok3 += 1
                    d2.append(B["met_live"] - A["met_live"])
                    w2 += B["met_live"] > A["met_live"]; l2 += B["met_live"] < A["met_live"]
                    t2 += B["met_live"] == A["met_live"]
            if not d1:
                continue
            m2 = f"{statistics.mean(d2):+.2f}" if d2 else "—"
            p2 = f"{sign_p(w2, l2):.4f}" if d2 else "—"
            c2 = f"{w2}/{l2}/{t2}" if d2 else "—"
            flag = f"**{ok3}/{len(slugs)}**" if ok3 >= 10 else f"⚠ {ok3}/{len(slugs)}"
            print(f"| {lbl} | {statistics.mean(d1):+.2f} | {w1}/{l1}/{t1} | {sign_p(w1,l1):.4f} | "
                  f"{m2} | {c2} | {p2} | {flag} |")
        print()

    json.dump({f"{a}|{s}": rec[(a, s)] for (a, s) in rec} | {"_bench": bench_of},
              open(f"{S}/explore_why_nomethod.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
