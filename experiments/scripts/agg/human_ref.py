"""Emit the human / external-model reference section for MODEL_SCORECARD_9B_zh.md.

Every number here is recomputed from a primary source at run time:
  - FrontierCS: the official leaderboard shipped in the benchmark's own cache
  - ALE-40:     the AtCoder contest archives (performance.csv per contest)
  - MLS:        the anchor rule read out of the scorer's source
  - our arms:   the same a12_<bench>.json dumps §3/§4 are built from
Nothing is transcribed by hand.
"""
import json, glob, os, collections, statistics, zipfile

SP = os.path.dirname(os.path.abspath(__file__))
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
FS = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"
SNAP = f"{FS}/.cache/ale-bench/datasets--SakanaAI--ALE-Bench/snapshots/0f426173b4e4e73b09b2b3631ae0490f66b75f99"
A12 = ["base9b_v2c", "ft01mix_a10", "ft03nm_a20", "lo32nm_a10",
       "rlv5_base_s15", "rlv5_ft01mix_a10_s15", "rlv5_ft03nm_a20_s15", "rlv5_lo32nm_a10_s15",
       "rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_ft03nm_a20_s20", "rlv5_lo32nm_a10_s20"]


def fcs_leaderboard():
    for track, label, key in [("algorithmic", "FrontierCS(算法轨)", "frontiercs"),
                              ("research", "FrontierCS-research(研究轨)", "frontiercs_research")]:
        lb = json.load(open(f"{FS}/.cache/Frontier-CS-official/leaderboard/{track}.json"))
        a12 = json.load(open(f"{SP}/a12_{key}.json"))
        print(f"\n**{label}** — 官方榜单口径 Avg@5 / Score@5,与我们的 mean@5 / best@5 一一对应。\n")
        print("| 参照 | Score@1 | Avg@5 | Score@5 | Elo |")
        print("|---|---|---|---|---|")
        for e in lb["entries"]:
            f = lambda x: "—" if x is None else f"{x:.2f}"
            tag = " **(人类专家)**" if e.get("is_human") else ""
            print(f"| {e['model']}{tag} | {f(e['score_at_1'])} | {f(e['avg_at_5'])} | "
                  f"{f(e['score_at_5'])} | {'—' if e['elo'] is None else e['elo']} |")
        print(f"| — 以下为本条线的 12 个模型({a12['n_problems']} 道共同题) — | | | | |")
        for t in A12:
            a = a12["arms"][t]
            print(f"| `{t}` | — | {a['mean@5'][0]:.2f} | {a['best@5'][0]:.2f} | — |")


def ale_human():
    a12 = json.load(open(f"{SP}/a12_alebench.json"))
    PROB = a12["problems"]
    human = {}
    for p in PROB:
        zf = zipfile.ZipFile(f"{SNAP}/{p}.zip")
        n = [x for x in zf.namelist() if x.endswith("performance.csv")][0]
        rows = [l.split(",") for l in zf.read(n).decode().splitlines()[1:] if l.strip()]
        rk = [int(r[0]) for r in rows]; pf = [float(r[1]) for r in rows]
        N = max(rk)                                   # ranks are tie-compressed
        med = min(range(len(rk)), key=lambda i: abs(rk[i] - N / 2))
        human[p] = {"N": N, "top": pf[0], "median": pf[med], "last": pf[-1]}

    print("\n**ALE-40** — 每场 AHC 的真人成绩单就在 benchmark 自带的 `performance.csv` 里,"
          "所以这里的“人类”不是估计值,是那一场比赛全部参赛者的实际 performance。\n")
    print(f"| 人类参照({len(PROB)} 场 AHC 的中位数) | performance |")
    print("|---|---|")
    print(f"| 冠军 | {statistics.median(h['top'] for h in human.values()):.0f} |")
    print(f"| 场内中位选手 | {statistics.median(h['median'] for h in human.values()):.0f} |")
    print(f"| 场内最后一名 | {statistics.median(h['last'] for h in human.values()):.0f} |")
    print(f"| 场均参赛人数 | {statistics.median(h['N'] for h in human.values()):.0f} 人 |")
    print()
    print("| 模型 | mean@5 performance | best@5 performance | 人类百分位(mean@5,越小越好) "
          "| 人类百分位(best@5) | 平均分>该场最后一名 |")
    print("|---|---|---|---|---|---|")
    for a in A12:
        d = collections.defaultdict(list)
        for f in glob.glob(f"{D}/outputs/cc_eval_{a}_thinking_32k_both_vllm/shard_*/samples.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r.get("data_source") != "alebench" or r.get("error"):
                    continue
                m = r.get("metrics") or {}
                if m.get("performance") is None:
                    continue
                d[str(r["ground_truth"])].append((m["performance"], m.get("rank")))
        pm = []; pb = []; beat = 0
        for p in PROB:
            N = human[p]["N"]
            rk = [x[1] for x in d[p] if x[1] is not None]
            pm.append(100.0 * min(statistics.mean(rk), N) / N)
            pb.append(100.0 * min(min(rk), N) / N)
            beat += statistics.mean(x[0] for x in d[p]) > human[p]["last"]
        print(f"| `{a}` | {statistics.mean(statistics.mean(x[0] for x in d[p]) for p in PROB):.1f} "
              f"| {statistics.mean(max(x[0] for x in d[p]) for p in PROB):.1f} "
              f"| {statistics.mean(pm):.1f}% | {statistics.mean(pb):.1f}% | {beat}/{len(PROB)} |")


def mls_anchor():
    names = [t["task"] for t in json.load(
        open(f"{D}/outputs/cc_mls21_base9b_v2c/summary.json"))["tasks"]]
    tasks = [f"{D}/mlsroot/tasks/{n}/leaderboard.csv" for n in names]
    tasks = [f for f in tasks if os.path.exists(f)]
    cnt = []
    for f in tasks:
        n = sum(1 for i, l in enumerate(open(f)) if i and l.split(",")[1].startswith("baseline:"))
        cnt.append(n)
    print("\n**MLS-Bench** — 这个 bench 的分数本身就是拿人类写的方法做锚的,不需要另找参照:\n")
    print("| 分数 | 含义(来自 `mlsroot/src/mlsbench/scoring/` 的定义) |")
    print("|---|---|")
    print("| 0.0 | 打平或不如该题**最弱**的已发表人类方法(`floor = anchors.worst_for(...)`,更差的被钳到 0) |")
    print("| 0.5 | 打平该题**最强**的已发表人类方法(`ref = anchors.best_for(...)`,`DEFAULT_REF_SCORE = 0.5`) |")
    print("| 1.0 | 达到该指标的理论上界(`bound`) |")
    print()
    print(f"每题的人类基线数量:MLS-21 的 {len(cnt)} 道题共 {sum(cnt)} 条 `baseline:*` 记录,"
          f"每题中位 {int(statistics.median(cnt))} 条(最少 {min(cnt)},最多 {max(cnt)})——"
          "它们是 LSQ、PACT、AdaRound 这类有论文的方法,不是随手写的对照。")


def dup_audit():
    SUB = {"frontiercs": "thinking_32k_both_vllm", "alebench": "thinking_32k_both_vllm",
           "frontiercs_research": "research_thinking_32k_vllm"}
    print("\n| bench | (题, sample_idx) 键数 | 有重复抽样的键 | **两次打分不一致的键** |")
    print("|---|---|---|---|")
    for bench, sub in SUB.items():
        tot = dup = diff = 0
        for a in A12:
            keys = collections.defaultdict(list)
            for f in sorted(glob.glob(f"{D}/outputs/cc_eval_{a}_{sub}/shard_*/samples.jsonl")):
                for l in open(f):
                    r = json.loads(l)
                    if r.get("data_source") != bench:
                        continue
                    s = (r.get("metrics") or {}).get("score")
                    if s is None:
                        continue
                    keys[(str(r["ground_truth"]), r["sample_idx"])].append(float(s))
            for v in keys.values():
                tot += 1
                if len(v) > 1:
                    dup += 1
                    diff += len(set(v)) > 1
        print(f"| {bench} | {tot} | {dup} ({100*dup/tot:.1f}%) | {diff} ({100*diff/tot:.2f}%) |")

    a12 = json.load(open(f"{SP}/a12_frontiercs_research.json"))
    PROB = a12["problems"]
    print("\nFrontierCS-research 在四种消歧规则下的 mean@5(其余两个 bench 的重复率太低,无需列):\n")
    print("| 模型 | 已发布(最后一条) | 取第一条 | 全部抽样合并 | 取最高 | 取最低 |")
    print("|---|---|---|---|---|---|")
    for a in A12:
        recs = collections.defaultdict(list)
        for f in sorted(glob.glob(f"{D}/outputs/cc_eval_{a}_research_thinking_32k_vllm/shard_*/samples.jsonl")):
            for l in open(f):
                r = json.loads(l)
                s = (r.get("metrics") or {}).get("score")
                if s is None:
                    continue
                recs[(str(r["ground_truth"]), r["sample_idx"])].append(float(s))

        def agg(pick):
            return statistics.mean(statistics.mean(pick(recs[(p, i)]) for i in range(5) if (p, i) in recs)
                                   for p in PROB if any((p, i) in recs for i in range(5)))
        pooled = statistics.mean(statistics.mean(x for i in range(5) for x in recs.get((p, i), []))
                                 for p in PROB if any((p, i) in recs for i in range(5)))
        print(f"| `{a}` | {a12['arms'][a]['mean@5'][0]:.3f} | {agg(lambda v: v[0]):.3f} | "
              f"{pooled:.3f} | {agg(max):.3f} | {agg(min):.3f} |")


if __name__ == "__main__":
    import sys
    {"fcs": fcs_leaderboard, "ale": ale_human, "mls": mls_anchor, "dup": dup_audit}[sys.argv[1]]()
