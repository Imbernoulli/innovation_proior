"""Dump the ALE-40 and MLS-21 corpora for the cross-domain hunt.

ALE draws live alongside FrontierCS in the "both" eval; MLS is agentic, so the
only place its reasoning lands is the per-task driver log, which also carries
the task prompt (including the named human baselines the agent has to beat).
"""
import json, os, glob, collections, shutil, re

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
S = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(S), "doc")
OUT = os.path.join(S, "all")
ARMS = ["base9b_v2c", "ft01mix_a10", "ft03nm_a20", "lo32nm_a10",
        "rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_ft03nm_a20_s20", "rlv5_lo32nm_a10_s20"]


def ale():
    a12 = json.load(open(f"{DOC}/a12_alebench.json"))
    common = a12["problems"]
    man = {p: {a: round(a12["per_problem"][a]["mean@5"].get(p, float("nan")), 2) for a in ARMS}
           for p in common}
    for arm in ARMS:
        per = collections.defaultdict(list)
        for f in glob.glob(f"{D}/outputs/cc_eval_{arm}_thinking_32k_both_vllm/shard_*/samples.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r.get("data_source") != "alebench" or r.get("error"):
                    continue
                m = r.get("metrics") or {}
                if m.get("performance") is None:
                    continue
                per[str(r["ground_truth"])].append(
                    (float(m["performance"]), int(r["sample_idx"]), int(m.get("rank") or -1), r["text"]))
        for p in common:
            cand = sorted(per.get(p, []), reverse=True)
            if not cand:
                continue
            d = f"{OUT}/alebench/{p}"
            os.makedirs(d, exist_ok=True)
            pf, si, rk, tx = cand[0]
            open(f"{d}/{arm}__best_s{si}__perf{pf:g}_rank{rk}.txt", "w").write(tx)
            pf2, si2, rk2, tx2 = cand[-1]
            open(f"{d}/{arm}__worst_s{si2}__perf{pf2:g}_rank{rk2}.txt", "w").write(tx2)
    json.dump(man, open(f"{OUT}/alebench/manifest_ale.json", "w"), indent=1)
    print("alebench problems dumped:", len(common))


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def mls():
    man = collections.defaultdict(dict)
    tasks = set()
    for arm in ARMS:
        for base in [f"{D}/outputs/cc_mls21_{arm}", f"{D}/outputs/cc_mls2p_{arm}"]:
            sm = f"{base}/summary.json"
            if os.path.exists(sm):
                for t in json.load(open(sm))["tasks"]:
                    if "scored" in str(t.get("status", "")):
                        man[t["task"]][arm] = round(float(t["score"]), 4)
            for lg in glob.glob(f"{base}/task_logs/*.log"):
                t = os.path.basename(lg)[:-4]
                tasks.add(t)
                d = f"{OUT}/mls/{t}"
                os.makedirs(d, exist_ok=True)
                # strip the ANSI colouring so a reader (and grep) sees plain text
                open(f"{d}/{arm}.log", "w").write(ANSI.sub("", open(lg, errors="replace").read()))
    json.dump(dict(man), open(f"{OUT}/mls/manifest_mls.json", "w"), indent=1)
    print("mls tasks dumped:", len(tasks))


ale()
mls()
