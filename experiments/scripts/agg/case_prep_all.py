"""Dump the FULL corpus (not the score-selected pool) for the cross-domain hunt.

case_prep.py only materialised problems where one of our arms already won, which
biases any count taken inside it. This dumps every common problem on both
FrontierCS tracks for all eight arms, plus the free-form cc_gen tasks, so a
search over it is unbiased. Scores stay in the filename; manifest_all.json keeps
the published mean@5 for every arm on every problem so a finder can check the
score without reopening the aggregates.
"""
import json, os, glob, collections, re

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
S = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(S), "doc")
OUT = os.path.join(S, "all")
ARMS = ["base9b_v2c", "ft01mix_a10", "ft03nm_a20", "lo32nm_a10",
        "rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_ft03nm_a20_s20", "rlv5_lo32nm_a10_s20"]
TAG = {"frontiercs": "cc_eval_%s_thinking_32k_both_vllm",
       "frontiercs_research": "cc_eval_%s_research_thinking_32k_vllm"}
GEN_TASKS = ["liveidea_gen", "review_weakness"]


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


def main():
    manifest = {}
    for bench in ["frontiercs_research", "frontiercs"]:
        a12 = json.load(open(f"{DOC}/a12_{bench}.json"))
        common = a12["problems"]
        manifest[bench] = {p: {a: round(a12["per_problem"][a]["mean@5"].get(p, float("nan")), 4)
                               for a in ARMS} for p in common}
        for arm in ARMS:
            per = collections.defaultdict(list)
            for f in glob.glob(f"{D}/outputs/{TAG[bench] % arm}/shard_*/samples.jsonl"):
                for l in open(f):
                    r = json.loads(l)
                    if r.get("data_source") != bench or r.get("error"):
                        continue
                    sc = (r.get("metrics") or {}).get("score")
                    if sc is None:
                        continue
                    per[str(r["ground_truth"])].append((float(sc), int(r["sample_idx"]), r["text"]))
            for p in common:
                cand = sorted(per.get(p, []), reverse=True)
                if not cand:
                    continue
                d = f"{OUT}/{bench}/{slug(p)}"
                os.makedirs(d, exist_ok=True)
                sc, si, tx = cand[0]
                open(f"{d}/{arm}__best_s{si}__score{sc:g}.txt", "w").write(tx)
                if len(cand) > 1:
                    sc2, si2, tx2 = cand[-1]
                    open(f"{d}/{arm}__worst_s{si2}__score{sc2:g}.txt", "w").write(tx2)
        print(bench, "problems dumped:", len(common))

    for task in GEN_TASKS:
        ids = set()
        for arm in ARMS:
            f = f"{D}/outputs/cc_gen_{arm}/samples.jsonl"
            for l in open(f):
                r = json.loads(l)
                if r["task"] != task or r["sample_idx"] != 0:
                    continue
                d = f"{OUT}/{task}/{slug(r['id'])}"
                os.makedirs(d, exist_ok=True)
                open(f"{d}/{arm}.txt", "w").write(r["text"])
                ids.add(r["id"])
        print(task, "ids dumped:", len(ids))
    json.dump(manifest, open(f"{OUT}/manifest_all.json", "w"), indent=1)


main()
