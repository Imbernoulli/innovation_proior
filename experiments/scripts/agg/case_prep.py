"""Build paired case bundles: same problem, our arm vs its stage-matched control.

Selection is score-driven so the case study cannot be a hunt for pretty prose:
a problem only enters the pool when our arm actually outscores the control on it.
Texts are dumped verbatim from the eval samples -- no truncation, no editing --
so every quote a reader sees can be grepped back to outputs/.
"""
import json, os, glob, collections, re

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
S = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(S), "doc")

ARMS = ["base9b_v2c", "ft01mix_a10", "ft03nm_a20", "lo32nm_a10",
        "rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_ft03nm_a20_s20", "rlv5_lo32nm_a10_s20"]
SFT_CTL, RL_CTL = "base9b_v2c", "rlv5_base_s20"
CONTRASTS = [(a, SFT_CTL) for a in ARMS[1:4]] + [(a, RL_CTL) for a in ARMS[5:8]]

TAG = {"frontiercs": "cc_eval_%s_thinking_32k_both_vllm",
       "frontiercs_research": "cc_eval_%s_research_thinking_32k_vllm"}


def load(bench, arm):
    """(problem -> [(score, sample_idx, text)]) for one arm on one bench."""
    out = collections.defaultdict(list)
    for f in glob.glob(f"{D}/outputs/{TAG[bench] % arm}/shard_*/samples.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("data_source") != bench or r.get("error"):
                continue
            sc = (r.get("metrics") or {}).get("score")
            if sc is None:
                continue
            out[str(r["ground_truth"])].append((float(sc), int(r["sample_idx"]), r["text"]))
    return out


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


def main():
    manifest = {}
    for bench in ["frontiercs_research", "frontiercs"]:
        a12 = json.load(open(f"{DOC}/a12_{bench}.json"))
        pp = a12["per_problem"]
        common = set(a12["problems"])
        texts = {a: load(bench, a) for a in ARMS}

        picked = collections.OrderedDict()
        for ours, ctl in CONTRASTS:
            d = [(pp[ours]["mean@5"][p] - pp[ctl]["mean@5"][p], p)
                 for p in common if p in pp[ours]["mean@5"] and p in pp[ctl]["mean@5"]]
            d.sort(reverse=True)
            for delta, p in d[:8]:
                if delta <= 0:
                    break
                picked.setdefault(p, []).append(
                    {"ours": ours, "control": ctl, "delta_mean@5": round(delta, 4),
                     "ours_mean@5": round(pp[ours]["mean@5"][p], 4),
                     "control_mean@5": round(pp[ctl]["mean@5"][p], 4)})

        for p in picked:
            pdir = f"{S}/{bench}/{slug(p)}"
            os.makedirs(pdir, exist_ok=True)
            for a in ARMS:
                cand = sorted(texts[a].get(p, []), reverse=True)
                if not cand:
                    continue
                sc, si, tx = cand[0]                       # best of the 5 draws
                open(f"{pdir}/{a}__best_s{si}__score{sc:g}.txt", "w").write(tx)
                if len(cand) > 1:
                    sc2, si2, tx2 = cand[-1]               # and the worst, for stability
                    open(f"{pdir}/{a}__worst_s{si2}__score{sc2:g}.txt", "w").write(tx2)
        manifest[bench] = {p: {"contrasts": v,
                               "all_arms_mean@5": {a: round(pp[a]["mean@5"].get(p, float("nan")), 4)
                                                   for a in ARMS}}
                           for p, v in picked.items()}
        print(bench, "problems picked:", len(picked))

    # liveidea_gen: no score, so dump every problem for all arms (sample 0).
    os.makedirs(f"{S}/liveidea_gen", exist_ok=True)
    ids = set()
    for a in ARMS:
        tag = "cc_gen_" + a
        f = f"{D}/outputs/{tag}/samples.jsonl"
        if not os.path.exists(f):
            print("MISSING", f); continue
        for l in open(f):
            r = json.loads(l)
            if r["task"] != "liveidea_gen" or r["sample_idx"] != 0:
                continue
            pdir = f"{S}/liveidea_gen/{slug(r['id'])}"
            os.makedirs(pdir, exist_ok=True)
            open(f"{pdir}/{a}.txt", "w").write(r["text"])
            ids.add(r["id"])
    print("liveidea_gen ids:", len(ids))
    json.dump(manifest, open(f"{S}/manifest.json", "w"), indent=1)


main()
