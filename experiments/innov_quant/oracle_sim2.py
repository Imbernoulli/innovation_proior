"""FCS samples vs the benchmark's shipped solutions (complete samples with code only).
Human references exist only for algorithmic problem 263 (reference1/2.cpp) and research nbody_simulation (reference_baseline.cpp);
the frontier-model pool (gemini3pro, gpt5.x, deepseekreasoner, grok4, trinity, ...) has ~45 solutions per algorithmic problem, 7 per research variant.
Writes oracle_metrics.csv: per sample sim_human, jac_frontier, sim_frontier, nearest_frontier, n_human, n_frontier."""
import json, glob, os, re, csv, difflib
from multiprocessing import Pool
O = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/.cache/Frontier-CS-official"; OUT = os.path.dirname(os.path.abspath(__file__))
def toks(code): return re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s]", code)
def grams(tk, n=4): return {tuple(tk[i:i + n]) for i in range(max(0, len(tk) - n + 1))}
def strip_comments(c): return re.sub(r"//[^\n]*|/\*.*?\*/|#[^\n]*", "", c, flags=re.S)
def model_of(f): return re.sub(r"_\d+$", "", os.path.basename(f).rsplit(".", 1)[0])
pool_cache = {}
def load_pool(bench, prob):
    key = (bench, prob)
    if key in pool_cache: return pool_cache[key]
    human, frontier = [], []
    if bench == "frontiercs":
        fs = glob.glob(f"{O}/algorithmic/solutions/{prob}/*.cpp")
    else:
        fs = glob.glob(f"{O}/research/solutions/{prob}/*.py") + glob.glob(f"{O}/research/solutions/{prob}/*.cpp")
    for f in fs:
        m = model_of(f); code = open(f, errors="replace").read()
        (human if m.startswith("reference") else frontier).append((m, code))
    pool_cache[key] = (human, frontier); return pool_cache[key]
def job(args):
    idx, bench, prob, code = args
    hu, fr = load_pool(bench, prob)
    tk = toks(strip_comments(code)); g = grams(tk)
    out = dict(idx=idx, n_human=len(hu), n_frontier=len(fr))
    out["sim_human"] = max((difflib.SequenceMatcher(None, tk, toks(strip_comments(c)), autojunk=False).ratio() for _, c in hu), default=None)
    best = (0.0, None, None)
    for m, c in fr:
        gc = grams(toks(strip_comments(c)))
        if g and gc:
            j = len(g & gc) / len(g | gc)
            if j > best[0]: best = (j, m, c)
    out["jac_frontier"] = best[0] if fr else None; out["nearest_frontier"] = best[1]
    out["sim_frontier"] = difflib.SequenceMatcher(None, tk, toks(strip_comments(best[2])), autojunk=False).ratio() if best[2] is not None else None
    return out
def main():
    rows = []
    for l in open(f"{OUT}/samples.jsonl"):
        s = json.loads(l)
        if s["bench"] in ("frontiercs", "frontiercs_research") and s["has_code"]:
            rows.append((len(rows), s["bench"], s["problem"], s["code"], s["arm"], s["sample_idx"], s["score"], s["in_common"], s["fam"], s["stage"], s["complete"]))
    res = {}
    with Pool(int(os.environ.get("NPROC", "12"))) as pool:
        for out in pool.imap_unordered(job, [(i, b, p, c) for i, b, p, c, *_ in rows], chunksize=8): res[out["idx"]] = out
    with open(f"{OUT}/oracle_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bench","problem","arm","sample_idx","score","in_common","fam","stage","complete","n_human","n_frontier","sim_human","jac_frontier","sim_frontier","nearest_frontier"])
        for i, b, p, c, arm, si, sc, ic, fam, stage, comp in rows:
            o = res[i]; w.writerow([b, p, arm, si, sc, ic, fam, stage, comp, o["n_human"], o["n_frontier"], o["sim_human"], o["jac_frontier"], o["sim_frontier"], o["nearest_frontier"]])
    print("rows", len(rows), "with human ref", sum(1 for i in res if res[i]["n_human"]))
if __name__ == "__main__": main()
