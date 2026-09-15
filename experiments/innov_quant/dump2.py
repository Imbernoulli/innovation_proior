"""v2 normalizer for the innovation analysis (replaces dump.py).
Changes vs v1: (1) paircommon dedupe rule (sorted shards, last non-error row wins); (2) a sample is COMPLETE only if
the text contains </think>; code is extracted from the final answer only (drafts inside truncated thinking are NOT code);
(3) same-protocol 2026 replicate arms (<arm>_y2026) are loaded as stage "rep" with control = the original arm, giving the
same-model cross-run similarity floor; (4) judge topology (partition, speedFactor) recorded per arm x bench.
Output: samples.jsonl, metrics.csv, coverage.json in this directory."""
import json, glob, re, os, csv, collections, difflib, sys
from multiprocessing import Pool
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"; OUT = os.path.dirname(os.path.abspath(__file__))
ARMS = {  # arm: (family, stage, control, sft_start)
    "base9b_v2c": ("9B", "base", "rlv5_base_s20", None), "ft01mix_a10": ("9B", "sft", "base9b_v2c", None), "ft03nm_a20": ("9B", "sft", "base9b_v2c", None), "lo32nm_a10": ("9B", "sft", "base9b_v2c", None),
    "rlv5_base_s20": ("9B", "rl_base", "base9b_v2c", None), "rlv5_ft01mix_a10_s20": ("9B", "rl_sft", "rlv5_base_s20", "ft01mix_a10"), "rlv5_ft03nm_a20_s20": ("9B", "rl_sft", "rlv5_base_s20", "ft03nm_a20"), "rlv5_lo32nm_a10_s20": ("9B", "rl_sft", "rlv5_base_s20", "lo32nm_a10"),
    "base4b": ("4B", "base", "rlv5_4b_base_s20", None), "4b_ft01mix_a10": ("4B", "sft", "base4b", None), "4b_lo32nm_a10": ("4B", "sft", "base4b", None),
    "rlv5_4b_base_s20": ("4B", "rl_base", "base4b", None), "rlv5_4b_ft01mix_a10_s20": ("4B", "rl_sft", "rlv5_4b_base_s20", "4b_ft01mix_a10"), "rlv5_4b_lo32nm_a10_s20": ("4B", "rl_sft", "rlv5_4b_base_s20", "4b_lo32nm_a10"),
    # same-protocol replicates (year sweep 2026 reruns): control = the original arm
    "base9b_v2c_y2026": ("9B", "rep", "base9b_v2c", None), "lo32nm_a10_y2026": ("9B", "rep", "lo32nm_a10", None), "rlv5_lo32nm_a10_s20_y2026": ("9B", "rep", "rlv5_lo32nm_a10_s20", None),
    "base4b_y2026": ("4B", "rep", "base4b", None), "4b_lo32nm_a10_y2026": ("4B", "rep", "4b_lo32nm_a10", None), "rlv5_4b_lo32nm_a10_s20_y2026": ("4B", "rep", "rlv5_4b_lo32nm_a10_s20", None)}
MAIN = [a for a, v in ARMS.items() if v[1] != "rep"]
REP_OF = {a: v[2] for a, v in ARMS.items() if v[1] == "rep"}          # replicate -> original
ORIG_REP = {v: a for a, v in REP_OF.items()}                           # original -> replicate
BENCHES = ["frontiercs", "alebench", "frontiercs_research"]
FENCE = re.compile(r"```[ \t]*([A-Za-z+#]*)[ \t]*\n(.*?)```", re.S)
TRUNC_TOK = 32700
SIM_CAP = 4000

def sub_of(bench): return "research_thinking_32k_vllm" if bench == "frontiercs_research" else "thinking_32k_both_vllm"
def files(arm, bench): return sorted(glob.glob(f"{D}/cc_eval_{arm}_{sub_of(bench)}/shard_*/samples.jsonl"))

def judge_meta(arm, bench):
    out = []
    for f in sorted(glob.glob(f"{D}/cc_eval_{arm}_{sub_of(bench)}/shard_*/judge_node_meta.json")):
        try: j = json.load(open(f)); out.append(dict(node=j.get("node"), partition=j.get("partition"), speed=(j.get("node_speed_calibration") or {}).get("speedFactor"), job=j.get("slurm_job_id")))
        except Exception: pass
    return out

def extract_code(final, bench):
    blocks = FENCE.findall(final or "")
    if blocks:
        lang, code = blocks[-1]; return code, lang.lower()
    m = re.search(r"^(import |from \S+ import|class |def |#include)", final or "", re.M)
    if m: return final[m.start():], "bare"
    return "", "none"

def toks(code): return re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s]", code)
def grams(tk, n=4): return {tuple(tk[i:i + n]) for i in range(max(0, len(tk) - n + 1))}
def code_metrics(code, lang):
    lines = [l for l in code.splitlines() if l.strip()]
    cm = "#" if lang in ("python", "py", "bare") else "//"
    loc = sum(1 for l in lines if not l.strip().startswith(cm))
    tk = toks(code)
    inc = [a or b for a, b in re.findall(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]|^\s*(?:import|from)\s+([\w.]+)", code, re.M)]
    return dict(loc=loc, n_tokens=len(tk), n_loops=len(re.findall(r"\b(for|while)\b", code)), n_if=len(re.findall(r"\bif\b", code)),
                n_func=len(re.findall(r"^\s*def \w+|^[A-Za-z_][\w:<>,\s\*&]*\s+\w+\s*\([^;]*\)\s*(?:const)?\s*\{", code, re.M)),
                n_numlit=len(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", code)), includes=sorted(set(inc)), n_ident=len(set(re.findall(r"[A-Za-z_]\w{2,}", code))))

def load(arm, bench):
    """paircommon rule: key=(gt, sample_idx); skip error rows; a later row (sorted shard order, file order) overwrites."""
    ok, seen = {}, set()
    for f in files(arm, bench):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("data_source") != bench: continue
            k = (str(r["ground_truth"]), int(r.get("sample_idx", -1))); seen.add(k)
            if r.get("error"): continue
            m = r.get("metrics") or {}; s = m.get("score", r.get("score"))
            if s is None: continue
            ok[k] = dict(score=float(s), score_unbounded=m.get("score_unbounded"), performance=m.get("performance"), rank=m.get("rank"),
                         completion_tokens=r.get("completion_tokens"), text=r.get("text") or "")
    return ok, seen

def normalize(bench, arm, k, v):
    fam, stage, ctrl, start = ARMS[arm]
    text = v.pop("text"); ct = v.get("completion_tokens") or 0
    i = text.rfind("</think>"); complete = i >= 0
    final = text[i + len("</think>"):] if complete else ""
    reasoning = text[:i] if complete else text
    code, lang = extract_code(final, bench) if complete else ("", "none")
    s = dict(bench=bench, fam=fam, arm=arm, stage=stage, problem=k[0], sample_idx=k[1], **v,
             text_len=len(text), reasoning_len=len(reasoning), final_len=len(final), complete=complete, trunc=ct >= TRUNC_TOK,
             lang=lang, has_code=bool(code.strip()), code=code)
    s.update(code_metrics(code, lang) if code.strip() else dict(loc=0, n_tokens=0, n_loops=0, n_if=0, n_func=0, n_numlit=0, includes=[], n_ident=0))
    return s

def sim_job(args):
    """one (bench, fam, problem): all samples of all arms in the family; returns per-sample similarity fields."""
    key, samples = args
    tk = {(s["arm"], s["sample_idx"]): toks(s["code"])[:SIM_CAP] for s in samples if s["has_code"]}  # cap: difflib is O(n*m); q90 is ~3.5k tokens
    gr = {k: grams(v) for k, v in tk.items()}
    by_arm = collections.defaultdict(list)
    for k in tk: by_arm[k[0]].append(k)
    def ratio(a, b): return difflib.SequenceMatcher(None, tk[a], tk[b], autojunk=False).ratio()
    def maxsim(a, arm):
        return max((ratio(a, b) for b in by_arm.get(arm, []) if b != a), default=None)
    out = {}
    for s in samples:
        a = (s["arm"], s["sample_idx"])
        if a not in tk: out[a] = dict(sim_ctrl=None, sim_start=None, sim_self=None, sim_rep=None, jac_pool=None); continue
        fam, stage, ctrl, start = ARMS[s["arm"]]
        r = dict(sim_ctrl=maxsim(a, ctrl), sim_start=maxsim(a, start) if start else None, sim_self=maxsim(a, s["arm"]),
                 sim_rep=maxsim(a, ORIG_REP[s["arm"]]) if s["arm"] in ORIG_REP else None)
        best = 0.0
        for b, g in gr.items():
            if b[0] == s["arm"] or b[0] not in MAIN: continue
            if g and gr[a]: best = max(best, len(g & gr[a]) / len(g | gr[a]))
        r["jac_pool"] = best; out[a] = r
    return key, out

def main():
    allsamples = []; coverage = {}
    for bench in BENCHES:
        per = {}
        for arm in ARMS:
            ok, seen = load(arm, bench)
            per[arm] = ok
            coverage[f"{bench}|{arm}"] = dict(rows_ok=len(ok), keys_seen=len(seen), problems=len({k[0] for k in ok}), judge=judge_meta(arm, bench))
        for fam in ("9B", "4B"):
            mains = [a for a in MAIN if ARMS[a][0] == fam]
            common = set.intersection(*[set(per[a]) for a in mains])
            common_probs = collections.Counter(k[0] for k in common)
            full = {p for p, c in common_probs.items() if c == 5}
            coverage[f"{bench}|{fam}|common"] = dict(common_keys=len(common), problems_all5=len(full), problems_any=len(common_probs))
            for arm in [a for a in ARMS if ARMS[a][0] == fam]:
                for k, v in per[arm].items():
                    s = normalize(bench, arm, k, dict(v)); s["in_common"] = k in common or (arm in REP_OF and k[0] in full)
                    s["in_full"] = k[0] in full
                    allsamples.append(s)
    groups = collections.defaultdict(list)
    for s in allsamples: groups[(s["bench"], s["fam"], s["problem"])].append(s)
    sims = {}
    with Pool(int(os.environ.get("NPROC", "12"))) as pool:
        done = 0
        for key, out in pool.imap_unordered(sim_job, sorted(groups.items(), key=lambda kv: -sum(len(s["code"]) for s in kv[1])), chunksize=1):
            for a, r in out.items(): sims[key + a] = r
            done += 1
            if done % 50 == 0: print("groups", done, "/", len(groups), flush=True)
    for s in allsamples: s.update(sims[(s["bench"], s["fam"], s["problem"], s["arm"], s["sample_idx"])])
    with open(f"{OUT}/samples.jsonl", "w") as f:
        for s in allsamples: f.write(json.dumps(s) + "\n")
    cols = [c for c in allsamples[0] if c not in ("code", "includes")]
    with open(f"{OUT}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for s in allsamples: w.writerow(s)
    json.dump(coverage, open(f"{OUT}/coverage.json", "w"), indent=1)
    print("samples", len(allsamples), "complete", sum(s["complete"] for s in allsamples), "has_code", sum(s["has_code"] for s in allsamples))
if __name__ == "__main__": main()
