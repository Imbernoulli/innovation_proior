#!/usr/bin/env python3
"""Controlled within-SFT comparison: innovation INTENT vs REALIZATION per alpha.

Same SFT run, same problems, only alpha varies. Read-only.
Metrics per (arm, alpha, benchmark):
  - named-method proposal share (TeeMOEA/EBO-M style detector), raw + per-10k-chars
  - multi-approach enumeration (distinct 'Approach k/Option k/...' labels), share of
    samples with >=2, mean distinct count, raw + per-10k-chars
  - KI density (already computed elsewhere, recomputed here on same text base)
  - realization: score>0 share among intent samples vs non-intent samples
  - within-problem diversity: distinct algorithm-fingerprint clusters among the 5
    samples of each problem (greedy clustering at Jaccard<0.4 on fingerprints)
  - similarity-to-base: per-sample Jaccard(fingerprint, union of base fingerprints on
    same problem); lower = more deviant from what base does (Research + FCS)
"""
import json, glob, os, re, sys, statistics
from collections import defaultdict

OUT = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs"

NAME_PATTERNS = [
    # acronym/CamelCase name followed by colon + Title-Case expansion (TeeMOEA: Tangential ...)
    re.compile(r"\b([A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)?)\s*:\s*(?:[A-Z][a-z]+[- ]){2,}"),
    re.compile(r"\b(?:I(?:'ll| will)?|let'?s|we(?:'ll| will)?)\s+call\s+(?:this|it|the)\b", re.I),
    re.compile(r"\b(?:dubbed|termed)\b", re.I),
    re.compile(r"\bcalled\s+[\"“']?([A-Z][A-Za-z0-9-]{2,})[\"”']?", ),
    re.compile(r"\b(?:novel|new)\s+(?:algorithm|method|approach|heuristic)\s+(?:called|named|I call)\b", re.I),
    re.compile(r"\bnovel contribution\b", re.I),
    re.compile(r"\bmy (?:own )?(?:algorithm|method),?\s+[A-Z][A-Za-z0-9-]{2,}", ),
]
ENUM_RE = re.compile(r"\b(?:Approach|Option|Idea|Method|Strategy|Plan|Candidate|Attempt)\s*#?\s*(\d+|[A-D])\b")
ALT_RE = re.compile(r"\balternativ|\banother (?:approach|idea|way)\b|\binstead,", re.I)
KI_RE = re.compile(r"key insight", re.I)

# ---- algorithm fingerprints ----
CPP_ALGO = {
    "dp": r"\bdp\b|\bdynamic programming\b|\bmemo",
    "greedy": r"\bgreedy\b",
    "bfs": r"\bbfs\b|\bbreadth.first\b",
    "dfs": r"\bdfs\b|\bdepth.first\b",
    "dijkstra": r"\bdijkstra\b",
    "flow": r"\bmax.?flow\b|\bmin.?cut\b|\bdinic\b",
    "sa": r"\bsimulated annealing\b|\banneal",
    "localsearch": r"\blocal search\b|\bhill.?climb",
    "random_restart": r"\brandom restart\b|\brestart",
    "genetic": r"\bgenetic\b|\bevolution",
    "segment_tree": r"\bsegment tree\b|\bfenwick\b|\bbit\b",
    "binary_search": r"\bbinary search\b",
    "sort": r"\bsort\b",
    "union_find": r"\bunion.?find\b|\bdsu\b",
    "bitmask": r"\bbitmask\b|\bbit mask\b",
    "backtrack": r"\bbacktrack",
    "randomized": r"\brandom",
    "constructive": r"\bconstruct",
    "math_closed": r"\bclosed.form\b|\bformula\b",
    "two_sat": r"\b2.?sat\b",
    "lp": r"\blinear programming\b|\bsimplex\b",
    "beam": r"\bbeam search\b",
    "montecarlo": r"\bmonte carlo\b",
}
PY_API = re.compile(r"\b(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)")
PY_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(")
PY_STOP = {"print","range","len","enumerate","zip","list","dict","set","tuple","float","int","str",
           "super","isinstance","getattr","setattr","hasattr","abs","min","max","sum","sorted","open",
           "self","return","array","zeros","ones","append","format","join","split","strip","items",
           "values","keys","get","pop","add","copy","type","input","map","filter","round","exp","log",
           "sqrt","mean","std","shape","reshape","astype","tolist","numpy","torch","init","forward","main"}
CPP_ALGO_RE = {k: re.compile(v, re.I) for k, v in CPP_ALGO.items()}

def strip_think(t):
    _, sep, suf = t.rpartition("</think>")
    return suf if sep else t

def fingerprint(ds, text):
    """method fingerprint from the ANSWER (post-think) part"""
    ans = strip_think(text)
    src = ans if ans.strip() else text  # fall back to think if no answer
    fp = set()
    if ds == "frontiercs":
        for k, r in CPP_ALGO_RE.items():
            if r.search(src):
                fp.add(k)
    else:
        for mmod in PY_API.findall(src):
            fp.add("mod:" + mmod.split(".")[0])
        for c in PY_CALL.findall(src):
            if c.lower() not in PY_STOP and not c.startswith("_"):
                fp.add("call:" + c)
    return fp

def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else None

def load(dirpat, ds_filter=None):
    seen = {}
    for f in sorted(glob.glob(os.path.join(dirpat, "shard_*", "samples.jsonl"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ds_filter and d["data_source"] != ds_filter:
                continue
            seen[(d["data_source"], str(d.get("ground_truth")), str(d["problem_idx"]), str(d["sample_idx"]))] = d
    return list(seen.values())

def sample_metrics(d):
    text = d.get("text") or ""
    named = []
    for p in NAME_PATTERNS:
        for mm in p.finditer(text):
            named.append(mm.group(0)[:60])
    enums = set(ENUM_RE.findall(text))
    alts = len(ALT_RE.findall(text))
    ki = len(KI_RE.findall(text))
    score = 0.0 if d.get("error") else float(d["metrics"].get("reward") or 0.0)
    return {
        "chars": len(text),
        "named": len(named) > 0,
        "named_matches": named[:3],
        "n_enum": len(enums),
        "alts": alts,
        "ki": ki,
        "score": score,
        "intent": (len(named) > 0) or (len(enums) >= 2),
    }

def agg(tag, samples, base_fps=None):
    n = len(samples)
    if n == 0:
        return None
    ms = [sample_metrics(d) for d in samples]
    total_chars = sum(m["chars"] for m in ms) or 1
    named_share = sum(m["named"] for m in ms) / n
    named_per10k = sum(len(m["named_matches"]) > 0 for m in ms) * 10000 / total_chars  # events per char
    multi_share = sum(m["n_enum"] >= 2 for m in ms) / n
    mean_enum = sum(m["n_enum"] for m in ms) / n
    enum_per10k = sum(m["n_enum"] for m in ms) * 10000 / total_chars
    alt_per10k = sum(m["alts"] for m in ms) * 10000 / total_chars
    ki_per10k = sum(m["ki"] for m in ms) * 10000 / total_chars
    intent = [m for m in ms if m["intent"]]
    nonint = [m for m in ms if not m["intent"]]
    conv_i = sum(1 for m in intent if m["score"] > 0) / len(intent) if intent else float("nan")
    conv_n = sum(1 for m in nonint if m["score"] > 0) / len(nonint) if nonint else float("nan")
    mean_chars = total_chars / n
    # within-problem diversity (distinct fingerprint clusters among 5 samples)
    byp = defaultdict(list)
    for d in samples:
        byp[(d["data_source"], str(d["problem_idx"]))].append(d)
    divs = []
    sim_to_base = []
    for key, ds_ in byp.items():
        fps = [fingerprint(d["data_source"], d.get("text") or "") for d in ds_]
        # greedy clustering
        clusters = []
        for fp in fps:
            for c in clusters:
                jv = jac(fp, c)
                if jv is not None and jv >= 0.4:
                    c |= fp
                    break
            else:
                clusters.append(set(fp))
        divs.append(len(clusters))
        if base_fps and key in base_fps:
            bu = base_fps[key]
            for fp in fps:
                jv = jac(fp, bu)
                if jv is not None:
                    sim_to_base.append(jv)
    return {
        "tag": tag, "n": n,
        "named_share": named_share,
        "multi_share": multi_share,
        "mean_enum": mean_enum,
        "enum_per10k": enum_per10k,
        "alt_per10k": alt_per10k,
        "ki_per10k": ki_per10k,
        "intent_share": len(intent) / n,
        "conv_intent": conv_i,
        "conv_nonintent": conv_n,
        "score_pos": sum(1 for m in ms if m["score"] > 0) / n,
        "mean_chars": mean_chars,
        "diversity_mean": sum(divs) / len(divs),
        "sim_to_base": (sum(sim_to_base) / len(sim_to_base)) if sim_to_base else None,
    }

def base_fp_union(samples):
    byp = defaultdict(set)
    for d in samples:
        byp[(d["data_source"], str(d["problem_idx"]))] |= fingerprint(d["data_source"], d.get("text") or "")
    return byp

ARMS = {
    "nom": {
        0.0: ("clean_start", None),
        0.05: ("clean_clean_nomaintain_wd01_a5",) * 1,
        0.10: ("clean_clean_nomaintain_wd01_a10",),
        0.20: ("clean_clean_nomaintain_wd01_a20",),
        0.30: ("clean_clean_nomaintain_wd01_a30",),
        0.50: ("clean_clean_nomaintain_wd01_a50",),
        1.0: ("clean_clean_nomaintain_wd01_sft",),
    },
    "full": {
        0.0: ("clean_start",),
        0.05: ("clean_clean_full_wd01_a5",),
        0.10: ("clean_clean_full_wd01_a10",),
        0.20: ("clean_clean_full_wd01_a20",),
        0.30: ("clean_clean_full_wd01_a30",),
        0.50: ("clean_clean_full_wd01_a50",),
        1.0: ("clean_clean_full_wd01_sft",),
    },
}

def research_dirs(tag):
    if tag == "clean_start":
        return [os.path.join(OUT, "cc_eval_q35_inst_start_research_research_thinking_32k_vllm"),
                os.path.join(OUT, "cc_eval_q35_inst_start_researchcpu_thinking_32k_vllm")]
    return [os.path.join(OUT, f"cc_eval_{tag}_research_thinking_32k_vllm")]

def main():
    results = {"fcs": {}, "research": {}}
    for arm, alphas in ARMS.items():
        # base fingerprints per benchmark
        base_tag = alphas[0.0][0]
        base_fcs = load(os.path.join(OUT, f"cc_eval_{base_tag}_thinking_32k_both_vllm"), "frontiercs")
        base_res = []
        for dd in research_dirs(base_tag):
            base_res += load(dd)
        bfp_fcs = base_fp_union(base_fcs)
        bfp_res = base_fp_union(base_res)
        for alpha, tt in sorted(alphas.items()):
            tag = tt[0]
            fcs = base_fcs if alpha == 0.0 else load(os.path.join(OUT, f"cc_eval_{tag}_thinking_32k_both_vllm"), "frontiercs")
            res = base_res if alpha == 0.0 else sum((load(dd) for dd in research_dirs(tag)), [])
            a1 = agg(tag, fcs, bfp_fcs)
            a2 = agg(tag, res, bfp_res)
            if a1: results["fcs"].setdefault(arm, {})[alpha] = a1
            if a2: results["research"].setdefault(arm, {})[alpha] = a2
            print(f"[{arm} a={alpha}] done", flush=True)
    json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "intent_results.json", "w"), indent=1)
    for bench in ("fcs", "research"):
        for arm in results[bench]:
            print(f"\n==== {bench} / {arm} ====")
            hdr = ["alpha", "named_share", "multi_share", "mean_enum", "enum_per10k", "ki_per10k",
                   "intent_share", "conv_intent", "conv_nonintent", "score_pos", "diversity_mean",
                   "sim_to_base", "mean_chars"]
            print("\t".join(hdr))
            for alpha, e in sorted(results[bench][arm].items()):
                row = [f"{alpha}"]
                for h in hdr[1:]:
                    v = e.get(h)
                    row.append(f"{v:.4g}" if isinstance(v, float) else str(v))
                print("\t".join(row))

if __name__ == "__main__":
    main()
