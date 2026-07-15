#!/usr/bin/env python3
"""Innovation-vs-alpha curve analysis for the model-soup question.

Reuses the EXACT documented protocols:
  - CASE_STUDY_clean_wd03_zh.md §6.4: think mean length (chars), "Key Insight" per
    10k chars, "Wait" per 10k chars, short-think (<10k chars) share.
  - SOUP_TRADEOFF_zh.md §1: innovation-register & pragmatic-register keyword lists,
    density per 1k chars of think text.
  - Competence: FCS = per-problem mean@5 of metrics.reward then macro-mean
    (dedup key=(data_source,problem_idx,sample_idx), last wins); ALE = metrics.performance.
  - Research: per-problem (ground_truth) mean@5, error->0, macro-mean.
  - MLS: summary.json mean over 20 tasks with unscored->0 (official None->0 protocol).
"""
import json, os, re, statistics, sys

OUT = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs"

# --- SOUP_TRADEOFF §1 keyword lists (verbatim translation of the documented regexes) ---
INNOV = [
    r"\bwait\b", r"\bactually\b", r"\bhmm\b", r"\breconsider", r"\brethink",
    r"\bre-examine", r"\bwhat if\b", r"\bis there a (?:better|smarter|cleverer|deeper)\b",
    r"\bbut (?:wait|actually)\b", r"\bhold on\b", r"\binteresting\b", r"\binsight",
    r"\bkey (?:idea|insight|observation)\b", r"\bnovel\b", r"\bclever", r"\belegant",
    r"\bgeneraliz\w*", r"\bdeeper\b", r"\bunderlying\b", r"\bwhy (?:does|is|would)\b",
    r"\bnot (?:obvious|trivial|standard)\b", r"\bnon-trivial\b",
    r"\bcan we (?:do|prove|show|construct)\b", r"\bperhaps\b", r"\bmaybe\b",
    r"\bconjecture", r"\balternativ\w*", r"\banother (?:approach|idea|way)\b",
    r"\bdifferent (?:approach|angle)\b", r"\boptimal\b",
    r"\bbetter (?:construction|bound|approach)\b", r"\bturns out\b", r"\bsurpris\w*",
    r"\bsubtle\b", r"\bmismatch\b",
    r"\bdoesn't make sense\b|\bdoesn't work\b|\bdoesn't hold\b", r"\bcontradic\w*",
    r"\brabbit hole\b",
]
PRAG = [
    r"\bimplement", r"\bjust (?:use|do|output|print|return|submit|go with)\b",
    r"\blet me (?:code|write|implement|just)\b",
    r"\bsimple (?:approach|solution|construction)\b", r"\bstraightforward\b",
    r"\bgood enough\b", r"\bshould work\b", r"\bthis works\b", r"\bedge case",
    r"\bcompile", r"\boutput format\b", r"\bprint\b", r"\bgreedy\b", r"\bbrute force\b",
    r"\bvalid (?:solution|output|answer)\b", r"\b(?:let me )?verify\b", r"\bcorrect(?:ness)?\b",
    r"\bmake sure\b", r"\bsubmit\b", r"\bfinal (?:answer|solution|code)\b",
    r"\bI'll (?:write|code|implement|use|go with)\b", r"\bconstraints\b", r"\bsafe\b",
    r"\bsimplest\b", r"\bpractical\b", r"\bfor now\b", r"\bwithin time\b",
    r"\befficient\b", r"O\(",
]
INNOV_RE = [re.compile(p, re.IGNORECASE) for p in INNOV]
PRAG_RE = [re.compile(p, re.IGNORECASE) for p in PRAG]
KEYINSIGHT_RE = re.compile(r"key insight", re.IGNORECASE)
WAIT_RE = re.compile(r"\bwait\b", re.IGNORECASE)


def load_samples(path):
    """dedup key=(data_source,problem_idx,sample_idx) last-wins across shards."""
    seen = {}
    import glob
    for f in sorted(glob.glob(os.path.join(path, "shard_*", "samples.jsonl"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen[(d["data_source"], str(d["problem_idx"]), str(d["sample_idx"]))] = d
    return list(seen.values())


def think_of(text):
    if not text:
        return ""
    head, sep, _ = text.rpartition("</think>")
    return head if sep else text


def count_hits(res, txt):
    return sum(len(r.findall(txt)) for r in res)


def fcs_stats(samples):
    rows = [d for d in samples if d["data_source"] == "frontiercs"]
    if not rows:
        return None
    byp = {}
    for d in rows:
        byp.setdefault(str(d["problem_idx"]), []).append(d)
    per_mean, per_best = [], []
    for p, ds in byp.items():
        sc = [float(d["metrics"].get("reward") or 0.0) if not d.get("error") else 0.0 for d in ds]
        per_mean.append(sum(sc) / len(sc))
        per_best.append(max(sc))
    nerr = sum(1 for d in rows if d.get("error"))
    # style metrics on think text
    thinks = [think_of(d.get("text") or "") for d in rows]
    tl = [len(t) for t in thinks]
    total_chars = sum(tl) or 1
    ki = sum(len(KEYINSIGHT_RE.findall(t)) for t in thinks)
    wa = sum(len(WAIT_RE.findall(t)) for t in thinks)
    inn = sum(count_hits(INNOV_RE, t) for t in thinks)
    prg = sum(count_hits(PRAG_RE, t) for t in thinks)
    trunc = sum(1 for d in rows if (d.get("completion_tokens") or 0) >= 32768)
    short = sum(1 for t in tl if t < 10000)
    sc_pos = sum(1 for d in rows if float(d["metrics"].get("reward") or 0.0) > 0)
    nothink = sum(1 for d in rows if "</think>" not in (d.get("text") or ""))
    return {
        "n_problems": len(byp), "n_samples": len(rows), "errors": nerr,
        "fcs_mean5": sum(per_mean) / len(per_mean),
        "fcs_best5": sum(per_best) / len(per_best),
        "think_mean_chars": sum(tl) / len(tl),
        "think_median_chars": statistics.median(tl),
        "short_think_share": short / len(rows),
        "trunc_share": trunc / len(rows),
        "score_pos_share": sc_pos / len(rows),
        "no_close_think_share": nothink / len(rows),
        "key_insight_per10k": ki * 10000 / total_chars,
        "wait_per10k": wa * 10000 / total_chars,
        "innov_per1k": inn * 1000 / total_chars,
        "prag_per1k": prg * 1000 / total_chars,
    }


def ale_stats(samples):
    rows = [d for d in samples if d["data_source"] == "alebench"]
    if not rows:
        return None
    byp = {}
    for d in rows:
        byp.setdefault(str(d["problem_idx"]), []).append(d)
    per = []
    for p, ds in byp.items():
        sc = [float(d["metrics"].get("performance") or 0.0) for d in ds]
        per.append(sum(sc) / len(sc))
    return {"ale_mean5": sum(per) / len(per), "n_tasks": len(byp)}


def research_stats(dirs):
    seen = {}
    import glob
    for path in dirs:
        for f in sorted(glob.glob(os.path.join(path, "shard_*", "samples.jsonl"))):
            for line in open(f, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen[(str(d["ground_truth"]), str(d["sample_idx"]))] = d
    if not seen:
        return None
    byp = {}
    for d in seen.values():
        byp.setdefault(str(d["ground_truth"]), []).append(d)
    per = []
    for p, ds in byp.items():
        sc = [0.0 if d.get("error") else float(d["metrics"].get("reward") or 0.0) for d in ds]
        per.append(sum(sc) / len(sc))
    nerr = sum(1 for d in seen.values() if d.get("error"))
    return {"research_mean5": sum(per) / len(per), "n_problems": len(byp),
            "n_samples": len(seen), "errors": nerr}


def mls_stats(tag):
    p = os.path.join(OUT, f"cc_mlsbench_cpu_{tag}", "summary.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    tot = 0.0
    n = s.get("n_tasks") or 20
    for t in s.get("tasks", []):
        # compound statuses ('timeout+scored', 'agent_failed+scored') still carry scores;
        # any non-None score counts, None -> 0 (official protocol)
        if t.get("score") is not None:
            tot += float(t["score"])
    return {"mls_none0": tot / n, "n_scored": s.get("n_scored"), "mls_reported": s.get("mean_score")}


ARMS = {
    # arm -> list of (alpha, both_tag, research_dirs(list of tags), mls_tag)
    "nomaintain_wd01": [
        (0.0, "clean_start", ["q35_inst_start_research", "q35_inst_start_researchcpu:combined"], "q35_start_devfix"),
        (0.05, "clean_clean_nomaintain_wd01_a5", ["clean_clean_nomaintain_wd01_a5"], "clean_clean_nomaintain_wd01_a5"),
        (0.10, "clean_clean_nomaintain_wd01_a10", ["clean_clean_nomaintain_wd01_a10"], "clean_clean_nomaintain_wd01_a10"),
        (0.20, "clean_clean_nomaintain_wd01_a20", ["clean_clean_nomaintain_wd01_a20"], "clean_clean_nomaintain_wd01_a20"),
        (0.30, "clean_clean_nomaintain_wd01_a30", ["clean_clean_nomaintain_wd01_a30"], "clean_clean_nomaintain_wd01_a30"),
        (0.50, "clean_clean_nomaintain_wd01_a50", ["clean_clean_nomaintain_wd01_a50"], "clean_clean_nomaintain_wd01_a50"),
        (1.0, "clean_clean_nomaintain_wd01_sft", ["clean_clean_nomaintain_wd01_sft"], "clean_nom_wd01_sft_devfix"),
    ],
    "full_wd01": [
        (0.0, "clean_start", ["q35_inst_start_research", "q35_inst_start_researchcpu:combined"], "q35_start_devfix"),
        (0.05, "clean_clean_full_wd01_a5", ["clean_clean_full_wd01_a5"], "clean_clean_full_wd01_a5"),
        (0.10, "clean_clean_full_wd01_a10", ["clean_clean_full_wd01_a10"], "clean_clean_full_wd01_a10"),
        (0.20, "clean_clean_full_wd01_a20", ["clean_clean_full_wd01_a20"], "clean_clean_full_wd01_a20"),
        (0.30, "clean_clean_full_wd01_a30", ["clean_clean_full_wd01_a30"], "clean_clean_full_wd01_a30"),
        (0.50, "clean_clean_full_wd01_a50", ["clean_clean_full_wd01_a50"], "clean_clean_full_wd01_a50"),
        (1.0, "clean_clean_full_wd01_sft", ["clean_clean_full_wd01_sft"], "clean_full_wd01_sft_devfix"),
    ],
    "wd03": [
        (0.0, "clean_start", None, "q35_start_devfix"),
        (0.10, "clnom_wd03_a10", ["clnom_wd03_a10"], "clnom_wd03_a10"),
        (0.20, "clnom_wd03_a20", ["clnom_wd03_a20"], None),
        (0.30, "clnom_wd03_a30", ["clnom_wd03_a30"], None),
        (0.50, "clnom_wd03_a50", ["clnom_wd03_a50"], None),
        (1.0, "clnom_wd03_sft", ["clnom_wd03_sft"], None),
    ],
    "newmt": [
        (0.0, "clean_start", None, "q35_start_devfix"),
        (0.10, "clnom_newmt_a10", ["clnom_newmt_a10"], "clnom_newmt_a10"),
        (0.50, "clnom_newmt_a50", ["clnom_newmt_a50"], None),
        (1.0, "clfull_newmt_sft", ["clfull_newmt_sft"], None),
    ],
    "old_method_q35a100": [
        (0.0, "q35_a100", None, None),
        (0.10, "q35_a100_method_soupa10", None, None),
        (0.20, "q35_a100_method_soupa20", None, None),
        (0.30, "q35_a100_method_soupa30", None, None),
        (0.50, "q35_a100_method_soupa50", None, None),
        (0.70, "q35_a100_method_soupa70", None, None),
        (1.0, "q35_a100_method", None, None),
    ],
}


def research_dir_for(tag):
    if tag.endswith(":combined"):
        base = tag.split(":")[0]
        return os.path.join(OUT, f"cc_eval_{base}_thinking_32k_vllm")
    return os.path.join(OUT, f"cc_eval_{tag}_research_thinking_32k_vllm")


def main():
    results = {}
    cache = {}
    for arm, rows in ARMS.items():
        results[arm] = []
        for alpha, both_tag, res_tags, mls_tag in rows:
            entry = {"alpha": alpha, "tag": both_tag}
            both_dir = os.path.join(OUT, f"cc_eval_{both_tag}_thinking_32k_both_vllm")
            if os.path.isdir(both_dir):
                if both_tag not in cache:
                    s = load_samples(both_dir)
                    cache[both_tag] = (fcs_stats(s), ale_stats(s))
                f, a = cache[both_tag]
                if f:
                    entry.update(f)
                if a:
                    entry.update(a)
            if res_tags:
                dirs = []
                for rt in res_tags:
                    d = research_dir_for(rt)
                    if os.path.isdir(d):
                        dirs.append(d)
                if dirs:
                    r = research_stats(dirs)
                    if r:
                        entry.update({("res_" + k if not k.startswith("research") else k): v
                                      for k, v in r.items()})
            if mls_tag:
                m = mls_stats(mls_tag)
                if m:
                    entry.update(m)
            results[arm].append(entry)
    json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "soup_alpha_results.json", "w"), indent=1)
    # pretty print
    for arm, rows in results.items():
        print(f"\n=== {arm} ===")
        hdr = ["alpha", "fcs_mean5", "fcs_best5", "ale_mean5", "research_mean5", "mls_none0",
               "think_mean_chars", "short_think_share", "trunc_share", "no_close_think_share",
               "key_insight_per10k", "wait_per10k", "innov_per1k", "prag_per1k", "score_pos_share", "errors"]
        print("\t".join(hdr))
        for e in rows:
            print("\t".join(
                (f"{e.get(h):.4g}" if isinstance(e.get(h), float) else str(e.get(h, "-")))
                for h in hdr))


if __name__ == "__main__":
    main()
