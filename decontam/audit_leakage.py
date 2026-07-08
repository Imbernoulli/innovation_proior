#!/usr/bin/env python3
"""
Data-leakage auditor for the innovation_proior SFT data vs the five eval
benchmarks (FrontierCS / ALE-Bench / ThetaEvolve / TTT-Discover / MLS-Bench).

Goes BEYOND the earlier n-gram audit (experiments/DATA_CONTAMINATION_AUDIT_zh.md):
it catches semantic / classic-problem-reconstruction leakage that n-gram cannot,
using a curated eval-task registry (decontam/eval_registry.json) plus a
deterministic MLS same-task overlap (source slug in MLS-Bench tasks/*).

Outputs (nothing here mutates the original sft/*.jsonl* or tag files):
  decontam/leakage_tags_sft.jsonl    line-aligned with sft/innovation_sft.jsonl(.gz)   (2698)
  decontam/leakage_tags_wave2.jsonl  line-aligned with sft/innovation_wave2_sft.jsonl  (758)
  decontam/leakage_tags_v4.jsonl     line-aligned with sft/innovation_v4_sft.jsonl     (346)
  decontam/summary.json              rollup counts + per-family / per-benchmark
  decontam/clean/*.jsonl.gz          decontaminated duplicates (contaminated rows dropped)
  decontam/removed/*.jsonl           the dropped rows (id + reason), for review

Each annotation row: {file,line,id,kind,leak,contaminated,exclude_from_summary,
                      benchmarks[],family,severity,reason}
Severity: CRITICAL > HIGH > MEDIUM > REVIEW > LOW > NONE.
"""
import json, gzip, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # innovation_proior/
DEC = ROOT / "decontam"
MLS_TASKS = Path("/srv/home/bohanlyu/MLS-Bench/tasks")

SEV_RANK = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "REVIEW": 2, "LOW": 1, "NONE": 0}

# Decontamination policy (2026-07-08 user directive; see experiments/DATA_LEAKAGE_AUDIT_zh.md):
#   * DROP entirely: discovery HEURISTIC-SEARCH / record constructions + AHC039 (they reconstruct the
#     eval instance). The generic problem/paper methods below are KEPT (public science, not overfit
#     to an eval instance).
#   * DROP FINALE TURN only: MLS same-task trajectories that inject a non-native stronger baseline.
#   * KEEP: MLS baseline ladders, standalone paper methods (incl. the finale methods), FCS-research
#     paper reconstructions, v4/wave2 synthetic (n-gram-clean).
KEEP_PAPER = {"circle-packing-in-square", "cap-set", "kissing-number", "fast-matrix-multiplication",
              "erdos-minimum-overlap", "autocorrelation-inequalities", "heilbronn-triangle",
              "low-autocorrelation-sequences", "sums-and-differences-sets"}
DISCOVERY_DROP_FAMS = {"circle_packing_n26", "erdos_min_overlap_c5", "autocorr_inequalities",
                       "hadamard_maxdet", "heilbronn_triangle", "cap_set", "kissing_number",
                       "fast_matrix_mult", "low_autocorr_binary_seq", "sums_diffs_finite_sets",
                       "ale_ahc039"}

def load_mls_slugs():
    slugs = set()
    for p in sorted(MLS_TASKS.iterdir()):
        if p.is_dir() and p.name not in ("deprecated",) and (p / "config.json").exists():
            slugs.add(p.name)
    return slugs

def _mls_native_baselines(slug):
    cfg = MLS_TASKS / slug / "config.json"
    if not cfg.exists():
        return set()
    try:
        d = json.load(open(cfg))
    except Exception:
        return set()
    b = d.get("baselines")
    names = set()
    if isinstance(b, dict):
        names |= set(b.keys())
    elif isinstance(b, list):
        for x in b:
            if isinstance(x, str):
                names.add(x)
            elif isinstance(x, dict):
                for k in ("name", "id", "slug", "label"):
                    if x.get(k):
                        names.add(x[k])
    return names

def load_type1_finale_offenders(mls_slugs):
    """MLS trajectories whose meta.json finale rung is NOT a native baseline
    (the user's 'Type 1' non-native-stronger-baseline leakage)."""
    offenders = {}
    for tdir in (ROOT / "trajectories").iterdir():
        if not tdir.is_dir() or tdir.name not in mls_slugs:
            continue
        meta = tdir / "meta.json"
        if not meta.exists():
            continue
        try:
            m = json.load(open(meta))
        except Exception:
            continue
        nb = _mls_native_baselines(tdir.name)
        for s in m.get("steps", []):
            if not s.get("finale"):
                continue
            sslug = s.get("slug", "")
            cand = {sslug, sslug.replace("-", "_"), sslug.replace("_", "-")}
            if not (cand & nb):
                offenders[tdir.name] = {"finale_slug": sslug, "method": s.get("method")}
    return offenders

def build_slug_map(reg):
    """slug -> dict(benchmarks, family, severity, reason)."""
    m = {}
    def put(slug, sev, ben=None, fam=None, reason=None):
        cur = m.get(slug)
        if cur is None or SEV_RANK[sev] > SEV_RANK[cur["severity"]]:
            m[slug] = {"benchmarks": sorted(set(benen)) if (benen:=(ben or [])) else [],
                       "family": fam, "severity": sev, "reason": reason}
    for fam in reg["families"]:
        b, name, note = fam["benchmarks"], fam["family"], fam.get("note", "")
        et = fam.get("eval_task", "")
        for slug in fam.get("critical_slugs", []):
            put(slug, "CRITICAL", b, name, f"Direct reconstruction of eval task [{et}]. {note}")
        for slug in fam.get("high_slugs", []):
            put(slug, "HIGH", b, name, f"Same eval-task family/ladder rung [{et}]. {note}")
        for slug in fam.get("medium_slugs", []):
            put(slug, "MEDIUM", b, name, f"Discovery-eval-family topic [{et}]; eval-set membership not locally confirmed. {note}")
    return m

def classify_slug(slug, kind, slug_map, mls_slugs, mls20, cp_review):
    """Return (leak, contaminated, benchmarks, family, severity, reason)."""
    # 1) explicit registry match (discovery / ALE / FCS-research reconstructions)
    if slug in slug_map:
        e = slug_map[slug]
        sev = e["severity"]
        contaminated = sev in ("CRITICAL", "HIGH")
        return (True, contaminated, e["benchmarks"], e["family"], sev, e["reason"])
    # 2) MLS-Bench same-task overlap (trajectory / agentic / method whose slug IS an MLS task)
    if slug in mls_slugs:
        ev = slug in mls20
        sev = "CRITICAL" if ev else "HIGH"
        reason = ("Same MLS-Bench task as an EVALUATED task (research question + fixed interface + "
                  "baseline ladder reproduced as a multi-turn %s; trajectories can inject non-native, "
                  "stronger-than-native baselines into context)." % kind) if ev else \
                 ("Same MLS-Bench task slug (research question + interface + baseline ladder reproduced "
                  "as multi-turn %s). Not clean held-out for MLS." % kind)
        return (True, True, ["MLS"], "mls_same_task", sev, reason)
    # 3) competitive-programming classics: could answer an FCS algorithmic problem (semantic; needs review)
    if slug in cp_review:
        return (True, False, ["FCS"], "cp_classic", "REVIEW",
                "Textbook competitive-programming algorithm; verify no FrontierCS algorithmic problem's "
                "intended solution is exactly this (n-gram cannot catch a paraphrased statement).")
    return (False, False, [], None, "NONE", "")

def annotate_main(slug_map, mls_slugs, mls20, cp_review, type1_offenders):
    """innovation_sft: kinds method / v4 / traj_* / agentic_* via _sft_tags.jsonl."""
    # map: finale method slug (and hyphen/underscore variants) -> the MLS task it is SOTA for
    finale_method_map = {}
    for task, off in type1_offenders.items():
        fs = off["finale_slug"]
        for v in {fs, fs.replace("-", "_"), fs.replace("_", "-")}:
            finale_method_map.setdefault(v, task)
    rows = []
    for i, l in enumerate(open(ROOT / "sft" / "_sft_tags.jsonl")):
        d = json.loads(l)
        raw_id, kind = d["id"], d["kind"]
        slug = re.sub(r"#.*$", "", raw_id)               # strip #rN framing
        type1 = False
        if kind == "v4":
            # synthetic FrontierSmith-style single-file C++ (ale-*/cp*): register-matched, not copied
            leak, cont, ben, fam, sev, reason = (
                True, False, ["FCS", "ALE"], "v4_synth_register", "MEDIUM",
                "Synthetic FCS/ALE-register single-file C++ (constructed, not copied); same input "
                "distribution as FCS/ALE eval. Verify no accidental match to a real benchmark problem.")
        else:
            leak, cont, ben, fam, sev, reason = classify_slug(slug, kind, slug_map, mls_slugs, mls20, cp_review)
            if (not leak) and kind == "method" and slug in finale_method_map:
                # standalone method that is ALSO the strongest-known method injected as an MLS finale
                mls_task = finale_method_map[slug]
                leak, cont, ben, fam, sev = True, False, ["MLS"], "mls_finale_method_standalone", "REVIEW"
                reason = ("Standalone method example that is the strongest-known method for MLS-Bench task "
                          "'%s' (injected there as a non-native finale). Present as a general 'innovation "
                          "prior' method, so kept, but it teaches the exact SOTA for that MLS task." % mls_task)
            if fam == "mls_same_task" and slug in type1_offenders and kind.startswith(("traj", "agentic")):
                type1 = True
                off = type1_offenders[slug]
                reason = ("TYPE-1 non-native stronger baseline: this MLS trajectory injects '%s' (%s) as a "
                          "finale endpoint that is NOT among the task's native baselines — extra capability "
                          "beyond what MLS-Bench discloses at inference. " % (off["finale_slug"], off["method"])) + reason
        rows.append({"file": "sft/innovation_sft.jsonl", "line": i, "id": raw_id, "kind": kind,
                     "leak": leak, "contaminated": cont, "type1_nonnative_finale": type1,
                     "exclude_from_summary": leak, "benchmarks": ben,
                     "family": fam, "severity": sev, "reason": reason})
    return rows

def annotate_wave2():
    rows = []
    for i, l in enumerate(open(ROOT / "sft" / "_wave2_tags.jsonl")):
        d = json.loads(l)
        rid, domain = d["id"], str(d.get("domain", ""))
        leak, cont, ben, fam, sev, reason = False, False, [], None, "NONE", ""
        if domain.startswith("fcs_codex"):
            leak, ben, fam, sev = True, ["FCS"], "wave2_fcs_codex", "MEDIUM"
            reason = ("Codex gpt-5.5 solutions to synthetic FCS-style problems (constructed, not real FCS "
                      "problem ids). Register-matched to FrontierCS; verify none reconstructs a real FCS problem.")
        elif domain == "code":
            leak, ben, fam, sev = True, ["FCS"], "wave2_code_bank", "REVIEW"
            reason = ("Competitive C++ from a public 'hardtests' bank; verify no overlap with FrontierCS "
                      "algorithmic problems (which may reuse public competitive problems).")
        elif domain == "math":
            leak, ben, fam, sev = True, ["THETA", "TTT"], "wave2_math", "REVIEW"
            reason = "Competition math; verify no overlap with ThetaEvolve/TTT-Discover math-discovery tasks."
        # reasoning / ifollow -> general breadth, not benchmark-specific -> NONE
        rows.append({"file": "sft/innovation_wave2_sft.jsonl", "line": i, "id": rid, "kind": "wave2:" + domain,
                     "leak": leak, "contaminated": cont, "exclude_from_summary": leak,
                     "benchmarks": ben, "family": fam, "severity": sev, "reason": reason})
    return rows

def annotate_v4():
    rows = []
    for i, l in enumerate(open(ROOT / "sft" / "_v4_tags.jsonl")):
        d = json.loads(l)
        rid = d["id"]
        rows.append({"file": "sft/innovation_v4_sft.jsonl", "line": i, "id": rid, "kind": "v4",
                     "leak": True, "contaminated": False, "exclude_from_summary": True,
                     "benchmarks": ["FCS", "ALE"], "family": "v4_synth_register", "severity": "MEDIUM",
                     "reason": ("Synthetic FCS/ALE-register single-file C++ (constructed, not copied); same "
                                "input distribution as FCS/ALE eval. Verify no accidental real-problem match.")})
    return rows

def write_clean(src_gz_candidates, tag_rows, clean_path, removed_path):
    """Drop rows with contaminated=True. Read src (gz or plain), keep line alignment."""
    src = None
    for c in src_gz_candidates:
        if Path(c).exists():
            src = c; break
    if src is None:
        return {"src": None, "kept": 0, "removed": 0, "note": "source jsonl not found; skipped clean copy"}
    opener = (lambda p: gzip.open(p, "rt")) if src.endswith(".gz") else (lambda p: open(p))
    removed = []
    kept = 0
    with opener(src) as fin, gzip.open(clean_path, "wt") as fout:
        for i, line in enumerate(fin):
            tag = tag_rows[i] if i < len(tag_rows) else {"contaminated": False}
            if tag.get("contaminated"):
                removed.append({"line": i, "id": tag.get("id"), "severity": tag.get("severity"),
                                "benchmarks": tag.get("benchmarks"), "family": tag.get("family"),
                                "reason": tag.get("reason")})
            else:
                fout.write(line); kept += 1
    with open(removed_path, "w") as fr:
        for r in removed:
            fr.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"src": src, "kept": kept, "removed": len(removed)}

def rollup(rows):
    from collections import Counter
    sev = Counter(r["severity"] for r in rows if r["leak"])
    ben = Counter()
    fam = Counter()
    cont = sum(1 for r in rows if r["contaminated"])
    for r in rows:
        if not r["leak"]:
            continue
        for b in r["benchmarks"]:
            ben[b] += 1
        fam[r["family"]] += 1
    return {"n": len(rows), "leak": sum(1 for r in rows if r["leak"]),
            "contaminated": cont, "by_severity": dict(sev),
            "by_benchmark": dict(ben), "by_family": dict(fam)}

def main():
    reg = json.load(open(DEC / "eval_registry.json"))
    mls_slugs = load_mls_slugs()
    mls20 = set(reg["mls_evaluated_20"])
    cp_review = set(reg["cp_classic_review_slugs"])
    slug_map = build_slug_map(reg)
    type1_offenders = load_type1_finale_offenders(mls_slugs)

    main_rows = annotate_main(slug_map, mls_slugs, mls20, cp_review, type1_offenders)
    wave2_rows = annotate_wave2()
    v4_rows = annotate_v4()

    # ---- decontam rules (the build_sft.py gate reads this) ----
    drop_method, drop_traj = set(), set()
    for r in main_rows:
        slug = r["id"].split("#")[0]
        if r["family"] in DISCOVERY_DROP_FAMS:
            if r["kind"] == "method" and slug not in KEEP_PAPER:
                drop_method.add(slug)
            elif r["kind"].startswith(("traj", "agentic")):
                drop_traj.add(slug)
    rules = {
        "drop_method_slugs": sorted(drop_method),
        "drop_traj_slugs": sorted(drop_traj),
        "type1_finale_traj": sorted(type1_offenders),
        "keep_paper_methods": sorted(KEEP_PAPER),
        "_policy": ("drop_method_slugs/drop_traj_slugs removed entirely (discovery heuristic-search/record "
                    "constructions + AHC039). type1_finale_traj: keep trajectory, skip finale rung. "
                    "Everything else kept. Per user directive 2026-07-08."),
    }
    json.dump(rules, open(DEC / "decontam_rules.json", "w"), indent=1, ensure_ascii=False)

    # ---- per-row decontam_action, folded into the annotations ----
    def action(r):
        slug = r["id"].split("#")[0]
        if slug in drop_method or slug in drop_traj:
            return "drop_row"
        if r.get("type1_nonnative_finale"):
            return "drop_finale_turn"
        if r["leak"]:
            return "keep_flagged"
        return "keep"
    for r in main_rows:
        r["decontam_action"] = action(r)
    for r in wave2_rows + v4_rows:
        r["decontam_action"] = "keep_flagged" if r["leak"] else "keep"

    for name, rows in [("sft", main_rows), ("wave2", wave2_rows), ("v4", v4_rows)]:
        with open(DEC / f"leakage_tags_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # denylist for build-time gate = the slugs actually removed (not the kept-but-flagged ladders)
    (DEC / "benchmark_denylist.txt").write_text("\n".join(sorted(drop_method | drop_traj)) + "\n")

    from collections import Counter
    act = Counter(r["decontam_action"] for r in main_rows)
    summary = {
        "mls_task_slugs_total": len(mls_slugs),
        "policy": "surgical (2026-07-08 user directive): see decontam_rules.json",
        "decontam_actions_sft": dict(act),
        "rules": {"drop_method_slugs": len(drop_method), "drop_traj_slugs": len(drop_traj),
                  "type1_finale_traj": len(type1_offenders)},
        "mls_type1_nonnative_finale": {
            "n_trajectories": len(type1_offenders),
            "offenders": {k: v["finale_slug"] for k, v in sorted(type1_offenders.items())},
        },
        "rollup": {"sft": rollup(main_rows), "wave2": rollup(wave2_rows), "v4": rollup(v4_rows)},
        "clean_build": ("Produced by the build_sft.py gate (INNOVATION_DECONTAM=1, reads decontam_rules.json): "
                        "decontam/clean_rebuilt/innovation_sft.jsonl(.gz). wave2/v4 are kept unchanged "
                        "(n-gram-clean synthetic). Originals under sft/ are never modified."),
    }
    json.dump(summary, open(DEC / "summary.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
