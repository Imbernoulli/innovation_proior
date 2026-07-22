#!/usr/bin/env python3
"""curate_wave2b.py -- turn raw lens-agent seed candidates into the wave-2b pack.

Input : a JSON file with the seed-imagination workflow result
        ({"candidates": [{"lens": ..., "seeds": [...]}, ...]}).
Output: seeds/bulk_seed_packs/pack_w2b_0507_1006.jsonl (full corpus-schema rows)
        + a curation report on stdout.

Curation rules (the anti-wave-2-incident discipline):
  1. family must be unique: within the batch AND vs the existing corpus
     (token-set near-duplicates are rejected, not just exact matches).
  2. mechanisms signature (sorted tag set) must be unique within the batch --
     two seeds composing the same mechanisms are the same problem in costume.
  3. per-lens cap so no single lens dominates.
  4. format quotas (A/B/C/D/E) hit the target mix.
  5. IDs fsx_<TIER>_0507..1006 assigned in curated order.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parent

FORMAT_QUOTA = {"A": 150, "C": 130, "B": 110, "D": 50, "E": 60}   # = 500
LENS_CAP = 30
DANG = {"S": "核心", "A": "重要", "B": "应用前沿", "C": "方法与异域前沿",
        "N": "高新颖度", "G": "泛化补全"}
BRIEF_FILE = {"A": "AGENT_BRIEF.md", "B": "AGENT_BRIEF_PY_PROGRAM.md",
              "C": "AGENT_BRIEF_PY_STDOUT.md", "D": "AGENT_BRIEF_PY_STDOUT.md",
              "E": "AGENT_BRIEF_PY_STDOUT.md"}
EVAL_FORM = {"A": "quality-metric", "B": "quality-metric", "C": "quality-metric",
             "D": "flops", "E": "quality-metric"}


def kebab(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip().lower())
    return re.sub(r"-+", "-", s).strip("-")


def fam_tokens(fam: str) -> frozenset:
    drop = {"the", "a", "of", "and", "with", "for", "under", "on", "in", "via", "based"}
    return frozenset(t for t in kebab(fam).split("-") if t and t not in drop)


def near_dup(tok_a: frozenset, tok_b: frozenset) -> bool:
    if not tok_a or not tok_b:
        return False
    inter = len(tok_a & tok_b)
    return inter / min(len(tok_a), len(tok_b)) >= 0.75


def build_brief(row: dict) -> str:
    fmt = row["format"]
    return (f"[{fmt}|WAVE2B-INNOVATION] Author a NOVEL, deterministically-scored problem in family "
            f"'{row['family']}', objective={row['objective']}imize, skin '{row['theme']}'. "
            f"Compose ALL of these mechanisms into one objective: {', '.join(row['mechanisms'])}. "
            f"The strong solution must exploit: {row['innovation_hook']} "
            f"The generator must plant trap cases where the obvious greedy approach lands far from strong. "
            f"Follow {BRIEF_FILE[fmt]} AND AGENT_BRIEF_INNOVATION_ADDENDUM.md; acceptance needs harness PASS "
            f"plus strong-greedy>=0.06, strong<=0.92, greedy-trivial>=0.03.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates_json", type=Path)
    ap.add_argument("--out", type=Path,
                    default=HERE / "bulk_seed_packs" / "pack_w2b_0507_1006.jsonl")
    ap.add_argument("--start", type=int, default=507)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--keep-all", action="store_true",
                    help="no format quotas / lens caps / target cap: keep every seed that passes dedup")
    ap.add_argument("--exclude-pack", type=Path, default=None,
                    help="dedup against an already-frozen pack (families+mechanisms already used)")
    args = ap.parse_args()

    if args.keep_all:
        for k in FORMAT_QUOTA:
            FORMAT_QUOTA[k] = 10 ** 9
        globals()["LENS_CAP"] = 10 ** 9
        args.target = 10 ** 9

    payload = json.loads(args.candidates_json.read_text())
    cand_groups = payload["candidates"] if "candidates" in payload else payload

    existing_fams = [json.loads(l)["family"] for l in open(HERE / "seed_list.jsonl")]
    existing_toks = [fam_tokens(f) for f in existing_fams]

    pre_used_mech = set()
    if args.exclude_pack:
        for l in open(args.exclude_pack):
            r = json.loads(l)
            existing_toks.append(fam_tokens(r["family"]))
            if r.get("mechanisms"):
                pre_used_mech.add(tuple(sorted(kebab(m) for m in r["mechanisms"])))

    # flatten, keep lens provenance, preserve agent order (their best first is not
    # guaranteed, so we interleave lenses round-robin for fairness)
    per_lens = []
    for grp in cand_groups:
        lens, seeds = grp["lens"], grp["seeds"]
        per_lens.append([(lens, s) for s in seeds])
    interleaved = []
    idx = 0
    while any(per_lens):
        for lst in per_lens:
            if idx < len(lst):
                interleaved.append(lst[idx])
        idx += 1
        if idx > 100:
            break

    seen_fam_toks = list(existing_toks)
    seen_mech = set(pre_used_mech)
    lens_count = Counter()
    fmt_count = Counter()
    picked, rejected = [], Counter()
    quota_overflow = []   # structurally-fine seeds rejected only by format quota

    for lens, s in interleaved:
        fam = kebab(s.get("family", ""))
        fmt = s.get("format")
        mechs = tuple(sorted(kebab(m) for m in s.get("mechanisms", [])))
        if not fam or fmt not in FORMAT_QUOTA or not mechs:
            rejected["malformed"] += 1
            continue
        if lens_count[lens] >= LENS_CAP:
            rejected["lens-cap"] += 1
            continue
        toks = fam_tokens(fam)
        if any(near_dup(toks, t) for t in seen_fam_toks):
            rejected["family-near-dup"] += 1
            continue
        if mechs in seen_mech:
            rejected["mechanism-dup"] += 1
            continue
        if fmt_count[fmt] >= FORMAT_QUOTA[fmt]:
            rejected[f"quota-{fmt}"] += 1
            quota_overflow.append({**s, "family": fam, "mechanisms": list(mechs), "lens": lens})
            continue
        tier = s.get("suggested_tier") if s.get("suggested_tier") in DANG else "G"
        picked.append({**s, "family": fam, "mechanisms": list(mechs),
                       "tier": tier, "lens": lens})
        seen_fam_toks.append(toks)
        seen_mech.add(mechs)
        lens_count[lens] += 1
        fmt_count[fmt] += 1
        if len(picked) >= args.target:
            break

    # ---- retype pass: C-overflow -> A (same gen+checker FORM, C++ implementation) ----
    # A C-format construction problem is legitimately authorable as a testlib C++ problem;
    # only do this to fill the A quota, never the reverse (B/D/E are different forms).
    if len(picked) < args.target and fmt_count["A"] < FORMAT_QUOTA["A"]:
        for s in quota_overflow:
            if s.get("format") != "C" or fmt_count["A"] >= FORMAT_QUOTA["A"]:
                continue
            toks = fam_tokens(s["family"])
            mechs = tuple(s["mechanisms"])
            if any(near_dup(toks, t) for t in seen_fam_toks) or mechs in seen_mech:
                continue
            tier = s.get("suggested_tier") if s.get("suggested_tier") in DANG else "G"
            picked.append({**s, "format": "A", "tier": tier, "retyped_from": "C"})
            seen_fam_toks.append(toks)
            seen_mech.add(mechs)
            lens_count[s["lens"]] += 1
            fmt_count["A"] += 1
            rejected["quota-C"] -= 1
            rejected["retyped-C-to-A"] += 1
            if len(picked) >= args.target:
                break

    rows = []
    scales = ["small", "medium", "large"]
    for i, s in enumerate(picked):
        pid = f"fsx_{s['tier']}_{args.start + i:04d}"
        rows.append({
            "id": pid, "tier": s["tier"], "dang": DANG[s["tier"]],
            "format": s["format"], "brief_file": BRIEF_FILE[s["format"]],
            "eval_form": EVAL_FORM[s["format"]],
            "family": s["family"],
            "mechanisms": s["mechanisms"],
            "innovation_hook": s["innovation_hook"],
            "source_frameworks": [f"wave2b-lens:{s['lens']}"],
            "why_it_generalizes": s["why_it_generalizes"],
            "seed_example": s["seed_example"],
            "objective": s["objective"], "theme": s["theme"],
            "scale": scales[i % 3], "variant": 0,
            "brief": build_brief(s),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "picked": len(rows), "target": args.target,
        "by_format": dict(fmt_count), "by_tier": dict(Counter(r["tier"] for r in rows)),
        "by_lens": dict(lens_count), "rejected": dict(rejected),
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    if not args.keep_all and len(rows) < args.target:
        print(f"WARNING: only {len(rows)} < {args.target}; run a top-up imagination round "
              f"for the missing formats", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
