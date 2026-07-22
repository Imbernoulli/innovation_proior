#!/usr/bin/env python3
"""
build_seed_list.py -- generate the tiered seed list of problem specs from the researched,
cross-framework taxonomy (reports/taxonomy_proposal.json).

Spans FrontierCS/ALE + FunSearch/AlphaEvolve/OpenEvolve/ThetaEvolve/TTT-Discover +
Frontier-Eng + MLS-Bench, restricted to DETERMINISTIC scoring (no wall-time/GPU; FLOPs OK).

Importance tiers (档):
    S 核心            graph & combinatorial optimization        (Format A, testlib C++)   80
    A 重要            math-discovery / heuristic evolution       (Formats B/C/D)           70
    B 应用前沿        engineering + scientific optimization      (Formats B/D/E)           30
    C 方法与异域前沿  ML-method design + exotic construction     (Formats B/C)             20

Formats:
    A  testlib instance-based combinatorial opt (C++ gen.cpp + chk.cc)        -> AGENT_BRIEF.md
    C  Python constructive-artifact + verifier (gen.py + verify.py, stdin/out)-> AGENT_BRIEF_PY_STDOUT.md
    D  FLOPs / op-count deterministic kernel-surrogate (gen.py + counter.py)  -> AGENT_BRIEF_PY_STDOUT.md
    E  symbolic regression w/ held-out split (gen.py + verify.py)             -> AGENT_BRIEF_PY_STDOUT.md
    B  evolve-a-heuristic against a frozen evaluator (evaluator.py)           -> AGENT_BRIEF_PY_PROGRAM.md

Each spec fixes only the scaffold; the generation agent instantiates the full, novel problem.
Default plan sums to 200; --per-tier N plans N per tier. Use --full500 to assemble
the canonical 500-spec base corpus. Use --current to assemble the checked-in current
corpus: base 500 + subagent supplements + bulk seed packs.
"""
import argparse, hashlib, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAX = HERE.parent / "reports" / "taxonomy_proposal.json"

THEMES = [
    "power grid", "deep-sea cable network", "interstellar relay", "coral reef survey",
    "subway system", "drone delivery swarm", "vineyard irrigation", "glacier sensor net",
    "festival stage layout", "warehouse robotics", "asteroid mining", "pandemic contact net",
    "quantum lab wiring", "railway freight yard", "wildlife corridor", "smart-city lighting",
    "orbital debris cleanup", "archaeology dig grid", "volcano monitoring", "e-sports arena",
    "bakery supply chain", "telescope array", "reservoir dam network", "polar research base",
    "traffic signal grid", "beehive apiary", "data-center cooling", "harbor container port",
    "ski resort lifts", "vaccine cold chain", "wind-farm turbines", "museum gallery tour",
    "salmon migration ladder", "carnival ride circuit", "lunar habitat", "recycling depot routes",
    "forest fire watchtowers", "tide pool ecology", "mountain rescue relays", "rooftop gardens",
    "satellite ground stations", "greenhouse zones", "highway toll gantries", "aquarium plumbing",
    "cave mapping expedition", "wind tunnel sensors", "solar farm inverters", "geothermal wells",
]
SCALES = ["small", "medium", "large"]        # instance-size sweep for generalization
FORMAT_BRIEF = {
    "A": "AGENT_BRIEF.md",
    "C": "AGENT_BRIEF_PY_STDOUT.md",
    "D": "AGENT_BRIEF_PY_STDOUT.md",
    "E": "AGENT_BRIEF_PY_STDOUT.md",
    "B": "AGENT_BRIEF_PY_PROGRAM.md",
}


def pick(lst, *keys):
    h = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return lst[int(h, 16) % len(lst)]


def infer_objective(text):
    t = text.lower()
    if "maxim" in t: return "max"
    if "minim" in t: return "min"
    return "optimize"


def build(per_tier=None, batch=1, avoid_file=None):
    """batch>1 continues variant/theme rotation and offsets IDs by 200*(batch-1), so batch 2
    yields fresh (family,theme,variant) scaffolds with IDs 201..400 that don't collide with batch 1.
    avoid_file: an existing seed_list.jsonl whose (family,theme) pairs are excluded (extra distinctness)."""
    tax = json.loads(TAX.read_text())
    specs = []
    avoid = set()
    if avoid_file and Path(avoid_file).exists():
        for l in open(avoid_file):
            s = json.loads(l); avoid.add((s["family"], s["theme"]))
    for tier in tax["tiers"]:
        tname = tier["tier"]
        fams = tier["families"]
        target = per_tier if per_tier else tier.get("default_count", len(fams))
        used = set(avoid)
        for k in range(target):
            i = target * (batch - 1) + k          # continue rotation across batches
            fam = fams[i % len(fams)]
            rep = i // len(fams)
            # theme unique per (family, variant); scale sweeps for generalization
            ti = int(hashlib.sha256(f"{tname}|{fam['family']}|{i}".encode()).hexdigest(), 16) % len(THEMES)
            for _ in range(len(THEMES)):
                theme = THEMES[ti]
                if (fam["family"], theme) not in used:
                    break
                ti = (ti + 1) % len(THEMES)
            used.add((fam["family"], theme))
            scale = SCALES[rep % len(SCALES)]
            fmt = fam["format"]
            idx = 200 * (batch - 1) + len(specs) + 1
            spec = {
                "id": f"fsx_{tname}_{idx:04d}",
                "tier": tname,
                "dang": tier.get("dang", ""),
                "format": fmt,
                "brief_file": FORMAT_BRIEF[fmt],
                "eval_form": fam["eval_form"],
                "family": fam["family"],
                "source_frameworks": fam.get("source_frameworks", []),
                "why_it_generalizes": fam.get("why_it_generalizes", ""),
                "seed_example": fam.get("example", ""),
                "objective": infer_objective(fam.get("example", "")),
                "theme": theme,
                "scale": scale,
                "variant": rep,
                "brief": (f"[{fmt}|{fam['eval_form']}] Instantiate a NOVEL, deterministically-scored "
                          f"problem in family '{fam['family']}' (tier {tname}), inspired by "
                          f"{','.join(fam.get('source_frameworks', [])[:2])}. Vary it to a '{scale}'-scale "
                          f"instance, variant #{rep}, skinned as '{theme}'. Anchor idea: {fam.get('example','')}"),
            }
            specs.append(spec)
    return specs


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def build_full500():
    specs = []
    specs.extend(build(batch=1))
    specs.extend(load_jsonl(HERE / "seed_list_b2.jsonl"))
    specs.extend(load_jsonl(HERE / "novelty_seeds.jsonl"))
    specs.extend(load_jsonl(HERE / "gen_seeds.jsonl"))
    ids = [s["id"] for s in specs]
    if len(specs) != 500 or len(set(ids)) != 500:
        raise SystemExit(f"full500 assembly invariant failed: {len(specs)} specs, {len(set(ids))} unique ids")
    return specs


def build_current():
    specs = build_full500()
    specs.extend(load_jsonl(HERE / "subagent_seeds.jsonl"))
    for path in sorted((HERE / "bulk_seed_packs").glob("pack_*.jsonl")):
        specs.extend(load_jsonl(path))
    ids = [s["id"] for s in specs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"current assembly invariant failed: {len(specs)} specs, {len(set(ids))} unique ids")
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(HERE / "seed_list.jsonl"))
    ap.add_argument("--per-tier", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--avoid", default=None, help="existing seed_list to exclude (family,theme) pairs from")
    ap.add_argument("--full500", action="store_true",
                    help="assemble the canonical 500-spec base corpus")
    ap.add_argument("--current", action="store_true",
                    help="assemble the current checked-in corpus")
    args = ap.parse_args()
    if args.current:
        specs = build_current()
    elif args.full500:
        specs = build_full500()
    else:
        specs = build(per_tier=args.per_tier, batch=args.batch, avoid_file=args.avoid)
    with open(args.out, "w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")
    from collections import Counter
    print(f"wrote {len(specs)} specs -> {args.out}")
    print("per tier:", dict(Counter(s["tier"] for s in specs)))
    print("per format:", dict(Counter(s["format"] for s in specs)))
    print("families:", len(set(s["family"] for s in specs)),
          "| unique (family,theme,variant):",
          len(set((s["family"], s["theme"], s["variant"]) for s in specs)), "/", len(specs))


if __name__ == "__main__":
    main()
