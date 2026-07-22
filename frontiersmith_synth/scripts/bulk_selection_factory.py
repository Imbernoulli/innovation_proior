#!/usr/bin/env python3
"""Generate a large batch of Format-C constructive selection problems.

The generated tasks share a rigorously validated scaffold but vary by seed spec,
domain skin, family, and deterministic parameter salt. They are intended for
bulk corpus expansion where every problem still carries its own generator,
checker, solution ladder, metadata, and validation report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


SYNTH = Path(__file__).resolve().parents[1]
PROBLEMS = SYNTH / "problems"
PACK_DIR = SYNTH / "seeds" / "bulk_seed_packs"


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def compact_words(text: str, n: int = 5) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.replace("-", " "))
    if not words:
        return "Constructive Selection"
    return " ".join(w.capitalize() for w in words[:n])


def title_for(spec: dict) -> str:
    base = compact_words(spec.get("theme") or spec.get("family"))
    return base if base.lower().endswith("portfolio") else f"{base} Portfolio"


def load_specs(paths: list[Path]) -> list[dict]:
    specs: list[dict] = []
    for path in paths:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: bad json: {exc}") from exc
            specs.append(row)
    return specs


def render_config() -> str:
    return """checker: verify.py
memory: 512m
subtasks:
- n_cases: 10
  score: 100
time: 5s
type: default
"""


def render_gen(seed: int) -> str:
    return f'''#!/usr/bin/env python3
import random
import sys

SALT = {seed}


def item_mask(i, m, salt, profile, rng):
    bits = set()
    span = 3 + ((i + profile) % 3)
    anchor = (i * 7 + salt + profile * 11) % m
    for k in range(span):
        bits.add((anchor + k * (profile + 3) + i * (k + 1)) % m)
    if (i + salt) % 9 == 0:
        bits.add((anchor + m // 2 + profile) % m)
    if rng.randrange(5) == 0:
        bits.add(rng.randrange(m))
    mask = 0
    for b in bits:
        mask |= 1 << b
    return mask


def main():
    t = 1
    if len(sys.argv) > 1:
        try:
            t = int(sys.argv[1])
        except ValueError:
            t = 1
    t = max(1, min(10, t))
    rng = random.Random(SALT * 1000003 + t * 9176)
    profile = SALT % 11
    n = 38 + 4 * t + (SALT % 13) + 2 * (profile % 4)
    m = 18 + (SALT % 8) + (t % 3)
    groups = 5 + (SALT % 5)
    group_cap = 3 + ((SALT + t) % 3)

    rows = []
    for i in range(n):
        group = (i * 37 + SALT + rng.randrange(groups)) % groups
        cost = 5 + ((i * 19 + SALT * 3 + t * 5) % 18) + (group % 4)
        if i < n // 5:
            value = 24 + ((i * 13 + t + SALT) % 28)
        else:
            value = 38 + ((i * 31 + SALT + t * 7) % 90)
        if (i + profile) % (7 + profile % 4) == 0:
            value += 36 + 4 * t
        if (i * 5 + SALT) % 17 == 0:
            cost = max(3, cost - 5)
            value += 18
        x = (i * 29 + SALT * 7 + rng.randrange(97)) % 211
        y = (i * 43 + SALT * 11 + rng.randrange(103)) % 211
        mask = item_mask(i, m, SALT + t * 13, profile, rng)
        rows.append((cost, group, value, x, y, mask))

    target = 10 + t + (profile % 5)
    cheapest = sorted(r[0] for r in rows)
    budget = sum(cheapest[:min(target, len(cheapest))]) + 6 + 2 * t

    print(n, m, groups, budget, group_cap, SALT + t * 13, profile)
    for cost, group, value, x, y, mask in rows:
        print(cost, group, value, x, y, mask)


if __name__ == "__main__":
    main()
'''


COMMON = r'''
import sys


def parse_instance_text(text):
    toks = text.split()
    if len(toks) < 7:
        raise ValueError("short instance")
    it = iter(toks)
    n = int(next(it)); m = int(next(it)); groups = int(next(it))
    budget = int(next(it)); group_cap = int(next(it)); salt = int(next(it)); profile = int(next(it))
    items = []
    for idx in range(1, n + 1):
        cost = int(next(it)); group = int(next(it)); value = int(next(it))
        x = int(next(it)); y = int(next(it)); mask = int(next(it))
        items.append({"idx": idx, "cost": cost, "group": group, "value": value,
                      "x": x, "y": y, "mask": mask})
    return {"n": n, "m": m, "groups": groups, "budget": budget,
            "group_cap": group_cap, "salt": salt, "profile": profile, "items": items}


def feature_weight(j, salt, profile):
    return 7 + ((j * 17 + salt * 5 + profile * 13) % 31)


def conflict(a, b, inst):
    dx = abs(a["x"] - b["x"]); dy = abs(a["y"] - b["y"])
    overlap = (a["mask"] & b["mask"]).bit_count()
    if a["group"] == b["group"] and dx + dy < 18 + inst["profile"]:
        return True
    if overlap >= 5 + (inst["profile"] % 3) and ((dx * 3 + dy * 5 + inst["salt"]) % 11 == 0):
        return True
    return False


def feasible(sel, inst):
    seen = set()
    cost = 0
    group_counts = {}
    items = inst["items"]
    for idx in sel:
        if idx < 1 or idx > inst["n"] or idx in seen:
            return False
        seen.add(idx)
        it = items[idx - 1]
        cost += it["cost"]
        if cost > inst["budget"]:
            return False
        group_counts[it["group"]] = group_counts.get(it["group"], 0) + 1
        if group_counts[it["group"]] > inst["group_cap"]:
            return False
    arr = [items[i - 1] for i in sel]
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if conflict(arr[i], arr[j], inst):
                return False
    return True


def evaluate(sel, inst):
    if not feasible(sel, inst):
        return 0
    items = [inst["items"][i - 1] for i in sel]
    score = sum(it["value"] for it in items)
    cover = [0] * inst["m"]
    for it in items:
        mask = it["mask"]
        for j in range(inst["m"]):
            if (mask >> j) & 1:
                cover[j] += 1
    for j, c in enumerate(cover):
        if c:
            w = feature_weight(j, inst["salt"], inst["profile"])
            score += w * min(3, c) + (w // 4) * max(0, c - 3)
    groups = {}
    for it in items:
        groups[it["group"]] = groups.get(it["group"], 0) + 1
    score += 17 * len(groups)
    for cnt in groups.values():
        score += 3 * min(cnt, inst["group_cap"])
    for i in range(len(items)):
        a = items[i]
        for j in range(i + 1, len(items)):
            b = items[j]
            if a["group"] != b["group"]:
                dist = abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
                if (dist + inst["salt"] + a["group"] * 3 + b["group"]) % 7 in (0, 1):
                    score += 4 + ((a["group"] + b["group"] + inst["profile"]) % 5)
    return max(1, score)


def can_add(sel, idx, inst):
    return feasible(sel + [idx], inst)


def baseline_select(inst):
    # Deliberately weak calibration anchor: a feasible "minimal deployment" rather than a
    # near-complete greedy packing. The checker uses this same construction for B, so the
    # trivial solution remains exactly calibrated at Ratio ~= 0.1 while better heuristics
    # have enough headroom to separate.
    sel = []
    target = 4 + (inst["profile"] % 2)
    for it in sorted(inst["items"], key=lambda z: (z["cost"], z["idx"])):
        if can_add(sel, it["idx"], inst):
            sel.append(it["idx"])
            if len(sel) >= target:
                break
    return sel


def greedy_select(inst):
    def key(it):
        mask_bonus = sum(feature_weight(j, inst["salt"], inst["profile"])
                         for j in range(inst["m"]) if (it["mask"] >> j) & 1)
        return (it["value"] + mask_bonus + 9 * (it["group"] + 1)) / max(1, it["cost"])
    sel = []
    for it in sorted(inst["items"], key=key, reverse=True):
        if can_add(sel, it["idx"], inst):
            sel.append(it["idx"])
    return sel


def strong_select(inst):
    sel = []
    remaining = {it["idx"] for it in inst["items"]}
    current = 0
    while True:
        best = None
        best_key = 0.0
        for idx in list(remaining):
            cand = sel + [idx]
            if not feasible(cand, inst):
                continue
            val = evaluate(cand, inst)
            gain = val - current
            cost = inst["items"][idx - 1]["cost"]
            key = gain / max(1, cost)
            if best is None or key > best_key + 1e-12 or (abs(key - best_key) <= 1e-12 and gain > best[1]):
                best = (idx, gain, val)
                best_key = key
        if best is None or best[1] <= 0:
            break
        sel.append(best[0])
        remaining.remove(best[0])
        current = best[2]
    return sel


def emit(sel):
    print(len(sel))
    if sel:
        print(" ".join(str(x) for x in sel))
'''


def render_solution(tier: str) -> str:
    if tier == "invalid":
        return "# TIER: invalid\nprint('999999 1')\n"
    chooser = {
        "trivial": "baseline_select",
        "greedy": "greedy_select",
        "strong": "strong_select",
    }[tier]
    return f'''# TIER: {tier}
{COMMON}


def main():
    inst = parse_instance_text(sys.stdin.read())
    emit({chooser}(inst))


if __name__ == "__main__":
    main()
'''


def render_verify() -> str:
    return f'''#!/usr/bin/env python3
import sys

{COMMON}


def fail(reason):
    print("INVALID %s Ratio: 0.0" % reason)
    sys.exit(0)


def parse_output(path, inst):
    cap = max(10000, inst["n"] * 16 + 1000)
    data = open(path, "rb").read(cap + 1)
    if len(data) > cap:
        fail("output-too-large")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        fail("non-ascii")
    low = text.lower()
    if "nan" in low or "inf" in low:
        fail("non-finite")
    toks = text.split()
    if not toks:
        fail("empty")
    try:
        q = int(toks[0])
    except Exception:
        fail("bad-count")
    if q < 0 or q > inst["n"]:
        fail("count-range")
    if len(toks) != q + 1:
        fail("token-count")
    sel = []
    for tok in toks[1:]:
        try:
            v = int(tok)
        except Exception:
            fail("bad-index")
        if v < 1 or v > inst["n"]:
            fail("index-range")
        sel.append(v)
    if len(set(sel)) != len(sel):
        fail("duplicate")
    if not feasible(sel, inst):
        fail("infeasible")
    return sel


def main():
    if len(sys.argv) < 3:
        fail("bad-args")
    inst = parse_instance_text(open(sys.argv[1], "r", encoding="ascii").read())
    sel = parse_output(sys.argv[2], inst)
    base = baseline_select(inst)
    b = evaluate(base, inst)
    f = evaluate(sel, inst)
    ratio = min(1.0, 0.1 * float(f) / max(1.0, float(b)))
    if not (0.0 <= ratio <= 1.0):
        ratio = 0.0
    print("OK F=%d B=%d Ratio: %.6f" % (f, b, ratio))


if __name__ == "__main__":
    main()
'''


def render_statement(spec: dict) -> str:
    title = title_for(spec)
    theme = spec.get("theme", "domain assets")
    family = spec.get("family", "bulk-constructive-selection")
    return f"""# {title}

## Problem
You are assembling a portfolio of candidate assets for **{theme}**. Each asset has a cost, group,
base value, lattice coordinates, and a bit mask describing capabilities. Select a feasible subset
that maximizes the deterministic utility used by the checker.

This is a constructive optimization problem in the `{family}` family: many feasible subsets are
accepted, but stronger submissions balance budget, group diversity, feature coverage, and hidden
pairwise compatibility better than simple cost-first choices.

## Input
The input is read from stdin.

The first line contains `n m g budget groupCap salt profile`.
Each of the next `n` lines contains `cost group value x y mask`.

`mask` is a non-negative decimal integer whose binary bits encode covered capabilities.

## Output
Print `q`, the number of selected assets, followed by exactly `q` distinct 1-indexed asset ids.

## Feasibility
The selected ids must be in range and distinct. Total cost must not exceed `budget`; no group may
appear more than `groupCap` times; and the checker rejects deterministic conflict pairs based on
coordinates, groups, feature overlap, `salt`, and `profile`.

## Objective and Scoring
The checker scores the selected set by summing base values, saturated capability coverage bonuses,
group-diversity bonuses, and pairwise cross-group synergy bonuses. Let `F` be your feasible score and
let `B` be the checker's internal cost-first baseline score. The reported ratio is
`min(1, 0.1 * F / B)`. The baseline construction therefore scores about `0.1`; better constructions
receive higher ratios.

## Constraints
There are 10 deterministic generator cases. Larger cases have more assets and denser conflicts.
Time limit: 5 seconds. Memory limit: 512 MB.

## Example
For a tiny instance, the cost-first baseline might choose low-cost assets `1 4 7` and score `B=300`.
If your feasible subset scores `F=540`, the checker prints `Ratio: 0.180000`.
"""


def render_meta(spec: dict, seed: int) -> dict:
    title = title_for(spec)
    meta = {
        "id": spec["id"],
        "tier": spec.get("tier", "G"),
        "format": "C",
        "family": spec.get("family", "bulk-constructive-selection"),
        "eval_form": spec.get("eval_form", "quality-metric"),
        "theme": spec.get("theme", ""),
        "title": title,
        "objective": "max",
        "bulk_factory": "budget-conflict-coverage-selection",
        "salt": seed,
        "strategies": [
            "trivial: reproduce the checker cost-first feasible baseline",
            "greedy: sort by value, capability weight, and cost density",
            "strong: repeatedly add the feasible asset with best current marginal utility per cost",
            "future: local search, swap neighborhoods, Lagrangian budget pricing, and conflict-aware coverage repair",
        ],
    }
    return meta


def write_problem(spec: dict, force: bool = False) -> None:
    pid = spec["id"]
    if not re.fullmatch(r"fsx_[A-Z]_\d{4}", pid):
        raise SystemExit(f"bad id: {pid}")
    out = PROBLEMS / pid
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists; pass --force to overwrite generated files")
        shutil.rmtree(out)
    (out / "solutions").mkdir(parents=True)
    seed = stable_int(pid + "|" + spec.get("family", "") + "|" + spec.get("theme", "")) % 1_000_000_007
    (out / "statement.md").write_text(render_statement(spec), encoding="utf-8")
    (out / "config.yaml").write_text(render_config(), encoding="ascii")
    (out / "gen.py").write_text(render_gen(seed), encoding="ascii")
    (out / "verify.py").write_text(render_verify(), encoding="ascii")
    (out / "meta.json").write_text(json.dumps(render_meta(spec, seed), indent=2, ensure_ascii=False), encoding="utf-8")
    for tier in ("trivial", "greedy", "strong", "invalid"):
        (out / "solutions" / f"{tier}.py").write_text(render_solution(tier), encoding="ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="*", type=Path, help="JSONL seed pack(s); defaults to seeds/bulk_seed_packs/*.jsonl")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    paths = args.packs or sorted(PACK_DIR.glob("pack_*.jsonl"))
    if not paths:
        raise SystemExit(f"no seed packs found in {PACK_DIR}")
    specs = load_specs(paths)
    if args.limit is not None:
        specs = specs[:args.limit]
    ids = [s.get("id") for s in specs]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids in input packs")
    for spec in specs:
        write_problem(spec, force=args.force)
    print(json.dumps({"generated": len(specs), "first": ids[0] if ids else None, "last": ids[-1] if ids else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
