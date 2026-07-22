#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0818 -- "Ridge-Blind Prospecting: Self-Tuning a
Black-Box Landscape Scan" (family: ruggedness-adaptive-blackbox; format B,
quality-metric).

THEME.  A prospecting drone must locate the richest deposit in a bounded
survey box.  The ore-richness field is a deterministic, hidden, continuous
function of position: some survey boxes are one smooth "swell" (unimodal),
others are studded with several separate deposits of different grade, laced
with a noisy, high-frequency mineral-vein texture (multimodal / rugged).  The
drone starts its run parked over a locally promising outcrop and is handed a
small SET OF STRUCTURED PILOT READINGS taken before the run: a few short-hop
probes right around the start (to sense local texture / correlation length)
plus a coarse sweep of single readings scattered near every named survey
anomaly across the whole box (to sense which anomalies are worth a look).
With a fixed remaining drilling BUDGET, the drone must decide where to sink
its wells.

This composes three mechanisms into one objective:
  - ruggedness-correlation-probe: the local short-hop probes reveal how fast
    the field changes over small distances -- a proxy for the field's
    correlation length / multimodality.
  - adaptive-stepsize-population: the spread (jitter radius) of the wells
    sunk around any anchor should shrink for a smooth, long-correlation
    field and grow for a rugged, short-correlation one.
  - elite-seeded-restart: on a rugged field the single best-looking anchor
    near the start is not enough -- wells should be seeded from several of
    the best PILOT anchors (including ones the coarse sweep found far from
    the start), not just refined locally.

TRAP.  The obvious recipe is "trust the neighbourhood you started in": pick
the best of the readings taken right around the start, and drill a tight
fixed-radius cluster of wells around it.  On instances built with several
deposits, the start is deliberately parked near a real but SECONDARY deposit
while a taller deposit sits elsewhere in the box, only visible in the coarse
sweep. A recipe that ignores the coarse sweep never drills near it and is
capped at the secondary deposit's grade, however precisely it refines there.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "dim": D, "box": [[lo_0,hi_0], ..., [lo_{D-1},hi_{D-1}]],
     "budget": Q,
     "start": [x_0..x_{D-1}], "start_value": float,
     "local_probes": [{"x":[..D..], "value": float, "r": float}, ...],
     "scan_probes":  [{"x":[..D..], "value": float}, ...]}
  stdout: ONE JSON object:
    {"queries": [[x_0..x_{D-1}], ...]}   # 1..Q points, each inside the box

  VALID iff "queries" is a list of length 1..Q, every element a length-D list
  of finite numbers (not bool) with each coordinate within the box bounds
  (small floating tolerance).  Any violation, a crash, timeout, or non-JSON
  output makes that instance score 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes,
ITSELF, the true hidden field f (a sum of separated Gaussian "deposits" plus
a bounded oscillatory texture term) and three references:
    f_lo   = value achieved by an internal BLIND uniform-random well sweep
             with the SAME budget Q (a weak reference)
    f_hi   = an analytic upper bound on f over the whole box (the tallest
             deposit's peak height plus the texture term's amplitude plus a
             small overlap slack) -- generally UNREACHABLE, so it leaves
             headroom
    f_cand = max(start_value, local/scan pilot values, f at every submitted,
             valid query)
  and normalizes with an affine anchor (the weak sweep -> 0.1, the loose
  upper bound -> 1.0):
    r = clamp( 0.1 + 0.9 * (f_cand - f_lo) / max(1e-9, f_hi - f_lo), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The hidden
field, deposit layout, and both references are computed by THIS parent
process, so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / (1 << 53)          # float in [0,1)

    def uni(lo, hi):
        return lo + nxt() * (hi - lo)

    def unit_vec(d):
        while True:
            v = [uni(-1.0, 1.0) for _ in range(d)]
            n = math.sqrt(sum(c * c for c in v))
            if n > 1e-6:
                return [c / n for c in v]

    return uni, unit_vec, nxt


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------- instance construction ------------------------
H = 10.0            # fixed global-deposit peak height, all instances

def _build_layout(seed, D, K, box_half):
    """Place K Gaussian deposits: index 0 is the tallest (global), index 1
    (if K>1) is the 'decoy' the start point is parked near.  Non-overlapping
    by construction (rejection placement); decoy kept well separated from
    the global deposit so a local-only search cannot stumble onto it."""
    uni, unit_vec, nxt = _rng(seed)
    heights = [H] + [uni(0.55, 0.85) * H for _ in range(K - 1)]
    widths = [uni(0.65, 1.05) for _ in range(K)]
    margin = box_half * 0.8
    centers = []
    for k in range(K):
        placed = None
        for _try in range(400):
            c = [uni(-margin, margin) for _ in range(D)]
            ok = True
            for j, cc in enumerate(centers):
                need = 1.15 * (widths[k] + widths[j])
                if k == 1 and j == 0:
                    need = max(need, 6.0)          # decoy <-> global separation
                if _dist(c, cc) < need:
                    ok = False
                    break
            if ok:
                placed = c
                break
        centers.append(placed if placed is not None else c)
    return heights, widths, centers


def _rug_params(seed, D):
    uni, unit_vec, nxt = _rng(seed)
    f1 = [uni(1.0, 2.2) for _ in range(D)]
    f2 = [uni(1.0, 2.2) for _ in range(D)]
    p1 = uni(0.0, 2 * math.pi)
    p2 = uni(0.0, 2 * math.pi)
    return f1, p1, f2, p2


def _make_f(heights, widths, centers, rug_amp, f1, p1, f2, p2):
    def f(x):
        s = 0.0
        for h, w, c in zip(heights, widths, centers):
            d2 = sum((xi - ci) ** 2 for xi, ci in zip(x, c))
            s += h * math.exp(-d2 / (2.0 * w * w))
        g1 = math.sin(sum(a * b for a, b in zip(f1, x)) + p1)
        g2 = math.cos(sum(a * b for a, b in zip(f2, x)) + p2)
        s += rug_amp * (g1 * g2)
        return s
    return f


def _clip_pt(x, box):
    return [min(max(xi, lo), hi) for xi, (lo, hi) in zip(x, box)]


def _build_instance(seed, D, K, box_half, rug_amp, tag, budget=20):
    heights, widths, centers = _build_layout(seed, D, K, box_half)
    f1, p1, f2, p2 = _rug_params(seed * 3 + 7, D)
    rug_amp_abs = rug_amp * H
    f = _make_f(heights, widths, centers, rug_amp_abs, f1, p1, f2, p2)
    box = [(-box_half, box_half)] * D

    decoy_idx = 1 if K > 1 else 0
    uni, unit_vec, nxt = _rng(seed * 5 + 11)
    dvec = unit_vec(D)
    off = widths[decoy_idx] * 1.15
    start = _clip_pt([c + off * u for c, u in zip(centers[decoy_idx], dvec)], box)
    start_value = f(start)

    # local (short-hop) probes: 3 random directions x 2 radii, from start
    local_probes = []
    uni2, unit_vec2, nxt2 = _rng(seed * 13 + 29)
    for _ in range(3):
        dirv = unit_vec2(D)
        for r in (0.4, 1.2):
            x = _clip_pt([s + r * u for s, u in zip(start, dirv)], box)
            local_probes.append({"x": x, "value": f(x), "r": _dist(x, start)})

    # scan (coarse sweep) probes: one small-jitter reading near EACH deposit
    scan_probes = []
    uni3, unit_vec3, nxt3 = _rng(seed * 17 + 41)
    for k in range(K):
        dirv = unit_vec3(D)
        jit = uni3(0.0, 0.35) * widths[k]
        x = _clip_pt([c + jit * u for c, u in zip(centers[k], dirv)], box)
        scan_probes.append({"x": x, "value": f(x)})

    # weak reference: blind uniform random well sweep with the SAME budget
    # (no credit for the pilot readings -- only for wells actually drilled)
    ubase, _, _ = _rng(seed * 23 + 53)
    best_base = -1e18
    for _ in range(budget):
        x = [ubase(*b) for b in box]
        v = f(x)
        if v > best_base:
            best_base = v

    f_hi = H + rug_amp_abs + 0.05 * H     # analytic upper bound + small overlap slack

    return {
        "name": f"survey_{seed}_{tag}",
        "dim": D, "box": [list(b) for b in box], "budget": budget,
        "start": start, "start_value": start_value,
        "local_probes": local_probes, "scan_probes": scan_probes,
        "_f": f, "_f_lo": best_base, "_f_hi": f_hi,
    }


def _build_instances():
    specs = [
        # (seed, D, K, box_half, rug_amp_frac, tag)
        (301, 2, 1, 13.0, 0.020, "smooth"),
        (302, 2, 1, 12.0, 0.015, "smooth"),
        (303, 2, 1, 14.0, 0.025, "smooth"),
        (304, 2, 3, 13.0, 0.300, "rugged"),
        (305, 2, 4, 14.0, 0.350, "rugged"),
        (306, 2, 3, 15.0, 0.280, "rugged"),
        (307, 2, 5, 16.0, 0.380, "rugged"),
        (308, 2, 4, 13.0, 0.320, "rugged"),
        (309, 3, 4, 10.0, 0.300, "rugged3d"),
        (310, 3, 5, 11.0, 0.350, "rugged3d"),
    ]
    return [_build_instance(seed, D, K, bh, ra, tag) for (seed, D, K, bh, ra, tag) in specs]


# ----------------------------- answer validation -----------------------------
_EPS = 1e-6

def _score_answer(inst, answer):
    """Return best_found (float) or None if the answer is invalid. Credit
    comes ONLY from the candidate's own submitted, valid queries -- the
    pilot readings (start_value / local_probes / scan_probes) are
    information for deciding where to drill, not free score."""
    if not isinstance(answer, dict):
        return None
    queries = answer.get("queries")
    if not isinstance(queries, list) or not (1 <= len(queries) <= inst["budget"]):
        return None
    D = inst["dim"]
    box = inst["box"]
    f = inst["_f"]
    best = -1e18
    for q in queries:
        if not isinstance(q, list) or len(q) != D:
            return None
        pt = []
        for xi, (lo, hi) in zip(q, box):
            if isinstance(xi, bool) or not isinstance(xi, (int, float)):
                return None
            if xi != xi or xi in (float("inf"), float("-inf")):
                return None
            if xi < lo - _EPS or xi > hi + _EPS:
                return None
            pt.append(xi)
        v = f(pt)
        if v > best:
            best = v
    return best


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {"name": inst["name"], "dim": inst["dim"], "box": inst["box"],
                   "budget": inst["budget"], "start": inst["start"],
                   "start_value": inst["start_value"],
                   "local_probes": inst["local_probes"],
                   "scan_probes": inst["scan_probes"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            best = _score_answer(inst, ans)
        except Exception:
            best = None
        if best is None:
            vec.append(0.0)
            continue
        f_lo, f_hi = inst["_f_lo"], inst["_f_hi"]
        denom = max(1e-9, f_hi - f_lo)
        r = 0.1 + 0.9 * (best - f_lo) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
