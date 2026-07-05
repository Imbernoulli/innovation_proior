#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0150 -- "Tide Pool Reserve: Intertidal Habitat Delineation"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A marine reserve manager surveys a rocky intertidal shore modeled as a
W x H coordinate grid.  On the shore live N focal organisms (a limpet colony, an
anemone cluster, a kelp holdfast, ...).  Organism i sits at an integer survey
point (x_i, y_i) and needs a protected TIDE POOL of a target habitat area a_i to
sustain its population through the tidal cycle.  The manager must delineate N
axis-aligned rectangular pool boundaries -- one per organism -- that (a) stay
inside the shore [0,W] x [0,H], (b) never overlap another pool (interiors must be
disjoint; touching edges are fine), and (c) SHOULD each enclose their organism's
survey point and match its target area as closely as possible.

This is the classic AtCoder-heuristic-contest "area-target rectangle placement"
task (AHC001 lineage) reframed as a STATIC, offline instance scored by a fixed
deterministic formula -- no wall-time, no GPU.  The manager runs a delineation
heuristic; the model supplies the rectangles.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "W": int, "H": int, "n": N (int),
             "x": [x_0..x_{N-1}],   # organism survey x, 0 <= x_i <  W
             "y": [y_0..y_{N-1}],   # organism survey y, 0 <= y_i <  H
             "a": [a_0..a_{N-1}]}   # target habitat areas, a_i >= 1
  stdout: ONE JSON object:
            {"rects": [[x1_0,y1_0,x2_0,y2_0], ..., [x1_{N-1},...,y2_{N-1}]]}
          integer coordinates with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

  A delineation is VALID iff `rects` is a list of exactly N rectangles, each four
  integers obeying the bound/order constraints above, AND no two rectangles have
  overlapping interiors.  Any structural violation, an overlap, a crash, a timeout,
  or non-JSON => that instance scores 0.0.  (A rectangle that fails to enclose its
  own survey point is still structurally VALID but earns 0 quality for that pool --
  exactly as in the reference contest scorer.)

SCORING (deterministic; no wall-time -- an operation budget of ~1e7 is the intended
design ceiling and is enforced only via the subprocess timeout, never the score).
  Per pool i with rectangle area s_i:
      if the rectangle encloses (x_i, y_i):  p_i = 1 - min(a_i,s_i)/max(a_i,s_i)
                                             q_i = 1 - p_i * p_i           (in [0,1])
      else:                                  q_i = 0
  Instance raw quality  Q = mean_i q_i   (in [0,1], the AHC001 normalized score).
  We anchor with the manager's trivial 1x1 delineation (one unit cell per organism):
      Q_base = raw quality of the all-1x1 layout   (weak reference, ~0)
  and normalize (trivial -> 0.1, the unreachable perfect layout -> 1.0):
      r = clamp( 0.1 + 0.9 * (Q_cand - Q_base) / max(1e-9, 1.0 - Q_base), 0, 1 )
  Because the targets are packed to ~80-90% of the shore area, no layout can hit
  every target without overlaps, so even strong local-search packers stay well
  below 1.0 -> genuine headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The reference
(1x1 baseline, the scorer) is computed by THIS parent process, so a frame-walking
candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_instance(seed, n, W, H, fill):
    """Deterministic tide-pool instance. Distinct integer survey points, target
    areas summing to ~`fill` fraction of the shore area."""
    ni = _rng(seed)
    xs, ys = [], []
    used = set()
    guard = 0
    while len(xs) < n and guard < 100000:
        guard += 1
        px = ni(0, W - 1)
        py = ni(0, H - 1)
        if (px, py) in used:
            continue
        used.add((px, py))
        xs.append(px)
        ys.append(py)
    # random positive weights -> target areas that sum to ~fill * W * H
    weights = [ni(10, 100) for _ in range(n)]
    sw = sum(weights)
    total = int(fill * W * H)
    areas = []
    for w in weights:
        a = (w * total) // sw
        if a < 1:
            a = 1
        areas.append(a)
    return {"name": f"shore{seed}", "W": W, "H": H, "n": n,
            "x": xs, "y": ys, "a": areas}


def _build_instances():
    """Deterministic instance distribution. (seed, n, W, H, fill)."""
    specs = [
        (101, 25, 1000, 1000, 0.92),
        (102, 30, 1000, 1000, 0.94),
        (103, 35, 1000, 1000, 0.95),
        (104, 30, 1000, 1000, 0.96),
        (105, 40, 1000, 1000, 0.95),
        (106, 35, 1000, 1000, 0.97),
        (107, 45, 1000, 1000, 0.96),
        (108, 40, 1000, 1000, 0.97),
        # harder / larger held-out instances (denser packing, more pools)
        (211, 50, 1000, 1000, 0.97),
        (212, 55, 1000, 1000, 0.98),
        (213, 60, 1000, 1000, 0.98),
        (214, 50, 1000, 1000, 0.99),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- validation + scoring ------------------------
def _overlap(a, b):
    """True iff rectangles a,b (x1,y1,x2,y2) have overlapping interiors."""
    return (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def _validate(inst, answer):
    """Return the list of validated integer rectangles, or None if infeasible."""
    if not isinstance(answer, dict):
        return None
    rects = answer.get("rects")
    if not isinstance(rects, list):
        return None
    N = inst["n"]
    W, H = inst["W"], inst["H"]
    if len(rects) != N:
        return None
    out = []
    for rc in rects:
        if not isinstance(rc, (list, tuple)) or len(rc) != 4:
            return None
        for v in rc:
            if isinstance(v, bool) or not isinstance(v, int):
                return None
        x1, y1, x2, y2 = rc
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            return None
        out.append((x1, y1, x2, y2))
    # pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            if _overlap(out[i], out[j]):
                return None
    return out


def _quality(inst, rects):
    """Mean per-pool AHC001 quality in [0,1] for a validated layout."""
    N = inst["n"]
    xs, ys, areas = inst["x"], inst["y"], inst["a"]
    tot = 0.0
    for i in range(N):
        x1, y1, x2, y2 = rects[i]
        # enclosure: point must lie inside [x1,x2) x [y1,y2)
        if not (x1 <= xs[i] < x2 and y1 <= ys[i] < y2):
            continue
        s = (x2 - x1) * (y2 - y1)
        a = areas[i]
        p = 1.0 - min(a, s) / max(a, s)
        tot += 1.0 - p * p
    return tot / N


def _baseline_quality(inst):
    """Manager's trivial all-1x1 delineation: one unit cell per organism."""
    rects = [(inst["x"][i], inst["y"][i], inst["x"][i] + 1, inst["y"][i] + 1)
             for i in range(inst["n"])]
    return _quality(inst, rects)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = _baseline_quality(inst)
        denom = 1.0 - q_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "W": inst["W"], "H": inst["H"],
                  "n": inst["n"], "x": list(inst["x"]),
                  "y": list(inst["y"]), "a": list(inst["a"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            rects = _validate(inst, ans)
        except Exception:
            rects = None
        if rects is None:
            vec.append(0.0)
            continue
        q_cand = _quality(inst, rects)
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
