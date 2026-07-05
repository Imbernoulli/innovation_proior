#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0120 -- "Greenhouse Zone Partition"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  An automated greenhouse is a rectangular grid of H x W growing cells.
Each cell holds a plant whose *ideal air temperature* is a fixed integer
`ideal[i][j]` (the crop mix has already been decided).  The climate system
cannot heat every cell independently: it partitions the grid into ZONES.  A zone
is any set of cells sharing one label; all cells in a zone are driven to a single
common temperature by that zone's controller.  Two costs trade off against each
other:

  * MISMATCH cost -- each cell pays (ideal - zone_temperature)^2.  For a fixed
    partition the controller always picks the temperature that MINIMIZES this,
    i.e. the mean of the ideals in the zone, so the evaluator computes the
    mismatch of a zone directly as its sum of squared deviations from its mean.
  * WALL cost -- every pair of orthogonally adjacent cells that lie in DIFFERENT
    zones needs a thermal divider between them, costing `wall_penalty` each.

Total cost = MISMATCH + wall_penalty * (# adjacent cross-zone pairs).  The task
is a static AtCoder-heuristic-contest instance: given the fixed field, output a
labelling that MINIMIZES total cost.  One giant zone pays zero walls but huge
mismatch; one-cell-per-zone pays zero mismatch but a wall between every neighbour
(and `wall_penalty` is set so full fragmentation is WORSE than a single zone).
The good partitions live in between, and finding them is open-ended: threshold /
quantile bucketing, band segmentation, and agglomerative region merging are all
viable and none is optimal.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "H": int, "W": int, "wall_penalty": int,
             "ideal": [[int,...],... ]     # H rows x W cols of integer temps}
  stdout: ONE JSON object:
            {"labels": [l_0, ..., l_{H*W-1}]}
          a flat ROW-MAJOR list of exactly H*W non-negative integers; cell (i,j)
          has label labels[i*W + j].  Label values are arbitrary non-negative
          ints; a "zone" is the set of cells sharing a label.

  VALID iff `labels` is a list of exactly H*W non-negative integers (bools and
  non-ints rejected).  Invalid output, wrong length, a crash, a timeout, or
  non-JSON -> that instance scores 0.0.

COMPUTE BUDGET.  Think of the controller as allowed on the order of 1e7
elementary operations per instance; the grids are small enough that quantile,
band, and agglomerative-merge heuristics all fit comfortably.  The harness
enforces this only as a safety timeout -- the SCORE never depends on wall-clock
time, so results are fully reproducible.

SCORING (deterministic; no wall-time).  Per instance:
    q_base = total cost of the ONE-ZONE labelling (all cells one zone: mismatch =
             sum of squared deviations from the global mean, zero walls) -- the
             weak reference.
    q_lb   = 0  (an unreachable ideal: zero mismatch AND zero walls at once).
    q_cand = total cost of the candidate labelling.
  Normalized with an affine anchor (one-zone -> 0.1, the q_lb ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / (q_base - q_lb), 0, 1 )
  A candidate reproducing the one-zone reference scores ~0.1; doing worse scores
  < 0.1 (clamped at 0).  Because q_lb = 0 is unreachable (any mismatch reduction
  costs walls), even strong region-merging heuristics stay well below 1.0 ->
  headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(global mean, q_base) are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing useful.

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

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_field(seed, H, W, M, noise_amp):
    """Deterministic smooth temperature field: `M` Gaussian hot/cold spots on an
    ambient base, plus bounded integer noise.  Returns H rows x W cols of ints."""
    ni = _rng(seed)
    ambient = 30
    spots = []
    for _ in range(M):
        cy = ni(0, H - 1)
        cx = ni(0, W - 1)
        amp = ni(15, 55) * (1 if ni(0, 9) < 7 else -1)   # mostly hot, some cold
        rho = ni(12, 32) / 10.0                           # 1.2 .. 3.2
        spots.append((cy, cx, amp, rho))
    field = []
    for i in range(H):
        row = []
        for j in range(W):
            t = float(ambient)
            for (cy, cx, amp, rho) in spots:
                d2 = (i - cy) * (i - cy) + (j - cx) * (j - cx)
                t += amp * math.exp(-d2 / (2.0 * rho * rho))
            t += ni(-noise_amp, noise_amp)
            ti = int(round(t))
            if ti < 0:
                ti = 0
            if ti > 120:
                ti = 120
            row.append(ti)
        field.append(row)
    return field


def _build_instances():
    """Deterministic instance family. (seed, H, W, M, noise_amp, wall_penalty)."""
    specs = [
        (101, 6, 6, 2, 2, 120),
        (102, 6, 7, 3, 3, 140),
        (103, 7, 7, 3, 2, 150),
        (104, 7, 8, 3, 3, 130),
        (105, 8, 7, 4, 2, 160),
        (106, 8, 8, 3, 3, 150),
        (107, 6, 8, 2, 4, 110),
        (108, 8, 8, 4, 2, 170),
        # harder / larger held-out instances (more spots, more noise, bigger)
        (211, 9, 9, 5, 4, 160),
        (212, 9, 10, 5, 3, 180),
        (213, 10, 9, 6, 4, 150),
        (214, 10, 10, 6, 5, 170),
    ]
    out = []
    for seed, H, W, M, noise_amp, wp in specs:
        field = _build_field(seed, H, W, M, noise_amp)
        out.append({"name": f"gh{seed}", "H": H, "W": W,
                    "wall_penalty": wp, "ideal": field})
    return out


# ----------------------------- references ----------------------------------
def _one_zone_cost(inst):
    """Cost of putting every cell in a single zone: SSE around the global mean,
    zero walls."""
    flat = [t for row in inst["ideal"] for t in row]
    n = len(flat)
    s = sum(flat)
    ss = sum(t * t for t in flat)
    return ss - (s * s) / n            # sum of squared deviations from the mean


# ----------------------------- validation + scoring ------------------------
def _cost(inst, answer):
    """Validate answer against the instance. Return total cost or None."""
    if not isinstance(answer, dict):
        return None
    labels = answer.get("labels")
    if not isinstance(labels, list):
        return None
    H = inst["H"]
    W = inst["W"]
    N = H * W
    if len(labels) != N:
        return None
    for v in labels:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0:
            return None
    flat = [t for row in inst["ideal"] for t in row]
    # mismatch: per-label sum of squared deviations from label mean
    cnt = {}
    ssum = {}
    ssq = {}
    for idx, lab in enumerate(labels):
        t = flat[idx]
        cnt[lab] = cnt.get(lab, 0) + 1
        ssum[lab] = ssum.get(lab, 0) + t
        ssq[lab] = ssq.get(lab, 0) + t * t
    mismatch = 0.0
    for lab in cnt:
        c = cnt[lab]
        mismatch += ssq[lab] - (ssum[lab] * ssum[lab]) / c
    # walls: adjacent cross-zone pairs (right + down neighbours)
    walls = 0
    for i in range(H):
        base = i * W
        for j in range(W):
            lab = labels[base + j]
            if j + 1 < W and labels[base + j + 1] != lab:
                walls += 1
            if i + 1 < H and labels[base + W + j] != lab:
                walls += 1
    return mismatch + inst["wall_penalty"] * walls


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = _one_zone_cost(inst)
        denom = q_base                       # q_lb = 0
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "H": inst["H"], "W": inst["W"],
                  "wall_penalty": inst["wall_penalty"],
                  "ideal": [list(r) for r in inst["ideal"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _cost(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_base - q_cand) / denom
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
