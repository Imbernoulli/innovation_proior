#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0140 -- "Terraced Vineyard: Drip-Emitter Placement"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A large terraced vineyard is modelled as an N x N grid of vine plots.  A
dry spell has left each plot with an integer *moisture deficit* w[r][c] >= 0 (how
badly that plot needs water; the sun-baked slopes and low hollows form clustered
high-deficit "hot zones" over a mild background).  The estate can install at most
K drip-irrigation *emitters*.  An emitter dropped on plot (r, c) wets every plot
within Chebyshev radius R of it -- the (2R+1) x (2R+1) square block centred on
(r, c), clipped to the grid.  A plot that is wet by AT LEAST ONE emitter has its
deficit fully relieved and contributes its deficit w to the harvest recovered; a
plot no emitter reaches contributes nothing (double-wetting a plot helps no more
than wetting it once).

Your job: choose where to drop the (up to) K emitters to RECOVER AS MUCH total
deficit as possible.  This is weighted maximum coverage with square footprints:
NP-hard, submodular, so marginal-gain greedy is strong but not optimal, and
relocation / swap local search does better -- there is real headroom.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": int, "R": int, "K": int,
             "grid": [[w_00, ...], ...]   # N rows x N cols of integers w >= 0}
  stdout: ONE JSON object:
            {"emitters": [[r0, c0], [r1, c1], ...]}
          a list of AT MOST K plots (each 0 <= r,c < N) to drop an emitter on.
          Fewer than K is allowed (but usually wasteful); duplicate positions are
          allowed (but wasteful -- they wet the same block).

  An answer is VALID iff `emitters` is a list of length <= K, each element a
  [row, col] pair of integers with 0 <= row < N and 0 <= col < N.  Wrong shape,
  more than K emitters, a non-integer / out-of-range coordinate, a crash, a
  timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the PARENT computes three
references from the full grid:
    win(r,c)  = sum of w over the (2R+1)^2 block centred at (r,c), clipped;
    UB        = sum of the K LARGEST win(r,c) values   (a loose upper bound on the
                best achievable union, since any K blocks' union <= their summed
                weight <= the K largest block weights);
    weak      = the UNION deficit recovered by dropping emitters on the K plots
                with the largest win(r,c) (deterministic tie-break) -- these pile
                onto the hottest zone and overlap heavily, a deliberately weak ref;
    cand      = the UNION deficit recovered by the candidate's emitters.
  and normalizes with an affine anchor (weak -> 0.1, UB -> 1.0):
    r = clamp( 0.1 + 0.9 * (cand - weak) / max(1e-9, UB - weak), 0, 1 )
  Reproducing the top-win pile-up scores ~0.1; spreading emitters to cover many
  hot zones scores higher.  Because UB double-counts overlaps it is unreachable,
  so even good spreaders stay well below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(win table, UB, weak union) are computed by THIS parent process, so a
frame-walking / introspecting candidate learns nothing useful and cannot game the
normalization.

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
def _build_grid(seed, N, nb, peak_lo, peak_hi, slope, bg_hi):
    """Deterministic N x N integer moisture-deficit grid: mild random background
    plus `nb` square 'hot zones' (linear cones), which cluster demand so that a
    naive top-block pile-up overlaps badly while spreading covers more."""
    ni = _rng(seed)
    grid = [[ni(0, bg_hi) for _ in range(N)] for _ in range(N)]
    for _ in range(nb):
        br = ni(0, N - 1)
        bc = ni(0, N - 1)
        peak = ni(peak_lo, peak_hi)
        for r in range(N):
            dr = r - br
            if dr < 0:
                dr = -dr
            for c in range(N):
                dc = c - bc
                if dc < 0:
                    dc = -dc
                d = dr if dr > dc else dc          # Chebyshev distance
                add = peak - slope * d
                if add > 0:
                    grid[r][c] += add
    return grid


def _build_instances():
    """Deterministic instance family. (seed, N, R, K, nb, peak_lo, peak_hi, slope, bg_hi, dist)."""
    specs = [
        (101, 30, 2, 10, 4, 30, 55, 6, 3, "fewhot"),
        (102, 30, 2, 12, 5, 25, 50, 5, 4, "fewhot"),
        (103, 34, 2, 12, 6, 30, 60, 6, 3, "midhot"),
        (104, 34, 3, 10, 5, 35, 65, 7, 4, "midhot"),
        (105, 36, 2, 14, 7, 25, 55, 5, 4, "midhot"),
        (106, 30, 2, 10, 3, 40, 70, 6, 2, "fewhot"),
        (107, 38, 2, 14, 8, 30, 60, 6, 5, "manyhot"),
        (108, 36, 3, 12, 6, 30, 55, 6, 4, "midhot"),
        # harder / larger held-out instances
        (211, 44, 2, 16, 9, 30, 65, 6, 4, "manyhot"),
        (212, 44, 3, 14, 7, 35, 70, 7, 5, "midhot"),
        (213, 48, 2, 18, 11, 25, 60, 5, 5, "manyhot"),
        (214, 48, 3, 16, 8, 40, 75, 7, 6, "midhot"),
    ]
    out = []
    for seed, N, R, K, nb, pl, ph, sl, bg, dist in specs:
        grid = _build_grid(seed, N, nb, pl, ph, sl, bg)
        out.append({"name": f"vineyard{seed}", "N": N, "R": R, "K": K,
                    "grid": grid, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _prefix(grid, N):
    """2D prefix-sum table, size (N+1) x (N+1)."""
    P = [[0] * (N + 1) for _ in range(N + 1)]
    for r in range(N):
        row = grid[r]
        Pr = P[r]
        Pr1 = P[r + 1]
        acc = 0
        for c in range(N):
            acc += row[c]
            Pr1[c + 1] = Pr[c + 1] + acc
    return P


def _block_sum(P, N, r, c, R):
    r0 = r - R
    if r0 < 0:
        r0 = 0
    c0 = c - R
    if c0 < 0:
        c0 = 0
    r1 = r + R + 1
    if r1 > N:
        r1 = N
    c1 = c + R + 1
    if c1 > N:
        c1 = N
    return P[r1][c1] - P[r0][c1] - P[r1][c0] + P[r0][c0]


def _win_table(P, N, R):
    """win(r,c) = block weight of the (2R+1)^2 window centred at (r,c)."""
    return [[_block_sum(P, N, r, c, R) for c in range(N)] for r in range(N)]


def _topk_positions(win, N, K):
    """The K plots with the largest win value; deterministic tie-break by
    (-win, r*N + c).  Returns a list of (r, c)."""
    order = sorted(((-win[r][c], r * N + c, r, c) for r in range(N) for c in range(N)))
    return [(t[2], t[3]) for t in order[:K]]


def _union_value(grid, N, R, emitters):
    """Total deficit of plots wet by at least one emitter (each counted once)."""
    covered = [[False] * N for _ in range(N)]
    for (r, c) in emitters:
        r0 = r - R if r - R > 0 else 0
        c0 = c - R if c - R > 0 else 0
        r1 = r + R + 1 if r + R + 1 < N else N
        c1 = c + R + 1 if c + R + 1 < N else N
        for rr in range(r0, r1):
            cov = covered[rr]
            for cc in range(c0, c1):
                cov[cc] = True
    tot = 0
    for r in range(N):
        row = grid[r]
        cov = covered[r]
        for c in range(N):
            if cov[c]:
                tot += row[c]
    return tot


# ----------------------------- validation ----------------------------------
def _parse_emitters(inst, answer):
    """Validate answer against the instance. Return a list of (r,c) tuples or None."""
    if not isinstance(answer, dict):
        return None
    em = answer.get("emitters")
    if not isinstance(em, list):
        return None
    N = inst["N"]
    K = inst["K"]
    if len(em) > K:
        return None
    out = []
    for pair in em:
        if not isinstance(pair, list) or len(pair) != 2:
            return None
        r, c = pair
        if isinstance(r, bool) or isinstance(c, bool):
            return None
        if not isinstance(r, int) or not isinstance(c, int):
            return None
        if r < 0 or r >= N or c < 0 or c >= N:
            return None
        out.append((r, c))
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        N, R, K = inst["N"], inst["R"], inst["K"]
        grid = inst["grid"]
        P = _prefix(grid, N)
        win = _win_table(P, N, R)

        # references
        topk = _topk_positions(win, N, K)
        weak = _union_value(grid, N, R, topk)
        flat = sorted((win[r][c] for r in range(N) for c in range(N)), reverse=True)
        ub = sum(flat[:K])
        denom = ub - weak
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "N": N, "R": R, "K": K,
                  "grid": [list(row) for row in grid]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            emitters = _parse_emitters(inst, ans)
        except Exception:
            emitters = None
        if emitters is None:
            vec.append(0.0)
            continue
        cand_val = _union_value(grid, N, R, emitters)

        r = 0.1 + 0.9 * (cand_val - weak) / denom
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
