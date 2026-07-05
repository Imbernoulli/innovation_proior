#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0130 -- "Riverside Freight Yard: Classification Track Plan"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A railway freight yard receives a batch of inbound freight cars that must be
sorted onto a fixed set of CLASSIFICATION TRACKS before outbound trains are built.
Each car belongs to a destination BLOCK (the outbound train / grouping it is bound
for).  A yardmaster must assign every car to exactly one classification track,
trading off three costs that make hump-yard planning hard:

  * SPLIT cost  -- cars of the SAME block that end up on DIFFERENT tracks must be
                   re-coupled later; every same-block pair that is split costs.
  * MIX cost    -- cars of DIFFERENT blocks sharing ONE track have to be shuffled
                   apart when the outbound train is pulled; every different-block
                   pair (b, b') that shares a track costs W[b][b'] -- some block
                   pairs are cheap to co-locate, others expensive (a symmetric
                   affinity matrix), so WHICH blocks share a track matters.
  * OVERFLOW    -- a track has a physical car capacity; every car over capacity
                   incurs a heavy per-car penalty.

There are more blocks than tracks (so some mixing is unavoidable) and a few large
blocks exceed a single track's capacity (so some splitting is unavoidable).  The
yardmaster minimizes total cost; there is no clean optimum -- it is a graph
partition / clustering trade-off with capacity limits.

This is an OFFLINE heuristic-contest instance: a FIXED, deterministically generated
yard, an official deterministic contest score, and no wall-time in the objective.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n_cars": N, "n_tracks": K, "cap": C, "n_blocks": B,
             "block": [b_0, ..., b_{N-1}],        # block id of each car, 0 <= b_i < B
             "mix_w": [[...], ...],               # B x B symmetric, 0 diagonal
             "split_pen": int, "over_pen": int}
  stdout: ONE JSON object:
            {"assign": [t_0, ..., t_{N-1}]}        # track of each car, 0 <= t_i < K

  A plan is VALID iff `assign` is a list of exactly N integers, each in [0, K).
  Invalid output, wrong length, an out-of-range track, a crash, a timeout, or
  non-JSON -> that instance scores 0.0.  (Capacity OVERFLOW is allowed but pays the
  heavy over_pen; it is a cost, not an invalidation.)

SCORING (deterministic; no wall-time).  Per instance we compute the official cost
of the plan and three references:
    cost(plan) = over_pen*overflow
               + sum over same-track different-block pairs (i,j) of W[b_i][b_j]
               + split_pen*split_pairs
    q_lb   = a provable LOWER BOUND on cost (unreachable ideal)
    q_base = cost of the internal ROUND-ROBIN plan (car i -> track i % K), a weak
             but balanced reference layout
    q_cand = cost of the candidate plan
  normalized with an affine anchor (weak baseline -> 0.1, lower bound -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A plan matching round-robin scores ~0.1; a plan reaching the (generally
  unreachable) lower bound scores 1.0; doing worse than round-robin scores < 0.1.

  Because q_lb ignores the mixing forced by having more blocks than tracks, even
  strong local-search planners stay strictly below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(lower bound, round-robin baseline) and the true scoring are computed by THIS
parent process, so a frame-walking / introspecting candidate learns nothing.

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


def _c2(x):
    return x * (x - 1) // 2


# ----------------------------- instance family -----------------------------
def _build_cars(seed, n_blocks, n_big, small_lo, small_hi, big_lo, big_hi):
    """Deterministically build the per-car block list.

    n_big blocks are LARGE (sizes in [big_lo, big_hi], chosen to exceed a track's
    capacity so they must be split); the remaining blocks are small (sizes in
    [small_lo, small_hi]).  Cars are then shuffled so a round-robin-by-index
    layout scatters every block across the tracks (a genuinely weak baseline).
    """
    ni = _rng(seed)
    sizes = []
    for b in range(n_blocks):
        if b < n_big:
            sizes.append(ni(big_lo, big_hi))
        else:
            sizes.append(ni(small_lo, small_hi))
    cars = []
    for b, sz in enumerate(sizes):
        cars.extend([b] * sz)
    # deterministic Fisher-Yates shuffle
    for i in range(len(cars) - 1, 0, -1):
        j = ni(0, i)
        cars[i], cars[j] = cars[j], cars[i]
    return cars


def _build_affinity(seed, n_blocks, w_lo, w_hi):
    """Symmetric B x B mix-affinity matrix, zero diagonal, entries in [w_lo, w_hi].
    A skew makes some block pairs cheap to co-locate and others expensive, turning
    track assignment into a weighted graph-partitioning trade-off."""
    ni = _rng(seed ^ 0x9E3779B9)
    W = [[0] * n_blocks for _ in range(n_blocks)]
    for a in range(n_blocks):
        for b in range(a + 1, n_blocks):
            # skew toward the low end, with occasional expensive pairs
            v = ni(w_lo, w_hi)
            if ni(0, 99) < 25:
                v = ni((w_lo + w_hi) // 2, w_hi)
            W[a][b] = v
            W[b][a] = v
    return W


def _derive_geometry(cars, n_blocks):
    """Given the car list, deterministically choose capacity C and track count K
    so that: (a) K*C >= N (no forced overflow), (b) B > K (forced mixing), and
    (c) the large blocks exceed C (forced splitting)."""
    n = len(cars)
    avg = n / n_blocks
    cap = int(round(1.55 * avg))
    if cap < 4:
        cap = 4
    # aim for ~1.10 * N total capacity across tracks (tight: forces hard packing)
    k = -(-int(round(1.10 * n)) // cap)      # ceil
    # guarantee more blocks than tracks (forced mixing)
    if k >= n_blocks:
        k = n_blocks - 1
    if k < 2:
        k = 2
    return cap, k


def _build_instances():
    """Deterministic instance family. Fields per spec:
       (seed, n_blocks, n_big, small_lo, small_hi, big_lo, big_hi)."""
    specs = [
        (1301, 11, 2, 4, 12, 22, 30),
        (1302, 12, 2, 3, 11, 20, 28),
        (1303, 13, 2, 4, 13, 24, 32),
        (1304, 10, 1, 5, 14, 26, 34),
        (1305, 12, 3, 3, 10, 20, 26),
        (1306, 14, 2, 4, 12, 24, 30),
        (1307, 11, 2, 5, 15, 26, 34),
        (1308, 13, 3, 3, 11, 22, 30),
        # harder / larger held-out instances
        (1311, 16, 3, 4, 13, 26, 36),
        (1312, 15, 2, 3, 12, 28, 38),
        (1313, 17, 4, 4, 11, 24, 34),
        (1314, 18, 3, 5, 14, 30, 42),
    ]
    out = []
    for (seed, nb, nbig, slo, shi, blo, bhi) in specs:
        cars = _build_cars(seed, nb, nbig, slo, shi, blo, bhi)
        cap, k = _derive_geometry(cars, nb)
        W = _build_affinity(seed, nb, 1, 8)
        out.append({
            "name": f"yard{seed}",
            "n_cars": len(cars),
            "n_tracks": k,
            "cap": cap,
            "n_blocks": nb,
            "block": cars,
            "mix_w": W,
            "split_pen": 3,
            "over_pen": 200,
        })
    return out


# ----------------------------- true cost -----------------------------------
def _cost(inst, assign):
    """Validate + score. Return cost (int) or None if the plan is invalid."""
    if not isinstance(assign, list):
        return None
    N = inst["n_cars"]
    K = inst["n_tracks"]
    B = inst["n_blocks"]
    C = inst["cap"]
    block = inst["block"]
    if len(assign) != N:
        return None
    W = inst["mix_w"]
    counts = [0] * K
    cnt = [[0] * B for _ in range(K)]
    for i, t in enumerate(assign):
        if isinstance(t, bool) or not isinstance(t, int):
            return None
        if t < 0 or t >= K:
            return None
        counts[t] += 1
        cnt[t][block[i]] += 1
    overflow = sum(max(0, counts[k] - C) for k in range(K))
    # weighted mix cost: sum over tracks of sum_{b<b'} cnt[b]*cnt[b']*W[b][b']
    mix_cost = 0
    for k in range(K):
        row = cnt[k]
        for b in range(B):
            cb = row[b]
            if cb:
                Wb = W[b]
                for b2 in range(b + 1, B):
                    cb2 = row[b2]
                    if cb2:
                        mix_cost += cb * cb2 * Wb[b2]
    # split cost: same-block pairs on different tracks
    Q = 0
    for k in range(K):
        row = cnt[k]
        for b in range(B):
            v = row[b]
            if v > 1:
                Q += v * (v - 1) // 2
    Nb = [0] * B
    for b in block:
        Nb[b] += 1
    sum_block = sum(_c2(x) for x in Nb)
    split_pairs = sum_block - Q
    return inst["over_pen"] * overflow + mix_cost + inst["split_pen"] * split_pairs


# ----------------------------- references ----------------------------------
def _baseline_assign(inst):
    """Sequential block-packing reference plan: group cars by block id and fill
    tracks to capacity in order.  Whole blocks stay contiguous (little splitting),
    tracks are packed to capacity (no overflow while K*C >= N), and mixing only
    occurs at block/track boundaries -- a decent but affinity-blind construction."""
    N = inst["n_cars"]
    K = inst["n_tracks"]
    C = inst["cap"]
    block = inst["block"]
    order = sorted(range(N), key=lambda i: block[i])
    assign = [0] * N
    t = 0
    used = 0
    for i in order:
        if used >= C and t < K - 1:
            t += 1
            used = 0
        assign[i] = t
        used += 1
    return assign


def _baseline_cost(inst):
    return _cost(inst, _baseline_assign(inst))


def _lower_bound(inst):
    """Provable lower bound on cost.
       overflow >= max(0, N - K*C); split >= sum over blocks of the minimum split
       forced by capacity (pack each block into ceil(Nb/C) full tracks); mix >= 0."""
    N = inst["n_cars"]
    K = inst["n_tracks"]
    C = inst["cap"]
    B = inst["n_blocks"]
    block = inst["block"]
    Nb = [0] * B
    for b in block:
        Nb[b] += 1
    over_lb = max(0, N - K * C)
    split_lb = 0
    for nb in Nb:
        if nb > C:
            full = nb // C
            rem = nb % C
            together_max = full * _c2(C) + _c2(rem)
            split_lb += _c2(nb) - together_max
    return inst["over_pen"] * over_lb + inst["split_pen"] * split_lb


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = _baseline_cost(inst)
        q_lb = _lower_bound(inst)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {
            "name": inst["name"], "n_cars": inst["n_cars"],
            "n_tracks": inst["n_tracks"], "cap": inst["cap"],
            "n_blocks": inst["n_blocks"], "block": list(inst["block"]),
            "mix_w": [list(row) for row in inst["mix_w"]],
            "split_pen": inst["split_pen"], "over_pen": inst["over_pen"],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        assign = ans.get("assign") if isinstance(ans, dict) else None
        try:
            q_cand = _cost(inst, assign)
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
