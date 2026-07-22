#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0894 -- "Binder Hub: Crosstalk-Coupled Line Vectoring"
(family: coupled-water-filling-rates; format B, quality-metric).

THEME.  A vectoring hub feeds a total transmit-power budget P across N subscriber
lines bundled in the same cable binder.  Lines are NOT independent: line j's
transmitted power leaks into line i's receiver as crosstalk, governed by a hidden
NxN coupling matrix A (A[i][j] = crosstalk gain from transmitter j into receiver
i; A[i][i] = 0).  The hub can also perform SUCCESSIVE INTERFERENCE CANCELLATION
(SIC): it commits to a joint DECODING ORDER over the N lines.  When a line is
decoded, the crosstalk contributed by every line decoded EARLIER has already been
learned and is subtracted (cancelled); crosstalk from lines decoded LATER (not yet
known) still corrupts the current line's signal and is treated as noise.  So a
line decoded first suffers interference from everyone; a line decoded last
suffers none.  The candidate chooses BOTH a power split p_i (sum p_i <= P) and a
decoding order -- both interact through the coupling matrix.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "budget": P (float),
             "gain": [g_0..g_{N-1}], "noise": [n_0..n_{N-1}],
             "coupling": NxN list of lists, coupling[i][j] = A[i][j] >= 0, diag 0}
  stdout: ONE JSON object:
            {"power": [p_0, ..., p_{N-1}], "order": [perm of 0..N-1]}
          order[0] is decoded FIRST (sees all other lines' interference),
          order[-1] is decoded LAST (sees none -- everyone else already cancelled).

  VALID iff: power is a list of N finite non-negative numbers with
  sum(power) <= budget + 1e-6*max(1,budget); order is a list of exactly N
  distinct integers that is a permutation of range(N).  Any violation, a crash,
  a timeout, or non-JSON output -> that instance scores 0.0.

RATE MODEL (ground truth; computed by score(), never revealed to the candidate
as a formula it can shortcut -- it must read gain/noise/coupling and act).  For
decode position k, line c = order[k]:
    I_c   = sum_{m>k} coupling[c][order[m]] * power[order[m]]      (uncancelled)
    R_c   = log2(1 + gain[c]*power[c] / (noise[c] + I_c))
    total = sum_c R_c

SCORING (deterministic, no wall-time).  Per instance the evaluator computes,
itself, TWO references using the SAME rate model as above:
    UB   = the coupling-FREE convex upper bound: the unique global optimum of
           sum_i log2(1 + gain_i*p_i/noise_i) s.t. sum p_i <= P, p_i >= 0 (closed
           -form water-filling).  Because coupling entries are >= 0, ANY feasible
           (power, order) can only ever reach a real rate <= UB -- so UB is a
           valid, always-true convex upper bound, never actually reachable when
           coupling is nonzero.
    BASE = the "obvious" recipe scored with the REAL (coupled) rate model: take
           the coupling-free water-filling power split above and decode in the
           trivial index order 0,1,...,N-1 (no attempt at cancellation ordering
           or interference-aware re-optimization).
  and normalizes with an affine anchor (BASE -> 0.1, UB -> 1.0):
    r = clamp( 0.1 + 0.9 * (R_cand - BASE) / max(1e-9, UB - BASE), 0, 1 )
  Reproducing BASE exactly scores ~0.1; approaching the (generally unreachable
  when coupling > 0) UB scores near 1.0.  Doing worse than BASE scores < 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  UB and BASE are
computed by THIS parent process, so a frame-walking / introspecting candidate
learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun


# ----------------------------- shared math ----------------------------------
def _waterfill(g, n, P):
    """Standard water-filling: maximize sum log2(1+g_i*p_i/n_i) s.t. sum p_i<=P,
    p_i>=0, via bisection on the water level mu. g_i, n_i > 0 always."""
    N = len(g)
    if N == 0:
        return []
    lo, hi = 0.0, max(n[i] / g[i] for i in range(N)) + P + 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        s = sum(max(0.0, mid - n[i] / g[i]) for i in range(N))
        if s > P:
            hi = mid
        else:
            lo = mid
    mu = (lo + hi) / 2.0
    p = [max(0.0, mu - n[i] / g[i]) for i in range(N)]
    tot = sum(p)
    if tot > P and tot > 1e-12:
        scale = P / tot
        p = [x * scale for x in p]
    return p


def _rate_given_order(g, n, A, p, order):
    """Ground-truth sum-rate under successive-cancellation decode order `order`."""
    N = len(g)
    total = 0.0
    for k, c in enumerate(order):
        I = 0.0
        row = A[c]
        for m in range(k + 1, N):
            d = order[m]
            I += row[d] * p[d]
        total += math.log2(1.0 + g[c] * p[c] / (n[c] + I))
    return total


# ----------------------------- instance family -------------------------------
def _gen_one(seed, N, kind, csz):
    rng = random.Random(seed)
    g = [round(rng.uniform(0.5, 4.0), 4) for _ in range(N)]
    n = [round(rng.uniform(0.6, 1.6), 4) for _ in range(N)]
    A = [[0.0] * N for _ in range(N)]
    if kind == "mild":
        for i in range(N):
            for j in range(N):
                if i != j:
                    A[i][j] = round(rng.uniform(0.0, 0.04), 4)
    else:
        # trap: a "hot cluster" of low-index lines with boosted direct gain
        # (so coupling-free water-filling over-feeds them) and ASYMMETRIC
        # crosstalk within the cluster: the lower-index member of every pair
        # suffers strongly from the higher-index one, and only weakly the
        # other way -- so the trivial identity decode order (low index first)
        # is the WORST possible order for this cluster (every low-index victim
        # is decoded before its high-index aggressor, so nothing gets
        # cancelled for it), while reversing the cluster's decode priority is
        # close to the best possible order.
        cluster = list(range(csz))
        for c in cluster:
            g[c] = round(g[c] * rng.uniform(1.6, 2.2), 4)
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                i, j = cluster[a], cluster[b]         # i < j
                A[i][j] = round(rng.uniform(0.5, 0.95), 4)   # i suffers a lot from j
                A[j][i] = round(rng.uniform(0.02, 0.06), 4)  # j barely suffers from i
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if i in cluster and j in cluster:
                    continue
                A[i][j] = round(rng.uniform(0.0, 0.03), 4)
    P = round(1.6 * N, 3)
    return {"name": f"cwf{seed}", "n": N, "budget": P, "gain": g, "noise": n,
            "coupling": A, "kind": kind}


def _build_instances():
    specs = [
        # (seed, N, kind, cluster_size)
        (7001, 6, "mild", 0),
        (7002, 7, "mild", 0),
        (7003, 8, "mild", 0),
        (7004, 6, "trap", 3),
        (7005, 7, "trap", 4),
        (7006, 8, "trap", 4),
        (7007, 9, "trap", 5),
        (7008, 7, "trap", 3),
        (7009, 9, "mild", 0),    # harder / held-out mild
        (7010, 10, "trap", 5),   # harder / held-out trap
    ]
    return [_gen_one(*s) for s in specs]


# ----------------------------- validation ------------------------------------
def _validate(ans, N, P):
    if not isinstance(ans, dict):
        return None
    power = ans.get("power")
    order = ans.get("order")
    if not isinstance(power, list) or len(power) != N:
        return None
    p = []
    for x in power:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if xf != xf or xf in (float("inf"), float("-inf")) or xf < -1e-9:
            return None
        p.append(max(0.0, xf))
    tol = 1e-6 * max(1.0, P)
    if sum(p) > P + tol:
        return None
    if not isinstance(order, list) or len(order) != N:
        return None
    seen = set()
    ordi = []
    for x in order:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x < 0 or x >= N or x in seen:
            return None
        seen.add(x)
        ordi.append(x)
    if len(seen) != N:
        return None
    return p, ordi


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        g, n, A, P, N = inst["gain"], inst["noise"], inst["coupling"], inst["budget"], inst["n"]
        p_ub = _waterfill(g, n, P)
        ub = sum(math.log2(1.0 + g[i] * p_ub[i] / n[i]) for i in range(N))
        base = _rate_given_order(g, n, A, p_ub, list(range(N)))
        denom = max(1e-9, ub - base)

        public = {"name": inst["name"], "n": N, "budget": P,
                  "gain": list(g), "noise": list(n),
                  "coupling": [row[:] for row in A]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            res = _validate(ans, N, P)
        except Exception:
            res = None
        if res is None:
            vec.append(0.0)
            continue
        p_c, order_c = res
        try:
            r_cand = _rate_given_order(g, n, A, p_c, order_c)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (r_cand - base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
