#!/usr/bin/env python3
# Deterministic checker for ESG-screen-portfolio (format C, minimize factor tracking error).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys
import math

TOL = 1e-6
SUMTOL = 1e-4


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def is_finite(x):
    return x == x and x != float("inf") and x != float("-inf")


def parse_instance(path):
    toks = open(path).read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    N = int(nxt())
    S = int(nxt())
    K = S + 1
    T = float(nxt())
    F = [[float(nxt()) for _ in range(K)] for _ in range(K)]
    sectors, sizes, esg, w, cap, d = [], [], [], [], [], []
    for _ in range(N):
        sectors.append(int(nxt()))
        sizes.append(float(nxt()))
        esg.append(float(nxt()))
        w.append(float(nxt()))
        cap.append(float(nxt()))
        d.append(float(nxt()))
    return N, S, K, T, F, sectors, sizes, esg, w, cap, d


def factor_gap_energy(devs, sectors, sizes, F, K):
    """devs = x - w per name. Returns e^T F e for the induced factor-exposure gap e."""
    e = [0.0] * K
    for i, dv in enumerate(devs):
        if dv == 0.0:
            continue
        sec = sectors[i]
        e[sec] += dv
        e[K - 1] += dv * sizes[i]
    Fe = [sum(F[k][j] * e[j] for j in range(K)) for k in range(K)]
    return sum(e[k] * Fe[k] for k in range(K))


def equal_waterfill(target, caps):
    """Cap-respecting equal-share allocation: split `target` as evenly as possible over
    all indices, never exceeding caps[i], spillover from capped entries redistributed
    evenly among the rest. List-based (no dict/set ordering involved) -> deterministic."""
    n = len(caps)
    alloc = [0.0] * n
    active = [True] * n
    remaining = target
    guard = 0
    while remaining > 1e-12 and guard < n + 5:
        idxs = [i for i in range(n) if active[i]]
        if not idxs:
            break
        share = remaining / len(idxs)
        newly_sat = []
        for i in idxs:
            headroom = caps[i] - alloc[i]
            if share >= headroom - 1e-12:
                alloc[i] += headroom
                remaining -= headroom
                newly_sat.append(i)
        if not newly_sat:
            for i in idxs:
                alloc[i] += share
            remaining = 0.0
            break
        for i in newly_sat:
            active[i] = False
        guard += 1
    return alloc


def tracking_error(x, w, sectors, sizes, F, K, d):
    devs = [x[i] - w[i] for i in range(len(x))]
    quad = factor_gap_energy(devs, sectors, sizes, F, K)
    spec = sum(d[i] * devs[i] * devs[i] for i in range(len(x)))
    val = quad + spec
    if val < 0.0:
        val = 0.0  # guard tiny negative from floating error; F is PSD by construction
    return math.sqrt(val + 1e-12)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")

    try:
        N, S, K, T, F, sectors, sizes, esg, w, cap, d = parse_instance(sys.argv[1])
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) != N:
        fail("expected exactly %d numbers, got %d" % (N, len(otoks)))

    x = []
    for i in range(N):
        try:
            v = float(otoks[i])
        except Exception:
            fail("non-numeric weight at position %d" % i)
        if not is_finite(v):
            fail("non-finite weight at position %d" % i)
        x.append(v)

    eligible = [esg[i] >= T for i in range(N)]

    for i in range(N):
        if x[i] < -TOL:
            fail("negative weight at name %d" % i)
        if not eligible[i]:
            if abs(x[i]) > TOL:
                fail("excluded name %d (esg=%.3f < T=%.3f) has nonzero weight %.8f" %
                     (i, esg[i], T, x[i]))
        else:
            if x[i] > cap[i] + TOL:
                fail("name %d weight %.8f exceeds substitution capacity cap=%.8f" %
                     (i, x[i], cap[i]))

    ssum = sum(x)
    if abs(ssum - 1.0) > SUMTOL:
        fail("weights sum to %.8f, must sum to 1.0 (tol %.1e)" % (ssum, SUMTOL))

    # clamp tiny negative/overshoot noise from floating input before scoring
    xc = [max(0.0, v) for v in x]

    Fval = tracking_error(xc, w, sectors, sizes, F, K, d)

    # internal trivial baseline: as-equal-as-possible weight across eligible names,
    # water-filled against each name's own substitution cap so it stays feasible
    elig_idx = [i for i in range(N) if eligible[i]]
    if not elig_idx:
        fail("no eligible names in instance")  # generator guards against this
    elig_caps = [cap[i] for i in elig_idx]
    elig_alloc = equal_waterfill(1.0, elig_caps)
    base_x = [0.0] * N
    for j, i in enumerate(elig_idx):
        base_x[i] = elig_alloc[j]
    Bval = tracking_error(base_x, w, sectors, sizes, F, K, d)

    sc = min(1000.0, 100.0 * Bval / max(1e-9, Fval))
    print("TE=%.8f B=%.8f Ratio: %.6f" % (Fval, Bval, sc / 1000.0))


if __name__ == "__main__":
    main()
