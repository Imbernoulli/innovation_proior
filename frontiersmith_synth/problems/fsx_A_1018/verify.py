#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the Quasi-Stationary
Casino Floor problem. Prints exactly one line ending in `Ratio: <float in [0,1]>`
and always exits 0.

Score = the participant's interior (substochastic) transition block's Perron
value (survival rate), normalised against the checker's own internal baseline
construction, via a deterministic sparse power iteration (log-sum growth-rate
estimator -- robust to periodic / near-periodic matrices).
"""
import math, sys

TOL = 1e-6
M_ITERS = 4000       # fixed iteration budget for the power-iteration estimator
BASE_ALPHA = 0.08    # checker's own baseline: mostly favours the shortcut door


def die(reason):
    print("infeasible: %s -- Ratio: 0.0" % reason)
    sys.exit(0)


def read_floats(tokens, k, ctx):
    if len(tokens) < k:
        die("not enough numbers for %s" % ctx)
    vals = []
    for t in tokens[:k]:
        try:
            v = float(t)
        except ValueError:
            die("non-numeric token %r in %s" % (t, ctx))
        if not math.isfinite(v):
            die("non-finite value in %s" % ctx)
        vals.append(v)
    return vals, tokens[k:]


def rho_est(arcs, n, iters=M_ITERS):
    """Deterministic spectral-radius estimate of a nonnegative sparse matrix
    given as arcs[i] = [(j, weight), ...] (row i's outgoing entries). Uses the
    growth-rate (log-sum) form of power iteration, which converges to the
    Perron value even when the dominant eigenvalue is not simple (periodic /
    cyclic support), unlike a plain fixed-point iterate."""
    if n == 0:
        return 0.0
    v = [1.0 / n] * n
    logsum = 0.0
    for _ in range(iters):
        w = [0.0] * n
        for i in range(n):
            s = 0.0
            for j, wt in arcs[i]:
                s += wt * v[j]
            w[i] = s
        nrm = sum(w)
        if nrm <= 0.0:
            return 0.0
        logsum += math.log(nrm)
        v = [x / nrm for x in w]
    return math.exp(logsum / iters)


def main():
    if len(sys.argv) != 4:
        print("usage: verify.py <in> <out> <ans> -- Ratio: 0.0")
        sys.exit(0)
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        in_tokens = f.read().split()
    p = 0

    def take(k):
        nonlocal p
        vals = in_tokens[p:p + k]
        p += k
        return vals

    Nc, K = int(take(1)[0]), int(take(1)[0])
    if Nc < 2 or K < 1:
        die("degenerate instance")
    f_ring, f_short, f_hret, f_hring = (float(x) for x in take(4))
    delta_lo = [float(x) for x in take(Nc)]
    delta_hi = [float(x) for x in take(K)]
    shortcut_t = [int(x) for x in take(Nc)]
    hret_t = [int(x) for x in take(K)]

    n = Nc + K

    # ---- read + validate the participant artifact ----
    try:
        out_tokens = open(outf).read().split()
    except Exception as e:
        die("cannot read output: %s" % e)

    need = 3 * n
    if len(out_tokens) != need:
        die("expected exactly %d numbers, got %d" % (need, len(out_tokens)))

    vals, out_tokens = read_floats(out_tokens, need, "output")
    rows = [vals[3 * i:3 * i + 3] for i in range(n)]

    arcs = [[] for _ in range(n)]
    for i in range(Nc):
        p_loop, p_short, p_cash = rows[i]
        if p_loop < f_ring - TOL:
            die("corridor %d: loop-door %.6f below floor %.6f" % (i, p_loop, f_ring))
        if p_short < f_short - TOL:
            die("corridor %d: shortcut-door %.6f below floor %.6f" % (i, p_short, f_short))
        if p_cash < delta_lo[i] - TOL:
            die("corridor %d: cashier-door %.6f below floor %.6f" % (i, p_cash, delta_lo[i]))
        s = p_loop + p_short + p_cash
        if abs(s - 1.0) > 1e-4:
            die("corridor %d: row sums to %.6f, not 1" % (i, s))
        arcs[i].append(((i + 1) % Nc, p_loop))
        arcs[i].append((Nc + shortcut_t[i], p_short))
    for k in range(K):
        p_ret, p_ring, p_cash = rows[Nc + k]
        if p_ret < f_hret - TOL:
            die("feature %d: return-door %.6f below floor %.6f" % (k, p_ret, f_hret))
        if p_ring < f_hring - TOL:
            die("feature %d: ring-door %.6f below floor %.6f" % (k, p_ring, f_hring))
        if p_cash < delta_hi[k] - TOL:
            die("feature %d: cashier-door %.6f below floor %.6f" % (k, p_cash, delta_hi[k]))
        s = p_ret + p_ring + p_cash
        if abs(s - 1.0) > 1e-4:
            die("feature %d: row sums to %.6f, not 1" % (k, s))
        arcs[Nc + k].append((hret_t[k], p_ret))
        arcs[Nc + k].append((Nc + (k + 1) % K, p_ring))

    F = rho_est(arcs, n)

    # ---- checker's own internal baseline construction (positive, feasible) ----
    base_arcs = [[] for _ in range(n)]
    for i in range(Nc):
        remaining = 1.0 - f_ring - f_short - delta_lo[i]
        p_loop = f_ring + BASE_ALPHA * remaining
        p_short = f_short + (1.0 - BASE_ALPHA) * remaining
        base_arcs[i].append(((i + 1) % Nc, p_loop))
        base_arcs[i].append((Nc + shortcut_t[i], p_short))
    for k in range(K):
        remaining = 1.0 - f_hret - f_hring - delta_hi[k]
        p_ret = f_hret + BASE_ALPHA * remaining
        p_ring = f_hring + (1.0 - BASE_ALPHA) * remaining
        base_arcs[Nc + k].append((hret_t[k], p_ret))
        base_arcs[Nc + k].append((Nc + (k + 1) % K, p_ring))
    B = rho_est(base_arcs, n)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("F=%.8f B=%.8f -- Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
