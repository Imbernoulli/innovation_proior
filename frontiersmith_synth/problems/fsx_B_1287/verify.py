#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for dynamic-hedge-rebalance.

Reads the instance from <in>, the participant's rebalancing sequence h_1..h_N from
<out>, validates feasibility strictly, computes the total hedging cost F (gamma-drift
+ transaction cost, minimize), builds the checker's own "never rebalance" baseline B,
and prints the normalized ratio.
"""
import sys, math

POS_BOUND = 3.0
EPS_TRADE = 1e-9


def fail(msg):
    print("INFEASIBLE: %s Ratio: 0.0" % msg)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    S = [float(toks[idx + i]) for i in range(N + 1)]; idx += N + 1
    D = [float(toks[idx + i]) for i in range(N + 1)]; idx += N + 1
    G = [float(toks[idx + i]) for i in range(N + 1)]; idx += N + 1
    cost_prop = float(toks[idx]); idx += 1
    cost_fixed = float(toks[idx]); idx += 1
    return N, S, D, G, cost_prop, cost_fixed


def total_cost(N, S, D, G, cost_prop, cost_fixed, h):
    """h has length N+1 (h[0]..h[N]); h[0] is the fixed initial hedge."""
    total = 0.0
    prev = h[0]
    for t in range(1, N + 1):
        drift = G[t] * (prev - D[t]) ** 2
        dtr = abs(h[t] - prev)
        trade = (cost_fixed if dtr > EPS_TRADE else 0.0) + cost_prop * dtr * S[t]
        total += drift + trade
        prev = h[t]
    return total


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    N, S, D, G, cost_prop, cost_fixed = read_instance(in_path)

    with open(out_path) as f:
        raw = f.read().split()

    if len(raw) != N:
        fail("expected exactly %d numbers (h_1..h_N), got %d" % (N, len(raw)))

    h = [D[0]]  # h[0] fixed by the instance (start delta-neutral)
    for tok in raw:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite value %r" % tok)
        if v < -POS_BOUND or v > POS_BOUND:
            fail("position %.6g outside feasible bound [-%.1f, %.1f]" % (v, POS_BOUND, POS_BOUND))
        h.append(v)

    F = total_cost(N, S, D, G, cost_prop, cost_fixed, h)
    if not math.isfinite(F):
        fail("non-finite objective")

    # internal baseline: the trivial feasible construction -- never rebalance at all,
    # hold the initial delta-neutral hedge for the whole path.
    h_static = [D[0]] * (N + 1)
    B = total_cost(N, S, D, G, cost_prop, cost_fixed, h_static)
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
