#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the island-microgrid
hourly-tariff problem. Prints "... Ratio: <float in [0,1]>" on its own final
line and exits 0.
"""
import math
import sys

T = 24


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    P_MIN = float(toks[idx]); idx += 1
    P_MAX = float(toks[idx]); idx += 1
    ALPHA = float(toks[idx]); idx += 1
    TOL_FRAC = float(toks[idx]); idx += 1
    EPS_MOD = int(toks[idx]); idx += 1
    L = []
    need = []
    rate = []
    for _ in range(N):
        row = [int(toks[idx + k]) for k in range(T)]
        idx += T
        nd = int(toks[idx]); idx += 1
        r = int(toks[idx]); idx += 1
        L.append(row)
        need.append(nd)
        rate.append(r)
    return N, P_MIN, P_MAX, ALPHA, TOL_FRAC, EPS_MOD, L, need, rate


def eps(L, i, t, eps_mod):
    return ((L[i][t] * 37 + i * 101 + t * 7) % eps_mod) / 10000.0


def replay(L, need, rate, price, eps_mod):
    """Replays the SAME published homogeneous greedy-arbitrage algorithm for
    every household: rank the 24 hours by (price[t] + household/hour tie-break),
    then fill need[i] starting from the cheapest ranked hour, up to rate[i] per
    hour, until fully charged."""
    N = len(L)
    G = [0.0] * T
    for i in range(N):
        order = sorted(range(T), key=lambda t: (price[t] + eps(L, i, t, eps_mod), t))
        rem = need[i]
        ci = [0] * T
        for t in order:
            if rem <= 0:
                break
            c = min(rate[i], rem)
            ci[t] = c
            rem -= c
        for t in range(T):
            G[t] += L[i][t] + ci[t]
    return G


def no_response_baseline(L, need, rate):
    """The checker's own trivial feasible reference: what happens with NO
    demand response at all -- every household just charges starting at
    hour 0, in strict calendar order, ignoring price entirely."""
    N = len(L)
    G = [sum(L[i][t] for i in range(N)) for t in range(T)]
    for i in range(N):
        rem = need[i]
        for t in range(T):
            if rem <= 0:
                break
            c = min(rate[i], rem)
            G[t] += c
            rem -= c
    return G


def objective(G, alpha):
    peak = max(G)
    cost = sum(alpha * g * g for g in G)
    return peak + cost


def revenue(price, G):
    return sum(price[t] * G[t] for t in range(T))


def fail(msg):
    print(f"{msg} Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, P_MIN, P_MAX, ALPHA, TOL_FRAC, EPS_MOD, L, need, rate = read_instance(in_path)

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except Exception:
        fail("cannot read output.")
        return

    if len(out_toks) != T:
        fail(f"expected exactly {T} prices, got {len(out_toks)}.")
        return

    price = []
    for tok in out_toks:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token in output.")
            return
        if not math.isfinite(v):
            fail("non-finite price in output.")
            return
        price.append(v)

    tol_bound = 1e-6
    for v in price:
        if v < P_MIN - tol_bound or v > P_MAX + tol_bound:
            fail(f"price {v} outside [{P_MIN}, {P_MAX}].")
            return
    price = [min(P_MAX, max(P_MIN, v)) for v in price]

    P0 = (P_MIN + P_MAX) / 2.0
    G0 = replay(L, need, rate, [P0] * T, EPS_MOD)
    R0 = P0 * sum(G0)

    G = replay(L, need, rate, price, EPS_MOD)
    R = revenue(price, G)

    tol = TOL_FRAC * max(1e-9, R0)
    if abs(R - R0) > tol:
        fail(f"revenue-neutrality violated: R={R:.3f} target={R0:.3f} tol={tol:.3f}.")
        return

    F = objective(G, ALPHA)

    Gnr = no_response_baseline(L, need, rate)
    B = objective(Gnr, ALPHA)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("OK. Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
