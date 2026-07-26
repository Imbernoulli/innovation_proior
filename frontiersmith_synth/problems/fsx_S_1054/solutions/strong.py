# TIER: strong
"""The insight: since every household replays the SAME published algorithm,
the tariff designer is really scheduling the whole fleet through the price
vector -- so the danger is not "is the signal cheap enough", it's "is the
signal bold enough to override every household's own per-hour tie-break".
Each household's replay breaks near-ties using a small idiosyncratic value
derived from its OWN load data (at most EPS_MOD/10000 apart). Any price
excursion much larger than that noise floor makes the WHOLE homogeneous
fleet agree on a ranking and pile onto the same hours -- exactly the
mechanism the greedy time-of-use block falls into.

So: stay almost perfectly flat. Keep every hour's price within a small
fraction of that same noise floor, so no single hour's price gap can ever
dominate a household's own tie-break -- charge sessions stay scattered
across all 24 hours by construction. Within that tiny budget, still tilt
fractionally against the hours where the fleet's OWN background demand is
already high (and fractionally toward the naturally quiet hours), so the
harmless scatter also drifts, in aggregate, away from the fixed peak.
"""
import sys

T = 24
AMPL = 0.008


def read_instance():
    toks = sys.stdin.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    P_MIN = float(toks[idx]); idx += 1
    P_MAX = float(toks[idx]); idx += 1
    ALPHA = float(toks[idx]); idx += 1
    TOL_FRAC = float(toks[idx]); idx += 1
    EPS_MOD = int(toks[idx]); idx += 1
    L, need, rate = [], [], []
    for _ in range(N):
        row = [int(toks[idx + k]) for k in range(T)]
        idx += T
        nd = int(toks[idx]); idx += 1
        r = int(toks[idx]); idx += 1
        L.append(row); need.append(nd); rate.append(r)
    return N, P_MIN, P_MAX, ALPHA, TOL_FRAC, EPS_MOD, L, need, rate


def eps(L, i, t, eps_mod):
    return ((L[i][t] * 37 + i * 101 + t * 7) % eps_mod) / 10000.0


def replay(L, need, rate, price, eps_mod):
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


def revenue(price, G):
    return sum(price[t] * G[t] for t in range(T))


def push_to_tol(L, need, rate, base, R0, P_MIN, P_MAX, eps_mod, tol_frac, steps=50):
    P0 = (P_MIN + P_MAX) / 2.0

    def price_for(s):
        return [max(P_MIN, min(P_MAX, P0 + s * (b - P0))) for b in base]

    def rev_diff(s):
        pr = price_for(s)
        G = replay(L, need, rate, pr, eps_mod)
        return revenue(pr, G) - R0

    tol = tol_frac * R0
    best_s = 0.0
    for k in range(steps + 1):
        s = k / steps
        if abs(rev_diff(s)) <= tol:
            best_s = s
        else:
            break
    return price_for(best_s)


def main():
    N, P_MIN, P_MAX, ALPHA, TOL_FRAC, EPS_MOD, L, need, rate = read_instance()
    P0 = (P_MIN + P_MAX) / 2.0
    G0 = replay(L, need, rate, [P0] * T, EPS_MOD)
    R0 = P0 * sum(G0)

    aggL = [sum(L[i][t] for i in range(N)) for t in range(T)]
    lo, hi = min(aggL), max(aggL)
    span = max(1, hi - lo)
    base = [P0 + AMPL * (2 * (aggL[t] - lo) / span - 1) for t in range(T)]

    price = push_to_tol(L, need, rate, base, R0, P_MIN, P_MAX, EPS_MOD, TOL_FRAC)
    print(" ".join("%.6f" % p for p in price))


if __name__ == "__main__":
    main()
