# TIER: trivial
"""Ascending 'cheap at night, pricier by day' ramp -- the very first idea most
people reach for ("night power should be cheapest"). Because it is a single
monotone price curve, the published per-household algorithm ranks the 24
hours in EXACTLY the same order (hour 0 cheapest, ... hour 23 priciest) for
every household in the fleet, regardless of their own load shape. That
recreates full calendar-order synchronization -- the checker's own
"do-nothing" reference construction -- so this reproduces the baseline.
"""
import sys

T = 24


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

    base = [P_MIN + t * (P_MAX - P_MIN) / (T - 1) for t in range(T)]
    price = push_to_tol(L, need, rate, base, R0, P_MIN, P_MAX, EPS_MOD, TOL_FRAC)
    print(" ".join("%.6f" % p for p in price))


if __name__ == "__main__":
    main()
