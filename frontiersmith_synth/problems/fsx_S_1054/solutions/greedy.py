# TIER: greedy
"""The textbook time-of-use tariff: a fixed cheap 'off-peak' window and a
fixed expensive 'peak' window, revenue-balanced. This is the standard
recipe a competent engineer reaches for -- and it looks reasonable, because
within the off-peak block ties are still broken by each household's own
epsilon, spreading the block's demand over several hours rather than one.
But the block is bold enough (far wider than the households' own tie-break
noise) that essentially the WHOLE fleet's charge session lands inside that
one narrow window regardless of which specific hour each household prefers
within it -- manufacturing a new, synchronized rebound peak there. It never
looks at the fleet's actual scale before committing to the same fixed
window every time.
"""
import sys

T = 24
VALLEY_START, VALLEY_WIDTH = 0, 8
PEAK_START, PEAK_WIDTH = 17, 6


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

    base = [P0] * T
    for k in range(VALLEY_WIDTH):
        base[(VALLEY_START + k) % T] = P_MIN
    for k in range(PEAK_WIDTH):
        base[(PEAK_START + k) % T] = P_MAX

    price = push_to_tol(L, need, rate, base, R0, P_MIN, P_MAX, EPS_MOD, TOL_FRAC)
    print(" ".join("%.6f" % p for p in price))


if __name__ == "__main__":
    main()
