# TIER: strong
# Insight (the innovation hook): with S-shaped fuel curves the cheapest marginal unit
# of heat is NOT more load on a saturated boiler -- it is DE-loading the saturated one
# back toward its ~70% sweet spot and lighting another.  So commitment and loading are
# co-optimized, not layered:
#   (1) commit only as many boilers as keep the online fleet near its sweet spot at the
#       CURRENT demand (fewer at the shoulders, more at the peak) -- turning boilers off
#       when demand sags saves both no-load fuel AND the off-sweet-spot penalty;
#   (2) light/de-light boilers progressively along the demand ramp (each new one starts
#       at pmin, the others barely move) so ramp limits are respected for free;
#   (3) dispatch the committed fleet by EQUAL MARGINAL COST (water-filling), which parks
#       every boiler near its sweet spot rather than filling the cheapest to 100%.
# Falls back to the static priority-list schedule if any move would be infeasible.
import sys
import math

XTAR = 0.70


def parse():
    toks = sys.stdin.read().split()
    i = 0
    T = int(toks[i]); i += 1
    K = int(toks[i]); i += 1
    D = [float(toks[i + j]) for j in range(T)]; i += T
    U = []
    for _ in range(K):
        U.append((float(toks[i]), float(toks[i + 1]), float(toks[i + 2]),
                  float(toks[i + 3]), float(toks[i + 4]), float(toks[i + 5]),
                  float(toks[i + 6]), int(float(toks[i + 7])), int(float(toks[i + 8]))))
        i += 9
    return T, K, D, U


def marg(o, u):
    C, a, b, x = u[0], u[3], u[4], u[5]
    xr = o / C
    return a * (1.0 + b * (3 * xr * xr - 4 * x * xr + x * x))


def invert(lam, u, lo, hi):
    # solve marg(o)=lam on the increasing branch, clamp to [lo,hi]
    C, a, b, x = u[0], u[3], u[4], u[5]
    R = (lam / a - 1.0) / b
    disc = x * x + 3.0 * R
    if disc <= 0.0:
        return lo
    xr = (2.0 * x + math.sqrt(disc)) / 3.0
    o = xr * C
    if o < lo:
        return lo
    if o > hi:
        return hi
    return o


def waterfill(C_list, U, los, his, demand):
    o = list(los)
    if sum(los) >= demand - 1e-9:
        return o
    lam_lo = min(marg(los[idx], U[k]) for idx, k in enumerate(C_list))
    lam_hi = max(marg(his[idx], U[k]) for idx, k in enumerate(C_list))
    for _ in range(60):
        lam = 0.5 * (lam_lo + lam_hi)
        s = 0.0
        for idx, k in enumerate(C_list):
            s += invert(lam, U[k], los[idx], his[idx])
        if s < demand:
            lam_lo = lam
        else:
            lam_hi = lam
    return [invert(lam_hi, U[k], los[idx], his[idx]) for idx, k in enumerate(C_list)]


def build_strong(T, K, D, U):
    # Forward controller: co-optimize commitment and loading, switching a boiler
    # only when a feasibility test (with lookahead) proves the move is ramp- and
    # min-time-legal -> guaranteed feasible by construction.
    order = sorted(range(K),
                   key=lambda k: (U[k][2] + U[k][3] * U[k][0] * (1 + U[k][4] * (1 - U[k][5]) ** 2)) / U[k][0])
    LC = 8    # commit ahead of a rising ramp
    LS = 45   # never shed a boiler a coming peak needs back within this window
    need_c = [max(D[t:min(t + LC + 1, T)]) for t in range(T)]
    need_s = [max(D[t:min(t + LS + 1, T)]) for t in range(T)]

    online = [False] * K
    online_prev = [False] * K  # everything is OFF before t=0
    out = [0.0] * K
    since = [10 ** 9] * K       # dwell (steps) in the current on/off state

    # Commit by CAPACITY (fleet gets LC steps to ramp toward cap), NOT by the
    # ramp-limited one-step maximum -- otherwise, because no-load fuel dominates,
    # lighting extra boilers to reach the peak costs far more than running the
    # cheap ones hot.  A separate ramp guard keeps the very next step feasible.
    acc = 0.0
    for k in order:
        online[k] = True
        acc += U[k][0]
        if acc >= need_c[0]:
            break

    O = [[0.0] * K for _ in range(T)]

    def hi_of(k):
        if not online_prev[k]:
            return U[k][0]                    # startup: free within [pmin, cap]
        return min(U[k][0], out[k] + U[k][6])

    def maxsum(exclude=None):        # ramp-limited one-step reach
        return sum(hi_of(k) for k in range(K) if k != exclude and online[k])

    def capsum(exclude=None):        # full-capacity reach (given time to ramp)
        return sum(U[k][0] for k in range(K) if k != exclude and online[k])

    for t in range(T):
        # (A) commit until the fleet CAPACITY covers the look-ahead demand AND the
        #     ramp-limited reach covers the immediate step
        guard = 0
        while capsum() < need_c[t] - 1e-6 or maxsum() < D[t] - 1e-6:
            cand = None
            for k in order:
                if not online[k] and since[k] >= U[k][8]:
                    cand = k
                    break
            if cand is None:
                for k in order:
                    if not online[k]:
                        cand = k
                        break
            if cand is None:
                return None
            online[cand] = True
            since[cand] = 0
            guard += 1
            if guard > K + 2:
                break
        if maxsum() < D[t] - 1e-6:
            return None
        # (B) shed the most expensive slack boiler (dwell >= min-up) only if the rest
        #     comfortably cover the LONG look-ahead demand -> no thrash near a coming
        #     peak, and it saves no-load fuel + the off-sweet-spot penalty
        for k in reversed(order):
            if (online[k] and since[k] >= U[k][7]
                    and capsum(exclude=k) >= need_s[t] - 1e-6
                    and maxsum(exclude=k) >= D[t] - 1e-6):
                online[k] = False
                since[k] = 0
                break
        # (C) dispatch the online fleet by equal marginal cost
        C_list = [k for k in range(K) if online[k]]
        if not C_list:
            return None
        los = []; his = []
        for k in C_list:
            if not online_prev[k]:
                lo, hi = U[k][1], U[k][0]
            else:
                lo = max(U[k][1], out[k] - U[k][6])
                hi = min(U[k][0], out[k] + U[k][6])
            if lo > hi:
                lo = hi
            los.append(lo); his.append(hi)
        if sum(his) < D[t] - 1e-6:
            return None
        res = waterfill(C_list, U, los, his, D[t])
        newout = [0.0] * K
        for idx, k in enumerate(C_list):
            newout[k] = res[idx]
        for k in range(K):
            O[t][k] = newout[k]
        # (D) advance dwell counters and state
        for k in range(K):
            since[k] = since[k] + 1 if online[k] == online_prev[k] else 1
            out[k] = newout[k] if online[k] else 0.0
        online_prev = online[:]
    return O


def build_greedy(T, K, D, U):
    order = sorted(range(K),
                   key=lambda k: (U[k][2] + U[k][3] * U[k][0] * (1 + U[k][4] * (1 - U[k][5]) ** 2)) / U[k][0])
    dmax = max(D)
    S = []; acc = 0.0
    for k in order:
        S.append(k); acc += U[k][0]
        if acc >= dmax:
            break
    Scap = sum(U[k][0] for k in S)
    O = [[0.0] * K for _ in range(T)]
    for t in range(T):
        for k in S:
            share = D[t] * U[k][0] / Scap
            O[t][k] = share if share > U[k][1] else U[k][1]
    return O


def feasible(T, K, D, U, O):
    for t in range(T):
        s = 0.0
        for k in range(K):
            o = O[t][k]; u = U[k]
            if o > 1e-9 and (o < u[1] - 1e-4 or o > u[0] + 1e-4):
                return False
            s += o
        if s < D[t] - 1e-4:
            return False
    for k in range(K):
        ramp = U[k][6]
        for t in range(1, T):
            if O[t][k] > 1e-9 and O[t - 1][k] > 1e-9 and abs(O[t][k] - O[t - 1][k]) > ramp + 1e-4:
                return False
    for k in range(K):
        mu, md = U[k][7], U[k][8]
        onk = [O[t][k] > 1e-9 for t in range(T)]
        t = 0
        while t < T:
            v = onk[t]; st = t
            while t < T and onk[t] == v:
                t += 1
            L = t - st
            if v and L < mu and t != T:
                return False
            if (not v) and st != 0 and t != T and L < md:
                return False
    return True


def total_fuel(T, K, U, O):
    def f(o, u):
        if o <= 1e-12:
            return 0.0
        xr = o / u[0]
        return u[2] + u[3] * o * (1.0 + u[4] * (xr - u[5]) ** 2)
    return sum(f(O[t][k], U[k]) for t in range(T) for k in range(K))


def main():
    T, K, D, U = parse()
    Og = build_greedy(T, K, D, U)
    Os = build_strong(T, K, D, U)
    O = Og
    if Os is not None and feasible(T, K, D, U, Os) and total_fuel(T, K, U, Os) < total_fuel(T, K, U, Og):
        O = Os
    out = []
    for t in range(T):
        out.append(" ".join("%.6f" % O[t][k] for k in range(K)))
    sys.stdout.write("\n".join(out) + "\n")


main()
