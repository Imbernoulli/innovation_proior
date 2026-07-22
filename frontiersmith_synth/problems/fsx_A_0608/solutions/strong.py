# TIER: strong
# Insight: stop dispatching trains one-at-a-time by local priority.  Choose a
# GLOBAL MEETING PATTERN -- a periodic template (cadence period P with per-
# direction phases) and a coherent wave order -- and fit its phase to the release
# times.  Because lateness is governed by the pattern, not by individual ASAP
# decisions, the strong solver enumerates a small family of global templates:
#   * fast waves grouped by direction (east-first / west-first),
#   * cadence slotting (period P, phases phiE/phiW) snapped to releases,
#   * deferral of slow low-value "blocker" trains behind the fast waves,
# runs each through the same feasibility engine, and keeps the cheapest.  The
# earliest-due-date order is included only as a fallback candidate.
import sys


def phys_block(k, di, S):
    return k if di == 0 else (S - 2 - k)


def dwell_station(k, di, S):
    return k if di == 0 else (S - 1 - k)


def earliest_free(intervals, lb, dur):
    x = lb
    changed = True
    while changed:
        changed = False
        for (s, e) in intervals:
            if x < e and s < x + dur:
                x = e
                changed = True
    return x


def cap_ok(intervals, ns, ne, cap):
    if ne <= ns:
        return True
    ev = []
    for (s, e) in intervals:
        a = max(s, ns); b = min(e, ne)
        if a < b:
            ev.append((a, 1)); ev.append((b, -1))
    if not ev:
        return 1 <= cap
    ev.sort()
    cur = 0; mx = 0
    for (_, dl) in ev:
        cur += dl
        if cur > mx:
            mx = cur
    return mx + 1 <= cap


class Sched:
    def __init__(self, S, cap):
        self.S = S; self.cap = cap
        self.block_iv = [[] for _ in range(S - 1)]
        self.stn_iv = [[] for _ in range(S)]
        self.gmax = 0

    def _fill(self, di, r, h, t):
        S = self.S
        e = [0] * (S - 1)
        lb = max(t, r)
        prev_arr = None
        for k in range(S - 1):
            b = phys_block(k, di, S)
            x = earliest_free(self.block_iv[b], lb, h)
            if k >= 1:
                st = dwell_station(k, di, S)
                if not cap_ok(self.stn_iv[st], prev_arr, x, self.cap[st]):
                    return None
            e[k] = x
            prev_arr = x + h
            lb = x + h
        return e

    def place(self, di, r, h, tau):
        S = self.S
        base = max(tau, r)
        e = None
        for step in range(0, 80):
            e = self._fill(di, r, h, base + step)
            if e is not None:
                break
        if e is None:
            e = self._fill(di, r, h, self.gmax + 1)
        for k in range(S - 1):
            b = phys_block(k, di, S)
            self.block_iv[b].append((e[k], e[k] + h))
            self.block_iv[b].sort()
            if k >= 1:
                st = dwell_station(k, di, S)
                ds = e[k - 1] + h; de = e[k]
                if de > ds:
                    self.stn_iv[st].append((ds, de))
        arr = e[S - 2] + h
        if arr > self.gmax:
            self.gmax = arr
        return e


def cost_of(res, trains, S):
    F = 0
    for j, (di, r, dd, w, v, h) in enumerate(trains):
        arr = res[j][S - 2] + h
        F += w * (arr - r) + v * max(0, arr - dd)
    return F


def run(order, targets, trains, S, cap):
    sc = Sched(S, cap)
    res = [None] * len(trains)
    for j in order:
        di, r, dd, w, v, h = trains[j]
        res[j] = sc.place(di, r, h, targets[j])
    return res


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    S = int(next(it)); TMAX = int(next(it))
    cap = [int(next(it)) for _ in range(S)]
    N = int(next(it))
    trains = []
    for _ in range(N):
        di = int(next(it)); r = int(next(it)); dd = int(next(it))
        w = int(next(it)); v = int(next(it)); h = int(next(it))
        trains.append((di, r, dd, w, v, h))

    hmax = max(t[5] for t in trains)
    idxs = list(range(N))

    best_res = None
    best_cost = None

    def consider(order, targets):
        nonlocal best_res, best_cost
        res = run(order, targets, trains, S, cap)
        c = cost_of(res, trains, S)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_res = res

    rel = [trains[j][1] for j in range(N)]

    # candidate 0: EDD fallback
    edd = sorted(idxs, key=lambda j: (trains[j][2], trains[j][1], j))
    consider(edd, list(rel))

    # priority key that defers slow low-value blockers: high v & fast go first
    def wave_key(j):
        di, r, dd, w, v, h = trains[j]
        return (h, -v, r, j)          # fast first, then high tardiness-weight

    # direction-grouped waves: run one direction's fast wave, then the other
    for first in (0, 1):
        order = sorted(idxs, key=lambda j: (0 if trains[j][0] == first else 1,
                                            wave_key(j)))
        consider(order, list(rel))

    # blocker deferral: everyone by release, but slow-low-value trains sent last
    defer = sorted(idxs, key=lambda j: (1 if (trains[j][5] == hmax and trains[j][4] <= 1) else 0,
                                        trains[j][1], j))
    consider(defer, list(rel))

    # cadence slotting: period P, per-direction phases fit to releases
    periods = sorted(set([2 * hmax, 3 * hmax, 4 * hmax, hmax + 1, 2 * hmax + 1]))
    for P in periods:
        if P <= 0:
            continue
        step = 1 if P <= 10 else 2
        for phiE in range(0, P, step):
            for phiW in range(0, P, step):
                targets = [0] * N
                for j in range(N):
                    di, r, dd, w, v, h = trains[j]
                    phi = phiE if di == 0 else phiW
                    targets[j] = r + ((phi - r) % P)
                order = sorted(idxs, key=lambda j: (targets[j], trains[j][5], j))
                consider(order, targets)

    out = []
    for j in range(N):
        out.append(" ".join(str(x) for x in best_res[j]))
    sys.stdout.write("\n".join(out) + "\n")


main()
