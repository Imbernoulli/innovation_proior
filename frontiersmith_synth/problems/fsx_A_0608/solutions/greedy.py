# TIER: greedy
# The obvious approach: earliest-due-date dispatch.  Sort trains by due date and
# send each one as soon as feasible (ASAP) given what is already on the line.
# It faithfully follows priorities but is myopic: a slow, tight-due train is sent
# first and crawls across the line, forcing long one-sided waits behind and against
# it -- exactly the cascade the planted instances punish.
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

    order = sorted(range(N), key=lambda j: (trains[j][2], trains[j][1], j))
    sc = Sched(S, cap)
    res = [None] * N
    for j in order:
        di, r, dd, w, v, h = trains[j]
        res[j] = sc.place(di, r, h, r)

    out = []
    for j in range(N):
        out.append(" ".join(str(x) for x in res[j]))
    sys.stdout.write("\n".join(out) + "\n")


main()
