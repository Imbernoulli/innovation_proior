# TIER: greedy
"""Each day, search for the mix maximizing TODAY's achievable value:
(feasible total mass at this ratio) * (rate*inhibition quality). This is
a real, nontrivial per-day optimization -- it reads the synergy table
and respects the inhibition thresholds -- but it is completely oblivious
to the adaptation state: it never looks at what it fed yesterday, and it
has no notion that jumping to a different mix costs anything. Self-
contained (no imports beyond stdlib); re-derives a fresh answer every
single day from today's available stock alone."""
import sys
from collections import deque


def read_instance():
    data = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = data[pos[0]]
        pos[0] += 1
        return v

    T = int(nxt()); K = int(nxt())
    alpha_milli = int(nxt())
    switch_cost_i100 = int(nxt())
    cap = int(nxt())
    c = [int(nxt()) for _ in range(K)]
    s = [[0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            v = int(nxt())
            s[i][j] = v; s[j][i] = v
    thr = [int(nxt()) for _ in range(K)]
    pen = [int(nxt()) for _ in range(K)]
    shelf = [int(nxt()) for _ in range(K)]
    M0 = [int(nxt()) for _ in range(K)]
    arr = [[int(nxt()) for _ in range(K)] for _ in range(T)]
    n_spikes = int(nxt())
    spike = [[None] * K for _ in range(T)]
    for _ in range(n_spikes):
        d = int(nxt()); typ = int(nxt()); amt = int(nxt()); sh = int(nxt())
        if 0 <= d < T and 0 <= typ < K:
            spike[d][typ] = (amt, sh)
    return T, K, cap, c, s, thr, pen, shelf, arr, spike


def quality(K, c, s, thr, pen, x):
    rate = 0.0
    for k in range(K):
        rate += c[k] * x[k]
    for i in range(K):
        xi = x[i]
        if xi <= 0.0:
            continue
        for j in range(i + 1, K):
            rate += s[i][j] * xi * x[j]
    if rate < 0.0:
        rate = 0.0
    inh = 1.0
    for k in range(K):
        thr_frac = thr[k] / 1000.0
        if x[k] > thr_frac:
            factor = 1.0 - pen[k] * (x[k] - thr_frac)
            if factor < 0.0:
                factor = 0.0
            inh *= factor
    return rate * inh


def best_mix(K, score_fn, extra_starts=None):
    starts = [[1.0 / K] * K]
    for k in range(K):
        v = [0.15 / (K - 1) if K > 1 else 0.0] * K
        v[k] = 0.85 if K > 1 else 1.0
        ssum = sum(v)
        starts.append([e / ssum for e in v])
    if extra_starts:
        starts.extend(extra_starts)
    best_x, best_v = None, None
    for x0 in starts:
        x = list(x0)
        cur = score_fn(x)
        step = 0.16
        while step > 1e-4:
            improved = True
            while improved:
                improved = False
                for i in range(K):
                    for j in range(K):
                        if i == j:
                            continue
                        if x[i] < step - 1e-12:
                            break
                        nx = list(x)
                        nx[i] -= step
                        nx[j] += step
                        v = score_fn(nx)
                        if v > cur + 1e-12:
                            x, cur = nx, v
                            improved = True
            step *= 0.5
        if best_v is None or cur > best_v:
            best_v, best_x = cur, x
    return best_x


def main():
    T, K, cap, c, s, thr, pen, shelf, arr, spike = read_instance()
    inv = [deque() for _ in range(K)]
    csum = sum(c)
    out_lines = []
    for t in range(T):
        for k in range(K):
            dq = inv[k]
            while dq and (t - dq[0][0]) >= dq[0][2]:
                dq.popleft()
            if arr[t][k] > 0:
                dq.append([t, float(arr[t][k]), shelf[k]])
            if spike[t][k]:
                amt, sh = spike[t][k]
                if amt > 0:
                    dq.append([t, float(amt), sh])
        avail = [sum(b[1] for b in inv[k]) for k in range(K)]

        def dayscore(x, avail=avail):
            support = [k for k in range(K) if x[k] > 1e-9]
            if not support:
                return 0.0
            tm = cap
            for k in support:
                r = avail[k] / x[k]
                if r < tm:
                    tm = r
            if tm <= 0:
                return 0.0
            return tm * quality(K, c, s, thr, pen, x)

        pos_types = [k for k in range(K) if avail[k] > 1e-9]
        extra = []
        if pos_types:
            u = [0.0] * K
            for k in pos_types:
                u[k] = 1.0 / len(pos_types)
            extra.append(u)
        extra.append([c[k] / csum for k in range(K)])
        x = best_mix(K, dayscore, extra_starts=extra)

        support = [k for k in range(K) if x[k] > 1e-9]
        tm = cap
        for k in support:
            r = avail[k] / x[k]
            if r < tm:
                tm = r
        if tm < 0:
            tm = 0.0
        row = [x[k] * tm for k in range(K)]
        out_lines.append(" ".join("%.6f" % v for v in row))
        for k in range(K):
            need = row[k]
            dq = inv[k]
            while need > 1e-6 and dq:
                b = dq[0]
                take = need if need < b[1] else b[1]
                b[1] -= take
                need -= take
                if b[1] <= 1e-6:
                    dq.popleft()
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
