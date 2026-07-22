# TIER: strong
# Insight: the machines interfere through a shared environmental state (humidity),
# so the drying multiplier (1+c*W)^2 makes per-coat throughput W/(1+cW)^2, which is
# maximized at an INTERIOR steady-state concurrency K* = 1/c -- not at W = P (maximal
# fan-out).  Reformulate the schedule as a concurrency-capped admission problem, seed
# the cap analytically from the congestion law, then sweep nearby caps and keep
# whichever cap actually simulates to the smallest makespan.
import sys
from fractions import Fraction as Fr


SCALE = 10 ** 9


def rnd(fr):
    # snap UP to the printable 1e-9 grid at every time advance, so our own
    # bookkeeping matches EXACTLY what the checker re-derives from the
    # (grid-snapped) apply times we output -- prevents ~1e-9 false precedence
    # violations from independent end-of-run rounding.
    scaled = -(-(fr.numerator * SCALE) // fr.denominator)
    return Fr(scaled, SCALE)


def fmt(fr):
    iv = int(fr * SCALE)
    ip, fp = divmod(iv, SCALE)
    return "%d.%09d" % (ip, fp)


def run_capped(n, base, succ, heads, c, cap):
    ready = list(heads)
    apply_time = [None] * n
    wet = {}
    cur = Fr(0)
    admitted = 0
    makespan = Fr(0)
    guard = 0
    guard_max = 20 * n + 100
    while admitted < n or wet:
        guard += 1
        if guard > guard_max:
            break
        while ready and len(wet) < cap:
            ready.sort()
            idx = ready.pop(0)
            apply_time[idx] = cur
            wet[idx] = Fr(base[idx])
            admitted += 1
        if not wet:
            break
        w = len(wet)
        mult = (1 + c * w) * (1 + c * w)
        dt = min(r * mult for r in wet.values())
        new_cur = rnd(cur + dt)
        actual_dt = new_cur - cur
        cur = new_cur
        for k in list(wet.keys()):
            wet[k] = wet[k] - actual_dt / mult
        done = [k for k, r in wet.items() if r <= 0]
        for k in done:
            if cur > makespan:
                makespan = cur
            del wet[k]
            s = succ[k]
            if s != -1:
                ready.append(s)
    return apply_time, makespan


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    cn = int(next(it))
    cd = int(next(it))
    c = Fr(cn, cd)
    base, succ, heads = [], [], []
    for _p in range(P):
        k = int(next(it))
        start = len(base)
        for _j in range(k):
            base.append(int(next(it)))
            succ.append(-1)
        for j in range(k - 1):
            succ[start + j] = start + j + 1
        heads.append(start)
    n = len(base)
    if n == 0:
        return

    # analytic seed from the congestion law: throughput K/(1+cK)^2 peaks at K*=1/c
    k0 = max(1, round(1 / c)) if c > 0 else n
    cands = set()
    for dk in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 9):
        v = k0 + dk
        if 1 <= v <= n:
            cands.add(v)
    for v in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, n):
        if 1 <= v <= n:
            cands.add(v)

    best_apply, best_ms = None, None
    for K in sorted(cands):
        at, ms = run_capped(n, base, succ, heads, c, K)
        if best_ms is None or ms < best_ms:
            best_apply, best_ms = at, ms

    out = [fmt(t) for t in best_apply]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
