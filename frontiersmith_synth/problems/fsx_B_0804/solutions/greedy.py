# TIER: greedy
# The "obvious" recipe: apply every coat the instant it becomes eligible
# (maximal fan-out / ASAP), ignoring the shared-humidity coupling entirely.
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
        actual_dt = new_cur - cur   # grid-snapped elapsed time (>= dt)
        cur = new_cur
        for k in list(wet.keys()):
            wet[k] = wet[k] - actual_dt / mult
        done = [k for k, r in wet.items() if r <= 0]
        for k in done:
            del wet[k]
            s = succ[k]
            if s != -1:
                ready.append(s)
    return apply_time


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
    apply_time = run_capped(n, base, succ, heads, c, n)  # no concurrency cap
    out = [fmt(t) for t in apply_time]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
