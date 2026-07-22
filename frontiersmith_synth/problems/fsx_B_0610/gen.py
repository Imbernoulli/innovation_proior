import sys

# Deterministic instance generator for the catalyst reactor-farm schedule.
# `python3 gen.py <testId>` prints ONE instance to stdout.  Seeded by testId only.
#
# Instance (stdin for the solver):
#   line 1: T R Q d L
#   line 2: L floats  e[0..L-1]      marginal conversion of the w-th fresh feed unit
#   line 3: T floats  p[0..T-1]      unit price at each step (revenue per unit SOLD)
#   line 4: T ints    cap[0..T-1]    market demand cap at each step (max units sold)
#
# Structure planted on purpose:
#  * throughput-dependent decay: e[] is a concave, strictly-decreasing curve; the w-th
#    unit of feed run since the last regeneration converts at rate e[min(w,L-1)].
#  * demand-anticipation: most steps are cheap low-demand valleys (p=1, small cap);
#    a handful are short high-price high-demand PEAK windows.
#  * offline-regeneration: resetting a reactor to fresh costs d fully-offline steps.
#  * TRAP: the peak windows are positioned inside the aged tail of a run-flat-out
#    threshold policy's decay cycle, so the naive "run at max, regenerate when stale"
#    recipe always meets the peaks with spent catalyst.


def gen(testId):
    idx = min(max(testId, 1), 10) - 1
    T_list = [24, 30, 36, 48, 60, 72, 90, 120, 150, 180]
    R_list = [1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    d_list = [3, 3, 4, 4, 4, 5, 5, 6, 6, 6]
    T = T_list[idx]; R = R_list[idx]; d = d_list[idx]
    Q = 10; L = 50
    emax = 1.0; emin = 0.15

    state = (testId * 2654435761 + 1013904223) & 0xffffffff

    def rnd():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7fffffff
        return state / 0x7fffffff

    # concave, strictly-decreasing marginal-conversion curve, floored at emin
    gamma = 0.85 + 0.5 * rnd()
    e = []
    for w in range(L):
        frac = 1.0 - w / (L - 1)
        e.append(emin + (emax - emin) * (frac ** gamma))
    e[L - 1] = emin
    for w in range(1, L):          # enforce non-increasing after fp
        if e[w] > e[w - 1]:
            e[w] = e[w - 1]

    cap_valley = 5 * R
    cap_peak = Q * R
    p = [1.0] * T
    cap = [cap_valley] * T

    # place peaks in the aged tail of a run-flat-out threshold policy's cycle
    steps_to_floor = (L + Q - 1) // Q          # steps at max to reach the floor
    period_g = steps_to_floor + d
    pw = 2                                       # peak window width
    c = 1
    while True:
        pt = c * period_g + (steps_to_floor - 1)  # aged tail of cycle c
        if pt + pw - 1 >= T - 1:
            break
        pp = 5.0 + 5.0 * rnd()                    # peak price in [5,10)
        for k in range(pw):
            tt = pt + k
            if 0 <= tt < T:
                p[tt] = pp
                cap[tt] = cap_peak
        c += 1

    out = []
    out.append("%d %d %d %d %d" % (T, R, Q, d, L))
    out.append(" ".join("%.6f" % x for x in e))
    out.append(" ".join("%.6f" % x for x in p))
    out.append(" ".join("%d" % x for x in cap))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    gen(int(sys.argv[1]))
