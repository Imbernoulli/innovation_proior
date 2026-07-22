# TIER: trivial
"""
Reproduces the checker's own internal baseline: release a small, level-
INDEPENDENT trickle every step (a fixed fraction of turbine capacity, not
of the current stock). It neither reacts to a rising level nor anticipates
incoming inflow, so it routinely lets reservoirs fill up and force-spill
most of the incoming water for zero credit.
"""
import sys

if __name__ == "__main__":
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); T = int(next(it))
    C = []; Rmax = []; DELAY = []; INIT = []
    for _ in range(N):
        c = float(next(it)); rmax = float(next(it))
        next(it); next(it)  # a_i, b_i unused by this policy
        d = int(next(it))
        init = float(next(it))
        C.append(c); Rmax.append(rmax); DELAY.append(d); INIT.append(init)
    inflow = []
    for _ in range(N):
        inflow.append([float(next(it)) for _ in range(T)])

    rate_frac = 0.05
    level = list(INIT)
    passthrough = [[0.0] * T for _ in range(N)]
    release = [[0.0] * T for _ in range(N)]

    for t in range(T):
        for i in range(N):
            inflow_it = inflow[i][t]
            if i > 0:
                s = t - DELAY[i]
                if s >= 0:
                    inflow_it += passthrough[i - 1][s]
            rel = min(rate_frac * Rmax[i], level[i])
            pre = level[i] - rel + inflow_it
            if pre > C[i]:
                spill = pre - C[i]
                new_level = C[i]
            else:
                spill = 0.0
                new_level = max(0.0, pre)
            passthrough[i][t] = rel + spill
            release[i][t] = rel
            level[i] = new_level

    out = []
    for i in range(N):
        out.append(" ".join("%.6f" % v for v in release[i]))
    sys.stdout.write("\n".join(out) + "\n")
