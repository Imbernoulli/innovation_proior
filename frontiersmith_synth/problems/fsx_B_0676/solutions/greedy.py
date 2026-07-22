# TIER: greedy
"""
The obvious "textbook" dispatcher: at every step, at every dam, release as
much water as the turbine allows (min(level, Rmax)). Instantaneous power is
release * gain(level_before_release), which is linear-increasing in
release, so myopically this looks unbeatable. In practice it drains every
reservoir toward empty almost immediately, so gain(level) sits near its
floor a_i for most of the horizon, and it dumps large, un-buffered pulses
onto the next reservoir with no regard for that dam's capacity/turbine
limits (forced spill cascades downstream on flood test cases).
"""
import sys


def main():
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
            rel = min(level[i], Rmax[i])  # maximize instantaneous output
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


if __name__ == "__main__":
    main()
