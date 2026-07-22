#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>

Scores a River Staircase dispatch schedule.

Instance (<in>):
    line 1:            N T
    next N lines:       C_i  Rmax_i  a_i  b_i  delay_i  init_i
                         (reservoir i, i=1..N, upstream(0)->downstream(N-1);
                          delay_1 is always 0 / unused)
    next N lines:       T space separated floats = localInflow_i[1..T]

Participant artifact (<out>):
    N lines, each with exactly T space separated non-negative finite floats:
    release_i[1..T]  (the water released through reservoir i's turbine at
    each step; must respect 0 <= release_i[t] <= min(level_i[t], Rmax_i)).

Physics (identical for participant replay and internal baseline):
    inflow_i[t]   = localInflow_i[t]
                    + passthrough_{i-1}[t - delay_i]   (0 if i==1 or t-delay_i<1)
    pre_i[t]      = level_i[t] - release_i[t] + inflow_i[t]
    if pre_i[t] > C_i:  spill_i[t] = pre_i[t]-C_i ; level_i[t+1] = C_i
    else:               spill_i[t] = 0            ; level_i[t+1] = pre_i[t]
    passthrough_i[t] = release_i[t] + spill_i[t]     (flows on downstream)
    power_i[t]    = release_i[t] * gain_i(level_i[t]),   gain_i(L)=a_i+b_i*sqrt(L/C_i)
    objective F   = sum over i,t of power_i[t]           (spilled water: 0 power)

The internal baseline B replays the SAME physics with a naive, non-adaptive
"bleed a fixed fraction of current level every step" release policy.
"""
import sys, math

EPS = 1e-3


def fail(msg):
    print("INVALID: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it)); T = int(next(it))
    C = []; Rmax = []; A = []; Bc = []; DELAY = []; INIT = []
    for _ in range(N):
        C.append(float(next(it))); Rmax.append(float(next(it)))
        A.append(float(next(it))); Bc.append(float(next(it)))
        DELAY.append(int(next(it))); INIT.append(float(next(it)))
    inflow = []
    for _ in range(N):
        row = [float(next(it)) for _ in range(T)]
        inflow.append(row)
    return N, T, C, Rmax, A, Bc, DELAY, INIT, inflow


def gain(level, C_i, a_i, b_i):
    x = level / C_i
    if x < 0.0:
        x = 0.0
    return a_i + b_i * math.sqrt(x)


def simulate(N, T, C, Rmax, A, Bc, DELAY, INIT, inflow, release):
    """release[i][t] (0-indexed i,t) already validated in-range per step
    (release <= level and <= Rmax) by the caller as it walks forward, since
    level[i][t] is only known once the simulation reaches step t."""
    level = [INIT[i] for i in range(N)]
    passthrough = [[0.0] * T for _ in range(N)]
    total = 0.0
    for t in range(T):
        for i in range(N):
            inflow_it = inflow[i][t]
            if i > 0:
                d = DELAY[i]
                s = t - d
                if s >= 0:
                    inflow_it += passthrough[i - 1][s]
            rel = release[i][t]
            if rel < -EPS:
                return None
            if rel > level[i] + EPS or rel > Rmax[i] + EPS:
                return None
            rel = min(max(rel, 0.0), min(level[i], Rmax[i]))
            pre = level[i] - rel + inflow_it
            if pre > C[i] + EPS:
                spill = pre - C[i]
                new_level = C[i]
            else:
                spill = 0.0
                new_level = max(0.0, pre)
            passthrough[i][t] = rel + spill
            total += rel * gain(level[i], C[i], A[i], Bc[i])
            level[i] = new_level
    return total


def baseline_release(N, T, C, Rmax, A, Bc, DELAY, INIT, inflow, rate_frac=0.05):
    """Naive feasible construction: release a small, level-INDEPENDENT
    trickle every step (a fixed fraction of turbine capacity, not of the
    current stock). It neither reacts to a rising level nor anticipates
    incoming inflow, so on anything but a trickle-sized inflow it lets the
    reservoir fill up and force-spill most of the water for zero credit."""
    level = [INIT[i] for i in range(N)]
    passthrough = [[0.0] * T for _ in range(N)]
    total = 0.0
    for t in range(T):
        for i in range(N):
            inflow_it = inflow[i][t]
            if i > 0:
                d = DELAY[i]
                s = t - d
                if s >= 0:
                    inflow_it += passthrough[i - 1][s]
            rel = min(rate_frac * Rmax[i], level[i])
            pre = level[i] - rel + inflow_it
            if pre > C[i] + EPS:
                spill = pre - C[i]
                new_level = C[i]
            else:
                spill = 0.0
                new_level = max(0.0, pre)
            passthrough[i][t] = rel + spill
            total += rel * gain(level[i], C[i], A[i], Bc[i])
            level[i] = new_level
    return total


def parse_output(path, N, T):
    with open(path) as f:
        lines = [ln for ln in f.read().split("\n")]
    # strip trailing blank lines
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != N:
        fail("expected %d lines, got %d" % (N, len(lines)))
    release = []
    for i, ln in enumerate(lines):
        toks = ln.split()
        if len(toks) != T:
            fail("row %d: expected %d values, got %d" % (i, T, len(toks)))
        row = []
        for tok in toks:
            try:
                v = float(tok)
            except ValueError:
                fail("row %d: non-numeric token %r" % (i, tok))
            if not math.isfinite(v):
                fail("row %d: non-finite value" % i)
            if v < -EPS:
                fail("row %d: negative release" % i)
            row.append(max(0.0, v))
        release.append(row)
    return release


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>")
        print("Ratio: 0.0")
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, T, C, Rmax, A, Bc, DELAY, INIT, inflow = read_instance(in_path)

    release = parse_output(out_path, N, T)

    F = simulate(N, T, C, Rmax, A, Bc, DELAY, INIT, inflow, release)
    if F is None:
        fail("release exceeds available level or turbine capacity at some step")

    B = baseline_release(N, T, C, Rmax, A, Bc, DELAY, INIT, inflow)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f baseline=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
