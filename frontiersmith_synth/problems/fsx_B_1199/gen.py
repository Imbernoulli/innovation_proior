#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN season to stdout.

River flow after the snow is gone.  A hidden watershed carries ONE storage
state -- snowpack -- that accumulates whenever precipitation falls while it is
cold and releases as melt whenever it is warm enough (a degree-heat melt law
capped by whatever snow remains).  Antecedent soil moisture (a fast-decaying
memory of recent rain and meltwater) separately amplifies how efficiently
today's rain turns into flow.  Each testId fixes a DIFFERENT hidden watershed.

The solver only ever SEES this TRAIN season, which is a RAIN-DOMINATED stretch
(mild, mostly-above-freezing weather with only a few brief cold snaps).  In
that regime the snowpack stays near empty almost the whole time, so a
memoryless "flow reacts to today's rain" view already explains most of the
training variance.  The HELD-OUT grading season lives in a completely
different regime -- a real winter-into-spring cycle where a large snowpack
builds up over many quiet cold weeks and then drives flow for a long stretch
even on days with zero rain -- and it is regenerated only inside the checker;
it is never printed here.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows
"<precip> <temp> <flow> <snow_proxy>". `snow_proxy` is a NOISY, coarsely
rounded sensor reading of the (otherwise hidden) snowpack level -- it lets the
solver notice the storage mechanism exists during the few cold snaps in
training, but it is NEVER available at grading time (the held-out rollout
only ever sees precip/temp). The exact hidden law (freeze threshold, melt
rate, moisture decay, seed) is NOT printed.
"""
import sys, random, math

CAP = 8.0


def hidden_params(t):
    """Hidden watershed for this test id (duplicated in gen.py AND verify.py,
    never printed)."""
    rng = random.Random(9130001 + t * 7919)
    Tf = rng.uniform(-0.05, 0.05)        # freeze threshold on the temp axis
    k_melt = rng.uniform(0.35, 0.55)     # melt rate per degree-heat above Tf
    gamma = rng.uniform(0.55, 0.75)      # antecedent-moisture decay per tick
    eta = rng.uniform(0.30, 0.60)        # meltwater -> soil moisture coupling
    alpha_r = rng.uniform(0.55, 0.80)    # rain quickflow coefficient
    kappa = rng.uniform(0.80, 1.60)      # moisture amplification of quickflow
    alpha_m = rng.uniform(0.70, 0.95)    # meltflow coefficient
    b0 = rng.uniform(0.05, 0.09)         # baseflow
    return Tf, k_melt, gamma, eta, alpha_r, kappa, alpha_m, b0


def simulate(precip, temp, params, sigma, seed):
    """Roll the true single-bucket watershed forward. Returns (flow, snow_trace)."""
    Tf, k_melt, gamma, eta, alpha_r, kappa, alpha_m, b0 = params
    rng = random.Random(seed)
    sw = 0.0
    am = 0.0
    flow = []
    snow_trace = []
    for i in range(len(precip)):
        pv, tv = precip[i], temp[i]
        snow_in = pv if tv < Tf else 0.0
        rain_in = pv if tv >= Tf else 0.0
        melt_potential = k_melt * max(0.0, tv - Tf)
        melt = min(sw, melt_potential)
        sw = min(CAP, sw + snow_in - melt)
        am_prev = am
        am = gamma * am_prev + rain_in + eta * melt
        quick = alpha_r * rain_in * (1.0 + kappa * am_prev)
        meltflow = alpha_m * melt
        q = b0 + quick + meltflow + rng.gauss(0.0, sigma)
        flow.append(max(0.0, q))
        snow_trace.append(sw)
    return flow, snow_trace


def train_weather(t, n):
    """Mild, mostly-above-freezing RAIN-season weather; a few brief, shallow
    cold snaps that build only a little transient snow (just enough to hint
    the storage mechanism exists, never enough to look like a real winter)."""
    rng = random.Random(55001 + t * 104729)
    per1 = rng.uniform(70, 110)
    per2 = rng.uniform(150, 240)
    ph1 = rng.uniform(0, 6.283185)
    ph2 = rng.uniform(0, 6.283185)
    ndips = rng.randint(2, 3)
    centers = sorted(rng.uniform(0.05 * n, 0.95 * n) for _ in range(ndips))
    depths = [rng.uniform(0.40, 0.60) for _ in range(ndips)]
    widths = [rng.uniform(4, 9) for _ in range(ndips)]

    temp = []
    for i in range(n):
        v = 0.30 + 0.15 * math.sin(2 * math.pi * i / per1 + ph1) \
                 + 0.05 * math.sin(2 * math.pi * i / per2 + ph2)
        for c, dp, wd in zip(centers, depths, widths):
            v -= dp * math.exp(-0.5 * ((i - c) / wd) ** 2)
        v += rng.gauss(0.0, 0.02)
        temp.append(v)

    precip = []
    for i in range(n):
        if rng.random() < 0.20:
            precip.append(min(1.4, rng.gammavariate(2.0, 0.18)))
        else:
            precip.append(0.0)
    return precip, temp


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = 340 - 6 * (t - 1)
    sigma = 0.018 + 0.003 * (t - 1)

    params = hidden_params(t)
    precip, temp = train_weather(t, n)
    flow, snow_trace = simulate(precip, temp, params, sigma, 2231 + t * 13)

    prng = random.Random(66001 + t * 9973)
    lines = ["%d %d" % (n, t)]
    for i in range(n):
        proxy = round(snow_trace[i] + prng.gauss(0.0, 0.10), 3)
        lines.append("%.6f %.6f %.6f %.6f" % (precip[i], temp[i], flow[i], proxy))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
