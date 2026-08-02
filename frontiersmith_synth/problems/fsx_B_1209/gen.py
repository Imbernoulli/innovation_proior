#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

Permafrost thaw forecast.  A ground-monitoring station logs a normalised
surface energy forcing f[t] (positive = net heat into the ground, negative =
net heat out) and a ground thermal index G[t].  Underneath, the ground tracks
a hidden, UNPRINTED energy accumulator: while the floored running sum of f
stays below a hidden latent-heat capacity, G is pinned at a frozen baseline
regardless of how f wobbles (a Stefan-problem latent-heat plateau).  Only once
the accumulator crosses that capacity does an active (thawed) layer start
growing -- roughly as the square root of the excess energy, amplified by an
insulation-loss feedback and capped at a bounded maximum depth.

Each testId fixes a DIFFERENT hidden station (capacity, growth rate, feedback
strength, baseline, noise).  The TRAIN window handed to the solver is always
chosen to end WHILE the station is still on the frozen plateau -- so G looks
essentially flat/insensitive to f for the entire visible record.  The GRADED
window is a longer continuation of the same forcing process, regenerated only
inside the checker; it may cross into the thaw regime.  Never printed: the
hidden capacity, growth/feedback constants, or the accumulator itself -- only
(f, G) training pairs.
"""
import sys, random, math

N_HELD = 480          # length of the (unprinted) graded continuation
DMAX = 3.0             # public: active-layer saturation depth


def params(t):
    """Hidden per-station physics. Lives in gen AND checker, never printed."""
    rng = random.Random(9010013 + t * 7919)
    trend = rng.uniform(0.00060, 0.00110)
    cyc_amp1 = rng.uniform(0.42, 0.58)
    cyc_amp2 = rng.uniform(0.10, 0.18)
    per1 = rng.uniform(40, 70)
    per2 = rng.uniform(11, 19)
    ph1 = rng.uniform(0, 6.283185)
    ph2 = rng.uniform(0, 6.283185)
    proc_noise = rng.uniform(0.02, 0.035)
    kappa = rng.uniform(0.45, 1.00)
    gamma = rng.uniform(0.15, 0.45)
    eta = rng.uniform(0.45, 1.00)
    obs_sigma = 0.02 + 0.004 * (t - 1)
    g_plateau = rng.uniform(0.25, 0.55)
    return dict(trend=trend, cyc_amp1=cyc_amp1, cyc_amp2=cyc_amp2, per1=per1,
                per2=per2, ph1=ph1, ph2=ph2, proc_noise=proc_noise, kappa=kappa,
                gamma=gamma, eta=eta, obs_sigma=obs_sigma, g_plateau=g_plateau)


# target fraction of the graded window at which the hidden capacity is set to
# be crossed (>=3 EARLY entries are the trap: greedy sees a long flat visible
# record and nothing else). Indexed by testId 1..10.
FRACS = [0.75, 0.85, 0.90, 0.65, 0.15, 0.20, 0.30, 0.60, 0.70, 0.25]


def n_train_for(t):
    return 300 - 6 * (t - 1)


def forcing_series(t, n, p):
    rng = random.Random(55010 + t * 104729)
    f = []
    for i in range(1, n + 1):
        v = (p['cyc_amp1'] * math.sin(2 * math.pi * i / p['per1'] + p['ph1'])
             + p['cyc_amp2'] * math.sin(2 * math.pi * i / p['per2'] + p['ph2'])
             + p['trend'] * i
             + rng.gauss(0.0, p['proc_noise']))
        f.append(v)
    return f


def hidden_capacity(t, f, n_train):
    """Pick LH so the (feedback-free) floored accumulator first reaches it at
    FRACS[t-1] of the way through the graded window -- i.e. AFTER n_train."""
    total = len(f)
    E, Es_raw = 0.0, []
    for x in f:
        E = max(0.0, E + x)
        Es_raw.append(E)
    target = max(n_train + int(FRACS[t - 1] * N_HELD), n_train + 5)
    target = min(target, total)
    return Es_raw[target - 1]


def simulate(t, f, n_train, p, LH):
    """Full hidden process: latent-heat plateau -> Stefan sqrt growth once the
    accumulator crosses LH -> insulation feedback amplifies further forcing ->
    active-layer depth saturates at DMAX. Returns the full G[] trace."""
    total = len(f)
    rng = random.Random(770013 + t * 31)
    E = 0.0
    D = 0.0
    G = []
    for idx in range(total):
        amp = 1.0 + p['gamma'] * (D / DMAX)
        E = max(0.0, E + f[idx] * amp)
        if E >= LH:
            draw = p['kappa'] * math.sqrt(max(0.0, E - LH))
            D = DMAX * math.tanh(draw / DMAX)
        else:
            D = 0.0
        g = p['g_plateau'] + p['eta'] * D + rng.gauss(0.0, p['obs_sigma'])
        G.append(g)
    return G


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_train = n_train_for(t)
    p = params(t)
    total = n_train + N_HELD
    f = forcing_series(t, total, p)
    LH = hidden_capacity(t, f, n_train)
    G = simulate(t, f, n_train, p, LH)

    out = ["%d %d" % (n_train, t)]
    for i in range(n_train):
        out.append("%.6f %.6f" % (f[i], G[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
