#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

Theme: a fisherman tracks a school of fish along a long sorted channel of
positions (a "key index" from 0 to N-1, like a huge sorted array of buoy
markers). Each day t he takes one noisy position fix. The school's TRUE
position follows a hidden drift law: a steady current component (linear in
t), a seasonal eddy (a sinusoid), and a slowly strengthening rip current
(a cubic term in t) that is a minor wobble across the training days but
keeps accelerating -- by the graded (held-out, FUTURE) window it dominates.

STDOUT prints ONLY: a header "T_train t N W H" then T_train rows "i obs_i".
The hidden law parameters, its exact shape, and the graded window are never
printed -- they live only inside verify.py (regenerated from the test id).
"""
import sys, random, math

N_KEYSPACE = 1_000_000
SEED_LAW = 90176
SEED_NOISE = 555001


def law_params(t):
    """Hidden drift-law parameters for this test id. Copied verbatim into
    verify.py so both sides derive identical numbers from the test id alone."""
    rng = random.Random(SEED_LAW + t * 7919)
    B0 = rng.uniform(0.30 * N_KEYSPACE, 0.45 * N_KEYSPACE)
    R = rng.uniform(70.0, 150.0)
    A_s = rng.uniform(2500.0, 7000.0)
    P = rng.uniform(35.0, 65.0)
    phase = rng.uniform(0.0, 2 * math.pi)
    gamma = rng.uniform(0.06, 0.14)
    T_train = 380 - 18 * (t - 1)
    H = 2 * T_train
    C3 = gamma * R / (T_train ** 2)
    sigma_obs = 250.0 + 40.0 * (t - 1)
    sigma_target = 0.35 * sigma_obs
    W = (3 * N_KEYSPACE) // 100
    return dict(N=N_KEYSPACE, B0=B0, R=R, A_s=A_s, P=P, phase=phase, C3=C3,
                T_train=T_train, H=H, sigma_obs=sigma_obs,
                sigma_target=sigma_target, W=W)


def law_value(t, p):
    """Noiseless TRUE school position at day t (float, unclamped)."""
    return (p['B0'] + p['R'] * t
            + p['A_s'] * math.sin(2 * math.pi * t / p['P'] + p['phase'])
            + p['C3'] * (t ** 3))


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    p = law_params(t)
    rng_noise = random.Random(SEED_NOISE + t * 104729)

    lines = ["%d %d %d %d %d" % (p['T_train'], t, p['N'], p['W'], p['H'])]
    for i in range(1, p['T_train'] + 1):
        val = law_value(i, p) + rng_noise.gauss(0.0, p['sigma_obs'])
        obs = int(round(clamp(val, 0, p['N'] - 1)))
        lines.append("%d %d" % (i, obs))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
