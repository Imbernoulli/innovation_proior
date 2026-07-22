#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN log to stdout.

Latent tool-wear recursion identification.  A rival factory's cutting tool
has a hidden internal wear level W that updates ONCE PER JOB from its value
after the previous job:

    W_i = f(W_{i-1}, gap_i, load_i, material_i)

`f` is a hidden recursive law (per test id) that partially recovers the tool
during idle gaps and accumulates wear from load/material in a functional
shape that is NEVER revealed (and never printed) -- it lives only in this
generator's / the checker's private construction below.  Every job sequence
(training or grading) starts fresh at W_0 = 0 -- sequences are independent
tool histories, not one continuous log.

Once W_i is known, the OBSERVED processing time follows a GIVEN, fixed
formula (constants are printed in the header, so this half is not hidden):

    T_i = BASE[material_i] * (1 + ALPHA * W_i) + noise

STDOUT prints ONLY: a header (t, n, alpha, base0..2) and n training rows of
(gap, load, material, T).  No hidden recursion parameter, no RNG seed, no
wear value is ever printed -- the solver must infer the recursion from how
consecutive (gap, load, material, T) rows relate to each other.
"""
import sys, math, random

SEED0 = 926000
SALT_HP = 0
SALT_TRAIN_SEQ = 1
SALT_TRAIN_NOISE = 2

SIGMA_FRAC = 0.28  # observation noise sigma is this fraction of the
                    # instance's own (ALPHA * mean BASE) signal scale, so the
                    # signal-to-noise ratio -- and hence the score's dynamic
                    # range between the no-wear baseline and a perfect
                    # predictor -- stays roughly constant across test ids
                    # instead of swinging with the random ALPHA/BASE draw.


def rng_for(t, salt):
    return random.Random(SEED0 + t * 104729 + salt * 999983)


def hidden_params(t):
    """Per-test-id hidden recursion + observation constants. Never printed."""
    rng = rng_for(t, SALT_HP)
    decay = rng.uniform(0.08, 0.22)          # idle-recovery rate (HIDDEN)
    wear_rate = rng.uniform(0.006, 0.014)    # load-driven gain rate (HIDDEN)
    mat_mult = [rng.uniform(0.7, 1.0),       # per-material wear severity (HIDDEN)
                rng.uniform(1.0, 1.3),
                rng.uniform(1.3, 1.7)]
    alpha = rng.uniform(0.8, 1.4)            # observation formula (GIVEN in input)
    base = [rng.uniform(8.0, 12.0) for _ in range(3)]
    wcap = 1.0
    return dict(decay=decay, wear_rate=wear_rate, mat_mult=mat_mult,
                alpha=alpha, base=base, wcap=wcap)


def train_gap(rng):
    if rng.random() < 0.08:
        return rng.uniform(6.0, 10.0)
    return rng.uniform(0.0, 3.0)


def train_load(rng):
    return rng.uniform(2.0, 6.0)


def train_mat(rng):
    return rng.choices([0, 1, 2], weights=[0.4, 0.35, 0.25])[0]


def gen_rows(rng, n, gap_fn, load_fn, mat_fn):
    rows = []
    for i in range(n):
        gap = 0.0 if i == 0 else gap_fn(rng)
        load = load_fn(rng)
        mat = mat_fn(rng)
        rows.append((gap, load, mat))
    return rows


def true_wear_step(W, gap, load, mat, hp):
    Wd = W * math.exp(-hp['decay'] * gap)
    gain = hp['wear_rate'] * hp['mat_mult'][mat] * (load ** 1.5) * (1.0 - Wd / hp['wcap'])
    Wn = Wd + gain
    if Wn < 0.0:
        Wn = 0.0
    if Wn > hp['wcap'] * 2.0:
        Wn = hp['wcap'] * 2.0
    return Wn


def simulate_T(rows, hp, noise_rng, noise_sigma):
    W = 0.0
    Ts = []
    for (gap, load, mat) in rows:
        W = true_wear_step(W, gap, load, mat, hp)
        T = hp['base'][mat] * (1.0 + hp['alpha'] * W) + noise_rng.gauss(0.0, noise_sigma)
        Ts.append(T)
    return Ts


def n_train_for(t):
    return 20 + 6 * (t - 1)


def noise_sigma_for(hp):
    return SIGMA_FRAC * hp['alpha'] * (sum(hp['base']) / 3.0)


def train_data(t):
    hp = hidden_params(t)
    n = n_train_for(t)
    seq_rng = rng_for(t, SALT_TRAIN_SEQ)
    rows = gen_rows(seq_rng, n, train_gap, train_load, train_mat)
    noise_rng = rng_for(t, SALT_TRAIN_NOISE)
    Ts = simulate_T(rows, hp, noise_rng, noise_sigma_for(hp))
    return hp, n, rows, Ts


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    hp, n, rows, Ts = train_data(t)
    out = []
    out.append("%d %d %.6f %.6f %.6f %.6f" % (t, n, hp['alpha'], hp['base'][0], hp['base'][1], hp['base'][2]))
    for (gap, load, mat), T in zip(rows, Ts):
        out.append("%.6f %.6f %d %.6f" % (gap, load, mat, T))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
