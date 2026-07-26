#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

A materials lab logs the DIELECTRIC BREAKDOWN VOLTAGE V of a thin insulating
film against two rig settings: film THICKNESS d (nanometres) and ambient
TEMPERATURE T (kelvin). Every sweep is confined to a SAFE MID-RANGE window
the rig can hold without damaging the sample: d in [40, 90] nm, T in
[280, 340] K. Each testId is a different film/dielectric-stack combination.

The true breakdown law is a HIDDEN two-mechanism law. Two independent
failure channels each have their own critical voltage, and the film fails
via whichever channel gives way at the LOWER voltage (a weakest-link
system), blended with a small smoothing width so the crossover is
physically continuous:

    V1(d,T) = V1ref * (d/D_REF)^p1 * (T/T_REF)^q1   -- "avalanche-like"
                                                          channel: needs MORE
                                                          voltage as T rises
                                                          (q1 > 0) and scales
                                                          close to linearly
                                                          with thickness
                                                          (p1 near 1)
    V2(d,T) = V2ref * (d/D_REF)^p2 * (T/T_REF)^q2   -- "tunneling-like"
                                                          channel: needs LESS
                                                          voltage as T rises
                                                          (q2 < 0) and scales
                                                          SUB-linearly with
                                                          thickness (p2 < p1)
    V(d,T)  = softmin_k(V1, V2)
            = m - (1/k) * ln( exp(-k*(V1-m)) + exp(-k*(V2-m)) ),  m=min(V1,V2)

Both channels are ordinary power laws (dimensionally sensible: a fixed
reference thickness/temperature and a single multiplicative scale), so the
observed voltage is everywhere finite, positive, continuous, and -- within
each channel's own regime -- monotonic in d and in T. The softmin makes the
crossover itself continuous (no jump) instead of a hard kink.

Inside the sampled window BOTH channels are comparable in size (the window
straddles the transition), so a single global power-law regression fits the
training rows nearly perfectly -- nothing in-sample screams "two channels."
Far outside the window, in the four EXTREME corners of the (d,T) square,
one channel decisively wins and the curve saturates onto that channel's own
asymptote -- an asymptote a single global power law never bends towards.
Which channel wins at a given far corner depends on BOTH d and T jointly
(the switch is a tilted curve in the (d,T) plane, not an axis-aligned
threshold), so the winner at the two "mixed" corners (thin+hot, thick+cold)
varies from instance to instance and cannot be read off a fixed rule.

STDOUT prints ONLY a header "<testId> <N>" then N rows "d T V" (one
measurement per line). theta/coefficients/seed are NEVER printed here.
"""
import sys
import random
import math

D_LO, D_HI = 40.0, 90.0
T_LO, T_HI = 280.0, 340.0
D_REF = 65.0
T_REF = 310.0
N_TRAIN = 140
NOISE_SIGMA = 0.020   # small multiplicative log-noise (measurement floor)

SEED_BASE = 114100
SEED_MULT = 7919
ROW_SEED_BASE = 22100
ROW_SEED_MULT = 131


def hidden_law(t):
    """Hidden two-channel law for this test id. Lives in gen AND checker; never printed."""
    rng = random.Random(SEED_BASE + t * SEED_MULT)
    p1 = rng.uniform(1.00, 1.20)
    q1 = rng.uniform(0.55, 0.85)
    p2 = rng.uniform(0.45, 0.65)
    q2 = rng.uniform(-0.75, -0.45)
    V1ref = rng.uniform(63.0, 80.0)
    V2ref = rng.uniform(63.0, 80.0)
    k = rng.uniform(0.05, 0.09)
    return p1, q1, p2, q2, V1ref, V2ref, k


def branch1(d, T, p1, q1, V1ref):
    return V1ref * (d / D_REF) ** p1 * (T / T_REF) ** q1


def branch2(d, T, p2, q2, V2ref):
    return V2ref * (d / D_REF) ** p2 * (T / T_REF) ** q2


def v_true(d, T, params):
    p1, q1, p2, q2, V1ref, V2ref, k = params
    v1 = branch1(d, T, p1, q1, V1ref)
    v2 = branch2(d, T, p2, q2, V2ref)
    m = min(v1, v2)
    z = math.exp(-k * (v1 - m)) + math.exp(-k * (v2 - m))
    return m - math.log(z) / k


def train_rows(t):
    params = hidden_law(t)
    rng = random.Random(ROW_SEED_BASE + t * ROW_SEED_MULT)
    rows = []
    # a jittered grid over d and T so rows are not a perfect regular lattice
    side = int(round(N_TRAIN ** 0.5))
    while side * side < N_TRAIN:
        side += 1
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N_TRAIN:
                break
            fd = (i + rng.uniform(0.08, 0.92)) / side
            fT = (j + rng.uniform(0.08, 0.92)) / side
            fd = min(0.999999, max(0.000001, fd))
            fT = min(0.999999, max(0.000001, fT))
            d = D_LO + fd * (D_HI - D_LO)
            T = T_LO + fT * (T_HI - T_LO)
            clean = v_true(d, T, params)
            noisy = clean * math.exp(rng.gauss(0.0, NOISE_SIGMA))
            rows.append((d, T, noisy))
            idx += 1
    rng.shuffle(rows)
    rows = rows[:N_TRAIN]
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = train_rows(t)
    out = ["%d %d" % (t, len(rows))]
    for d, T, V in rows:
        out.append("%.8f %.8f %.8f" % (d, T, V))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
