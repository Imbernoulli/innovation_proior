#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy harbor tide-height log to stdout.

Hidden law (never printed): the tide height is a sum of FOUR sinusoidal
"gears". THREE of them are locked in an EXACT small-integer frequency ratio
n1:n2:n3 around one shared, unknown base "celestial gear" rate f0 (a hidden
commensurability). The FOURTH is a genuinely free interloper frequency f4,
picked well away from any low-order rational relation to f0.

The TRAIN log the solver sees spans a window SHORTER than the full locked
super-period (1/f0), sampled at a modest rate -- inside that window the three
locked gears sit closer together in frequency than the window's own spectral
resolution, so they cannot be told apart by treating them as four independent,
unrelated frequencies. The HELD-OUT grading window (regenerated ONLY inside
the grader) starts several super-periods later, where any per-frequency
estimation error has had time to turn into a large, essentially uncorrelated
phase error -- unless the three locked gears were fit through their EXACT
integer relationship using one shared parameter.

STDOUT prints ONLY: header "<n> <test_id>" then n rows "<t> <y>". The hidden
integers, the base rate, amplitudes, phases and seeds are NEVER printed.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
CANDIDATE_TRIPLES = [(2, 3, 7), (3, 4, 5), (2, 5, 7), (3, 5, 8),
                      (2, 3, 11), (4, 5, 7), (2, 7, 9), (3, 7, 8)]
F0_LO, F0_HI = 0.0045, 0.0065      # shared base "gear" rate (period 154..222)
A_LO, A_HI = 0.6, 1.4
F4_LO, F4_HI = 0.16, 0.30          # free interloper, well above the locked band
NOISE_TRAIN = 0.08
DT_TRAIN = 0.5
T_TRAIN_LO, T_TRAIN_HI = 120.0, 190.0   # difficulty ladder: shorter -> harder


def n_train_for(t):
    frac = max(0.0, min(1.0, (t - 1) / 9.0))
    span = T_TRAIN_HI - frac * (T_TRAIN_HI - T_TRAIN_LO)
    return int(round(span / DT_TRAIN))


def params(t):
    """Hidden tide law for this test id (identical in gen.py and verify.py)."""
    rng = random.Random(424242 + t * 97531)
    n1, n2, n3 = rng.choice(CANDIDATE_TRIPLES)
    f0 = rng.uniform(F0_LO, F0_HI)
    A = [rng.uniform(A_LO, A_HI) for _ in range(4)]
    phi = [rng.uniform(0.0, 2 * math.pi) for _ in range(4)]
    while True:
        f4 = rng.uniform(F4_LO, F4_HI)
        ok = True
        for m in range(1, 26):
            if abs(f4 - m * f0) < 0.02:
                ok = False
                break
        if ok:
            break
    return n1, n2, n3, f0, f4, A, phi


def true_y(tt, n1, n2, n3, f0, f4, A, phi):
    w1, w2, w3, w4 = 2 * math.pi * n1 * f0, 2 * math.pi * n2 * f0, 2 * math.pi * n3 * f0, 2 * math.pi * f4
    return (A[0] * math.sin(w1 * tt + phi[0]) + A[1] * math.sin(w2 * tt + phi[1]) +
            A[2] * math.sin(w3 * tt + phi[2]) + A[3] * math.sin(w4 * tt + phi[3]))


def gen_train(t):
    n1, n2, n3, f0, f4, A, phi = params(t)
    n_train = n_train_for(t)
    rng = random.Random(1111 + t * 13)
    rows = []
    for i in range(n_train):
        tt = i * DT_TRAIN
        y = true_y(tt, n1, n2, n3, f0, f4, A, phi) + rng.gauss(0.0, NOISE_TRAIN)
        rows.append((tt, y))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for tt, y in rows:
        out.append("%.6f %.8g" % (tt, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
