#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample of a sled's stick-slip pull
response to stdout.

Physical picture (sled on waxed wood, jerky pulls): before each pull the sled
has been resting motionless for r seconds. A pull ramps the applied force up
to a peak F and holds it (quasi-steady). Two things can happen:

  * If F is not enough to overcome the (aged) static-friction hold, the sled
    stays put but SLOWLY CREEPS a little (a logarithmic function of the rest
    time -- longer quiescence lets more microscopic contact junctions form,
    so the creep grows with ln(1+r)).
  * If F exceeds the hold, the sled breaks free and slides under KINETIC
    friction, producing a much larger net displacement that depends on the
    EXCESS force over the (lower) kinetic floor -- and, once sliding, no
    longer depends on how long it had rested.

Which branch applies is itself governed by an aging static-friction hold that
RISES logarithmically with rest time -- a longer nap makes the sled harder to
budge. This is a hybrid automaton: one guard, two trivial per-mode laws. The
hidden constants (and the exact guard) are NEVER printed -- only sampled
(F, r, y) triples, recorded under GENTLE, modest-range pulls. Held-out grading
uses more aggressive pulls / longer naps and lives only inside the checker.

STDOUT: a header "<n_train> <test_id>" then n_train lines "F r y".
"""
import sys, math, random


def true_y(F, r, F0, A, FK, KC, AKIN, BKIN):
    """Hybrid automaton: aging guard picks the active mode; each mode is a
    trivially simple function of (F, r)."""
    L = math.log(1.0 + r)
    hold = F0 + A * L
    if F <= hold:
        return KC * F * L                 # stuck: logarithmic creep only
    else:
        return AKIN * (F - FK) + BKIN     # sliding: linear excess-force law


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    # Fixed hidden law (same mechanism for every test id -- only the sampled
    # data changes with difficulty). Never printed to stdout.
    F0, A, FK, KC, AKIN, BKIN = 3.0, 1.1, 1.2, 0.04, 0.9, 0.25

    n_train = 90 - 4 * (t - 1)
    sigma_add = 0.03
    sigma_rel = 0.02 + 0.02 * (t - 1)
    f_max_train = 7.0
    r_max_train = 15.0

    rnd = random.Random(900000 + 137 * t)
    rows = []
    for _ in range(n_train):
        F = rnd.uniform(0.0, f_max_train)
        r = rnd.uniform(0.0, r_max_train)
        yt = true_y(F, r, F0, A, FK, KC, AKIN, BKIN)
        noise_sd = sigma_add + sigma_rel * abs(yt)
        y = yt + rnd.gauss(0.0, noise_sd)
        rows.append((F, r, y))

    out = ["%d %d" % (n_train, t)]
    for F, r, y in rows:
        out.append("%.6f %.6f %.6f" % (F, r, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
