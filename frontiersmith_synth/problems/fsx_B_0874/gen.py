#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE sparse eclipse-timing TRAIN log to stdout.

Theme: moons of a far planet keep running late.  A newly charted moon's transit
(eclipse) epochs t_k = t0 + P*k + c*k^2 + A*sin(2*pi*k/M + phi) + noise, where
k is the orbital-cycle number.  Three mechanisms are baked into the SAME law:

  - secular-drift:        the c*k^2 term (a slow, ever-GROWING timing offset --
                            e.g. tidal deceleration lengthening the orbit).
  - superposed-periods:    the A*sin(...) term (a BOUNDED wobble at a second,
                            hidden period M, e.g. a resonant tug from a sibling
                            moon).
  - sparse-event-times:    only a random subset of the theoretical cycles is
                            ever OBSERVED (transits are missed on many passes:
                            bad weather, wrong hemisphere, instrument downtime),
                            so the solver sees gapped (k, t_k) pairs, not a
                            dense signal.

Each testId fixes a DIFFERENT hidden moon (period P, offset t0, drift c,
wobble A/M/phi, timing jitter sigma) via a seeded RNG.  The observing campaign
covers only a few years; the solver is graded on epochs several YEARS beyond
that campaign (see verify.py), where c*k^2 and A*sin(...) behave very
differently: one keeps growing, the other stays bounded.

STDOUT prints ONLY a header "<n_train> <test_id>" followed by n_train lines
"<k> <t_k>" (cycle number, observed epoch time in days).  The hidden law
(t0, P, c, A, M, phi, sigma) is NEVER printed -- it lives only in this
function and in verify.py's identical copy of it.
"""
import sys, math, random

BASE = 90210
TRAIN_DAYS = 1400.0     # ~3.8 years of observing campaign
HELD_LO_MULT = 1.7      # held-out cycles start at 1.7x the campaign's cycle count
HELD_HI_MULT = 2.3      # ... and run out to 2.3x  (roughly 6.5-8.8 years out)
P_LO, P_HI = 8.0, 16.0  # plausible short-period moon, in days
NH = 26                 # held-out grading points (used by verify.py only)


def params(t):
    """Hidden law for this test id (lives in gen.py AND verify.py, never printed)."""
    rng = random.Random(BASE + t * 1000003)
    P = rng.uniform(P_LO, P_HI)
    t0 = rng.uniform(0.0, 300.0)
    Ktrainmax = int(TRAIN_DAYS // P)
    trap = (t % 2 == 1)
    if trap:
        # long-period wobble: LESS than one full cycle fits inside the campaign,
        # so in-window it looks like just another slow curve -- degenerate with
        # the quadratic drift.  The drift itself is the dominant true signal.
        M = rng.uniform(1.6, 2.5) * Ktrainmax
        c = rng.uniform(2.2e-4, 4.0e-4)
    else:
        # short-period wobble: several full cycles fit inside the campaign, so
        # it is genuinely identifiable.  Drift is present but negligible.
        M = rng.uniform(0.12, 0.28) * Ktrainmax
        c = rng.uniform(0.3e-6, 2.0e-6)
    A = rng.uniform(0.6, 1.5)
    phi = rng.uniform(0.0, 2 * math.pi)
    sigma = rng.uniform(0.07, 0.11)
    return dict(P=P, t0=t0, Ktrainmax=Ktrainmax, M=M, c=c, A=A, phi=phi,
                sigma=sigma, trap=trap)


def true_val(k, pr):
    return (pr['t0'] + pr['P'] * k + pr['c'] * k * k
            + pr['A'] * math.sin(2 * math.pi * k / pr['M'] + pr['phi']))


def gen_train(t, pr):
    """Sparse, gapped observed cycles (only SOME transits are ever caught)."""
    rng2 = random.Random(BASE + 7 + t * 668211)
    Ktrainmax = pr['Ktrainmax']
    keep_p = rng2.uniform(0.55, 0.8)
    ks = [k for k in range(0, Ktrainmax + 1) if rng2.random() < keep_p]
    if 0 not in ks:
        ks = [0] + ks
    if Ktrainmax not in ks:
        ks = ks + [Ktrainmax]
    ks = sorted(set(ks))
    if len(ks) < 35:
        cand = [k for k in range(Ktrainmax + 1) if k not in ks]
        rng2.shuffle(cand)
        need = 35 - len(ks)
        ks = sorted(set(ks + cand[:need]))
    rng3 = random.Random(BASE + 11 + t * 97711)
    obs = [(k, true_val(k, pr) + rng3.gauss(0.0, pr['sigma'])) for k in ks]
    return obs


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    pr = params(t)
    obs = gen_train(t, pr)
    out = ["%d %d" % (len(obs), t)]
    for k, v in obs:
        out.append("%d %.6f" % (k, v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
