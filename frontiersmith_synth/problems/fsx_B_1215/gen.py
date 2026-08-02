#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE catalogue of past unrest episodes to stdout.

Hidden model (mirrored byte-for-byte in verify.py). Each past unrest episode has
a latent "unrest magnitude" U that drives THREE observable summary statistics
logged by the monitoring network:

    ACC  = peak deformation ACCELERATION during the episode  (A0 + A1*U + noise)
    SEIS = background seismic energy-release rate            (S0 + S1*U + noise)
    INFL = cumulative magma-chamber INFLATION index          (I0 + I1*U + noise)

ACC and SEIS are driven by U exactly like INFL is -- they rise whenever unrest
intensifies, REGARDLESS of how the episode ends. Only INFL (how much magma
actually accumulated) governs whether the episode culminates in eruption. The
true eruption probability is

    P(erupt | INFL) = sigmoid( k1 * (INFL - Vstar) )

and the logged outcome is a single Bernoulli draw from that probability --
"erupted" (1) or a FAILED INTRUSION (0, magma stalled short of the surface).
Because ACC/SEIS are confounded with INFL only THROUGH the shared U, episodes
that "looked" the same on the accelerometer and seismometer can end very
differently -- and failed intrusions (0) heavily outnumber eruptions (1) in
this catalogue, exactly as in the real record.

The TRAIN catalogue printed here comes from a MODERATE unrest-magnitude band.
The HELD-OUT grading episodes (generated only inside verify.py, never seen
here) come from a markedly MORE INTENSE recent bout of regional unrest -- a
genuinely different input region -- while the underlying magma-volume law
governing eruption is unchanged.

STDOUT prints ONLY: header "<n> <test_id>" then n rows "<ACC> <INFL> <SEIS>
<erupted>". The hidden law, its coefficients, and the seeds are NEVER printed.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
SEED_LAW = 700000
SEED_TRAIN = 710000
UTR_LO, UTR_HI = 0.15, 0.75
SIG_A, SIG_S, SIG_I = 0.35, 0.40, 0.35


def n_train(t):
    return 80 + 30 * (t - 1)


def law(t):
    """Hidden generative law for this test id (identical in gen.py and verify.py)."""
    rng = random.Random(SEED_LAW + t * 91013)
    A0 = rng.uniform(0.6, 1.4); A1 = rng.uniform(3.0, 5.0)
    S0 = rng.uniform(0.4, 1.2); S1 = rng.uniform(2.2, 4.2)
    I0 = rng.uniform(0.2, 0.5); I1 = rng.uniform(1.6, 2.6)
    Vstar = rng.uniform(1.6, 2.2); k1 = rng.uniform(1.8, 3.0)
    return dict(A0=A0, A1=A1, S0=S0, S1=S1, I0=I0, I1=I1, Vstar=Vstar, k1=k1)


def sigmoid(x):
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def gen_episodes(t, lo, hi, n, seed, seed_mult):
    p = law(t)
    rng = random.Random(seed + t * seed_mult)
    rows = []
    for _ in range(n):
        U = rng.uniform(lo, hi)
        ACC = p['A0'] + p['A1'] * U + rng.gauss(0.0, SIG_A)
        SEIS = p['S0'] + p['S1'] * U + rng.gauss(0.0, SIG_S)
        INFL = p['I0'] + p['I1'] * U + rng.gauss(0.0, SIG_I)
        pt = sigmoid(p['k1'] * (INFL - p['Vstar']))
        erupt = 1 if rng.random() < pt else 0
        rows.append((ACC, INFL, SEIS, erupt))
    return rows


def gen_train(t):
    return gen_episodes(t, UTR_LO, UTR_HI, n_train(t), SEED_TRAIN, 13)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for ACC, INFL, SEIS, erupt in rows:
        out.append("%.8g %.8g %.8g %d" % (ACC, INFL, SEIS, erupt))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
