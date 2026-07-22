#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

A bakery logs dough RISE RATE r (arbitrary rate units) against proofing-box
temperature T (Kelvin), sweeping only a WARM, mid-range proofing window (about
25-42 C) that a home/commercial proofer can actually hold steady. Each testId
is a DIFFERENT dough/yeast-strain combination.

The true rate law is a HIDDEN two-channel crossover:

    k1(T) = A1 * exp(-theta1 / T)     -- fermentation channel: rate RISES
                                          with T (ordinary positive apparent
                                          activation temperature theta1)
    k2(T) = A2 * exp(+theta2 / T)     -- network/enzyme-stability channel:
                                          its rate FALLS as T rises (an
                                          effective NEGATIVE apparent
                                          activation temperature -theta2)
    r(T)  = k1(T) * k2(T) / (k1(T) + k2(T))     -- the two channels combine
                                                    like resistors IN SERIES
                                                    (harmonic mean): whichever
                                                    channel is momentarily
                                                    slower BOTTLENECKS the
                                                    observed rise rate.

Inside the proofing window the fermentation channel is
faster everywhere, but the stability channel is falling toward it, so the
observed rate is a slightly damped, gently curving version of pure
fermentation kinetics -- a straight line in Arrhenius coordinates (ln r vs
1/T) fits it beautifully. Far ABOVE the window (the oven-overshoot regime)
the stability channel becomes the bottleneck and the rate turns over and
STALLS; far BELOW the window (fridge retard) the fermentation channel is the
clear bottleneck and the rate collapses faster than the in-window trend
would suggest. Both held-out regimes are regenerated only inside the
checker -- NEVER printed here, and neither are theta1, A1, theta2, A2, or any
seed.

STDOUT prints ONLY: a header "<testId> <N>" then N rows "T r", one proofing
run per line.
"""
import sys, random, math

T_LO = 298.0          # 25 C
T_HI = 315.0          # 42 C
T_REF = (T_LO + T_HI) / 2.0
N_TRAIN = 120
NOISE_SIGMA = 0.012    # small multiplicative log-noise (measurement floor)


def hidden_law(t):
    """Hidden crossover law for this test id. Lives in gen AND grader; never printed."""
    rng = random.Random(925000 + t * 7919)
    theta1 = rng.uniform(2000.0, 2800.0)     # fermentation channel steepness
    theta2 = rng.uniform(900.0, 1500.0)      # stability channel steepness
    k1_ref = rng.uniform(400.0, 900.0)       # k1 value at T_REF
    k2_ref = rng.uniform(1500.0, 3000.0)     # k2 value at T_REF
    A1 = k1_ref * math.exp(theta1 / T_REF)
    A2 = k2_ref * math.exp(-theta2 / T_REF)
    return A1, theta1, A2, theta2


def rate_true(T, A1, theta1, A2, theta2):
    k1 = A1 * math.exp(-theta1 / T)
    k2 = A2 * math.exp(theta2 / T)
    return k1 * k2 / (k1 + k2)


def train_rows(t):
    A1, theta1, A2, theta2 = hidden_law(t)
    rng = random.Random(11000 + t * 13)
    rows = []
    for i in range(N_TRAIN):
        # jittered sweep across the proofing window so points aren't a
        # perfect deterministic grid
        frac = (i + rng.uniform(0.05, 0.95)) / N_TRAIN
        frac = min(0.999999, max(0.000001, frac))
        T = T_LO + frac * (T_HI - T_LO)
        clean = rate_true(T, A1, theta1, A2, theta2)
        noisy = clean * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        rows.append((T, noisy))
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = train_rows(t)
    out = ["%d %d" % (t, len(rows))]
    for T, r in rows:
        out.append("%.8f %.8f" % (T, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
