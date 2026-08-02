#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN learning-curve trace to stdout.

A hidden model family has test error that shrinks with training-set size `n`
along a power law that eventually flattens to a strictly positive IRREDUCIBLE
ERROR FLOOR (label noise / Bayes error -- more data can never erase it):

    err(n) -> floor + A * n^(-alpha)          as n grows "large"

But the VISIBLE small-data range you get here is not one clean regime. Early
on (small n) the curve decays along a much STEEPER exponent -- an artifact of
a fast initial representation-learning burst -- before bending over into the
true, much SHALLOWER asymptotic regime the floor+power-law describes. Both
regimes are stitched together continuously at a hidden break point `n_break`
that this file never prints; only the resulting noisy (n, err) rows are shown.

You must forecast error at HELD-OUT scales -- 3x to 2000x past the largest
`n` you are shown -- where the mechanism is regenerated (fresh, unseen)
entirely inside the grader. The break point, floor, and both exponents are
NEVER printed by this script; the ground truth lives only in verify.py.

STDOUT format: a header "<m> <testId>" then `m` rows "<n> <err>" (n ascending
integers, err a noisy positive float). Nothing else is printed.
"""
import sys, random


def params(t):
    """Hidden learning-curve law for this test id (lives in gen AND verify,
    never printed). Two power-law regimes joined continuously at n_break."""
    rng = random.Random(90210 + t * 104729)
    floor = rng.uniform(0.03, 0.11)
    alpha_slow = rng.uniform(0.28, 0.50)                 # true asymptotic exponent
    alpha_fast = alpha_slow + rng.uniform(0.55, 1.05)    # steep early-burst exponent
    A_slow = rng.uniform(0.8, 2.2)
    n_min = 40 + (t % 4) * 10
    scale_mult = 55.0 + 7.0 * t                          # visible span widens with t
    n_max = int(n_min * scale_mult)
    frac = rng.uniform(0.22, 0.38)                       # break sits early in log-range
    n_break = int(round(n_min * (n_max / n_min) ** frac))
    A_fast = A_slow * (n_break ** (alpha_fast - alpha_slow))   # continuity at n_break
    m = max(9, 14 - (t - 1) // 3)                         # fewer points as t grows
    sigma = 0.004 + 0.0012 * t                            # measurement noise sd
    return floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break, n_min, n_max, m, sigma


def true_err(n, floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break):
    if n < n_break:
        return floor + A_fast * (n ** (-alpha_fast))
    return floor + A_slow * (n ** (-alpha_slow))


def train_ns(n_min, n_max, m):
    ns = []
    prev = 0
    for i in range(m):
        f = i / (m - 1) if m > 1 else 0.0
        n = int(round(n_min * (n_max / n_min) ** f))
        if n <= prev:
            n = prev + 1
        ns.append(n)
        prev = n
    return ns


def main():
    if len(sys.argv) < 2:
        print("1 1")
        print("100 0.5")
        return
    t = int(sys.argv[1])
    floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break, n_min, n_max, m, sigma = params(t)
    ns = train_ns(n_min, n_max, m)
    rng = random.Random(555 + t * 7919)
    print(f"{m} {t}")
    for n in ns:
        e = true_err(n, floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break)
        y = e + rng.gauss(0.0, sigma)
        y = max(1e-4, y)
        print(f"{n} {y:.8f}")


if __name__ == "__main__":
    main()
