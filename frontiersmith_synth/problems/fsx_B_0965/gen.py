#!/usr/bin/env python3
# gen.py -- prints ONE instance (the TRAIN sample) for a given testId to stdout.
#   python3 gen.py <testId>       testId = 1..10  (difficulty ladder: train size shrinks,
#                                                    measurement noise rises, with testId)
#
# Theme: CENSUS CLERK'S NUMBER-STAMPED BRASS TAGS.  Every resident n = 1, 2, 3, ... wears a
# brass tag stamped with an auxiliary registry number f(n) by a hand-cranked press.  f is a
# fixed but UNKNOWN function of the tag number.  The clerk's ledger records the tag number n
# and the press's stamped reading for n = 1 .. N_train (small serial numbers only); some
# stampings are worn/mis-struck, adding small integer noise.  A separate archive box (which the
# clerk never opens) holds tags n = 100000 .. 1000000 -- the grader alone evaluates there.
#
# The hidden law and the noise seed live here AND (independently, byte-identical) inside
# verify.py. This file prints DATA ROWS ONLY -- no seed, no law, no coefficients -- so the
# solver must DISCOVER the functional family purely from the numbers.
#
# Hidden law (NOT printed): f is MULTIPLICATIVE -- f(1) = 1, and for coprime m, n,
# f(m*n) = f(m)*f(n).  On a prime power p^k it is a simple deduction rule
#     f(p^k) = p^k - a(p) * p^(k-1)
# where the integer discount a(p) depends only on which of THREE classes p falls in:
# p == 2, p % 4 == 1, or p % 4 == 3.  (a(2), a(1 mod 4), a(3 mod 4)) vary per testId.
import sys, random

# ---- ladder (identical in verify.py) ----
COEF_TABLE = {
    1:  (1, 1, 1),
    2:  (0, 2, 0),
    3:  (1, 0, 2),
    4:  (0, 1, 2),
    5:  (1, 2, 1),
    6:  (0, 1, 1),
    7:  (1, 0, 1),
    8:  (0, 2, 1),
    9:  (1, 1, 2),
    10: (1, 2, 0),
}   # testId -> (a2 in {0,1}, a(p%4==1) in {0,1,2}, a(p%4==3) in {0,1,2})

# N_train is the SAME for every testId on purpose -- if it varied per testId it would be a
# hard, directly-readable side channel (line 1 of stdin) that reveals which law applies without
# looking at a single observation value. Only the noise level ramps with testId; the solver
# must actually read the (n, obs) values to tell instances apart.
NTRAIN_FIXED = 3000
SIGMA_TABLE = {1: 0.4, 2: 0.5, 3: 0.6, 4: 0.7, 5: 0.8,
               6: 1.0, 7: 1.1, 8: 1.3, 9: 1.5, 10: 1.8}


def coef_for(test_id):
    a2, a1, a3 = COEF_TABLE[test_id]
    return a2, a1, a3


def pp_value(p, k, a2, a1, a3):
    if p == 2:
        a = a2
    elif p % 4 == 1:
        a = a1
    else:
        a = a3
    return p ** k - a * p ** (k - 1)


def f_true(n, a2, a1, a3):
    if n == 1:
        return 1
    m = n
    d = 2
    result = 1
    while d * d <= m:
        if m % d == 0:
            k = 0
            while m % d == 0:
                m //= d
                k += 1
            result *= pp_value(d, k, a2, a1, a3)
        d += 1
    if m > 1:
        result *= pp_value(m, 1, a2, a1, a3)
    return result


def gen_train(test_id):
    a2, a1, a3 = coef_for(test_id)
    ntrain = NTRAIN_FIXED
    sigma = SIGMA_TABLE[test_id]
    rng = random.Random(4_100_000 + test_id * 131)
    rows = []
    for n in range(1, ntrain + 1):
        true_v = f_true(n, a2, a1, a3)
        noise = int(round(rng.gauss(0.0, sigma)))
        noise = max(-5, min(5, noise))
        obs = true_v + noise
        if obs < 0:
            obs = 0
        rows.append((n, obs))
    return rows


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(test_id)
    out = [str(len(rows))]
    for (n, obs) in rows:
        out.append("%d %d" % (n, obs))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
