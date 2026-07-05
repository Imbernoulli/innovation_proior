#!/usr/bin/env python3
"""Instance generator for fsx_G_0447 (format D, flop-cse-arith-dag).

`python3 gen.py <testId>` prints ONE instance to stdout.
testId 1..10 = difficulty ladder (small -> large / more sharing).
All randomness is seeded by testId ONLY (deterministic).

Instance = a SET of target polynomial expressions over input variables
x_0 .. x_{n-1}.  Each target y_j is  sum_t c * (product of variables).
The terms are drawn from a shared POOL of monomials, so the same monomials
(and variable sub-products) recur across targets -> genuine common-
subexpression / sub-product sharing opportunities for a compiler backend.

STDOUT format:
  n m
  then m target blocks, each:
    K                         # number of terms in this target
    c d v1 v2 ... vd          # (K lines) coeff c, degree d, then d var indices
"""
import sys
import random


def build(testId):
    t = testId
    rng = random.Random(20260701 + 7919 * t)

    n = 5 + t                       # 6 .. 15 input variables
    m = 3 + t                       # 4 .. 13 targets
    P = 5 + 2 * t                   # pool of distinct monomials
    dmin = 2
    dmax = 2 + min(3, (t + 1) // 2)  # 2 .. 5 max degree
    Tterms = min(P, 3 + t // 2)     # terms per target
    coeffs = [-3, -2, -1, 1, 2, 3]

    # --- build a pool of DISTINCT monomials (each = sorted tuple of var ids) ---
    pool = []
    seen = set()
    attempts = 0
    while len(pool) < P and attempts < 50 * P:
        attempts += 1
        d = rng.randint(dmin, min(dmax, n))
        mono = tuple(sorted(rng.sample(range(n), d)))
        if mono in seen:
            continue
        seen.add(mono)
        pool.append(mono)
    # fallback if collisions starved the pool
    if len(pool) < 2:
        pool = [tuple(sorted(rng.sample(range(n), min(dmin, n)))) for _ in range(max(2, P))]

    Tt = min(Tterms, len(pool))

    targets = []
    for _ in range(m):
        idxs = rng.sample(range(len(pool)), Tt)
        terms = []
        for k, pi in enumerate(idxs):
            c = rng.choice(coeffs)
            if k == 0:
                c = abs(c)          # first term positive (clean accumulation)
            terms.append((c, pool[pi]))
        targets.append(terms)

    return n, m, targets


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n, m, targets = build(testId)
    out = []
    out.append("%d %d" % (n, m))
    for terms in targets:
        out.append(str(len(terms)))
        for (c, mono) in terms:
            out.append("%d %d %s" % (c, len(mono), " ".join(str(v) for v in mono)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
