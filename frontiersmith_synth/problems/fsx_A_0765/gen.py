#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy "ornament ledger" to stdout.

Backstory (for gen.py's own bookkeeping only -- NOT printed): a hidden beauty
law scores small square ornament grids by combining three measurements:

    D  raw symmetry-defect count   (how many cells break the grid's own
                                     rotation/reflection symmetry)
    K  motif count                 (how many separate ink blobs the grid has)
    H  spacing entropy             (Shannon entropy of the gaps between
                                     motifs, already scale-free)

together with two purely geometric bookkeeping numbers that come from the
grid's side length `n` and its **fold order** `g` (the order of the
rotation/reflection group the ornament repeats under -- g=8 is an ordinary
4-fold "rosette", larger g means a finer kaleidoscope):

    A  = n*n            total cells (area)
    M  = (n*n) // g      number of symmetry ORBITS at that fold order

The TRAIN ledger the solver sees comes from a single workshop that only ever
carved ordinary 4-fold rosettes: every training row has g = 8, so across all
of training M is just A / 8 -- a fixed multiple of the area, and the two
numbers carry IDENTICAL information. The held-out grading ledger (regenerated
ONLY inside the checker) is sampled from finer, larger kaleidoscopes with
g in {16, 24, 32, 40} and much bigger n, where M and A/8 pull apart sharply.

STDOUT prints ONLY: header "<n_rows> <test_id>" then n_rows data rows
"n g D M A K H B" (B = the hidden beauty score, the training LABEL). The law,
its weights and the seeds are never printed -- only these numeric columns.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
N_TRAIN = 70
N_TRAIN_LO, N_TRAIN_HI = 5, 14
G_TRAIN = 8
NOISE_TRAIN = 0.15


def params(t):
    """Hidden beauty law for this test id (identical in gen.py and verify.py)."""
    rng = random.Random(500000 + t * 8161063)
    w1 = rng.uniform(0.8, 1.6)     # weight on the orbit-normalised defect term
    w2 = rng.uniform(1.0, 2.4)     # weight on the (already scale-free) entropy term
    w3 = rng.uniform(1.0, 2.5)     # weight on the area-normalised motif term
    w4 = rng.uniform(0.6, 1.2)     # additive offset (keeps B positive)
    return w1, w2, w3, w4


def true_B(D, M, H, K, A, w1, w2, w3, w4):
    """B = w1*sqrt(D/M) + w2*H + w3*(K/A) + w4."""
    return w1 * math.sqrt(D / M) + w2 * H + w3 * (K / A) + w4


def gen_rows(t, n_rows, n_lo, n_hi, g_choices, noise_sigma, seed_base):
    w1, w2, w3, w4 = params(t)
    rng = random.Random(seed_base + t * 131)
    rows = []
    for _ in range(n_rows):
        n = rng.randint(n_lo, n_hi)
        g = rng.choice(g_choices)
        A = n * n
        if g > A:
            g = 1
        M = max(1, A // g)
        rho_D = rng.uniform(0.6, 2.4)          # per-instance defect roughness
        D = max(1, round(M * rho_D))
        rho_K = rng.uniform(0.03, 0.35)        # per-instance motif density
        K = max(1, round(A * rho_K))
        H = rng.uniform(0.4, 3.2)
        B = true_B(D, M, H, K, A, w1, w2, w3, w4) + rng.gauss(0.0, noise_sigma)
        rows.append((n, g, D, M, A, K, H, B))
    return rows


def gen_train(t):
    return gen_rows(t, N_TRAIN, N_TRAIN_LO, N_TRAIN_HI, [G_TRAIN], NOISE_TRAIN, 111)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for n, g, D, M, A, K, H, B in rows:
        out.append("%d %d %d %d %d %d %.8g %.8g" % (n, g, D, M, A, K, H, B))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
