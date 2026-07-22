#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE train sample to stdout.

Family: integer-recurrence-recovery.  A hidden integer sequence a(n) is a sum
of a handful of pure geometric terms c_i * r_i^n (a constant-coefficient
linear recurrence sequence -- equivalently a rational generating function
P(x)/Q(x) with Q(x) = prod (1 - r_i x)).  One term (root R) has a LARGE
coefficient and dominates every term you are shown.  The remaining terms
("trap" roots, each R+1, R+2, ...) have coefficient exactly +-1 -- utterly
negligible next to the dominant term throughout the visible window -- but
their roots are slightly LARGER than R, so each one eventually overtakes the
dominant term at some far index.  The visible window ends well before any
crossover; the grading (held out, inside the checker) probes indices well
past several of them.

Each testId fixes a DIFFERENT hidden law (order D = 2..6, growing with
testId).  STDOUT prints ONLY a header "<T> <testId>" then T rows "n a(n)"
for n = 0..T-1.  The roots, coefficients, and true order are NEVER printed.
"""
import sys


def num_trap(t):
    if t <= 2:
        return 2
    if t <= 4:
        return 3
    if t <= 6:
        return 4
    return 5


_C_BASE = {3: 150, 4: 280, 5: 1400, 6: 13000}


def law(t):
    """Hidden law for test id t (lives in gen AND checker, never printed).
    C_BASE grows fast with D so that EVERY trap root's crossover against the
    dominant term stays past the visible window T (verified numerically at
    authoring time), even though the trap coefficients are always +-1."""
    R = 9
    d = num_trap(t)
    D = 1 + d
    T = 2 * D + 6
    C = _C_BASE[D] + 8 * t
    roots = [R] + [R + k for k in range(1, d + 1)]
    coeffs = [C] + [1 if k % 2 == 1 else -1 for k in range(1, d + 1)]
    return roots, coeffs, D, T


def a_of(n, roots, coeffs):
    return sum(c * (r ** n) for c, r in zip(coeffs, roots))


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    roots, coeffs, D, T = law(t)
    out = ["%d %d" % (T, t)]
    for n in range(T):
        out.append("%d %d" % (n, a_of(n, roots, coeffs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
