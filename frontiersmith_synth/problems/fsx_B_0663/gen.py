#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance to stdout. Deterministic: seeded only by testId."""
import sys


def mul(A, B, p):
    a, b, c, d = A
    e, f, g, h = B
    return ((a * e + b * g) % p, (a * f + b * h) % p,
            (c * e + d * g) % p, (c * f + d * h) % p)


def good_hint(k, p):
    """A generically strong (subgroup-escaping) courier set built from short words
    in two fixed determinant-1 seeds. Products of det-1 matrices are automatically
    det-1, no modular inverse needed."""
    P = (1, 1, 1, 2)
    Q = (2, 1, 1, 1)
    if k == 2:
        return [P, Q]
    if k == 3:
        return [P, Q, mul(P, Q, p)]
    raise ValueError("k must be 2 or 3")


def trap_hint(k, p):
    """A deliberately subgroup-trapped courier set: k copies of the SAME upper-
    triangular unipotent shear. Feasible (det=1) but generates only a tiny cyclic
    subgroup regardless of p, k, r."""
    return [(1, 1, 0, 1) for _ in range(k)]


# testId -> (p, k, r, trap)
TESTS = {
    1: (23, 2, 3, False),
    2: (31, 2, 3, False),
    3: (41, 2, 4, True),
    4: (53, 2, 4, False),
    5: (67, 3, 3, True),
    6: (79, 3, 3, False),
    7: (227, 3, 4, False),
    8: (281, 3, 4, True),
    9: (131, 2, 4, False),
    10: (337, 3, 4, True),
}


def main():
    tid = int(sys.argv[1])
    p, k, r, trap = TESTS[tid]
    hint = trap_hint(k, p) if trap else good_hint(k, p)
    out = [f"{p} {k} {r}"]
    for (a, b, c, d) in hint:
        out.append(f"{a % p} {b % p} {c % p} {d % p}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
