#!/usr/bin/env python3
"""gen.py <testId> -- emits one quotient-balanced-partition instance to stdout.

Instance = a prime p (n = p-1 trees, the nonzero residues mod p, split into k
equal-size plots) plus a fixed public battery of two Fourier-dual balance
tests: ROW tests (contiguous windows in natural VALUE order) and GRAFT tests
(residue buckets in discrete-LOG order w.r.t. a fixed primitive root).

Deterministic: testId selects a hardcoded config; no randomness anywhere.
"""
import sys

# (p, k, d_bads, d_good) per testId (1..10), a small->large/adversarial ladder.
# d_bads are divisors of k (the ones a naive "index mod k" coset trick nails
# perfectly for orders that DON'T divide k but blows up on for orders that DO);
# d_good are extra graft tests unrelated to k's factorization. Every p was
# chosen so that n=p-1 is an exact multiple of k*lcm(d_bads), which is what
# lets a hybrid (coset skeleton, regrouped at modulus k*lcm(d_bads)) construction
# hit every plot size exactly.
CONFIGS = [
    (73,   4,  [2],     [3, 5]),
    (109,  6,  [3],     [5, 7]),
    (193,  8,  [2, 4],  [3, 7]),
    (401,  10, [5],     [3, 7]),
    (1009, 12, [3, 4],  [5, 7]),
    (1373, 14, [2, 7],  [3, 11]),
    (2269, 18, [2, 9],  [5, 7]),
    (2801, 20, [4, 5],  [3, 7]),
    (6337, 24, [3, 8],  [5, 7]),
    (8101, 30, [6, 10], [7, 11]),
]


def build_windows(n, k):
    bs = n // k
    ww = min(16, max(6, bs // 10))
    raw = [(1 + bs // 4, ww), (bs - ww // 2, ww), (1 + 2 * bs + bs // 3, ww)]
    out = []
    for t, w in raw:
        t = max(1, t)
        w = max(1, w)
        if t + w - 1 > n:
            w = n - t + 1
        out.append((t, w))
    return out


def main():
    test_id = int(sys.argv[1])
    p, k, d_bads, d_good = CONFIGS[test_id - 1]
    n = p - 1
    bs = n // k
    sizes = [bs] * k
    windows = build_windows(n, k)
    mults = list(d_bads) + list(d_good)

    out = []
    out.append(f"{p} {k}")
    out.append(" ".join(str(s) for s in sizes))
    out.append(str(len(windows)))
    for t, w in windows:
        out.append(f"{t} {w}")
    out.append(str(len(mults)))
    for d in mults:
        out.append(str(d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
