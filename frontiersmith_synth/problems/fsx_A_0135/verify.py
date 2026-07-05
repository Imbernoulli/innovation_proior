#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the tide-pool
first-autocorrelation-constant problem.

Reads the instance from <in> and the participant density vector f from <out>.
Feasibility (any violation -> `Ratio: 0.0`):
  * exactly n non-negative finite reals,
  * 0 <= f_i <= cap_i for every pool,
  * total placed population == S (within tolerance).
Objective (minimize):
  c1(f) = 2*n * max(conv(f,f)) / (sum f)^2
Baseline B: the checker's own trivial feasible construction -- pile the whole
population into the left-most pools ("fill from the left"), a naive placement.
Normalization (minimization): sc = min(1000, 100*B/F); Ratio = sc/1000.
"""
import sys


def read_tokens(path):
    with open(path) as fh:
        return fh.read().split()


def autocorr_peak(f):
    """max over k of sum_i f[i]*f[k-i]  (full self-convolution peak)."""
    n = len(f)
    best = 0.0
    # g[k] = sum_{i} f[i]*f[k-i], k = 0 .. 2n-2
    for k in range(2 * n - 1):
        lo = max(0, k - (n - 1))
        hi = min(k, n - 1)
        s = 0.0
        i = lo
        while i <= hi:
            s += f[i] * f[k - i]
            i += 1
        if s > best:
            best = s
    return best


def c1(f, n):
    tot = sum(f)
    if tot <= 0:
        return None
    return 2.0 * n * autocorr_peak(f) / (tot * tot)


def fill_left(cap, S):
    """Naive baseline: fill pools left-to-right to capacity until S is used."""
    f = [0.0] * len(cap)
    r = float(S)
    for i in range(len(cap)):
        take = cap[i] if cap[i] < r else r
        if take < 0:
            take = 0.0
        f[i] = float(take)
        r -= take
        if r <= 1e-12:
            break
    return f


def fail(reason):
    print("Reason: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    itok = read_tokens(in_path)
    n = int(itok[0])
    S = int(itok[1])
    cap = [int(x) for x in itok[2:2 + n]]
    if len(cap) != n:
        fail("bad instance")

    otok = read_tokens(out_path)
    if len(otok) != n:
        fail("expected %d densities, got %d" % (n, len(otok)))
    try:
        f = [float(x) for x in otok]
    except ValueError:
        fail("non-numeric density")

    tol = 1e-5 * max(1.0, S) + 1e-6
    for i in range(n):
        v = f[i]
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite density at pool %d" % i)
        if v < -1e-9:
            fail("negative density at pool %d" % i)
        if v > cap[i] + 1e-9:
            fail("density %.6g exceeds capacity %d at pool %d" % (v, cap[i], i))
    if abs(sum(f) - S) > tol:
        fail("total population %.9g != required %d" % (sum(f), S))

    F = c1(f, n)
    if F is None or F <= 0:
        fail("degenerate density (zero total)")

    # checker-built baseline
    B = c1(fill_left(cap, S), n)

    sc = 100.0 * B / max(1e-9, F)
    if sc > 1000.0:
        sc = 1000.0
    print("c1=%.6f baseline=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
