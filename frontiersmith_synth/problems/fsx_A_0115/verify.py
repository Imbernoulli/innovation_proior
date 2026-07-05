#!/usr/bin/env python3
# verify.py <in> <out> <ans>   -- deterministic scorer for the watchtower overlap constant.
# Exact integer arithmetic; prints "Ratio: <x in [0,1]>".  ans is ignored.
import sys

def read_ints(path):
    with open(path) as fh:
        return fh.read().split()

def max_selfconv(f, n):
    """max_j (f*f)[j], the self-convolution (f*f)[j] = sum_i f[i]*f[j-i],
    j = 0..2n-2.  Pure-integer O(n^2)."""
    best = 0
    for j in range(2 * n - 1):
        s = 0
        lo = max(0, j - n + 1)
        hi = min(j, n - 1)
        for i in range(lo, hi + 1):
            s += f[i] * f[j - i]
        if s > best:
            best = s
    return best

def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)

def main():
    inf, outf = sys.argv[1], sys.argv[2]

    itok = read_ints(inf)
    if len(itok) < 2:
        fail("bad instance")
    n = int(itok[0]); U = int(itok[1])

    toks = read_ints(outf)
    if len(toks) != n:
        fail("expected %d integers, got %d" % (n, len(toks)))
    f = []
    for tk in toks:
        try:
            v = int(tk)
        except ValueError:
            fail("non-integer output")
        if v < 0 or v > U:
            fail("value %d out of [0,%d]" % (v, U))
        f.append(v)

    S = sum(f)
    if S <= 0:
        fail("sum must be strictly positive")

    M = max_selfconv(f, n)          # = max_j (f*f)[j] > 0 since S > 0
    # objective c = 2*n*M/S^2  (reported for transparency); score uses val = 2/c = S^2/(n*M)
    c = 2.0 * n * M / (S * S)
    val = (S * S) / (n * M)         # exact ratio of integers, > 0
    ratio = 0.1 * (val ** 6)
    if ratio > 1.0:
        ratio = 1.0

    # internal baseline B: flat profile f=(1,..,1) -> M=n, S=n, c=2.0 (documented, sanity only)
    print("overlap_c=%.6f val=%.6f baseline_c=2.0 Ratio: %.6f" % (c, val, ratio))

if __name__ == "__main__":
    main()
