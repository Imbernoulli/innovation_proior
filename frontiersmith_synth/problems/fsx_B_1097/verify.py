#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for the subsequence-antichain layering problem.

Feasibility (ANY violation -> Ratio: 0.0):
  - output starts with an integer K, 0 <= K <= 100000, followed by exactly K tokens (strings)
  - each string has length in [1, Lmax] and uses only digit characters '0'..str(a-1)
  - the K strings are pairwise distinct
  - for each length l, at most cap[l] of the strings have that length
  - K <= T (global budget)
  - no string in the set is a scattered subsequence of another string in the set

Objective: F = sum over kept strings of weight[len(s)].  Internal baseline B is the
best a single-length-1 layer can score (which is also exactly what solutions/trivial.py
builds), so a submission that only reproduces that baseline lands at Ratio 0.1.
"""
import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def is_subseq(s, t):
    """Is s a scattered subsequence of t?"""
    it = iter(t)
    return all(ch in it for ch in s)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        a = int(next(it))
        Lmax = int(next(it))
        T = int(next(it))
        weight = [0] * (Lmax + 1)
        for l in range(1, Lmax + 1):
            weight[l] = int(next(it))
        cap = [0] * (Lmax + 1)
        for l in range(1, Lmax + 1):
            cap[l] = int(next(it))
    except Exception:
        fail("bad input")

    if a < 2 or a > 9 or Lmax < 1 or T < 0:
        fail("bad input ranges")

    valid_chars = set(str(d) for d in range(a))

    # ---- internal baseline B: the full length-1 layer (all `a` single-character strings) ----
    B = weight[1] * min(cap[1], a, T)
    B = max(1, B)

    # ---- parse participant output ----
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    if not raw:
        fail("empty output")
    try:
        k = int(raw[0])
    except Exception:
        fail("bad count token")
    if k < 0 or k > 100000:
        fail("count out of range")
    toks = raw[1:]
    if len(toks) != k:
        fail("count mismatch: header says %d, found %d strings" % (k, len(toks)))
    if k > T:
        fail("total count %d exceeds budget T=%d" % (k, T))

    seen = set()
    by_len_count = [0] * (Lmax + 1)
    strings = []
    for s in toks:
        if len(s) < 1 or len(s) > Lmax:
            fail("bad length %r" % s)
        for ch in s:
            if ch not in valid_chars:
                fail("bad character in %r" % s)
        if s in seen:
            fail("duplicate string %r" % s)
        seen.add(s)
        strings.append(s)
        by_len_count[len(s)] += 1

    for l in range(1, Lmax + 1):
        if by_len_count[l] > cap[l]:
            fail("length %d count %d exceeds cap %d" % (l, by_len_count[l], cap[l]))

    # ---- antichain (scattered-subsequence) check over all pairs ----
    order = sorted(range(len(strings)), key=lambda i: len(strings[i]))
    m = len(order)
    for ii in range(m):
        i = order[ii]
        si = strings[i]
        for jj in range(ii + 1, m):
            j = order[jj]
            sj = strings[j]
            if len(si) == len(sj):
                continue  # equal-length distinct strings can never be subsequences of each other
            if is_subseq(si, sj):
                fail("%r is a subsequence of %r" % (si, sj))

    F = sum(weight[len(s)] for s in strings)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d K=%d Ratio: %.6f" % (F, B, k, sc / 1000.0))


if __name__ == "__main__":
    main()
