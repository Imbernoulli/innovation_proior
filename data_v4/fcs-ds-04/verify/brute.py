#!/usr/bin/env python3
# Slow but obviously-correct oracle for the online range-rank / range-quantile
# problem. Reads the same stdin as sol.cpp, performs the IDENTICAL XOR-decoding
# of query parameters with the previous answer, and answers each query by a
# direct linear scan / sort over the slice a[l-1:r].
#
# Contract assumed (guaranteed by the problem statement after decoding):
#   values a[i] >= 0 ; for every query 1 <= l <= r <= n ;
#   for type-2 queries additionally 1 <= k <= r-l+1.
import sys

def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    def rd():
        nonlocal idx
        v = int(data[idx]); idx += 1
        return v

    n = rd(); q = rd()
    a = [rd() for _ in range(n)]

    out = []
    last = 0
    for _ in range(q):
        typ = rd()
        A = rd() ^ last
        B = rd() ^ last
        C = rd() ^ last
        if typ == 1:
            l, r, x = A, B, C
            ans = sum(1 for v in a[l - 1:r] if v <= x)
        else:
            l, r, k = A, B, C
            seg = sorted(a[l - 1:r])
            ans = seg[k - 1]
        out.append(str(ans))
        last = ans

    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
