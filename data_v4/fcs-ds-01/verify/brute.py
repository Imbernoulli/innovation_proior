#!/usr/bin/env python3
# Independent brute-force oracle for "offline range-distinct counting".
#
# Method: for each query (l, r) build a Python set of the slice a[l..r] and
# report its size. This is O(q * n) and obviously correct -- a set deduplicates
# by value, so its size is exactly the number of distinct values in the range.
# It is only used on small n, q in the differential tester.
#
# Input format (matches sol.cpp):
#   n q
#   a[0] a[1] ... a[n-1]
#   then q lines, each "l r" (0-indexed, inclusive).
import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    out = []
    for _ in range(q):
        l = int(next(it))
        r = int(next(it))
        out.append(str(len(set(a[l:r + 1]))))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
