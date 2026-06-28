#!/usr/bin/env python3
# Independent brute force oracle for "k-th smallest in [l, r]" (no updates).
# Obviously-correct method: for each query, copy the subarray a[l-1 .. r-1],
# sort it, and read off the (k-1)-th element (0-based). O(q * n log n).
import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1
    out = []
    for _ in range(q):
        l = int(data[idx]); idx += 1
        r = int(data[idx]); idx += 1
        k = int(data[idx]); idx += 1
        sub = sorted(a[l - 1:r])
        out.append(str(sub[k - 1]))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
