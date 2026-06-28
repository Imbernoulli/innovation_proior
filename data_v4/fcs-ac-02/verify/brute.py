#!/usr/bin/env python3
"""Independent oracle / checker for the 'graph with exactly k triangles' problem.

It does NOT reconstruct the same graph. Given the input k and the solution's output,
it validates the output as a graph and counts triangles directly (an O(n^3) /
common-neighbor method that is obviously correct), then asserts the count equals k.

Usage:
    brute.py <input_file> <output_file>
Prints "OK" on success, or "FAIL: <reason>" and exits non-zero on any problem.
"""
import sys


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    k = int(read_tokens(in_path)[0])

    out = read_tokens(out_path)
    NMAX = 1000

    # The problem guarantees a solution exists for 0 <= k <= 10**8, so a correct
    # solution must never print -1 in that range.
    if len(out) == 1 and out[0] == "-1":
        print("FAIL: printed -1 but a graph exists for this k")
        sys.exit(1)

    idx = 0
    try:
        n = int(out[idx]); idx += 1
        m = int(out[idx]); idx += 1
    except (IndexError, ValueError):
        print("FAIL: could not read n and m")
        sys.exit(1)

    if not (1 <= n <= NMAX):
        print(f"FAIL: n={n} out of range [1,{NMAX}]")
        sys.exit(1)
    if m < 0:
        print(f"FAIL: negative m={m}")
        sys.exit(1)

    edges = []
    seen = set()
    for _ in range(m):
        try:
            a = int(out[idx]); idx += 1
            b = int(out[idx]); idx += 1
        except (IndexError, ValueError):
            print("FAIL: not enough edge endpoints")
            sys.exit(1)
        if not (1 <= a <= n and 1 <= b <= n):
            print(f"FAIL: edge endpoint out of range: {a} {b}")
            sys.exit(1)
        if a == b:
            print(f"FAIL: self-loop at {a}")
            sys.exit(1)
        key = (min(a, b), max(a, b))
        if key in seen:
            print(f"FAIL: duplicate edge {a} {b}")
            sys.exit(1)
        seen.add(key)
        edges.append((a - 1, b - 1))

    if idx != len(out):
        print("FAIL: trailing tokens in output")
        sys.exit(1)

    # Count triangles directly via bitsets of neighbors; for each edge (u,v) the number
    # of common neighbors w with w>v is the triangles on that edge counted once.
    nbr = [0] * n  # bitmask of neighbors
    for a, b in edges:
        nbr[a] |= (1 << b)
        nbr[b] |= (1 << a)

    tri = 0
    for a in range(n):
        for b in range(a + 1, n):
            if nbr[a] & (1 << b):
                common = nbr[a] & nbr[b]
                # count set bits with index > b
                common >>= (b + 1)
                tri += bin(common).count("1")

    if tri != k:
        print(f"FAIL: graph has {tri} triangles, expected {k}")
        sys.exit(1)

    print("OK")


if __name__ == "__main__":
    main()
