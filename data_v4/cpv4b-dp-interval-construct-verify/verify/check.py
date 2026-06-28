#!/usr/bin/env python3
"""
Property verifier used by the oracle harness.

Usage: python3 check.py <n> < output_of_solution

A solution's output is ACCEPTED iff it is a valid answer for the given n:
  - exactly n integers,
  - strictly increasing,
  - first equals 0,
  - last <= M = 8*n*n,
  - all pairwise differences distinct (Sidon property).

Exit code 0 = accepted, 1 = rejected (with a reason on stderr).
"""
import sys

def main():
    n = int(sys.argv[1])
    toks = sys.stdin.read().split()
    if len(toks) != n:
        sys.stderr.write(f"expected {n} integers, got {len(toks)}\n"); sys.exit(1)
    x = list(map(int, toks))
    M = 8 * n * n
    if x[0] != 0:
        sys.stderr.write(f"first mark must be 0, got {x[0]}\n"); sys.exit(1)
    for i in range(1, n):
        if x[i] <= x[i-1]:
            sys.stderr.write(f"not strictly increasing at {i}: {x[i-1]} >= {x[i]}\n"); sys.exit(1)
    if x[-1] > M:
        sys.stderr.write(f"last mark {x[-1]} exceeds M={M}\n"); sys.exit(1)
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            d = x[j] - x[i]
            if d in seen:
                sys.stderr.write(f"duplicate difference {d}\n"); sys.exit(1)
            seen.add(d)
    sys.exit(0)

if __name__ == "__main__":
    main()
