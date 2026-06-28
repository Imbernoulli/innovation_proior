#!/usr/bin/env python3
"""Independent brute force / checker for the "aggressive transmitters" problem.

Usage:
    python3 brute.py <input_file> <solver_output_file>

It recomputes the canonical answer D* (the maximum achievable minimum pairwise
gap when choosing k of the n candidate positions) by exhaustively enumerating
all C(n, k) subsets -- a method with no shared logic with the binary-search
solution. It then checks that the solver's reported D equals D*, and that the
solver's printed witness placement is a genuine set of k distinct candidate
positions whose minimum pairwise gap equals D*.

Prints "OK" on success, or "MISMATCH: <reason>" on failure (exit code 1).
"""
import sys
from itertools import combinations


def solve_bruteforce(n, k, pos):
    best = -1
    # All pairwise gaps are minimised by adjacent elements once sorted, so for a
    # chosen subset the min gap is the min of consecutive differences in sorted
    # order. We still enumerate every size-k subset to stay obviously correct.
    for combo in combinations(sorted(pos), k):
        mn = min(combo[i + 1] - combo[i] for i in range(k - 1))
        if mn > best:
            best = mn
    return best


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    idx = 0
    n = int(inp[idx]); idx += 1
    k = int(inp[idx]); idx += 1
    pos = [int(inp[idx + i]) for i in range(n)]
    idx += n

    star = solve_bruteforce(n, k, pos)

    if len(out) < 1:
        print("MISMATCH: empty output")
        sys.exit(1)
    reported_D = int(out[0])
    if reported_D != star:
        print(f"MISMATCH: reported D={reported_D} but D*={star}")
        sys.exit(1)

    witness = [int(x) for x in out[1:1 + k]]
    if len(witness) != k:
        print(f"MISMATCH: witness has {len(witness)} positions, expected k={k}")
        sys.exit(1)
    if len(set(witness)) != k:
        print("MISMATCH: witness positions not distinct")
        sys.exit(1)

    from collections import Counter
    avail = Counter(pos)
    for w in witness:
        if avail[w] <= 0:
            print(f"MISMATCH: witness position {w} not an available candidate")
            sys.exit(1)
        avail[w] -= 1

    w_sorted = sorted(witness)
    mn = min(w_sorted[i + 1] - w_sorted[i] for i in range(k - 1))
    if mn != star:
        print(f"MISMATCH: witness min gap={mn} but D*={star}")
        sys.exit(1)

    print("OK")


if __name__ == "__main__":
    main()
