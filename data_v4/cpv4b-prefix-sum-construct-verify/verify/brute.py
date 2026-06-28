#!/usr/bin/env python3
"""
Independent brute-force ORACLE / CHECKER for the
"distinct prefix sums mod n permutation" problem.

Reads the same stdin as sol (a single integer n on stdin), but it ALSO needs
the candidate output to grade it, so it is driven by the test harness:

    python3 brute.py <n>            -> prints the canonical verdict:
                                       "INFEASIBLE"  if no permutation exists,
                                       "FEASIBLE"    if at least one exists.

    python3 brute.py <n> --check    -> reads a candidate solution on stdin
                                       (the full stdout produced by sol for
                                        input n) and prints "OK" iff that
                                       candidate is a correct answer for n,
                                       else "BAD: <reason>".

Feasibility is decided INDEPENDENTLY of sol's construction:
exhaustive search over permutations for small n.
"""
import sys
from itertools import permutations


def distinct_prefix_mod(perm, n):
    """True iff the prefix sums of perm are pairwise distinct modulo n."""
    s = 0
    seen = set()
    for x in perm:
        s += x
        r = s % n
        if r in seen:
            return False
        seen.add(r)
    return True


def feasible_bruteforce(n):
    """Exhaustively decide whether a valid permutation exists (small n only)."""
    for perm in permutations(range(1, n + 1)):
        if distinct_prefix_mod(perm, n):
            return True
    return False


def main():
    n = int(sys.argv[1])
    check = (len(sys.argv) > 2 and sys.argv[2] == "--check")

    feasible = feasible_bruteforce(n)

    if not check:
        print("FEASIBLE" if feasible else "INFEASIBLE")
        return

    data = sys.stdin.read().split()

    if not feasible:
        # The only correct output is a single token "-1".
        if data == ["-1"]:
            print("OK")
        else:
            print("BAD: expected -1 for infeasible n, got " + repr(data[:8]))
        return

    # Feasible: candidate must be a permutation of 1..n with distinct prefix
    # sums mod n. (It must NOT be -1.)
    if data == ["-1"]:
        print("BAD: claimed infeasible but a permutation exists")
        return
    try:
        perm = [int(t) for t in data]
    except ValueError:
        print("BAD: non-integer token in output")
        return
    if len(perm) != n:
        print("BAD: expected %d numbers, got %d" % (n, len(perm)))
        return
    if sorted(perm) != list(range(1, n + 1)):
        print("BAD: not a permutation of 1..n")
        return
    if not distinct_prefix_mod(perm, n):
        print("BAD: prefix sums are not distinct mod n")
        return
    print("OK")


if __name__ == "__main__":
    main()
