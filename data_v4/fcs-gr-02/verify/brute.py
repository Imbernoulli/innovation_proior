#!/usr/bin/env python3
# Independent brute force: try all 2^n assignments (n <= 20).
# Outputs only the decision line: "YES" or "NO".
# (Assignment correctness of the fast solver is checked separately by check.py,
#  because 2-SAT solutions are not unique.)
import sys
from itertools import product

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        return
    clauses = []
    for _ in range(m):
        i = int(next(it)); a = int(next(it))
        j = int(next(it)); b = int(next(it))
        clauses.append((i, a, j, b))

    # Try every assignment of the n boolean variables.
    sat = False
    for assign in product((0, 1), repeat=n):
        ok = True
        for (i, a, j, b) in clauses:
            # clause satisfied if (assign[i]==a) or (assign[j]==b)
            if (assign[i] == a) or (assign[j] == b):
                continue
            ok = False
            break
        if ok:
            sat = True
            break
    print("YES" if sat else "NO")

if __name__ == "__main__":
    main()
