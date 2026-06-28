#!/usr/bin/env python3
# Usage: check.py <input_file> <sol_output_file> <brute_decision: YES|NO>
# Validates the fast solver's output against the brute-force decision AND,
# when YES, checks that the returned assignment actually satisfies every clause.
import sys

def main():
    inp = open(sys.argv[1]).read().split()
    sol_lines = open(sys.argv[2]).read().split('\n')
    brute_dec = sys.argv[3].strip()

    it = iter(inp)
    n = int(next(it)); m = int(next(it))
    clauses = []
    for _ in range(m):
        i = int(next(it)); a = int(next(it))
        j = int(next(it)); b = int(next(it))
        clauses.append((i, a, j, b))

    # strip empty trailing lines
    lines = [l for l in sol_lines if l.strip() != ""]
    if not lines:
        # only valid if there is genuinely no output expected; treat as mismatch
        print("MISMATCH: empty solver output")
        sys.exit(1)
    dec = lines[0].strip()

    if dec != brute_dec:
        print(f"MISMATCH: decision sol={dec} brute={brute_dec}")
        sys.exit(1)

    if dec == "NO":
        sys.exit(0)

    # dec == YES: parse assignment
    if n == 0:
        sys.exit(0)
    if len(lines) < 2:
        print("MISMATCH: YES but no assignment line")
        sys.exit(1)
    vals = lines[1].split()
    if len(vals) != n:
        print(f"MISMATCH: assignment has {len(vals)} values, expected {n}")
        sys.exit(1)
    try:
        assign = [int(v) for v in vals]
    except ValueError:
        print("MISMATCH: non-integer assignment")
        sys.exit(1)
    for v in assign:
        if v not in (0, 1):
            print(f"MISMATCH: assignment value {v} not in {{0,1}}")
            sys.exit(1)
    for (i, a, j, b) in clauses:
        if (assign[i] == a) or (assign[j] == b):
            continue
        print(f"MISMATCH: clause ({i},{a})|({j},{b}) unsatisfied by assignment")
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
