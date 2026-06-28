#!/usr/bin/env python3
"""
Independent checker for the "Frequency Gap Planning" (Sidon set) problem.

The answer is NOT unique, so this brute force does not reproduce a single fixed
output. Instead it is an obviously-correct VALIDATOR of the stated property:

Given n on stdin (via argv[1] file) and a candidate output (n integers), accept
iff:
  * exactly n integers are printed,
  * all are distinct,
  * all lie in [0, 4*n*n],
  * the set is a Sidon set: all positive pairwise differences are distinct
    (equivalently all unordered pairwise sums are distinct).

Additionally, for small n it confirms (by exhaustive search) that a valid set of
size n within [0, 4*n*n] actually EXISTS, so a correct solver must produce one.

Usage:
  python3 brute.py <input_file> <output_file>
Prints "OK" and exits 0 on accept; prints a reason and exits 1 on reject.
"""
import sys

def read_ints(path):
    with open(path) as f:
        return f.read().split()

def is_sidon(vals):
    seen = set()
    m = len(vals)
    for i in range(m):
        for j in range(i + 1, m):
            d = vals[j] - vals[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def exists_small(n, L):
    # Backtracking existence check for tiny n: can we place n strictly increasing
    # values in [0, L] with all pairwise differences distinct? (Used only n<=9.)
    used_diffs = set()
    chosen = []

    def bt(start):
        if len(chosen) == n:
            return True
        for v in range(start, L + 1):
            ok = True
            newd = []
            for c in chosen:
                d = v - c
                if d in used_diffs:
                    ok = False
                    break
                newd.append(d)
            if ok:
                # also ensure the new diffs are internally distinct (they are, c distinct)
                for d in newd:
                    used_diffs.add(d)
                chosen.append(v)
                if bt(v + 1):
                    return True
                chosen.pop()
                for d in newd:
                    used_diffs.discard(d)
        return False

    return bt(0)

def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    n = int(read_ints(in_path)[0])
    L = 4 * n * n

    toks = read_ints(out_path)
    if len(toks) != n:
        print(f"WRONG: expected {n} integers, got {len(toks)}")
        sys.exit(1)
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        print("WRONG: non-integer token")
        sys.exit(1)

    if len(set(vals)) != n:
        print("WRONG: values not distinct")
        sys.exit(1)
    for v in vals:
        if v < 0 or v > L:
            print(f"WRONG: value {v} out of range [0,{L}]")
            sys.exit(1)

    svals = sorted(vals)
    if not is_sidon(svals):
        print("WRONG: not a Sidon set (a pairwise gap repeats)")
        sys.exit(1)

    # For tiny n, double-check existence is even possible (sanity on the bound).
    if n <= 9 and not exists_small(n, L):
        # This would mean the bound L is too tight to admit ANY valid set;
        # that would be a problem-statement bug, surfaced here.
        print(f"BUG: no valid set of size {n} exists within [0,{L}]")
        sys.exit(1)

    print("OK")
    sys.exit(0)

if __name__ == "__main__":
    main()
