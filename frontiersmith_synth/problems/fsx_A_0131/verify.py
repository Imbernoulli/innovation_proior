#!/usr/bin/env python3
"""Deterministic checker for the telescope-array cap-set problem (format C).

Usage: python3 verify.py <in> <out> <ans>   (ans is an empty placeholder)

Reads dimension n from <in> and the participant's antenna configuration from
<out>. Validates the cap-set feasibility strictly; on ANY violation prints
`Ratio: 0.0` (+ reason). Otherwise scores cardinality against the checker's own
diagonal baseline B = n + 1.
"""
import sys


def read_n(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return int(line.split()[0])
    raise ValueError("no n in input")


def parse_config(path, n):
    """Parse participant output -> list of vectors (tuples). Raises on format err."""
    vecs = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != n:
                raise ValueError("line has %d entries, expected %d" % (len(parts), n))
            v = []
            for p in parts:
                x = int(p)
                if x < 0 or x > 2:
                    raise ValueError("entry %d out of {0,1,2}" % x)
                v.append(x)
            vecs.append(tuple(v))
    return vecs


def has_line(vecs, n):
    """Return True iff some three distinct vectors x,y,z satisfy
    (x+y+z) % 3 == 0 componentwise. Early-exits on the first line found."""
    sset = set(vecs)
    L = vecs
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            c = tuple((-(a[k] + b[k])) % 3 for k in range(n))
            if c != a and c != b and c in sset:
                return True
    return False


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n = read_n(inf)

    # ---- checker's own trivial baseline: diagonal cap {0, e_1, ..., e_n} ----
    B = n + 1  # provably a valid cap set of size n+1 (positive baseline)

    try:
        vecs = parse_config(outf, n)
    except Exception as e:
        print("Ratio: 0.0 (bad output format: %s)" % e)
        return

    # feasibility: distinctness
    if len(set(vecs)) != len(vecs):
        print("Ratio: 0.0 (duplicate antenna vectors)")
        return

    # feasibility: no degenerate baseline triple
    if has_line(vecs, n):
        print("Ratio: 0.0 (degenerate baseline triple present)")
        return

    F = len(vecs)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
