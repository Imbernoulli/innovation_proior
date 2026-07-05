#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   (ans is an empty placeholder -- ignored)

Deterministic scorer for the near-orthogonal Latin squares construction problem.

Instance (<in>):  first line "n k".
Artifact (<out>): k*n*n integers = k Latin squares of order n, given as consecutive
                  blocks of n rows x n columns (block m = the m-th square).

Feasibility (any violation -> `Ratio: 0.0`):
  * exactly k*n*n integer tokens, every token a finite integer;
  * every entry in [0, n-1];
  * every one of the k squares is a valid Latin square (each row and each column is a
    permutation of {0,...,n-1}).

Objective (maximize):
  F = sum over all C(k,2) unordered pairs (p<q) of  D(L_p, L_q)
  where D(A,B) = number of DISTINCT ordered pairs (A[i][j], B[i][j]) over all cells.
  F = C(k,2)*n^2  iff the set is mutually orthogonal (unreachable for these orders).

Scoring:  internal baseline B = F of the cyclic construction the checker builds itself.
          sc = min(1000, 100 * F / max(1e-9, B));  print Ratio: sc/1000.
          (trivial cyclic submission -> 0.1; a 10x-better set would cap at 1.0.)
"""
import sys
from math import gcd


def _fail(reason):
    sys.stdout.write("infeasible (%s)  Ratio: 0.0\n" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    n = int(toks[0]); k = int(toks[1])
    return n, k


def parse_squares(path, n, k):
    """Strictly parse exactly k*n*n finite integers -> list of k n-by-n squares."""
    with open(path) as f:
        raw = f.read().split()
    need = k * n * n
    if len(raw) != need:
        _fail("expected %d integer tokens, got %d" % (need, len(raw)))
    vals = []
    for tok in raw:
        # reject non-finite / non-integer tokens (nan, inf, floats, garbage)
        low = tok.lower()
        if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            _fail("non-finite token")
        try:
            v = int(tok)
        except ValueError:
            _fail("non-integer token '%s'" % tok[:16])
        vals.append(v)
    squares = []
    idx = 0
    for _m in range(k):
        sq = []
        for _i in range(n):
            row = vals[idx:idx + n]; idx += n
            sq.append(row)
        squares.append(sq)
    return squares


def is_latin(sq, n):
    full = set(range(n))
    for row in sq:
        if len(row) != n:
            return False
        for v in row:
            if v < 0 or v >= n:
                return False
        if set(row) != full:
            return False
    for c in range(n):
        col = [sq[r][c] for r in range(n)]
        if set(col) != full:
            return False
    return True


def distinct_pairs(A, B, n):
    seen = set()
    for i in range(n):
        Ai = A[i]; Bi = B[i]
        for j in range(n):
            seen.add((Ai[j], Bi[j]))
    return len(seen)


def total_F(squares, n, k):
    F = 0
    for p in range(k):
        for q in range(p + 1, k):
            F += distinct_pairs(squares[p], squares[q], n)
    return F


def build_baseline(n, k):
    """Cyclic construction: k valid, distinct Latin squares.
    L_m[i][j] = (a_m * i + j + s_m) mod n, with a_m a residue coprime to n (so columns
    are permutations) and s_m an additive shift used when k exceeds the count of such
    residues.  For the chosen orders every coprime residue is EVEN, so no pair of these
    squares is orthogonal => the baseline is a genuine, beatable starting point."""
    coprimes = [a for a in range(1, n) if gcd(a, n) == 1]
    L = len(coprimes)
    squares = []
    for m in range(k):
        a = coprimes[m % L]
        s = m // L
        sq = [[(a * i + j + s) % n for j in range(n)] for i in range(n)]
        squares.append(sq)
    return squares


def main():
    if len(sys.argv) < 3:
        sys.stdout.write("usage error  Ratio: 0.0\n")
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, k = read_instance(in_path)

    squares = parse_squares(out_path, n, k)
    for m in range(k):
        if not is_latin(squares[m], n):
            _fail("square %d is not a Latin square" % m)

    F = total_F(squares, n, k)
    B = total_F(build_baseline(n, k), n, k)
    B = max(1e-9, float(B))

    sc = min(1000.0, 100.0 * F / B)
    sys.stdout.write("F=%d B=%d n=%d k=%d  Ratio: %.6f\n" % (F, int(B), n, k, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
