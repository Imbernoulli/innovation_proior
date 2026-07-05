#!/usr/bin/env python3
"""Deterministic checker for the mountain-rescue-relay additive-combinatorics problem.

Usage:  python3 verify.py <in> <out> <ans>     (ans is an ignored placeholder)

Instance (<in>):  a line  "n V".
Artifact (<out>): n distinct integers in [0, V] (whitespace-separated) = beacon altitudes A.

Objective (MAXIMIZE):  rho(A) = |A + A| / |A - A|
    A+A = { a+b : a,b in A }   (distinct pairwise sums   = "handshake codes")
    A-A = { a-b : a,b in A }   (distinct pairwise gaps    = "altitude gaps")
This is the classic "more sums than differences" (MSTD) exponent -- a genuinely OPEN
extremal problem: the supremum of rho over integer sets is not known and there is no
polynomial optimal construction. A generic (difference-spread / Sidon) layout gives
rho < 1; only carefully engineered sets exceed 1.

Scoring is exact rational arithmetic. The internal baseline B is the ratio of a
deterministic Sidon (Mian-Chowla) layout of the same size n -- a trivial, strongly
difference-heavy construction (B ~ 0.5). Maximization normalization:
    sc = min(1000, 100 * F / max(1e-9, B));  Ratio = sc / 1000.
So reproducing the Sidon baseline scores ~0.1 and a 10x-better ratio would cap at 1.0
(unreachable -- rho is bounded well below 10*B, leaving ample headroom).
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    n = int(toks[0])
    V = int(toks[1])
    return n, V


def parse_output(path, n, V):
    """Strictly parse the participant artifact. Returns (A_list, None) on success or
    (None, reason) on any feasibility violation."""
    try:
        with open(path) as f:
            raw = f.read()
    except Exception as e:
        return None, "unreadable output (%s)" % e
    toks = raw.split()
    if len(toks) != n:
        return None, "expected exactly %d integer tokens, got %d" % (n, len(toks))
    A = []
    for tk in toks:
        # strict integer parse: rejects floats, 'nan', 'inf', hex, empty, etc.
        try:
            v = int(tk)
        except (ValueError, TypeError):
            return None, "token '%s' is not an integer" % tk[:32]
        if v < 0 or v > V:
            return None, "altitude %d out of range [0,%d]" % (v, V)
        A.append(v)
    if len(set(A)) != n:
        return None, "altitudes are not all distinct"
    return A, None


def rho(A):
    """Exact |A+A| / |A-A| as a Python float from integer set cardinalities."""
    s = set()
    d = set()
    for a in A:
        for b in A:
            s.add(a + b)
            d.add(a - b)
    return len(s) / len(d), len(s), len(d)


def mian_chowla(n):
    """Deterministic greedy Sidon set of size n (all pairwise differences distinct)."""
    A = [0]
    diffs = set()
    x = 1
    while len(A) < n:
        ok = True
        nd = []
        for a in A:
            dd = x - a
            if dd in diffs:
                ok = False
                break
            nd.append(dd)
        if ok:
            A.append(x)
            diffs.update(nd)
        x += 1
    return A


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad invocation)")
        return
    inf, outf = sys.argv[1], sys.argv[2]
    n, V = read_instance(inf)

    A, reason = parse_output(outf, n, V)
    if A is None:
        print("Infeasible: %s" % reason)
        print("Ratio: 0.0")
        return

    F, nsum, ndiff = rho(A)

    # internal deterministic baseline: Sidon layout ratio (~0.5, difference-heavy)
    B_set = mian_chowla(n)
    Bval, _, _ = rho(B_set)

    sc = 100.0 * F / max(1e-9, Bval)
    if sc > 1000.0:
        sc = 1000.0
    ratio = sc / 1000.0
    print("n=%d |A+A|=%d |A-A|=%d rho=%.6f baseline=%.6f" % (n, nsum, ndiff, F, Bval))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
