#!/usr/bin/env python3
"""counter.py <in> <out> <ans>   (ans ignored)

Format D (FLOPs / op-count) checker for tensor-decomposition-rank.

The participant submits a rank-R decomposition of the watchtower alert tensor T:
    line 1:  R
    next R lines:  a rationals (u) then b rationals (v) then c rationals (w)
                   (each rational is an integer "p" or a fraction "p/q")
representing  T_hat[i][j][k] = sum_r u_r[i] * v_r[j] * w_r[k].

Gate 1 (EXACT equivalence): reconstruct with exact rational arithmetic; any entry
that differs from T -> Ratio: 0.0.  Non-finite / malformed tokens -> Ratio: 0.0.
Gate 2 (op count): score = R (number of scalar-multiply primitives; fewer is better).

Baseline B (checker-internal, trivial feasible construction) = number of nonzero
mode-3 fibers of T  (one rank-1 term per nonzero fiber decomposes T exactly).
    ratio = min(1.0, 0.1 * B / R)      -> trivial reproduces B => 0.1 ; caps at 1.0.
"""
import sys
from fractions import Fraction


def die(msg):
    print(msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    with open(path, "r") as f:
        return f.read().split()


def parse_frac(tok):
    # exact rational only; reject nan/inf/exponent/garbage
    t = tok.strip()
    if t == "" or "e" in t.lower() or "n" in t.lower() or "i" in t.lower():
        raise ValueError("bad token")
    return Fraction(t)


def main():
    if len(sys.argv) < 3:
        die("usage: counter.py in out ans")
    in_toks = read_tokens(sys.argv[1])
    # ---- parse instance ----
    it = iter(in_toks)
    try:
        a = int(next(it)); b = int(next(it)); c = int(next(it))
    except StopIteration:
        die("bad instance header")
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    try:
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    T[i][j][k] = int(next(it))
    except StopIteration:
        die("truncated instance")

    # baseline B = # nonzero mode-3 fibers
    B = 0
    for i in range(a):
        for j in range(b):
            if any(T[i][j][k] != 0 for k in range(c)):
                B += 1
    if B <= 0:
        B = 1

    # ---- parse participant output ----
    out_toks = read_tokens(sys.argv[2])
    if not out_toks:
        die("empty output")
    try:
        R = int(out_toks[0])
    except ValueError:
        die("R not an integer")
    if R < 0:
        die("R negative")
    cap = 5 * a * b * c + 5
    if R > cap:
        die("R exceeds sanity cap")
    per = a + b + c
    need = 1 + R * per
    if len(out_toks) != need:
        die("token count %d != expected %d" % (len(out_toks), need))

    terms = []
    idx = 1
    try:
        for _ in range(R):
            u = [parse_frac(out_toks[idx + t]) for t in range(a)]; idx += a
            v = [parse_frac(out_toks[idx + t]) for t in range(b)]; idx += b
            w = [parse_frac(out_toks[idx + t]) for t in range(c)]; idx += c
            terms.append((u, v, w))
    except (ValueError, ZeroDivisionError):
        die("non-finite or malformed rational")

    # ---- Gate 1: exact reconstruction ----
    That = [[[Fraction(0) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    for (u, v, w) in terms:
        for i in range(a):
            ui = u[i]
            if ui == 0:
                continue
            for j in range(b):
                uv = ui * v[j]
                if uv == 0:
                    continue
                row = That[i][j]
                for k in range(c):
                    if w[k] != 0:
                        row[k] += uv * w[k]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if That[i][j][k] != T[i][j][k]:
                    die("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    # ---- Gate 2: op count ----
    F = R if R > 0 else 1
    sc = min(1.0, 0.1 * B / F)
    print("terms=%d baseline=%d" % (R, B))
    print("Ratio: %.6f" % sc)


if __name__ == "__main__":
    main()
