#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for masked bilinear-rank-completion.

Feasibility: the submitted m x m matrix over F_p must reproduce every revealed
input entry exactly (finite, integer, in range).
Objective (minimize): the exact rank over F_p of the completed matrix, computed
by Gaussian elimination modulo the prime P given in the input. Baseline B is
the rank of the checker's own zero-fill of the hidden cells.
"""
import sys, re

INT_RE = re.compile(r"^[+-]?\d+$")


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def modinv(a, p):
    return pow(a % p, p - 2, p)


def rank_mod_p(mat, p, m):
    A = [row[:] for row in mat]
    rank = 0
    row = 0
    for col in range(m):
        piv = None
        for r in range(row, m):
            if A[r][col] % p != 0:
                piv = r
                break
        if piv is None:
            continue
        A[row], A[piv] = A[piv], A[row]
        inv = modinv(A[row][col], p)
        A[row] = [(x * inv) % p for x in A[row]]
        for r in range(m):
            if r != row and A[r][col] % p != 0:
                f = A[r][col]
                A[r] = [(A[r][c] - f * A[row][c]) % p for c in range(m)]
        row += 1
        rank += 1
        if row == m:
            break
    return rank


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        in_tokens = f.read().split()
    if len(in_tokens) < 2:
        fail("bad input file")
    m = int(in_tokens[0])
    p = int(in_tokens[1])
    grid_tokens = in_tokens[2:]
    if len(grid_tokens) != m * m:
        fail("bad input grid size")
    revealed = [[False] * m for _ in range(m)]
    input_val = [[0] * m for _ in range(m)]
    idx = 0
    for i in range(m):
        for j in range(m):
            tok = grid_tokens[idx]; idx += 1
            if tok == "?":
                revealed[i][j] = False
            else:
                if not INT_RE.match(tok):
                    fail("corrupt input token")
                revealed[i][j] = True
                input_val[i][j] = int(tok) % p

    try:
        with open(out_path) as f:
            out_text = f.read()
    except Exception:
        fail("cannot read output")
    out_tokens = out_text.split()
    if len(out_tokens) != m * m:
        fail(f"expected {m*m} integers, got {len(out_tokens)}")

    submitted = [[0] * m for _ in range(m)]
    idx = 0
    for i in range(m):
        for j in range(m):
            tok = out_tokens[idx]; idx += 1
            if not INT_RE.match(tok):
                fail(f"non-integer / non-finite token {tok!r} at ({i},{j})")
            v = int(tok)
            if v < 0 or v > p - 1:
                fail(f"value {v} out of range [0,{p-1}] at ({i},{j})")
            submitted[i][j] = v

    for i in range(m):
        for j in range(m):
            if revealed[i][j] and submitted[i][j] != input_val[i][j]:
                fail(f"mismatch at revealed cell ({i},{j}): expected {input_val[i][j]}, got {submitted[i][j]}")

    F = rank_mod_p(submitted, p, m)

    zero_fill = [[input_val[i][j] if revealed[i][j] else 0 for j in range(m)] for i in range(m)]
    B = rank_mod_p(zero_fill, p, m)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"F(rank)={F} B(baseline rank)={B}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
