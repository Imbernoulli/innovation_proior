# TIER: greedy
"""The obvious first attempt: the statement says the tapestry is woven from
"a few" thread patterns, so just assume the simplest common guess (rank 3),
find the fully-intact rows, and solve a single 3x3 linear system per row using
whichever 3 columns it happens to share with those rows -- no verification
that 3 is actually the right count, and no cross-check against the row's
OTHER revealed entries.

Whenever the true rank of a given tapestry is not 3, this single-shot
assumption is either unsolvable (over-guessing collapses the reference rows'
own row space) or solvable-but-wrong (under-guessing lands in the wrong
subspace) -- either way, every torn cell in every ordinary row gets filled in
with values that generically do not fit the true pattern at all, so the
completed tapestry ends up just as fragmented (full rank) as leaving it
blank."""
import sys

RANK_GUESS = 3


def modinv(a, p):
    return pow(a % p, p - 2, p)


def solve_linear(A, b, n, p):
    M = [A[i][:] + [b[i]] for i in range(n)]
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, n):
            if M[r][col] % p != 0:
                piv = r; break
        if piv is None:
            return None
        M[row], M[piv] = M[piv], M[row]
        inv = modinv(M[row][col], p)
        M[row] = [(x * inv) % p for x in M[row]]
        for r in range(n):
            if r != row and M[r][col] % p != 0:
                f = M[r][col]
                M[r] = [(M[r][c] - f * M[row][c]) % p for c in range(n + 1)]
        row += 1
    if row < n:
        return None
    return [M[i][n] for i in range(n)]


def main():
    data = sys.stdin.read().split()
    m = int(data[0]); p = int(data[1])
    tokens = data[2:2 + m * m]
    grid = [tokens[i * m:(i + 1) * m] for i in range(m)]
    revealed = [[grid[i][j] != "?" for j in range(m)] for i in range(m)]
    val = [[int(grid[i][j]) if revealed[i][j] else 0 for j in range(m)] for i in range(m)]

    ref_rows = [i for i in range(m) if all(revealed[i])]
    r = RANK_GUESS
    basis = ref_rows[:r]

    out = [[None] * m for _ in range(m)]
    if len(basis) < r:
        basis = []
    A_full = [[val[b][j] for b in basis] for j in range(m)] if basis else None

    for i in range(m):
        if i in basis:
            for j in range(m):
                out[i][j] = val[i][j] if revealed[i][j] else 0
            continue
        c = None
        if basis:
            revealed_cols = [j for j in range(m) if revealed[i][j]]
            if len(revealed_cols) >= r:
                cols_used = revealed_cols[:r]  # first r only, no robustness check
                A = [A_full[j] for j in cols_used]
                bb = [val[i][j] for j in cols_used]
                c = solve_linear(A, bb, r, p)
        for j in range(m):
            if revealed[i][j]:
                out[i][j] = val[i][j]
            elif c is not None:
                out[i][j] = sum(c[k] * A_full[j][k] for k in range(r)) % p
            else:
                out[i][j] = 0

    sys.stdout.write("\n".join(" ".join(str(x) for x in row) for row in out) + "\n")


if __name__ == "__main__":
    main()
