# TIER: strong
"""Exploit the bilinear structure directly.

1. Detect the fully-intact reference rows (no '?' in them at all) -- these are
   spike-free by construction, so they give a trustworthy row-space basis for
   free, with no bootstrapping needed.
2. For each candidate rank r' = 1..len(reference rows), use the first r'
   reference rows as a basis. For every other row, solve the r' x r' linear
   system given by several DIFFERENT subsets of that row's revealed columns
   (finite-field "closure of the solution variety": every subset yields an
   exact algebraic certificate) and keep the certificate that agrees with the
   most of the row's OTHER revealed entries -- a stained/torn cell is exactly
   the cell that disagrees with the consensus fit, so detecting spikes and
   completing the row are the same computation.
3. Pick the r' whose overall consensus rate is highest (this is how the true
   rank is discovered -- a wrong r' can never even produce an invertible
   system from genuinely rank-r data, so consensus collapses to ~0).
4. Fill torn cells with the consensus bilinear prediction; revealed cells are
   always copied verbatim (mandatory for feasibility), including any stains.
"""
import sys
import itertools

NTRIALS = 20


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


def fit_rank(val, revealed, m, p, ref_rows, r):
    basis = ref_rows[:r]
    A_full = [[val[b][j] for b in basis] for j in range(m)]
    out = [[None] * m for _ in range(m)]
    total_c = 0
    total_n = 0
    for i in range(m):
        if i in basis:
            for j in range(m):
                out[i][j] = val[i][j] if revealed[i][j] else 0
            continue
        revealed_cols = [j for j in range(m) if revealed[i][j]]
        c = None
        if len(revealed_cols) >= r:
            combos = list(itertools.islice(itertools.combinations(revealed_cols, r), NTRIALS))
            best_c, best_match = None, -1
            for combo in combos:
                A = [A_full[j] for j in combo]
                bb = [val[i][j] for j in combo]
                cc = solve_linear(A, bb, r, p)
                if cc is None:
                    continue
                match = sum(1 for j in revealed_cols
                            if (sum(cc[k] * A_full[j][k] for k in range(r)) % p) == val[i][j])
                if match > best_match:
                    best_match, best_c = match, cc
            c = best_c
            total_c += max(best_match, 0)
            total_n += len(revealed_cols)
        for j in range(m):
            if revealed[i][j]:
                out[i][j] = val[i][j]
            elif c is not None:
                out[i][j] = sum(c[k] * A_full[j][k] for k in range(r)) % p
            else:
                out[i][j] = 0
    consensus = total_c / total_n if total_n else 0.0
    return out, consensus


def main():
    data = sys.stdin.read().split()
    m = int(data[0]); p = int(data[1])
    tokens = data[2:2 + m * m]
    grid = [tokens[i * m:(i + 1) * m] for i in range(m)]
    revealed = [[grid[i][j] != "?" for j in range(m)] for i in range(m)]
    val = [[int(grid[i][j]) if revealed[i][j] else 0 for j in range(m)] for i in range(m)]

    ref_rows = [i for i in range(m) if all(revealed[i])]
    if not ref_rows:
        # No fully intact row survived: fall back to zero-fill (feasible, low quality).
        out = [[val[i][j] if revealed[i][j] else 0 for j in range(m)] for i in range(m)]
        sys.stdout.write("\n".join(" ".join(str(x) for x in row) for row in out) + "\n")
        return

    best = None
    for r in range(1, len(ref_rows) + 1):
        out, cons = fit_rank(val, revealed, m, p, ref_rows, r)
        if best is None or cons > best[0]:
            best = (cons, out)

    out = best[1]
    sys.stdout.write("\n".join(" ".join(str(x) for x in row) for row in out) + "\n")


if __name__ == "__main__":
    main()
