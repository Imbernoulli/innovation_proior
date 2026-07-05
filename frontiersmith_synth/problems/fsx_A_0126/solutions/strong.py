# TIER: strong
# Float-guided single-flip hill climbing on |det| with Sherman-Morrison inverse
# maintenance.  From many seeded invertible starts, repeatedly flip the free cell
# whose flip most multiplies |det|; keep the best grid.  Honors all cemented cells.
import sys, random, math


def inverse_and_logdet(M, n):
    """Gauss-Jordan inverse + log|det|.  Returns (inv, logdet) or (None, -inf)."""
    A = [row[:] for row in M]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ld = 0.0
    for k in range(n):
        piv = k
        best = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > best:
                best = abs(A[i][k]); piv = i
        if best < 1e-9:
            return None, -float("inf")
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            I[k], I[piv] = I[piv], I[k]
        akk = A[k][k]
        ld += math.log(abs(akk))
        invp = 1.0 / akk
        ak = A[k]; ik = I[k]
        for j in range(n):
            ak[j] *= invp
            ik[j] *= invp
        for i in range(n):
            if i == k:
                continue
            f = A[i][k]
            if f != 0.0:
                ai = A[i]; ii = I[i]
                for j in range(n):
                    ai[j] -= f * ak[j]
                    ii[j] -= f * ik[j]
    return I, ld


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    fixed = {}
    for _ in range(K):
        r = int(data[idx]); c = int(data[idx + 1]); v = int(data[idx + 2]); idx += 3
        fixed[(r, c)] = v

    rng = random.Random(7 + N)
    free = [(i, j) for i in range(N) for j in range(N) if (i, j) not in fixed]

    best = None
    best_ld = -float("inf")
    RESTARTS = 30
    STEPS = 3 * N

    for _ in range(RESTARTS):
        # build an invertible feasible start
        M = None
        for _try in range(12):
            cand = [[0] * N for _ in range(N)]
            for (r, c), v in fixed.items():
                cand[r][c] = v
            for (i, j) in free:
                cand[i][j] = rng.randint(0, 1)
            inv, ld = inverse_and_logdet(cand, N)
            if inv is not None:
                M = cand
                Ainv = inv
                logdet = ld
                break
        if M is None:
            continue

        for _step in range(STEPS):
            bi = bj = -1
            bfac = 1.0 + 1e-9
            bdelta = 0
            for (i, j) in free:
                delta = 1 - 2 * M[i][j]         # 0->1 : +1 ; 1->0 : -1
                f = 1.0 + delta * Ainv[j][i]    # det multiplier
                af = abs(f)
                if af > bfac:
                    bfac = af; bi = i; bj = j; bdelta = delta
            if bi < 0:
                break
            # apply flip + Sherman-Morrison inverse update
            factor = 1.0 + bdelta * Ainv[bj][bi]
            col_i = [Ainv[p][bi] for p in range(N)]
            row_j = Ainv[bj][:]
            coef = bdelta / factor
            for p in range(N):
                cip = col_i[p] * coef
                if cip != 0.0:
                    Ap = Ainv[p]
                    for q in range(N):
                        Ap[q] -= cip * row_j[q]
            M[bi][bj] ^= 1
            logdet += math.log(abs(factor))

        if logdet > best_ld:
            best_ld = logdet
            best = [row[:] for row in M]

    if best is None:
        best = [[0] * N for _ in range(N)]
        for (r, c), v in fixed.items():
            best[r][c] = v
        for i in range(N):
            if (i, i) not in fixed:
                best[i][i] = 1

    out = []
    for i in range(N):
        out.append(" ".join(str(x) for x in best[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
