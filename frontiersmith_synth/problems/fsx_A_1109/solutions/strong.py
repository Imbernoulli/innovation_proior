# TIER: strong
# Fill-aware scheduling: the meter charges CURRENT row nonzeros, and every rotation
# fills rows i,j with the union of their patterns -- so the real price of a pivot is
# its immediate cost PLUS the nonzeros it creates (a tax on every future pivot that
# touches those rows).  Score each candidate by energy-killed per effective cost
#     gain = A[i][j]^2          (E drops by exactly p^2 per rotation)
#     eff  = (nnz_i + nnz_j) + LAM * |pattern_i XOR pattern_j|
# and take the best ratio.  This deliberately prefers SMALLER pivots on sparse rows,
# inverting the textbook largest-pivot rule, and defers the big hub entries whose
# rotations would flood their partner rows with fill.
import sys
import math

LAM = 2.0


def rotate(A, i, j):
    p = A[i][j]
    if p == 0.0:
        return
    a = A[i][i]
    d = A[j][j]
    tau = (d - a) / (2.0 * p)
    if tau == 0.0:
        t = 1.0
    elif tau > 0.0:
        t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
    else:
        t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
    c = 1.0 / math.sqrt(1.0 + t * t)
    s = t * c
    n = len(A)
    Ai = A[i]
    Aj = A[j]
    for k in range(n):
        if k == i or k == j:
            continue
        aik = Ai[k]
        ajk = Aj[k]
        if aik == 0.0 and ajk == 0.0:
            continue
        ni = c * aik - s * ajk
        nj = s * aik + c * ajk
        Ai[k] = A[k][i] = ni
        Aj[k] = A[k][j] = nj
    A[i][i] = a - t * p
    A[j][j] = d + t * p
    Ai[j] = 0.0
    Aj[i] = 0.0


def main():
    data = sys.stdin.read().split()
    pos = 0
    n = int(data[pos]); B = int(data[pos + 1]); m = int(data[pos + 2]); pos += 3
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = float(int(data[pos + i]))
    pos += n
    for _ in range(m):
        i = int(data[pos]) - 1
        j = int(data[pos + 1]) - 1
        v = int(data[pos + 2])
        pos += 3
        A[i][j] = float(v)
        A[j][i] = float(v)

    plan = []
    spent = 0
    while True:
        pats = [set(k for k in range(n) if A[r][k] != 0.0) for r in range(n)]
        best = None
        bestsc = 0.0
        bestc = 0
        for i in range(n):
            Ai = A[i]
            si = pats[i]
            li = len(si)
            for j in range(i + 1, n):
                v = Ai[j]
                if v == 0.0:
                    continue
                sj = pats[j]
                c = li + len(sj)
                if spent + c > B:
                    continue
                fill = len(si ^ sj)
                eff = c + LAM * fill
                sc = (v * v) / eff
                if sc > bestsc:
                    bestsc = sc
                    best = (i, j)
                    bestc = c
        if best is None:
            break
        rotate(A, best[0], best[1])
        spent += bestc
        plan.append(best)

    out = [str(len(plan))]
    for (i, j) in plan:
        out.append("%d %d" % (i + 1, j + 1))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
