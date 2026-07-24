# TIER: greedy
# Classical largest-pivot Jacobi: always grind the biggest remaining off-diagonal
# entry whose current cost still fits the budget.  Ignores the fill-in tax entirely.
import sys
import math


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


def nnz(row):
    c = 0
    for v in row:
        if v != 0.0:
            c += 1
    return c


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
        best = None
        bestv = 0.0
        bestc = 0
        for i in range(n):
            Ai = A[i]
            for j in range(i + 1, n):
                v = Ai[j]
                if v != 0.0 and abs(v) > bestv:
                    c = nnz(A[i]) + nnz(A[j])
                    if spent + c <= B:
                        best = (i, j)
                        bestv = abs(v)
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
