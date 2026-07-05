# TIER: greedy
# Pick the cheaper unary orientation: row-wise vs column-wise. Since m < n, the
# column-wise construction usually needs fewer terms than the row-wise baseline.
import sys

def read_M():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    M = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    return n, m, M

def sgn(x):
    return (x > 0) - (x < 0)

def rowwise(n, m, M):
    terms = []
    for i in range(n):
        h = max((abs(M[i][j]) for j in range(m)), default=0)
        for l in range(1, h + 1):
            u = [0] * n; u[i] = 1
            v = [sgn(M[i][j]) if abs(M[i][j]) >= l else 0 for j in range(m)]
            terms.append((u, v))
    return terms

def colwise(n, m, M):
    terms = []
    for j in range(m):
        h = max((abs(M[i][j]) for i in range(n)), default=0)
        for l in range(1, h + 1):
            v = [0] * m; v[j] = 1
            u = [sgn(M[i][j]) if abs(M[i][j]) >= l else 0 for i in range(n)]
            terms.append((u, v))
    return terms

def emit(terms, n, m):
    if not terms:
        terms = [([1] + [0] * (n - 1), [0] * m)]
    outl = [str(len(terms))]
    for u, v in terms:
        outl.append(" ".join(map(str, u)))
        outl.append(" ".join(map(str, v)))
    sys.stdout.write("\n".join(outl) + "\n")

def main():
    n, m, M = read_M()
    rw = rowwise(n, m, M)
    cw = colwise(n, m, M)
    best = rw if len(rw) <= len(cw) else cw
    emit(best, n, m)

if __name__ == "__main__":
    main()
