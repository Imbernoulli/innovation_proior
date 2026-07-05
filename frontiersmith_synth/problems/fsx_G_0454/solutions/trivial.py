# TIER: trivial
# Row-wise unary construction == the checker's internal baseline.
# For each row i, build the row with max_j|M[i][j]| ternary "level" terms e_i (x) s^(l).
import sys

def read_M():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    M = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    return n, m, M

def sgn(x):
    return (x > 0) - (x < 0)

def main():
    n, m, M = read_M()
    terms = []
    for i in range(n):
        h = max((abs(M[i][j]) for j in range(m)), default=0)
        for l in range(1, h + 1):
            u = [0] * n
            u[i] = 1
            v = [sgn(M[i][j]) if abs(M[i][j]) >= l else 0 for j in range(m)]
            terms.append((u, v))
    if not terms:
        # degenerate zero matrix guard (never hit by gen)
        terms.append(([1] + [0] * (n - 1), [0] * m))
    outl = [str(len(terms))]
    for u, v in terms:
        outl.append(" ".join(map(str, u)))
        outl.append(" ".join(map(str, v)))
    sys.stdout.write("\n".join(outl) + "\n")

if __name__ == "__main__":
    main()
