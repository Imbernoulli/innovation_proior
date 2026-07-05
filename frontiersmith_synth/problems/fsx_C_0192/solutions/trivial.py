# TIER: trivial
"""Schedule nothing new -- echo the prefilled grid unchanged. F == B -> ratio ~0.1
(this is exactly the checker's internal baseline construction)."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    g = [[int(toks[idx + i * N + j]) for j in range(N)] for i in range(N)]
    out = []
    for i in range(N):
        out.append(' '.join(str(g[i][j]) for j in range(N)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
