# TIER: trivial
"""Baseline completion: the minimal 'triangular' sign matrix (+1 on/below the
diagonal, -1 above), with excavated cells overwritten. This is exactly the
checker's reference baseline, so it scores the calibrated ~0.10."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    vals = list(map(int, toks[1:1 + N * N]))
    G = [vals[i * N:(i + 1) * N] for i in range(N)]
    M = [[1 if j <= i else -1 for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if G[i][j] != 0:
                M[i][j] = G[i][j]
    out = []
    for i in range(N):
        out.append(" ".join(str(x) for x in M[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
