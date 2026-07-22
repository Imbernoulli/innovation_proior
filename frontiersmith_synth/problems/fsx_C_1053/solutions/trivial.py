# TIER: trivial
# Reproduces the checker's own internal baseline: place all K sensors on the first
# K node ids (0..K-1), ignoring the graph, weights, saturation and bonuses entirely.
import sys


def main():
    toks = sys.stdin.read().split()
    K = int(toks[2])
    print(" ".join(str(i) for i in range(K)))


if __name__ == "__main__":
    main()
