# TIER: invalid
# Emits garbage: every pipe (including cap=0 outlet meters, where a valve is
# forbidden) gets an astronomically large extra resistance, far past the
# feasibility bound. Must score 0.
import sys


def main():
    tok = sys.stdin.read().split()
    n_edges = int(tok[2])
    print(" ".join(["1e12"] * n_edges))


if __name__ == "__main__":
    main()
