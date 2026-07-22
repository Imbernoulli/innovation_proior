# TIER: invalid
"""Emits a garbage/infeasible artifact: wrong grid shape (N-1 columns) so the
checker's strict feasibility check must reject it with Ratio: 0.0."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    out = [str(N)]
    # deliberately wrong: only N-1 numbers per row (shape mismatch)
    row = " ".join(["1.2345"] * max(N - 1, 1))
    for _ in range(N):
        out.append(row)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
