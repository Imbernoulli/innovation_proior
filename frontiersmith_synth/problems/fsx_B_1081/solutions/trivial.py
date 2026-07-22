# TIER: trivial
"""One giant panel covering the whole skin: spend zero seam budget. This is
exactly the checker's own internal baseline construction, so it reproduces
Ratio ~= 0.1 by design. Every full-interior vertex stays fully surrounded by
the same panel, so the objective equals the worst-case (unrelieved) maximum
curvature of the whole surface."""
import sys


def main():
    data = sys.stdin.read().split()
    R = int(data[0]); C = int(data[1])
    N = 2 * R * C
    sys.stdout.write(" ".join(["0"] * N) + "\n")


if __name__ == "__main__":
    main()
