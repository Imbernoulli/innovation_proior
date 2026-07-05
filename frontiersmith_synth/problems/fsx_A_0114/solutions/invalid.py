# TIER: invalid
# Emits infeasible output: all modules stacked far outside the triangular plot.
import sys


def main():
    d = sys.stdin.read().split()
    n = int(d[0])
    out = ["9.0 9.0" for _ in range(n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
