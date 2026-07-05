# TIER: invalid
# Emits infeasible output: all depots stacked at the same out-of-region point.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    out = ["5.0 5.0" for _ in range(n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
