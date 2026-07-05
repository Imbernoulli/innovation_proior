# TIER: invalid
# Emit out-of-range entries (2s) -> fails the {0,1} feasibility check -> Ratio 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    out = []
    for i in range(N):
        out.append(" ".join("2" for _ in range(N)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
