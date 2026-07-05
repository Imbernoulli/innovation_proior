# TIER: invalid
# Emits an N x N matrix of the value 2 -- correct shape but entries outside {-1,1}, so the
# feasibility gate must reject it (score 0).
import sys


def main():
    N = int(sys.stdin.read().split()[0])
    row = " ".join("2" for _ in range(N))
    sys.stdout.write("\n".join(row for _ in range(N)) + "\n")


if __name__ == "__main__":
    main()
