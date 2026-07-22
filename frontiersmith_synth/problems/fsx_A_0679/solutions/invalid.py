# TIER: invalid
# Emits a structurally broken artifact: every cell is declared parity (zero
# raw/D cells), yet each parity line still references raw index 0. Since
# raw_count == 0, index 0 is always out of range -> the checker must reject
# this with Ratio: 0.0.
import sys


def main():
    R, C, p = map(int, sys.stdin.read().split())
    out = []
    for _ in range(R * C):
        out.append("P 1 0 1")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
