# TIER: invalid
# Stock SKU 1 in every slot -> aisle/column/loop distinctness violated and the
# prefilled givens are clobbered -> the checker must reject this (Ratio 0.0).
import sys


def main():
    data = [ln.split() for ln in sys.stdin.read().splitlines()]
    data = [t for t in data if t]
    N = int(data[0][0])
    out = "\n".join(" ".join("1" for _ in range(N)) for _ in range(N))
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
