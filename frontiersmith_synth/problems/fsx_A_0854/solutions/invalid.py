# TIER: invalid
# Builds a plausible-looking circuit (correct token shapes, valid gate
# references) but claims every target's value equals the SAME single input
# variable (wire 1) -- wrong for every target with weight >= 2, so the
# exact-equivalence gate rejects it -> Ratio 0.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it))
    for _ in range(R):
        k = int(next(it))
        for _ in range(k):
            next(it)

    lines = ["0"]              # G = 0 gates
    for _ in range(R):
        lines.append("1")      # claim every row equals wire 1 (almost always wrong)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
