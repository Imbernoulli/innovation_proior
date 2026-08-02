# TIER: invalid
"""Emits a structurally-plausible but infeasible artifact (non-finite coordinates)
-- must score 0.0 under strict feasibility checking."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it)); R = int(next(it)); c = float(next(it))
    for _ in range(R):
        next(it); next(it)
    k_cal = int(next(it))
    for _ in range(k_cal):
        for _ in range(2 + (R - 1)):
            next(it)
    k_test = int(next(it))
    out = ["nan nan" for _ in range(k_test)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
