# TIER: invalid
# Deliberately infeasible: injects a current pattern whose magnitude exceeds
# the stated per-electrode bound I_max (and, independently, does not sum to
# zero) -- must be rejected by the checker with Ratio: 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0

    def nxt():
        nonlocal ptr
        v = data[ptr]
        ptr += 1
        return v

    b = int(nxt()); L = int(nxt()); m = int(nxt()); q = int(nxt()); I_max = int(nxt())

    lines = []
    for _ in range(q):
        row = [0] * b
        row[0] = I_max * 5 + 7   # out of range AND (with all-else-0) sum != 0
        lines.append(" ".join(map(str, row)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
