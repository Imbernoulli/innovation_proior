# TIER: invalid
"""
Emits a schedule that releases far more water than any dam could ever hold
or pass (guaranteed to violate release <= min(level, Rmax) immediately) so
the checker must score it Ratio: 0.0.
"""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); T = int(next(it))

    out = []
    for _ in range(N):
        out.append(" ".join("999999.0" for _ in range(T)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
