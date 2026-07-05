# TIER: invalid
"""Emits pads far OUTSIDE the unit triangle -> feasibility fails -> Ratio 0.0."""
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    out = ["%.17g %.17g" % (5.0, 5.0) for _ in range(n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
