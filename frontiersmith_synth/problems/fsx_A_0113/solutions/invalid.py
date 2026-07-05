# TIER: invalid
"""Infeasible: stack every substation at the centre with a large radius, so all
coverage disks mutually overlap. The checker must reject this -> Ratio 0.0."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = ["0.5 0.5 0.3" for _ in range(n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
