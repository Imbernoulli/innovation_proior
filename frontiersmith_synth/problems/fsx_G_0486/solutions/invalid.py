# TIER: invalid
"""Infeasible: emits the three tags 00..0, 11..1, 22..2, which are collinear
(0+1+2 == 0 in every coordinate). The checker must reject -> Ratio 0.0."""
import sys


def main():
    n = None
    for tok in sys.stdin.read().split():
        try:
            n = int(tok); break
        except ValueError:
            continue
    if n is None:
        n = 1
    out = ["0" * n, "1" * n, "2" * n]
    sys.stdout.write(str(len(out)) + "\n" + "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
