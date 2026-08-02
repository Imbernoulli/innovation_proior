# TIER: invalid
"""Deliberately infeasible: claims ONE template equal to line 1's exact
tokens (no wildcards at all) and assigns every line to it. Any other line
that differs anywhere from line 1 breaks the template's constant claim ->
strict feasibility check must reject this with Ratio 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); W = int(next(it))
    lines = [[next(it) for _ in range(W)] for _ in range(N)]

    out = ["1"]
    out.append(" ".join(lines[0]))
    out.append(" ".join("1" for _ in range(N)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
