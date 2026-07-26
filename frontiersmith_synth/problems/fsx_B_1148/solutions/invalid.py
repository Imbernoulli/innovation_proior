# TIER: invalid
"""Deliberately infeasible artifact: claims to push a prop id that does not
exist (out of range 1..M) out of stage cell 0 in every window. Must score
Ratio: 0.0 under strict feasibility checking."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); S = int(next(it)); w = int(next(it))

    out_lines = []
    for _ in range(S - 1):
        out_lines.append("1")
        out_lines.append("PUSH 0 L %d" % (M + 999))  # prop id far out of range
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
