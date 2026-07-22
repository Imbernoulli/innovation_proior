# TIER: invalid
"""Deliberately infeasible artifact: emits OUTPUT with nothing ever pushed to
the stack (guaranteed stack underflow), plus some garbage-looking filler --
the checker must reject this with Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    _P = int(next(it))
    _N = int(next(it)); _M = int(next(it)); K = int(next(it))
    lines = ["PUSH 999999", "OP ADD"]  # PUSH references an out-of-range leaf, OP then underflows anyway
    for _ in range(K):
        lines.append("OUTPUT")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
