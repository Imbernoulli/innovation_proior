# TIER: invalid
"""Emits a grid with a value outside the declared palette -- must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))

    out = []
    for i in range(N):
        row = [str(99) for _ in range(N)]  # 99 is never a declared palette value
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
