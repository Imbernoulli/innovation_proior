# TIER: greedy
"""The obvious textbook recipe: never mind the rake, just shave every cell
that's out of tolerance down to exactly the threshold with point-polish
strokes. Simple, always correct, never looks at neighbours or shared
structure."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    T = int(data[idx]); idx += 1
    _MAXOPS = int(data[idx]); idx += 1
    h = [int(data[idx + i]) for i in range(N)]

    ops = []
    for i in range(N):
        excess = abs(h[i]) - T
        if excess > 0:
            ops.extend(["P %d" % i] * excess)

    out = [str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
