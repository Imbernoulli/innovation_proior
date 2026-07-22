# TIER: trivial
"""An unpracticed apprentice: for every cell above threshold, shave it all
the way down with point-polish strokes -- but paranoid about leaving any
residue, they always stroke it TWICE as many times as the textbook amount
needs (most of the extra strokes land on an already-flat cell and do
nothing, but each one still costs 1). Never touches the rake."""
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
            ops.extend(["P %d" % i] * (2 * excess))

    out = [str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
