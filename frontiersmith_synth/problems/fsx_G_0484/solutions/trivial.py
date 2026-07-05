# TIER: trivial
# Natural-order first-fit covering code -- identical to the checker's internal baseline,
# so it reproduces B and scores ~0.1.  Scan words in numeric order; every uncovered word
# becomes an anchor covering its radius-r ball.
import sys


def main():
    data = sys.stdin.read().split()
    n, r = int(data[0]), int(data[1])
    N = 1 << n
    masks = [m for m in range(N) if bin(m).count("1") <= r]

    covered = bytearray(N)
    out = []
    for p in range(N):
        if not covered[p]:
            out.append(p)
            for m in masks:
                covered[p ^ m] = 1

    w = sys.stdout.write
    for c in out:
        w(format(c, "0%db" % n) + "\n")


if __name__ == "__main__":
    main()
