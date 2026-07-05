# TIER: strong
# Full-ball max-coverage greedy.  For the first uncovered word p, consider EVERY word c that
# covers p (the whole radius-r ball around p) as a candidate anchor, and pick the one whose
# radius-r ball covers the most currently-uncovered words (ties -> smallest index).  This
# searches a strict superset of the greedy tier's candidates, so each anchor is chosen no
# worse, and the resulting code is markedly smaller.  Still far above the (open) optimum.
import sys


def main():
    data = sys.stdin.read().split()
    n, r = int(data[0]), int(data[1])
    N = 1 << n
    masks = [m for m in range(N) if bin(m).count("1") <= r]

    covered = bytearray(N)
    out = []
    ptr = 0
    while True:
        while ptr < N and covered[ptr]:
            ptr += 1
        if ptr >= N:
            break
        p = ptr
        best = -1
        bestc = p
        # candidates = ball(p, r) = {p ^ m}; each such c covers p (dist(p,c)=popcount(m)<=r)
        for mc in masks:
            c = p ^ mc
            nc = 0
            for m in masks:
                if not covered[c ^ m]:
                    nc += 1
            if nc > best or (nc == best and c < bestc):
                best = nc
                bestc = c
        out.append(bestc)
        for m in masks:
            covered[bestc ^ m] = 1

    w = sys.stdout.write
    for c in out:
        w(format(c, "0%db" % n) + "\n")


if __name__ == "__main__":
    main()
