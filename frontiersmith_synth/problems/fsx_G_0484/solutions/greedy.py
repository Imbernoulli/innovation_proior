# TIER: greedy
# Local max-coverage first-fit.  Process words in numeric order; for the first uncovered
# word p, instead of anchoring p itself, try p and its single-bit neighbours (all still cover
# p since r>=1) and anchor whichever covers the most currently-uncovered words.  A small,
# cheap look-ahead that beats the trivial baseline.
import sys


def main():
    data = sys.stdin.read().split()
    n, r = int(data[0]), int(data[1])
    N = 1 << n
    masks = [m for m in range(N) if bin(m).count("1") <= r]
    # candidate offsets from p: identity + all single-bit flips
    cand_off = [0] + [1 << i for i in range(n)]

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
        for off in cand_off:
            c = p ^ off
            nc = 0
            for m in masks:
                if not covered[c ^ m]:
                    nc += 1
            if nc > best:
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
