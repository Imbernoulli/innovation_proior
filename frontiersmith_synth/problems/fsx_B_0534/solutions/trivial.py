# TIER: trivial
# Baseline construction: a single straight firebreak wall down the middle of the
# map (the free column nearest the centre).  It halves the map but never adapts
# to where the fuel actually is, so the heavier half keeps a huge worst-case
# burn -> reproduces the checker baseline.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); F = int(it[p + 1]); K = int(it[p + 2]); p += 3
    ign_c = set()
    for _ in range(K):
        r = int(it[p]); c = int(it[p + 1]); p += 2
        ign_c.add(c)

    mid = N // 2
    col = None
    for d in range(N):
        for cand in (mid + d, mid - d):
            if 0 <= cand < N and cand not in ign_c:
                col = cand; break
        if col is not None:
            break

    out = [str(N)]
    for r in range(N):
        out.append("%d %d" % (r, col))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
