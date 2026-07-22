# TIER: greedy
# The obvious "textbook" move: split the relay budget EVENLY across every
# pair (floor(R/m) relays each, leftover unused) and chain them at equal
# spacing to MINIMIZE each hop's distance -- the natural instinct when the
# per-hop rate is log(1+SINR) and shorter hops mean less path-loss. This
# ignores (a) that pairs have very different intrinsic lengths so an equal
# split over-serves already-easy pairs and starves the hard one, and (b)
# that piling relays onto every pair inflates the number of simultaneously
# active co-hop transmitters, raising interference for everybody.
import sys


def main():
    t = sys.stdin.read().split()
    p = 0
    m = int(t[p]); p += 1
    R = int(t[p]); p += 1
    p += 5  # P, alpha, N0, Xmax, Ymax (unused)
    pairs = []
    for _ in range(m):
        sx = float(t[p]); sy = float(t[p + 1]); dx = float(t[p + 2]); dy = float(t[p + 3])
        p += 4
        pairs.append((sx, sy, dx, dy))

    k = R // m  # equal split, minimize per-hop distance for every pair

    out = []
    for (sx, sy, dx, dy) in pairs:
        line = [str(k)]
        for j in range(1, k + 1):
            frac = j / (k + 1)
            x = sx + (dx - sx) * frac
            y = sy + (dy - sy) * frac
            line.append("%.6f %.6f" % (x, y))
        out.append(" ".join(line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
