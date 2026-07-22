# TIER: greedy
# The obvious move: BLANKET conductive material around the hottest sources (a Manhattan
# disk of radius 3), highest power first, until the budget K is spent. This lowers the
# temperature right AT the hot cells but does nothing about the long low-conductivity path
# the heat must still cross to reach a vent -- so on deep sources it barely helps.
import sys

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); KHI = int(next(it)); K = int(next(it))
    S = int(next(it))
    sources = []
    for _ in range(S):
        r = int(next(it)); c = int(next(it)); p = int(next(it))
        sources.append((r, c, p))
    NV = int(next(it))
    vents = set()
    for _ in range(NV):
        r = int(next(it)); c = int(next(it)); vents.add((r, c))

    def interior(r, c):
        return 1 <= r <= H - 2 and 1 <= c <= W - 2

    up = []
    seen = set()
    for (r, c, p) in sorted(sources, key=lambda s: -s[2]):
        for rr in range(r - 3, r + 4):
            for cc in range(c - 3, c + 4):
                if abs(rr - r) + abs(cc - c) > 3:
                    continue
                if not interior(rr, cc) or (rr, cc) in vents or (rr, cc) in seen:
                    continue
                if len(up) >= K:
                    break
                seen.add((rr, cc)); up.append((rr, cc))
            if len(up) >= K:
                break
        if len(up) >= K:
            break

    out = [str(len(up))]
    for (r, c) in up:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
