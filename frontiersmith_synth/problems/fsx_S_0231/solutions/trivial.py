# TIER: trivial
# Reproduce the checker's calibration baseline exactly: anchored greedy with type 0 only.
import sys

def normalize(cells):
    mnx = min(x for x, y in cells); mny = min(y for x, y in cells)
    return frozenset((x - mnx, y - mny) for x, y in cells)

def transforms(cells):
    res, seen = [], set()
    cur = list(cells)
    for ref in range(2):
        c = [(-x, y) for x, y in cur] if ref else list(cur)
        for _ in range(4):
            s = normalize(c)
            if s not in seen:
                seen.add(s); res.append(s)
            c = [(y, -x) for x, y in c]
    return res

def anchored(orient):
    ax, ay = min(orient, key=lambda p: (p[1], p[0]))
    return [(x - ax, y - ay) for (x, y) in orient]

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    W = int(next(it)); H = int(next(it))
    R = int(next(it))
    blocked = set()
    for _ in range(R):
        ox = int(next(it)); oy = int(next(it)); blocked.add((ox, oy))
    for _ in range(W * H):
        next(it)  # skip demand grid
    P = int(next(it))
    shapes = []; stocks = []
    for _ in range(P):
        c = int(next(it)); s = int(next(it))
        cells = [(int(next(it)), int(next(it))) for _ in range(s)]
        shapes.append(cells); stocks.append(c)

    rels0 = [anchored(o) for o in transforms(shapes[0])]
    occ = set(blocked)
    used = 0
    placements = []
    for y in range(H):
        for x in range(W):
            if used >= stocks[0]:
                break
            if (x, y) in occ:
                continue
            for rel in rels0:
                cells = [(x + rx, y + ry) for (rx, ry) in rel]
                if all(0 <= cx < W and 0 <= cy < H and (cx, cy) not in occ for (cx, cy) in cells):
                    for cc in cells:
                        occ.add(cc)
                    placements.append((0, cells))
                    used += 1
                    break

    out = [str(len(placements))]
    for (t, cells) in placements:
        out.append(str(t) + " " + " ".join("%d %d" % (cx, cy) for (cx, cy) in cells))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
