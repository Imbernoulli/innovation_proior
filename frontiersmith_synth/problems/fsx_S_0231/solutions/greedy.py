# TIER: greedy
# Area-first greedy: at each uncovered cell deploy the largest-cell template that fits,
# mixing all types up to their stock. Weight-blind.
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
        next(it)
    P = int(next(it))
    shapes = []; stocks = []; sizes = []
    for _ in range(P):
        c = int(next(it)); s = int(next(it))
        cells = [(int(next(it)), int(next(it))) for _ in range(s)]
        shapes.append(cells); stocks.append(c); sizes.append(s)

    rels = [[anchored(o) for o in transforms(sh)] for sh in shapes]
    # priority order: largest cell count first
    order = sorted(range(P), key=lambda t: -sizes[t])

    occ = set(blocked)
    used = [0] * P
    placements = []
    for y in range(H):
        for x in range(W):
            if (x, y) in occ:
                continue
            done = False
            for t in order:
                if used[t] >= stocks[t]:
                    continue
                for rel in rels[t]:
                    cells = [(x + rx, y + ry) for (rx, ry) in rel]
                    if all(0 <= cx < W and 0 <= cy < H and (cx, cy) not in occ for (cx, cy) in cells):
                        for cc in cells:
                            occ.add(cc)
                        placements.append((t, cells))
                        used[t] += 1
                        done = True
                        break
                if done:
                    break

    out = [str(len(placements))]
    for (t, cells) in placements:
        out.append(str(t) + " " + " ".join("%d %d" % (cx, cy) for (cx, cy) in cells))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
