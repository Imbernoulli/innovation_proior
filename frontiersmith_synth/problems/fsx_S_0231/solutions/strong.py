# TIER: strong
# Seeded multi-restart demand-aware search. At each anchor cell (best-fit) deploy the
# type/orientation that services the most currently-uncovered traffic demand, steering
# large templates onto hotspots. Multiple restarts with perturbed anchor/type order;
# keep the highest-demand feasible packing. Deterministic (fixed seed).
import sys, random

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
    w = {}
    for y in range(H):
        for x in range(W):
            w[(x, y)] = int(next(it))
    P = int(next(it))
    shapes = []; stocks = []
    for _ in range(P):
        c = int(next(it)); s = int(next(it))
        cells = [(int(next(it)), int(next(it))) for _ in range(s)]
        shapes.append(cells); stocks.append(c)

    rels = [[anchored(o) for o in transforms(sh)] for sh in shapes]

    anchors = [(x, y) for y in range(H) for x in range(W) if (x, y) not in blocked]

    def run(seed):
        rng = random.Random(seed)
        occ = set(blocked)
        used = [0] * P
        placements = []
        order = anchors[:]
        # light perturbation of scan order to escape the pure row-major layout
        if seed % 2 == 1:
            rng.shuffle(order)
        for (x, y) in order:
            if (x, y) in occ:
                continue
            best = None
            best_val = -1
            for t in range(P):
                if used[t] >= stocks[t]:
                    continue
                for rel in rels[t]:
                    cells = [(x + rx, y + ry) for (rx, ry) in rel]
                    ok = True
                    val = 0
                    for (cx, cy) in cells:
                        if not (0 <= cx < W and 0 <= cy < H) or (cx, cy) in occ:
                            ok = False; break
                        val += w[(cx, cy)]
                    if not ok:
                        continue
                    # tiny deterministic tie-break jitter for restart diversity
                    jitter = rng.random() * 0.01
                    if val + jitter > best_val:
                        best_val = val + jitter
                        best = (t, cells)
            if best is not None:
                t, cells = best
                for cc in cells:
                    occ.add(cc)
                placements.append((t, cells))
                used[t] += 1
        score = sum(w[c] for (_, cells) in placements for c in cells)
        return score, placements

    best_score = -1
    best_pl = []
    for seed in range(40):
        sc, pl = run(1000 + seed)
        if sc > best_score:
            best_score = sc
            best_pl = pl

    out = [str(len(best_pl))]
    for (t, cells) in best_pl:
        out.append(str(t) + " " + " ".join("%d %d" % (cx, cy) for (cx, cy) in cells))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
