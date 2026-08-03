# TIER: strong
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

def anchored(o):
    ax, ay = min(o, key=lambda p: (p[1], p[0]))
    return [(x - ax, y - ay) for (x, y) in o]

def orient_rels(cells):
    return [anchored(o) for o in transforms(cells)]

def greedy_pack(W, H, blocked, cands, orient_perm=None):
    occ = set(blocked); used = {c[0]: 0 for c in cands}; placements = []
    for y in range(H):
        for x in range(W):
            if (x, y) in occ:
                continue
            for (tid, rels, stock) in cands:
                if used[tid] >= stock:
                    continue
                rr = rels if orient_perm is None else [rels[k] for k in orient_perm.get(tid, range(len(rels)))]
                done = False
                for rel in rr:
                    cells = [(x + rx, y + ry) for (rx, ry) in rel]
                    ok = all(0 <= cx < W and 0 <= cy < H and (cx, cy) not in occ for (cx, cy) in cells)
                    if ok:
                        for cc in cells:
                            occ.add(cc)
                        placements.append((tid, cells)); used[tid] += 1; done = True; break
                if done:
                    break
    return placements

def parse():
    d = iter(sys.stdin.read().split())
    W = int(next(d)); H = int(next(d)); R = int(next(d))
    bedrock = set()
    for _ in range(R):
        bedrock.add((int(next(d)), int(next(d))))
    P = int(next(d)); shapes = []; stocks = []
    for _ in range(P):
        c = int(next(d)); s = int(next(d)); cells = []
        for _ in range(s):
            cells.append((int(next(d)), int(next(d))))
        stocks.append(c); shapes.append(cells)
    return W, H, bedrock, shapes, stocks

def emit(pl):
    lines = [str(len(pl))]
    for (t, cells) in pl:
        parts = [str(t)]
        for (x, y) in cells:
            parts.append(str(x)); parts.append(str(y))
        lines.append(" ".join(parts))
    sys.stdout.write("\n".join(lines) + "\n")

def coverage(pl):
    return sum(len(c) for (_, c) in pl)

def main():
    W, H, bedrock, shapes, stocks = parse()
    P = len(shapes)
    rels = [orient_rels(sh) for sh in shapes]

    # restart 0 == deterministic largest-first greedy (guarantees strong >= greedy)
    order0 = sorted(range(P), key=lambda t: -len(shapes[t]))
    best = greedy_pack(W, H, bedrock, [(t, rels[t], stocks[t]) for t in order0])
    best_cov = coverage(best)

    # seeded multi-restart: shuffle type priority + orientation enumeration, keep best
    restarts = 140
    for r in range(restarts):
        rng = random.Random(917 * (r + 1) + 7 * P + 3 * (W * H))
        order = list(range(P))
        rng.shuffle(order)
        oper = {}
        for t in range(P):
            perm = list(range(len(rels[t])))
            rng.shuffle(perm)
            oper[t] = perm
        cands = [(t, rels[t], stocks[t]) for t in order]
        pl = greedy_pack(W, H, bedrock, cands, orient_perm=oper)
        cov = coverage(pl)
        if cov > best_cov:
            best_cov = cov; best = pl

    emit(best)

main()
