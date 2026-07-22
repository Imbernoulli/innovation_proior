# TIER: greedy
# The obvious approach: ignore style/vocabulary entirely and just maximize the
# minimum pairwise Hamming distance. Generate many random 4-connected glyphs within
# the ink budget, then greedily pick the N that are farthest apart. This buys big
# distinctness at the cost of a huge, noise-like motif vocabulary -- exactly the trap.
import sys, random

def rand_connected(rng, H, W, inkLo, inkHi):
    target = rng.randint(inkLo, inkHi)
    r0, c0 = rng.randrange(H), rng.randrange(W)
    ink = {(r0, c0)}
    frontier = [(r0, c0)]
    while len(ink) < target:
        nbrs = []
        for (r, c) in ink:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in ink:
                    nbrs.append((nr, nc))
        if not nbrs:
            break
        ink.add(rng.choice(nbrs))
    if len(ink) < inkLo:
        return None
    g = [[0] * W for _ in range(H)]
    for (r, c) in ink:
        g[r][c] = 1
    return g

def flat(g, H, W):
    return tuple(g[r][c] for r in range(H) for c in range(W))

def ham(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def main():
    d = sys.stdin.read().split()
    N, H, W = int(d[0]), int(d[1]), int(d[2])
    inkLo, inkHi = int(d[3]), int(d[4])
    rng = random.Random(20240611)
    cands = []
    seen = set()
    attempts = 0
    while len(cands) < 400 and attempts < 6000:
        attempts += 1
        g = rand_connected(rng, H, W, inkLo, inkHi)
        if g is None:
            continue
        f = flat(g, H, W)
        if f in seen:
            continue
        seen.add(f); cands.append(f)
    # greedy max-min-distance selection
    chosen = [cands[0]]
    while len(chosen) < N and len(chosen) < len(cands):
        best, bestd = None, -1
        for f in cands:
            if f in chosen:
                continue
            dmin = min(ham(f, c) for c in chosen)
            if dmin > bestd:
                bestd, best = dmin, f
        chosen.append(best)
    # pad if somehow short (shouldn't happen)
    while len(chosen) < N:
        chosen.append(chosen[-1])
    out = []
    for f in chosen[:N]:
        out.extend(str(x) for x in f)
    sys.stdout.write(" ".join(out) + "\n")

main()
