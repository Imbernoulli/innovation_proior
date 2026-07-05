# TIER: strong
import sys, itertools, random

def build(order, n):
    chosen = []
    cs = set()
    tups = {}
    for c in order:
        t = tuple(ord(ch) - 48 for ch in c)
        ok = True
        for a in chosen:
            ta = tups[a]
            third = "".join(chr(48 + ((3 - (ta[i] + t[i]) % 3) % 3)) for i in range(n))
            if third in cs and third != c and third != a:
                ok = False
                break
        if ok:
            chosen.append(c)
            cs.add(c)
            tups[c] = t
    return chosen

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); b = int(next(it))
    flooded = set(next(it) for _ in range(b))

    all_cells = ["".join(c) for c in itertools.product("012", repeat=n)]
    allowed = [c for c in all_cells if c not in flooded]

    # seeded multi-restart randomized greedy; keep the largest raid-free set
    best = build(allowed, n)  # start from lexicographic
    restarts = 160 if n <= 4 else 110
    for r in range(restarts):
        rng = random.Random(9137 * (r + 1) + 31 * n)
        order = allowed[:]
        rng.shuffle(order)
        cand = build(order, n)
        if len(cand) > len(best):
            best = cand

    print(len(best))
    if best:
        print("\n".join(best))

main()
