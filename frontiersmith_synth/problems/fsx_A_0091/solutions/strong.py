# TIER: strong
import sys, random

def can_add(chosen, by_row, by_col, w, t):
    for w2 in by_row.get(t, ()):
        d = w2 - w
        if d and (w, t + d) in chosen:
            return False
    for aw in by_row.get(t, ()):
        d = w - aw
        if d and (aw, t + d) in chosen:
            return False
    for at in by_col.get(w, ()):
        d = t - at
        if d and (w + d, at) in chosen:
            return False
    return True

def build(order):
    chosen = set(); by_row = {}; by_col = {}
    for (w, t) in order:
        if can_add(chosen, by_row, by_col, w, t):
            chosen.add((w, t))
            by_row.setdefault(t, []).append(w)
            by_col.setdefault(w, []).append(t)
    return chosen

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    m = int(next(it)); e = int(next(it))
    emb = set()
    for _ in range(e):
        w = int(next(it)); t = int(next(it))
        emb.add((w, t))

    allowed = [(w, t) for w in range(m) for t in range(m) if (w, t) not in emb]

    best = build(allowed)  # lexicographic seed
    restarts = 120 if m <= 12 else 70
    for r in range(restarts):
        rng = random.Random(7919 * (r + 1) + 31 * m)
        o = allowed[:]
        rng.shuffle(o)
        cand = build(o)
        if len(cand) > len(best):
            best = cand

    res = sorted(best)
    out = [str(len(res))]
    out.extend("%d %d" % (w, t) for (w, t) in res)
    sys.stdout.write("\n".join(out) + "\n")

main()
