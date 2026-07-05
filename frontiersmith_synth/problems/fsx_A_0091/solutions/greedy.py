# TIER: greedy
import sys

def can_add(chosen, by_row, by_col, w, t):
    # (w,t) as apex: need (w+d,t) and (w,t+d) both present
    for w2 in by_row.get(t, ()):
        d = w2 - w
        if d and (w, t + d) in chosen:
            return False
    # (w,t) as horizontal leg: apex (aw,t) in same window, completer (aw, t+d)
    for aw in by_row.get(t, ()):
        d = w - aw
        if d and (aw, t + d) in chosen:
            return False
    # (w,t) as vertical leg: apex (w,at) in same warehouse, completer (w+d, at)
    for at in by_col.get(w, ()):
        d = t - at
        if d and (w + d, at) in chosen:
            return False
    return True

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    m = int(next(it)); e = int(next(it))
    emb = set()
    for _ in range(e):
        w = int(next(it)); t = int(next(it))
        emb.add((w, t))

    order = [(w, t) for w in range(m) for t in range(m)]  # lexicographic
    chosen = set(); by_row = {}; by_col = {}
    for (w, t) in order:
        if (w, t) in emb:
            continue
        if can_add(chosen, by_row, by_col, w, t):
            chosen.add((w, t))
            by_row.setdefault(t, []).append(w)
            by_col.setdefault(w, []).append(t)

    res = sorted(chosen)
    out = [str(len(res))]
    out.extend("%d %d" % (w, t) for (w, t) in res)
    sys.stdout.write("\n".join(out) + "\n")

main()
