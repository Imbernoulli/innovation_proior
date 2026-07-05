# TIER: trivial
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    m = int(next(it)); e = int(next(it))
    emb = set()
    for _ in range(e):
        w = int(next(it)); t = int(next(it))
        emb.add((w, t))
    # delivery window t = 0, minus embargoed (matches checker baseline)
    chosen = [(w, 0) for w in range(m) if (w, 0) not in emb]
    out = [str(len(chosen))]
    out.extend("%d %d" % (w, t) for (w, t) in chosen)
    sys.stdout.write("\n".join(out) + "\n")

main()
