# TIER: trivial
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    alphabet = next(it)
    nP = int(next(it)); nN = int(next(it))
    P = [next(it) for _ in range(nP)]
    N = [next(it) for _ in range(nN)]
    # build trie over P; state id = distinct prefix
    ids = {"": 0}
    edges = []  # (from, sym, to)
    accept = set()
    for s in P:
        cur = ""
        for ch in s:
            nxt = cur + ch
            if nxt not in ids:
                ids[nxt] = len(ids)
                edges.append((ids[cur], ch, ids[nxt]))
            cur = nxt
        accept.add(ids[s])
    S = len(ids)
    starts = [0]
    acc = sorted(accept)
    out = []
    out.append(str(S))
    out.append(str(len(starts)) + " " + " ".join(map(str, starts)))
    out.append(str(len(acc)) + " " + " ".join(map(str, acc)))
    out.append(str(len(edges)))
    for (f, sym, t) in edges:
        out.append("%d %s %d" % (f, sym, t))
    sys.stdout.write("\n".join(out) + "\n")

main()
