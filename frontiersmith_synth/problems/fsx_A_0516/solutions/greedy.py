# TIER: greedy
"""The textbook grammar-compression recipe: RePair.  Repeatedly replace the most
frequent adjacent digram with a fresh non-terminal -> a straight-line grammar (SLP)
whose expansion equals T exactly.  This factors REPEATED SUBSTRINGS, but it cannot
express "apply the same rule across n levels": every level needs its OWN rule, so the
grammar size grows ~ n = log|T| and plateaus far above the true O(1) L-system.
This is the obvious approach an average strong coder writes first."""
import sys


def repair(seq):
    seq = list(seq)
    rules = []          # (name, [a, b])
    nid = 0
    while True:
        counts = {}
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            counts[p] = counts.get(p, 0) + 1
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        pair, cnt = best
        if cnt < 2:
            break
        name = "N%d" % nid
        nid += 1
        rules.append((name, [pair[0], pair[1]]))
        newseq = []
        i = 0
        n = len(seq)
        while i < n:
            if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                newseq.append(name)
                i += 2
            else:
                newseq.append(seq[i])
                i += 1
        seq = newseq
    return seq, rules


def emit(reduced, rules):
    depth = {}
    rmap = {name: rhs for name, rhs in rules}

    def d(sym):
        if sym not in rmap:
            return 0
        if sym in depth:
            return depth[sym]
        depth[sym] = 1 + max(d(c) for c in rmap[sym])
        return depth[sym]

    n = 0
    for name, _ in rules:
        n = max(n, d(name))
    for s in reduced:
        n = max(n, d(s))
    out = [str(n), " ".join(reduced), str(len(rules))]
    for name, rhs in rules:
        out.append(name + " " + " ".join(rhs))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    data = sys.stdin.read().split("\n")
    T = data[1].split() if len(data) > 1 else []
    reduced, rules = repair(T)
    emit(reduced, rules)


if __name__ == "__main__":
    main()
