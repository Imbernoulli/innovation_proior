import sys, random

# Datum-tree error-budget generator (family: datum-tree-error-budget).
# Structure per test:
#   * root feature 0 (the master datum)
#   * a two-sided "trunk": left chain L[0..g-1] and right chain R[0..g-1], both
#     rooted at 0. Trunk edges are SHARED by every cross-side critical pair.
#   * leaves hang under the trunk via private chains. Each leaf's TOP node may
#     attach to the DEEP end of the trunk (allowed[0], listed first -> baseline)
#     or to a SHALLOWER node (allowed[1]), skipping some trunk edges.
#   * critical pairs join a left leaf and a right leaf; three planted regimes:
#       - "bottleneck": high weight, SHORT private chains -> they set the
#         weighted max; the fix is to make their SHARED trunk edges precise.
#       - "lure": low weight, LONG private chains -> deepest nodes, highest RAW
#         stackup. A depth/longest-chain greedy wastes its precise slots here.
#       - "filler": mid weight/length.
# Trap: greedy (shallow tree + tighten the deepest/longest chains) never spends
# precise slots on the shared trunk, so the weighted-max bottleneck is untouched.

def gen(testId):
    rng = random.Random(90001 + testId * 1237)
    g   = 4 + testId              # trunk length each side
    s   = max(2, g // 3)          # how many trunk edges the shallow option skips
    TB, TBp = 60, 6              # trunk op error: base / precise
    LB, LBp = 40, 4              # private-chain op error: base / precise

    a = []; p = []; allowed = []
    def add(base, prec, allow):
        a.append(base); p.append(prec); allowed.append(list(allow))
        return len(a) - 1

    root = add(0, 0, [])
    L = []; prev = root
    for _ in range(g):
        prev = add(TB, TBp, [prev]); L.append(prev)
    R = []; prev = root
    for _ in range(g):
        prev = add(TB, TBp, [prev]); R.append(prev)

    Ldeep, Lshallow = L[-1], L[g - 1 - s]
    Rdeep, Rshallow = R[-1], R[g - 1 - s]

    def chain(top_allow, length):
        # private chain; TOP node carries the datum choice, rest forced; return leaf
        prev = add(LB, LBp, list(top_allow))
        for _ in range(length - 1):
            prev = add(LB, LBp, [prev])
        return prev

    pairs = []
    # bottleneck: high weight, short private chains
    nb = 3 + testId // 3
    for i in range(nb):
        ll = chain([Ldeep, Lshallow], 1)
        rl = chain([Rdeep, Rshallow], 1)
        pairs.append((ll, rl, 20 + (i % 5)))
    # lure: low weight, long private chains
    for i in range(3):
        ln = g + 2
        ll = chain([Ldeep, Lshallow], ln)
        rl = chain([Rdeep, Rshallow], ln)
        pairs.append((ll, rl, 2 + (i % 3)))
    # filler: mid weight, mid private chains
    nf = 2 + testId // 3
    for i in range(nf):
        ll = chain([Ldeep, Lshallow], 2)
        rl = chain([Rdeep, Rshallow], 2)
        pairs.append((ll, rl, 9 + (i % 4)))

    rng.shuffle(pairs)
    n = len(a)
    k = 2 * g

    out = ["%d %d" % (n, k),
           " ".join(map(str, a)),
           " ".join(map(str, p))]
    for i in range(n):
        al = allowed[i]
        out.append(" ".join(map(str, [len(al)] + al)))
    out.append(str(len(pairs)))
    for (u, v, w) in pairs:
        out.append("%d %d %d" % (u, v, w))
    return "\n".join(out) + "\n"

if __name__ == "__main__":
    sys.stdout.write(gen(int(sys.argv[1])))
