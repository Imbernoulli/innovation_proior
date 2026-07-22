# TIER: greedy
# The obvious optimization: generic common-subexpression elimination on the
# expanded monomials.  Repeatedly precompute the globally most frequent
# co-occurring variable pair as a shared product, collapse it inside every
# monomial that contains it, and finish with left-folds.  Beats trivial by
# sharing sub-products -- but, working only on the expanded monomial list, it
# CANNOT discover the hidden additive factorization, so it stays far above the
# strong circuit.
import sys
from collections import Counter

def read_instance(data):
    it = iter(data)
    p = int(next(it)); n = int(next(it)); k = int(next(it))
    polys = []
    for _ in range(k):
        M = int(next(it))
        mons = []
        for _ in range(M):
            c = int(next(it)) % p
            exps = tuple(int(next(it)) for _ in range(n))
            mons.append((c, exps))
        polys.append(mons)
    return p, n, k, polys

def main():
    data = sys.stdin.read().split()
    p, n, k, polys = read_instance(data)

    lines = []
    nxt = [n]
    def emit(s):
        idx = nxt[0]; lines.append(s); nxt[0] += 1; return idx

    # collect distinct monomials as multisets of variable node-ids (atoms)
    distinct = {}
    for mons in polys:
        for c, exps in mons:
            if exps in distinct:
                continue
            atoms = []
            for j in range(n):
                atoms += [j] * exps[j]
            distinct[exps] = atoms       # list of atom ids (>=1)

    # working copies (only support>=2 need products)
    work = {e: list(a) for e, a in distinct.items() if len(a) >= 2}
    prodcache = {}                       # frozen pair (x,y) sorted -> node id
    def product(x, y):
        key = (x, y) if x <= y else (y, x)
        if key not in prodcache:
            prodcache[key] = emit("mul %d %d" % (key[0], key[1]))
        return prodcache[key]

    # global most-frequent-pair CSE
    while True:
        cnt = Counter()
        for atoms in work.values():
            m = len(atoms)
            if m < 2:
                continue
            seen = set()
            for i in range(m):
                for jj in range(i + 1, m):
                    a, b = atoms[i], atoms[jj]
                    pr = (a, b) if a <= b else (b, a)
                    if pr not in seen:
                        seen.add(pr)
                        cnt[pr] += 1
        if not cnt:
            break
        # deterministic pick: highest count, then lexicographically smallest pair
        best = min(((-v, pr) for pr, v in cnt.items()))
        bestcount = -best[0]; pair = best[1]
        if bestcount < 2:
            break
        node = product(pair[0], pair[1])
        a, b = pair
        for atoms in work.values():
            if a in atoms and b in atoms:
                atoms.remove(a); atoms.remove(b); atoms.append(node)

    # finish remaining monomials by left-folding
    mono_final = {}
    for e, atoms in distinct.items():
        if len(atoms) == 1:
            mono_final[e] = atoms[0]
            continue
        cur = list(work[e])
        node = cur[0]
        for q in cur[1:]:
            node = product(node, q)
        mono_final[e] = node

    const_cache = {}
    def const(v):
        v %= p
        if v not in const_cache:
            const_cache[v] = emit("const %d" % v)
        return const_cache[v]

    outs = []
    for mons in polys:
        acc = None
        for c, exps in mons:
            mn = mono_final[exps]
            cn = const(c)
            term = emit("mul %d %d" % (cn, mn))   # scalar mult -> free
            acc = term if acc is None else emit("add %d %d" % (acc, term))
        if acc is None:
            acc = const(0)
        outs.append(acc)

    sys.stdout.write("%d\n" % len(lines))
    sys.stdout.write("\n".join(lines))
    if lines:
        sys.stdout.write("\n")
    sys.stdout.write("out " + " ".join(map(str, outs)) + "\n")

if __name__ == "__main__":
    main()
