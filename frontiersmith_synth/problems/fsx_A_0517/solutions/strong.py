# TIER: strong
# INSIGHT: the expanded targets are a multivariate factorization in disguise.
# Recover the hidden generative structure instead of rewriting monomials.
#   (1) Within each target, two variables share a hidden linear factor iff they
#       NEVER co-occur in a monomial (each monomial takes one variable per
#       factor).  Connected components of the non-co-occurrence graph = the
#       factor variable-blocks.
#   (2) Recover each block's linear form up to scale from coefficient ratios;
#       normalise it (leading coeff 1) so the SAME hidden factor gets ONE shared
#       node across all targets (dag-sharing).  The residual scalar is a free
#       constant multiply.
#   (3) Each target = scalar * product of its (free) linear factors -> only
#       (#factors - 1) genuine multiplications, vs thousands for monomials.
import sys

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

def emit_trivial(polys, p, n):
    # safety fallback (only if an instance is not multilinear); mirrors trivial.
    lines = []; nxt = [n]
    def emit(s):
        idx = nxt[0]; lines.append(s); nxt[0] += 1; return idx
    cc = {}
    def const(v):
        v %= p
        if v not in cc: cc[v] = emit("const %d" % v)
        return cc[v]
    pw = {}
    def var_pow(j, e):
        if e == 1: return j
        if (j, e) in pw: return pw[(j, e)]
        prev = var_pow(j, e - 1); node = emit("mul %d %d" % (prev, j)); pw[(j, e)] = node
        return node
    mc = {}
    def mono(exps):
        if exps in mc: return mc[exps]
        parts = [var_pow(j, exps[j]) for j in range(n) if exps[j]]
        cur = parts[0]
        for q in parts[1:]: cur = emit("mul %d %d" % (cur, q))
        mc[exps] = cur; return cur
    outs = []
    for mons in polys:
        acc = None
        for c, exps in mons:
            term = emit("mul %d %d" % (const(c), mono(exps)))
            acc = term if acc is None else emit("add %d %d" % (acc, term))
        outs.append(acc if acc is not None else const(0))
    return lines, outs

def main():
    data = sys.stdin.read().split()
    p, n, k, polys = read_instance(data)

    multilinear = all(all(e <= 1 for e in exps) for mons in polys for c, exps in mons)
    lines = []; nxt = [n]
    def emit(s):
        idx = nxt[0]; lines.append(s); nxt[0] += 1; return idx

    if not multilinear:
        lines, outs = emit_trivial(polys, p, n)
        sys.stdout.write("%d\n" % len(lines))
        sys.stdout.write("\n".join(lines))
        if lines: sys.stdout.write("\n")
        sys.stdout.write("out " + " ".join(map(str, outs)) + "\n")
        return

    const_cache = {}
    def const(v):
        v %= p
        if v not in const_cache:
            const_cache[v] = emit("const %d" % v)
        return const_cache[v]

    factor_cache = {}   # canonical key -> node computing the normalised linear form
    def factor_node(key):
        # key = tuple of (var, normalised_coeff), sorted; leading coeff == 1
        if key in factor_cache:
            return factor_cache[key]
        terms = []
        for (v, coef) in key:
            if coef % p == 1:
                terms.append(v)
            else:
                terms.append(emit("mul %d %d" % (const(coef), v)))  # scalar -> free
        node = terms[0]
        for q in terms[1:]:
            node = emit("add %d %d" % (node, q))
        factor_cache[key] = node
        return node

    outs = []
    for mons in polys:
        monos = {frozenset(j for j in range(n) if exps[j]): c for c, exps in mons}
        V = sorted({v for ms in monos for v in ms})
        idx = {v: i for i, v in enumerate(V)}
        # co-occurrence within this target
        coocc = set()
        for ms in monos:
            lst = list(ms)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a, b = lst[i], lst[j]
                    coocc.add((a, b) if a < b else (b, a))
        # union-find on non-co-occurring pairs -> factor blocks
        parent = list(range(len(V)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: parent[rx] = ry
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                a, b = V[i], V[j]
                if (a, b) not in coocc:
                    union(idx[a], idx[b])
        blocks = {}
        for v in V:
            r = find(idx[v]); blocks.setdefault(r, []).append(v)
        blocklist = [sorted(b) for b in blocks.values()]

        # reference monomial = one leading (min) variable per block
        vmins = [b[0] for b in blocklist]
        M0 = frozenset(vmins)
        coeff0 = monos[M0]
        inv0 = pow(coeff0, p - 2, p)

        keys = []
        for b in blocklist:
            vmin = b[0]
            rest = M0 - {vmin}
            pairs = []
            for v in b:
                if v == vmin:
                    nc = 1
                else:
                    Mv = frozenset(rest | {v})
                    nc = monos[Mv] * inv0 % p
                pairs.append((v, nc))
            keys.append(tuple(sorted(pairs)))

        # build circuit: product of factor nodes, then scalar gamma = coeff0
        fnodes = [factor_node(kk) for kk in keys]
        prod = fnodes[0]
        for q in fnodes[1:]:
            prod = emit("mul %d %d" % (prod, q))   # genuine (non-scalar) multiply
        gamma = coeff0 % p
        if gamma == 1:
            out_node = prod
        else:
            out_node = emit("mul %d %d" % (const(gamma), prod))  # scalar -> free
        outs.append(out_node)

    sys.stdout.write("%d\n" % len(lines))
    sys.stdout.write("\n".join(lines))
    if lines:
        sys.stdout.write("\n")
    sys.stdout.write("out " + " ".join(map(str, outs)) + "\n")

if __name__ == "__main__":
    main()
