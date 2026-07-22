# TIER: trivial
# Canonical baseline: evaluate every DISTINCT monomial once (shared univariate
# powers, one product-chain per monomial), then sum coeff*monomial per target.
# This reproduces the checker's baseline B -> ratio ~= 0.1.  No factor sharing.
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

def main():
    data = sys.stdin.read().split()
    p, n, k, polys = read_instance(data)

    lines = []
    nxt = [n]
    def emit(s):
        idx = nxt[0]; lines.append(s); nxt[0] += 1; return idx
    const_cache = {}
    def const(v):
        v %= p
        if v not in const_cache:
            const_cache[v] = emit("const %d" % v)
        return const_cache[v]

    # shared univariate powers: pow_node[(j,e)]
    powcache = {}
    def var_pow(j, e):
        if e == 1:
            return j
        key = (j, e)
        if key in powcache:
            return powcache[key]
        prev = var_pow(j, e - 1)
        node = emit("mul %d %d" % (prev, j))
        powcache[key] = node
        return node

    mono_cache = {}
    def mono_node(exps):
        if exps in mono_cache:
            return mono_cache[exps]
        parts = []
        for j in range(n):
            e = exps[j]
            if e:
                parts.append(var_pow(j, e))
        # parts non-empty (support>=1); left-fold
        cur = parts[0]
        for q in parts[1:]:
            cur = emit("mul %d %d" % (cur, q))
        mono_cache[exps] = cur
        return cur

    outs = []
    for mons in polys:
        acc = None
        for c, exps in mons:
            mn = mono_node(exps)
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
