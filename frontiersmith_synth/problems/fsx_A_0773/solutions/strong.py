# TIER: strong
# The insight: don't reason cell-by-cell. Recognize WHICH finite group (up to
# isomorphism) the ledger is isotopic to, from a small library of candidate groups of
# the right order, then use the quadrangle criterion to pin down the isotopy itself.
#
# Concretely: a Latin square isotopic to a group's Cayley table means there exist THREE
# unknown permutations (of rows, columns, symbols) such that grid[i][j] =
# sym_perm(M[row_perm[i]][col_perm[j]]) for the candidate group's table M. The KEY fact
# (the quadrangle criterion in operational form) is that grid[i][j]=grid[i'][j']=v with
# grid[i][j']=v2 known and grid[i'][j] known forces an EQUALITY between abstract group
# PRODUCTS: two surviving cells that show the SAME visible code v must correspond to the
# SAME product row_perm[i]*col_perm[j] in the candidate group, no matter how far apart
# they are in the ledger. That equality is a rigid combinatorial constraint we can
# search over directly: assign row_perm/col_perm (each a permutation of the group's n
# elements) one row/column at a time, using "most-constrained-first" ordering, and prune
# every branch that would make two same-valued cells correspond to different abstract
# products (or two different-valued cells correspond to the same product). Once
# row_perm/col_perm are pinned down consistently with every surviving cell, the symbol
# permutation (and hence literally the entire table) falls out for free -- cells nowhere
# near any surviving data get filled correctly too, because they are determined by the
# group law, not by local adjacency.
#
# If no candidate group in the small library fits (or the search is inconclusive within
# its node budget), fall back to the same constraint-propagation greedy used above --
# never worse than the recipe, and typically far better.
import sys, json, itertools

inst = json.load(sys.stdin)
n = inst["n"]
grid = inst["grid"]

NODE_BUDGET_PER_GROUP = 60000   # deterministic search-node cap (NOT wall-clock: keeps
                                  # the algorithm exactly reproducible on any machine)


# ----------------------------- finite-group Cayley tables (solver's own library) -----
def cyclic_table(n):
    return [[(i + j) % n for j in range(n)] for i in range(n)]


def prod_table(moduli):
    n = 1
    for m in moduli:
        n *= m

    def to_tuple(x):
        t = []
        for m in reversed(moduli):
            t.append(x % m); x //= m
        return tuple(reversed(t))

    def to_idx(t):
        x = 0
        for v, m in zip(t, moduli):
            x = x * m + v
        return x

    tab = [[0] * n for _ in range(n)]
    for i in range(n):
        ti = to_tuple(i)
        for j in range(n):
            tj = to_tuple(j)
            tk = tuple((a + b) % m for a, b, m in zip(ti, tj, moduli))
            tab[i][j] = to_idx(tk)
    return tab


def dihedral_table(m):
    n = 2 * m

    def idx(s, r):
        return s * m + (r % m)

    tab = [[0] * n for _ in range(n)]
    for i in range(n):
        s1, r1 = divmod(i, m)
        for j in range(n):
            s2, r2 = divmod(j, m)
            if s2 == 0:
                tab[i][j] = idx(s1, r1 + r2)
            else:
                tab[i][j] = idx((s1 + 1) % 2, r2 - r1)
    return tab


def quaternion8_table():
    names = ['1', '-1', 'i', '-i', 'j', '-j', 'k', '-k']
    units = {
        ('1', '1'): (1, '1'),
        ('i', 'i'): (-1, '1'), ('j', 'j'): (-1, '1'), ('k', 'k'): (-1, '1'),
        ('i', 'j'): (1, 'k'), ('j', 'k'): (1, 'i'), ('k', 'i'): (1, 'j'),
        ('j', 'i'): (-1, 'k'), ('k', 'j'): (-1, 'i'), ('i', 'k'): (-1, 'j'),
    }

    def unit_mul(u1, u2):
        if u1 == '1':
            return (1, u2)
        if u2 == '1':
            return (1, u1)
        return units[(u1, u2)]

    def mul1(a, b):
        sa = -1 if a.startswith('-') else 1
        sb = -1 if b.startswith('-') else 1
        ua = a[1:] if a.startswith('-') else a
        ub = b[1:] if b.startswith('-') else b
        s2, r = unit_mul(ua, ub)
        s = sa * sb * s2
        return r if s == 1 else ('-' + r)

    idxof = {nm: i for i, nm in enumerate(names)}
    tab = [[0] * 8 for _ in range(8)]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            tab[i][j] = idxof[mul1(a, b)]
    return tab


def symmetric3_table():
    perms = sorted(itertools.permutations(range(3)))
    idxof = {p: i for i, p in enumerate(perms)}

    def compose(p, q):
        return tuple(p[q[x]] for x in range(3))

    n = len(perms)
    tab = [[0] * n for _ in range(n)]
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            tab[i][j] = idxof[compose(p, q)]
    return tab


GROUP_LIB = {
    6: [cyclic_table(6), symmetric3_table()],
    8: [cyclic_table(8), prod_table([4, 2]), prod_table([2, 2, 2]),
        dihedral_table(4), quaternion8_table()],
    9: [cyclic_table(9), prod_table([3, 3])],
    10: [cyclic_table(10), dihedral_table(5)],
}


# ----------------------------- isotopy backtracking search ---------------------------
def try_isotopy_match(n, M, grid, node_budget):
    known_row = [[] for _ in range(n)]
    known_col = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v is not None:
                known_row[i].append((j, v)); known_col[j].append((i, v))

    row_perm = [None] * n
    col_perm = [None] * n
    row_used = [False] * n
    col_used = [False] * n
    val2prod = {}
    prod2val = {}
    nodes = [0]

    def undo_assign(added):
        for (v, prod) in added:
            del val2prod[v]; del prod2val[prod]

    def try_assign(kind, idx, g):
        added = []
        cells = known_row[idx] if kind == 'R' else known_col[idx]
        for (pos, v) in cells:
            if kind == 'R':
                if col_perm[pos] is None:
                    continue
                prod = M[g][col_perm[pos]]
            else:
                if row_perm[pos] is None:
                    continue
                prod = M[row_perm[pos]][g]
            if v in val2prod:
                if val2prod[v] != prod:
                    undo_assign(added); return None
            elif prod in prod2val:
                if prod2val[prod] != v:
                    undo_assign(added); return None
            else:
                val2prod[v] = prod; prod2val[prod] = v
                added.append((v, prod))
        return added

    def pick_var():
        best = None
        for i in range(n):
            if row_perm[i] is None:
                score = sum(1 for (j, v) in known_row[i] if col_perm[j] is not None)
                score2 = len(known_row[i])
                if best is None or (score, score2) > best[2]:
                    best = ('R', i, (score, score2))
        for j in range(n):
            if col_perm[j] is None:
                score = sum(1 for (i, v) in known_col[j] if row_perm[i] is not None)
                score2 = len(known_col[j])
                if best is None or (score, score2) > best[2]:
                    best = ('C', j, (score, score2))
        return best

    def backtrack():
        nodes[0] += 1
        if nodes[0] > node_budget:
            return False
        var = pick_var()
        if var is None:
            return True
        kind, idx, _ = var
        used = row_used if kind == 'R' else col_used
        for g in range(n):
            if used[g]:
                continue
            added = try_assign(kind, idx, g)
            if added is None:
                continue
            if kind == 'R':
                row_perm[idx] = g
            else:
                col_perm[idx] = g
            used[g] = True
            if backtrack():
                return True
            used[g] = False
            if kind == 'R':
                row_perm[idx] = None
            else:
                col_perm[idx] = None
            undo_assign(added)
        return False

    if not backtrack():
        return None
    leftover_prods = [p for p in range(n) if p not in prod2val]
    leftover_vals = [v for v in range(n) if v not in val2prod]
    for p, v in zip(leftover_prods, leftover_vals):
        prod2val[p] = v
    sym_perm = [prod2val[p] for p in range(n)]
    return row_perm, col_perm, sym_perm


def constraint_propagation_fill(n, grid):
    g = [row[:] for row in grid]
    changed = True
    while changed:
        changed = False
        row_missing = [set(range(n)) - set(x for x in g[i] if x is not None) for i in range(n)]
        col_missing = [set(range(n)) - set(g[i][j] for i in range(n) if g[i][j] is not None) for j in range(n)]
        for i in range(n):
            for j in range(n):
                if g[i][j] is None:
                    cand = row_missing[i] & col_missing[j]
                    if len(cand) == 1:
                        v = next(iter(cand))
                        g[i][j] = v
                        row_missing[i].discard(v); col_missing[j].discard(v)
                        changed = True
    for i in range(n):
        for j in range(n):
            if g[i][j] is None:
                used_row = set(x for x in g[i] if x is not None)
                used_col = set(g[k][j] for k in range(n) if g[k][j] is not None)
                cand = sorted(set(range(n)) - used_row - used_col)
                if cand:
                    g[i][j] = cand[0]
                else:
                    rem = sorted(set(range(n)) - used_row)
                    g[i][j] = rem[0] if rem else 0
    return g


def solve():
    for M in GROUP_LIB.get(n, []):
        res = try_isotopy_match(n, M, grid, NODE_BUDGET_PER_GROUP)
        if res is None:
            continue
        row_perm, col_perm, sym_perm = res
        filled = [[sym_perm[M[row_perm[i]][col_perm[j]]] for j in range(n)] for i in range(n)]
        if all(grid[i][j] is None or filled[i][j] == grid[i][j] for i in range(n) for j in range(n)):
            return filled
    return constraint_propagation_fill(n, grid)


print(json.dumps({"grid": solve()}))
