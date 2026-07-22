#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0773 -- "Restore the Shredded Royal Multiplication Ledger"
(family: masked-cayley-completion).

Setup: a royal treasury kept its unit-conversion / tribute-multiplication ledger as an
n x n table: row i, column j -> ledger[i][j], a "code" in 0..n-1. Unknown to anyone
reading the ledger today, the table was ORIGINALLY the Cayley (multiplication) table of
a finite group G of order n (rows/columns/symbols = group elements, cell = the product),
but a clerk long ago relabeled the rows, the columns, and the symbols independently
(three unrelated permutations) before the ledger was copied down -- so the visible table
is an ISOTOPE of a group table, not the group table itself. Fire then destroyed 55-80%
of the cells. The candidate must reconstruct the WHOLE table from the surviving fragment.

THE MATHEMATICAL HOOK (quadrangle criterion + isotopy reconstruction): a Latin square is
isotopic to the Cayley table of SOME group iff it satisfies the quadrangle criterion: if
three corners of two "rectangles" (row,col) agree, the fourth entries are forced equal.
Equivalently -- and this is what makes small fragments so informative -- any two rows of
such a table are related by a single FIXED permutation of symbols (independent of which
column you look at), and that permutation belongs to the tiny family of "left translations"
of the underlying group, not an arbitrary permutation of n! possibilities. So a handful of
correctly-placed surviving cells, combined with recognizing WHICH group (up to isomorphism)
generated the table, pins down literally every other cell -- even ones nowhere near any
surviving data. Local, cell-by-cell reasoning (classic Latin-square constraint propagation:
each empty cell must avoid the values already used in its row and column) has no access to
this global algebraic rigidity and stalls badly once erasure is heavy, because at that point
most cells have several locally-consistent candidates and no local rule to pick the right one.

The candidate is UNTRUSTED model output: it runs in an ISOLATED subprocess via `isorun`,
sees ONLY the public instance (n and the partial grid, with erased cells as null) on
stdin, and returns ONLY its answer on stdout.

Public instance (stdin JSON):
  {"n": <int>, "grid": [[code|null, ...], ...]}   # n x n, null = destroyed cell

Answer (stdout JSON):
  {"grid": [[code, ...], ...]}                    # n x n, every cell filled with an
                                                    # integer code in 0..n-1

Scoring (deterministic; no wall-time):
  The answer must reproduce every SURVIVING cell EXACTLY (any mismatch there is an
  inconsistent, malformed answer -> infeasible -> 0). Among the ORIGINALLY DESTROYED
  cells, let `correct` be the number matching the true hidden table, and `viol` be the
  number of row/column Latin-property violations (duplicate codes) anywhere in the
  returned grid. Then

      raw = correct / (#destroyed cells)  -  viol / (2*n*(n-1))     (clamped to [0,1])
      instance score = 0.9 * raw

  so a construction that is Latin-consistent everywhere but never exploits the group
  structure (getting only chance-level correctness on destroyed cells) scores low, a
  correct-but-imperfect Latin fill scores moderately, and full, structure-exploiting
  reconstruction (correct==destroyed, viol==0) scores 0.9 -- a cap that always leaves
  headroom above any reference solution. `Ratio` is the mean instance score over 10
  deterministic, seeded instances (several are held out at heavier erasure).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean score over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, itertools
import isorun

SCALE_CAP = 0.9


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


def _random_perm(r, n):
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        j = r(0, i)
        a[i], a[j] = a[j], a[i]
    return a


# ----------------------------- finite-group Cayley tables -------------------
def _cyclic_table(n):
    return [[(i + j) % n for j in range(n)] for i in range(n)]


def _prod_table(moduli):
    """Direct product of cyclic groups Z_{moduli[0]} x Z_{moduli[1]} x ... (mixed radix)."""
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


def _dihedral_table(m):
    """Dihedral group of order n=2m: element (s,r) = F^s R^r, index = s*m+r."""
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


def _quaternion8_table():
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


def _symmetric3_table():
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
    6: [("Z6", lambda: _cyclic_table(6)), ("S3", lambda: _symmetric3_table())],
    8: [("Z8", lambda: _cyclic_table(8)), ("Z4xZ2", lambda: _prod_table([4, 2])),
        ("Z2^3", lambda: _prod_table([2, 2, 2])), ("D4", lambda: _dihedral_table(4)),
        ("Q8", lambda: _quaternion8_table())],
    9: [("Z9", lambda: _cyclic_table(9)), ("Z3xZ3", lambda: _prod_table([3, 3]))],
    10: [("Z10", lambda: _cyclic_table(10)), ("D5", lambda: _dihedral_table(5))],
}


def _build_instance(seed, n, group_name, erase_frac):
    r = _rng(seed)
    M = dict(GROUP_LIB[n])[group_name]()
    pr = _random_perm(r, n); pc = _random_perm(r, n); ps = _random_perm(r, n)
    L = [[ps[M[pr[i]][pc[j]]] for j in range(n)] for i in range(n)]
    cells = [(i, j) for i in range(n) for j in range(n)]
    for k in range(len(cells) - 1, 0, -1):
        idx = r(0, k)
        cells[k], cells[idx] = cells[idx], cells[k]
    n_erase = int(round(erase_frac * n * n))
    erased = set(cells[:n_erase])
    grid = [[None if (i, j) in erased else L[i][j] for j in range(n)] for i in range(n)]
    public = {"n": n, "grid": grid}
    hidden = {"L": L, "erased": [[i, j] for (i, j) in sorted(erased)]}
    return {"public": public, "hidden": hidden}


# ----------------------------- instance family -------------------------------
def make_instances():
    specs = [
        # seed, n, group name, erasure fraction (fraction of the n*n cells destroyed)
        (101, 6, "S3", 0.60),
        (102, 6, "Z6", 0.65),
        (103, 8, "D4", 0.65),
        (104, 8, "Q8", 0.60),
        (105, 8, "Z4xZ2", 0.70),
        (106, 8, "Z2^3", 0.72),
        (107, 9, "Z9", 0.68),
        (108, 9, "Z3xZ3", 0.75),
        (109, 10, "D5", 0.72),
        (110, 10, "Z10", 0.80),   # held-out, heaviest erasure
    ]
    return [_build_instance(seed, n, gname, ef) for seed, n, gname, ef in specs]


# ----------------------------- scoring ---------------------------------------
def baseline(inst):
    """Reference-only: the raw correctness fraction of the naive per-row 'list the
    unused symbols in column order' fill (no group reasoning, no column awareness).
    Not used directly in the score formula (see score()); kept for audit/documentation
    per the evaluator contract."""
    pub = inst["public"]; n = pub["n"]; grid = pub["grid"]; L = inst["hidden"]["L"]
    fill = [row[:] for row in grid]
    for i in range(n):
        known = set(x for x in fill[i] if x is not None)
        missing = sorted(set(range(n)) - known)
        k = 0
        for j in range(n):
            if fill[i][j] is None:
                fill[i][j] = missing[k]; k += 1
    erased = inst["hidden"]["erased"]
    if not erased:
        return 1.0
    correct = sum(1 for (i, j) in erased if fill[i][j] == L[i][j])
    return correct / len(erased)


def score(inst, answer):
    """Strictly validate `answer` against the instance; return (ok, obj) with obj in
    [0,1] already the FINAL per-instance ratio (0.9 cap baked in)."""
    pub = inst["public"]; n = pub["n"]; grid = pub["grid"]
    L = inst["hidden"]["L"]; erased = inst["hidden"]["erased"]

    if not isinstance(answer, dict):
        return False, None
    ans_grid = answer.get("grid", None)
    if not isinstance(ans_grid, list) or len(ans_grid) != n:
        return False, None
    parsed = []
    for row in ans_grid:
        if not isinstance(row, list) or len(row) != n:
            return False, None
        prow = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int):
                return False, None
            if not (0 <= v <= n - 1):
                return False, None
            prow.append(v)
        parsed.append(prow)

    # every surviving (non-erased) cell must be reproduced EXACTLY
    for i in range(n):
        for j in range(n):
            if grid[i][j] is not None and parsed[i][j] != grid[i][j]:
                return False, None

    n_erased = len(erased)
    if n_erased == 0:
        return True, SCALE_CAP

    correct = sum(1 for (i, j) in erased if parsed[i][j] == L[i][j])
    row_viol = sum(n - len(set(row)) for row in parsed)
    col_viol = sum(n - len(set(parsed[i][j] for i in range(n))) for j in range(n))
    total_viol = row_viol + col_viol
    max_viol = 2 * n * (n - 1)

    raw = correct / n_erased
    pen = total_viol / max_viol if max_viol else 0.0
    obj = max(0.0, min(1.0, raw - pen))
    return True, obj * SCALE_CAP


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None:
            vec.append(0.0); continue
        vec.append(obj if (obj == obj and 0.0 <= obj <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
