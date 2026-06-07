# A 5-chromatic unit-distance graph: the chromatic number of the plane is at least 5

## Problem

Colour the plane so no two points at distance exactly 1 share a colour; CNP is the minimum
number of colours. By compactness, CNP ≥ 5 follows from a single finite unit-distance graph
(points in the plane, edges between unit-distant pairs) that admits no proper 4-colouring.
Since 1950 only 4 ≤ CNP ≤ 7 was known. The construction below exhibits such a non-4-colourable
finite unit-distance graph.

## Key idea

Engineer a contradiction around one feature: a **monochromatic √3-equilateral triple**.

- **Engine (rhombus).** A unit rhombus = two unit equilateral triangles sharing an edge. In
  any 3-colouring its two acute (60°) vertices — which are √3 apart — are forced to the same
  colour. The **Moser spindle** (two rhombi sharing an acute vertex, the far acute vertices
  rotated to unit distance by 2·arcsin(1/(2√3)), equivalently cos = 5/6 and sin = √11/6)
  is the 7-vertex, 11-edge graph with χ = 4 built from this.

- **Classifier (H).** The hexagon-with-centre, 7 vertices, 12 unit edges, has exactly four
  essentially-distinct colourings with at most four colours; two contain a monochromatic
  √3-triple ("structured"), two don't. So "is this copy of H structured?" is a binary handle
  on any 4-colouring.

- **Forcer (L).** Assemble 52 rotated copies of H (built in stages J→K→L) so that **every**
  4-colouring makes at least one copy structured. The construction is recursive spindle logic:
  J (31 v, 13 H) constrains six "linking" vertices to three colour-patterns; K (61 v) is two
  J's rotated by 2·arcsin(1/4) so corresponding linking vertices become a unit edge, forcing
  the pattern in which each linking diagonal is monochromatic; L (121 v) is two K's rotated
  about a linking vertex A by 2·arcsin(1/8), making B and B′ a unit edge — but L's structure
  forces B = A = B′, contradicting that edge unless some H is structured.

- **Anti-forcer (M).** Build a unit-distance graph M (1345 v) around one H from a dense,
  interlocking mesh of spindles so that **no** 4-colouring makes its central H structured. A
  mono triple is a local concentration of mono √3-pairs; densely interlocked spindles are
  used to constrain those pairs away from such concentration. Plain 60°-rotation meshes
  (only 3 edge classes, max degree 18) are too slack; enriching the edge directions with the
  offsets of tightly-linked spindles — angles i·arcsin(√3/2)+j·arcsin(1/√12), i∈0…5,
  j∈−2…2 (graph V, max degree 30) — stiffens the mesh enough to work.

- **Compose (N).** Glue 52 copies of M onto L so each M's central H is one of L's copies.
  L forces some H structured; M forbids that H structured; contradiction. N (20425 v after
  merging) is not 4-colourable, so **CNP ≥ 5**.

- **Shrink + certify (G).** Stepwise vertex deletions reduce N by ~13× to a 1581-vertex graph
  G, small enough that standard SAT solvers check both halves of χ(G) = 5 directly: a
  5-colouring exists, and the 4-colour CNF is UNSAT. Proof logging can make the UNSAT half
  independently checkable.

## Certifying (un)colourability

M's property is checked by a one-sided depth-first colourer: fix the central H to a structured
pattern (two cases suffice by symmetry), colour the rest most-constrained-vertex first, and
propagate forced colours; the spindle density makes the search collapse in minutes. This is
the standard one-hot SAT encoding with extra clauses fixing the central H pattern — x[v,c];
an at-least-one clause per vertex; an edge clause (¬x[u,c] ∨ ¬x[w,c]) per edge and colour.
UNSAT in that constrained formula means no 4-colouring makes the central H structured; for
the final graph G, UNSAT of the unconstrained 4-colour formula means non-4-colourability.

## Code

```python
from fractions import Fraction as F
from itertools import combinations

class Qsqrt:
    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = F(a)
        self.b = F(b)
        self.c = F(c)
        self.d = F(d)

    def __add__(s, o): o = _q(o); return Qsqrt(s.a + o.a, s.b + o.b, s.c + o.c, s.d + o.d)
    def __radd__(s, o): return s + o
    def __sub__(s, o): o = _q(o); return Qsqrt(s.a - o.a, s.b - o.b, s.c - o.c, s.d - o.d)
    def __rsub__(s, o): return _q(o) - s

    def __mul__(s, o):
        o = _q(o)
        a = s.a * o.a + 3 * s.b * o.b + 11 * s.c * o.c + 33 * s.d * o.d
        b = s.a * o.b + s.b * o.a + 11 * (s.c * o.d + s.d * o.c)
        c = s.a * o.c + s.c * o.a + 3 * (s.b * o.d + s.d * o.b)
        d = s.a * o.d + s.b * o.c + s.c * o.b + s.d * o.a
        return Qsqrt(a, b, c, d)

    def __rmul__(s, o): return s * o
    def __eq__(s, o): o = _q(o); return (s.a, s.b, s.c, s.d) == (o.a, o.b, o.c, o.d)
    def __hash__(s): return hash((s.a, s.b, s.c, s.d))
    def __repr__(s): return f"{s.a}+{s.b}√3+{s.c}√11+{s.d}√33"


def _q(x):
    return x if isinstance(x, Qsqrt) else Qsqrt(x, 0, 0, 0)


ZERO = Qsqrt(0)
ONE = Qsqrt(1)
SQRT3 = Qsqrt(0, 1, 0, 0)
SQRT11 = Qsqrt(0, 0, 1, 0)


def dist2(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def is_unit(p, q):
    return dist2(p, q) == ONE


def rhombus():
    A = (ZERO, ZERO)
    B = (SQRT3, ZERO)
    C = (SQRT3 * F(1, 2), Qsqrt(F(1, 2)))
    D = (SQRT3 * F(1, 2), Qsqrt(F(-1, 2)))
    V = {"A": A, "B": B, "C": C, "D": D}
    E = [("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    return V, E, ("A", "B")


def moser_spindle():
    cos_t = Qsqrt(F(5, 6))
    sin_t = SQRT11 * F(1, 6)
    base = {
        "O": (ZERO, ZERO),
        "P": (SQRT3, ZERO),
        "c": (SQRT3 * F(1, 2), Qsqrt(F(1, 2))),
        "d": (SQRT3 * F(1, 2), Qsqrt(F(-1, 2))),
    }

    def rot(p):
        x, y = p
        return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    V = {
        "O": base["O"], "P1": base["P"], "c1": base["c"], "d1": base["d"],
        "P2": rot(base["P"]), "c2": rot(base["c"]), "d2": rot(base["d"]),
    }
    E = [("O", "c1"), ("O", "d1"), ("P1", "c1"), ("P1", "d1"), ("c1", "d1"),
         ("O", "c2"), ("O", "d2"), ("P2", "c2"), ("P2", "d2"), ("c2", "d2"),
         ("P1", "P2")]
    return V, E


def k_colour(vertices, edges, k):
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    order = sorted(vertices, key=lambda v: -len(adj[v]))
    colour = {}

    def forbidden(v):
        return {colour[w] for w in adj[v] if w in colour}

    def search(i):
        if i == len(order):
            return True
        v = order[i]
        bad = forbidden(v)
        for c in range(k):
            if c in bad:
                continue
            colour[v] = c
            if search(i + 1):
                return True
            del colour[v]
        return False

    return dict(colour) if search(0) else None


def chromatic_number(vertices, edges, hi=7):
    for k in range(1, hi + 1):
        if k_colour(vertices, edges, k) is not None:
            return k
    return None


def dimacs_kcol(vertices, edges, k):
    idx = {v: i for i, v in enumerate(vertices)}
    var = lambda v, c: idx[v] * k + c + 1
    clauses = []
    for v in vertices:
        clauses.append([var(v, c) for c in range(k)])
    for v in vertices:
        for c1, c2 in combinations(range(k), 2):
            clauses.append([-var(v, c1), -var(v, c2)])
    for u, w in edges:
        for c in range(k):
            clauses.append([-var(u, c), -var(w, c)])
    lines = [f"p cnf {len(vertices) * k} {len(clauses)}"]
    lines += [" ".join(map(str, cl)) + " 0" for cl in clauses]
    return "\n".join(lines)


if __name__ == "__main__":
    V, E, (a, b) = rhombus()
    for u, w in E:
        assert is_unit(V[u], V[w]), (u, w, dist2(V[u], V[w]))
    assert dist2(V[a], V[b]) == Qsqrt(3)

    verts = list(V)
    adj = {v: set() for v in V}
    for u, w in E:
        adj[u].add(w)
        adj[w].add(u)

    def all_colourings(k):
        out = []
        def rec(i, col):
            if i == len(verts):
                out.append(dict(col))
                return
            v = verts[i]
            for c in range(k):
                if all(col.get(w) != c for w in adj[v]):
                    col[v] = c
                    rec(i + 1, col)
                    del col[v]
        rec(0, {})
        return out

    cols = all_colourings(3)
    assert len(cols) == 6
    assert all(col["A"] == col["B"] for col in cols)

    Vs, Es = moser_spindle()
    assert len(Vs) == 7 and len(Es) == 11
    for u, w in Es:
        assert is_unit(Vs[u], Vs[w]), (u, w, dist2(Vs[u], Vs[w]))
    assert chromatic_number(list(Vs), Es, hi=6) == 4
    assert k_colour(list(Vs), Es, 3) is None
    assert k_colour(list(Vs), Es, 4) is not None
    assert dimacs_kcol(list(Vs), Es, 3).splitlines()[0] == "p cnf 21 61"
    print("OK: rhombus forced pair and Moser spindle chi = 4")
```

Running it confirms the two load-bearing gadget facts exactly: the rhombus is unit-distance
with a √3 acute pair forced monochromatic under 3 colours, and the Moser spindle is exactly
4-chromatic. The larger construction uses those facts through L, M, N and the 1581-vertex G;
SAT verification of G's 5-colourability and non-4-colourability gives χ(G) = 5 and
establishes CNP ≥ 5.
