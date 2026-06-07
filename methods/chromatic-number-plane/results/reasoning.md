OK, let me think about this from scratch. I want to push the lower bound on the chromatic number of the plane past 4. The whole thing is famously stuck between 4 and 7, and the lower end hasn't moved since 1950. By the de Bruijn–Erdős compactness theorem the plane needs k colours iff every finite unit-distance graph does, so my real target is concrete: build a finite graph drawn in the plane with every listed edge exactly distance 1 that admits no proper 4-colouring. If I can draw such a graph with literal unit segments as edges, then any 4-colouring of the whole plane, restricted to my points, would 4-colour it — impossible — so five colours are needed. The difficulty is the tension baked into the problem: I get to control only *local* geometry (which points are unit apart), but non-4-colourability is a *global* property. I need local gadgets whose constraints accumulate into a global impossibility.

So first: what do I actually have in my hands? The clean compact witness for the lower bound 4 is the Moser spindle. Let me make sure I understand why it works, because whatever I build has to be made of the same kind of stuff. Take a rhombus with unit sides and angles 60°/120° — that's just two unit equilateral triangles glued along an edge. The short diagonal is the shared edge, length 1; the long diagonal joins the two acute (60°) corners. Its length is 2·cos30° = √3. Now 3-colour this rhombus. The two endpoints of the shared edge are adjacent, so they're two different colours, say 1 and 2. Each acute vertex is adjacent to both of those endpoints (it forms a triangle with them), so it can't be 1 and can't be 2 — it's forced to 3. Both acute vertices forced to 3. So in any 3-colouring of a unit rhombus, the two acute vertices, which sit √3 apart, are the same colour. Let me hold onto that, it feels like the engine.

The spindle then takes two such rhombi, glues them at one acute vertex, and rotates one until the two *far* acute vertices land exactly distance 1 apart, and adds that edge. The far vertices are each √3 from the shared vertex, so to bring their chord to 1 I rotate by 2·arcsin(1/(2√3)) — chord = 2·√3·sin(arcsin(1/(2√3))) = 1, good. The full rotation has cos θ = 1 - 2·(1/(2√3))² = 5/6 and sin θ = √11/6, so I can write the spindle in exact square-root arithmetic. Now 3-colour the spindle: by the rhombus fact, each far vertex equals the shared vertex's colour; so the two far vertices are equal; but they're joined by a unit edge — contradiction. Not 3-colourable. It *is* 4-colourable. χ = 4. Let me actually verify this algebraically rather than trust the picture — build the seven points, check all eleven edges really are unit length, run a colourer.

(I went and did this. Rhombus: all five edges exactly unit in exact √3-arithmetic, acute pair exactly √3 apart, and across all six proper 3-colourings the acute pair is monochromatic every single time. Spindle: 7 vertices, 11 edges, all unit in Q(√3,√11), and a backtracking colourer says not-3-colourable, is-4-colourable, χ=4. Good, the engine runs.)

Now I hit the wall. The spindle gives me χ=4, not 5. The naive hope is "stack more spindles and the number climbs." It doesn't. In a 4-colouring the spindle does not hand me a forced monochromatic √3-pair; it only says that the two distinguished √3-pairs sharing the glued acute vertex cannot both be monochromatic, because then the far acute vertices would be equal across their unit edge. That is useful pressure, but not a 5-chromatic obstruction by itself, and decades of bolting spindles together by hand have not produced anything that resisted 4 colours. The lesson I take: I'm not going to *increase* the chromatic number by piling on local 4-chromatic gadgets and hoping. I have to *engineer* a contradiction — set up a situation where one part of the graph forces a coloured feature to appear and another part forces that same feature never to appear.

So what feature? It should be the natural currency of these gadgets: a monochromatic √3-equilateral-triangle. Three mutually-√3 points all one colour. Three √3-monochromatic pairs concentrated at one place. The rhombi produce √3-pairs; if I can make a small gadget whose 4-colourings split cleanly into "contains such a triple" versus "doesn't", then "triple present?" becomes a binary handle I can push from both sides.

A small natural gadget has the right split. I want something whose triples are √3-equilateral. The vertices and centre of a regular hexagon of side 1: call it H, 7 points. The centre is unit-distant from all six corners (regular hexagon of side 1 has circumradius 1), and consecutive corners are unit-distant. That's 12 unit edges. A proper colouring cannot put the centre in a monochromatic triple with corners, because the centre touches every corner. So any triple has to live among the six corners, and the only corner triples with all pairwise distances √3 are the two alternating triples. Let me think about the colourings with at most four colours. Up to rotation, reflection, and permuting colour names, H has exactly four essentially distinct ones: two contain one of those alternating monochromatic triples, and two don't. So H is precisely a *colour classifier*: every 4-colouring of H is, locally, either "structured" (has the mono triple) or "unstructured" (doesn't). That's the slot I need.

Now the plan crystallises into two opposing jobs. First I need to assemble many rotated copies of H so that in every 4-colouring of the assembly, at least one copy of H is structured. Then I need a graph around a single H so that every 4-colouring leaves that central H unstructured. If I can do both and then glue them — make the central H of the second graph land on each copy of H in the first graph — one side says some H is structured and the other says that same H isn't. That's the whole architecture. Let me try to actually build each side.

I start the forcing side small and grow, watching what each rotation buys me. Put one H at the centre, six copies centred at distance 1 around it, six more centred at distance √3 — a 31-vertex graph, 13 copies of H; call it J. Why this shell pattern? Because distance 1 and √3 are exactly the distances at which H-copies share or nearly-share edges, so the copies interlock rather than float free, and their colourings become mutually constrained. Now enumerate the 4-colourings of J in which *no* copy of H is structured — I am assuming the forcing attempt fails and trying to see how constrained that assumption is. Grouping by whether the central H has two monochromatic √3-pairs or none, and then whether the surrounding copies do, it turns out there are only six essentially distinct such colourings. And — staring at those six — they all agree on a coarse feature: the six vertices at distance 2 from the centre (call them the *linking vertices*) take only three patterns. Either (a) all six match the centre's colour; or (b) four consecutive ones match the centre and the other two share a second colour; or (c) two *opposite* ones match the centre and the other four share a second colour. (I had to add a couple of auxiliary vertices — "black" vertices that can take any consistent colour — to kill some spurious extra colourings that would otherwise appear if those vertices were deleted; without them, seven of the copies could go unstructured in a way that breaks the linking-vertex regularity. Fine, keep them.)

Why do I care about the linking vertices? Because they're the handle for gluing two copies of J. Notice option (c): two opposite linking vertices match the centre, the other four are a second colour — which means each *linking diagonal* (a pair of opposite linking vertices) is monochromatic in case (c). That's a strong, exportable fact. Can I force case (c)? Take J and a second copy of J rotated about the origin by 2·arcsin(1/4) — the angle that brings *corresponding* linking vertices of the two copies to unit distance from each other. Call the union K, 61 vertices, 26 copies of H. Now in any 4-colouring of K with no structured H, both J's must individually be in one of (a)/(b)/(c); but the new unit edges between corresponding linking vertices rule out (a) and (b) and force *both* copies into (c). So all six linking diagonals of K are monochromatic. The rotation manufactured exactly one new constraint — a unit edge between linking vertices — and that one edge collapses three options to one. This is the spindle move generalised: rotate a rigid sub-structure until a chosen pair of vertices becomes a unit edge, and let that edge do combinatorial work.

Repeat the move one level up. Take K and a copy of K rotated about one linking vertex A by 2·arcsin(1/8) — the angle that brings the opposite linking vertex B to unit distance from its counterpart B′ in the rotated copy. Union them into L: 121 vertices, 52 copies of H. Now suppose, for contradiction, that L has a 4-colouring with no structured H. Then within each copy of K all linking diagonals are monochromatic, so A and B are the same colour (they're opposite ends of a diagonal), and A and B′ are the same colour. But B and B′ are now joined by a unit edge, so they must differ. That gives B = A = B′ and B ≠ B′. Hence in every 4-colouring of L, at least one of the 52 copies of H is structured — has a monochromatic triple. The recursion is the point: at each level I rotate to create one unit edge between a pair of vertices that the lower level has already forced to be monochromatic, and that edge breaks the assumption.

The forbidding side is harder, and it's where I burn through a wrong idea first. I want a graph M containing one H such that no 4-colouring of M makes that H structured — no 4-colouring puts a monochromatic √3-triple on it. Why should such a thing even exist? A monochromatic √3-triple is a local concentration of monochromatic √3-pairs — three of them meeting at one place. Spindles, on the other hand, do not let their two distinguished √3-pairs both be monochromatic: if both far vertices matched the shared acute vertex, the far vertices would match each other across their unit edge. So in a dense web of overlapping spindles, monochromatic √3-pairs should be forced to spread rather than pile up. Uniformity is the enemy of concentration. If I surround H with a dense enough mesh of interlocking spindles, a triple on the central H becomes the kind of local spike the mesh should forbid. That's the bet.

So I want graphs with very high spindle density. First attempt: I notice a nice 9-vertex graph T — a spindle plus two extra vertices P, Q chosen so that P, Q, and the spindle's tip X form a unit equilateral triangle, and P, Q lie on the line extending the spindle's base. Adding those two vertices lets a single spindle "host" extra structure cheaply. With the triangular symmetry, I can take three spindles into a 15-vertex graph U with full triangular (rotational + reflectional) symmetry. Then I tile by translations and 60° rotations of U, expecting enormous spindle density. And indeed I can get a 97-vertex graph carrying 78 spindles — dense. I write a custom program to test such graphs: does a 4-colouring exist that *does* make the central H structured? If yes, the graph fails as M; I want a graph where the answer is no.

I check examples well past a thousand vertices and none of them works — every one of these translation/60°-rotation graphs still admits a 4-colouring with a structured central H. Why? Because everything built from a single U and 60° rotations lives in too few orientations: the spindles occupy only six orientations, and the edges fall into just three equivalence classes under 60° rotation. With so few edge directions the mesh isn't *rigid* enough to smear the monochromatic pairs uniformly — there is enough slack for a colouring to pile three monochromatic √3-pairs onto the central H. Maximum vertex degree caps out around 18. The mesh is dense but not *interlocked* enough.

So I patch the source of the slack: add more edge directions. The natural place to find good new directions is the geometry of spindles that *share a lot of vertices* — two or three tightly-linked spindles sit at specific relative orientations, and those orientations give me new edge classes. Concretely I let edge directions be i·arcsin(√3/2) + j·arcsin(1/√12) off the x-axis, with i ∈ 0…5 and j ∈ −2…2. That's the six 60°-spaced families, each split into five by the spindle-linking offset arcsin(1/√12). Now the maximum vertex degree jumps from 18 to 30 — call this richer local graph V. More directions means a stiffer, more over-constrained mesh, which is exactly what I argued I need to forbid the local spike.

Now build M properly. Let W be the set of all points within distance √3 of the origin reachable as the sum of two edges of V (edges read as vectors) — a 301-vertex disk of the V-mesh. Take W together with six translates of it, each translate moving the origin onto a vertex of the central H. That's a 1345-vertex graph: a dense V-mesh wrapped around H, one lobe per hexagon vertex. Run the custom program: is there any 4-colouring of this graph in which the central H is structured? It comes back: none. So this graph is a valid M — every 4-colouring leaves its central H unstructured. The added edge directions were the whole fix; with only the original three classes the mesh wasn't tight enough, and with the spindle-linkage classes it is.

Let me pin down *how* I'm deciding "no 4-colouring makes the central H structured", because for a graph this size I can't enumerate colourings. But I don't need to. The question is one-sided — not "what is χ", just "does a 4-colouring with a particular feature exist" — and the graph is so edge- and spindle-dense that the search collapses fast. I fix the colours of the seven vertices of the central H to one of its structured patterns (by symmetry only the two essentially-distinct triple-containing patterns of H need checking), then do depth-first colouring of the rest: colour the next vertex with its first untried colour, then look at its uncoloured neighbours — if any already sees neighbours of all four colours I must backtrack; if any sees exactly three colours, its colour is *forced*, so I assign it immediately and cascade. Order the vertices so the most constrained come first — by how many spindles they're in, then degree, then how many unit triangles they're in. Because the mesh is so dense, only around 20 branch choices are usually needed before forced-colour propagation fills almost everything else, so the tree is tiny; it runs in minutes. This is exactly the one-hot SAT picture — x[v,c], at-least-one per vertex, and an edge clause (¬x[u,c] ∨ ¬x[w,c]) per edge and colour — solved by unit propagation plus backtracking, just specialised to exploit the density.

Now glue the two sides. Take L, where every 4-colouring makes some one of its 52 H-copies structured, and for each of those 52 copies drop in a copy of M, rotated and translated so M's central H lands exactly on that copy. Merge coincident vertices. Call the union N. Now suppose N had a proper 4-colouring. Restrict it to L: some copy of H is structured. But that copy is the central H of some embedded M; restrict the same colouring to that M, and that H is unstructured. The same H can't be both. So N has no proper 4-colouring. N is a unit-distance graph, everything was built from unit segments and rigid rotations, and after merging it has 20425 vertices. The chromatic number of the plane is at least 5.

Two honesty checks before I'm satisfied. Is N genuinely unit-distance — are all the edges real unit segments after the rotations by 2·arcsin(1/4), 2·arcsin(1/8), and the V-direction offsets? Yes: every piece is built from unit edges, rotations preserve distances, and the two staged rotations are chosen by the chord formula. For K the relevant radius is 2, so 2·2·sin(arcsin(1/4)) = 1; for L the relevant radius is 4, so 2·4·sin(arcsin(1/8)) = 1. The rhombus check verifies the √3 distance exactly, and the spindle check verifies the 2·arcsin(1/(2√3)) chord and χ = 4. And is the non-colourability argument trustworthy, given it leans on a custom program for M? That nags at me — custom code can be buggy, and a wrong UNSAT here would be a wrong theorem. The fix is size. If I can shrink N drastically, a standard SAT solver can re-verify non-4-colourability from scratch, with a certificate an independent checker can validate.

So shrink. The construction is stepwise, so I can attack it stepwise: inside each M-like piece, find vertices whose deletion still leaves the required "central H never structured" property intact, and delete them; then look for single new vertices whose addition lets me delete more than one old vertex. It's crude — just exploiting the structure I already have — but it works surprisingly well. Concretely the record graph G is assembled from a 40-point seed set S (with explicit coordinates in Q[√3, √11, √33]), its closure S_a under the 60° rotations and y-negation (397 vertices), a copy S_b rotated by 2·arcsin(1/4), their union Y with (1/3,0) and (−1/3,0) deleted, and finally two rotated copies of Y (about (−2,0) by π/2 ± arcsin(1/8)) unioned together. That collapses N by a factor of about 13 down to 1581 vertices — and it's small enough that standard SAT solvers can check both sides directly: a 5-colouring exists, while the 4-colour CNF is UNSAT. The lower bound is real.

Stepping back at the causal chain: the rhombus forces a √3-pair monochromatic under 3 colours; H turns "monochromatic √3-triple present?" into a clean binary classifier of 4-colourings; rotating rigid sub-structures by the angles that make a forced-equal pair into a unit edge (2·arcsin(1/4), then 2·arcsin(1/8)) lets me force, level by level, that L always has a structured H; densely interlocking spindles — enriched with extra edge directions from tightly-linked-spindle geometry to stiffen the mesh — spread the constrained √3-pairs enough that M's central H is never structured; gluing L to 52 copies of M makes the same H structured and unstructured, so N is not 4-colourable; and shrinking N to G brings the whole thing within reach of an independent SAT check. I want the executable core to check the two load-bearing gadgets and the colourability/SAT machinery:

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
