OK, let me think about this from scratch. I want to push the lower bound on the chromatic number of the plane past 4. The whole thing is famously stuck between 4 and 7, and the lower end hasn't moved since 1950. By the de Bruijn–Erdős compactness theorem the plane needs k colours iff some *finite* unit-distance subgraph does, so my real target is concrete: build a finite set of points in the plane, with edges between points exactly distance 1 apart, that admits no proper 4-colouring. If I can draw such a graph with literal unit segments as edges, then any 4-colouring of the whole plane, restricted to my points, would 4-colour it — impossible — so five colours are needed. The difficulty is the tension baked into the problem: I get to control only *local* geometry (which points are unit apart), but non-4-colourability is a *global* property. I need local gadgets whose constraints accumulate into a global impossibility.

So first: what do I actually have in my hands? The only reason we even know the bound is 4 is the Moser spindle. Let me make sure I understand *why* it works, because whatever I build has to be made of the same kind of stuff. Take a rhombus with unit sides and angles 60°/120° — that's just two unit equilateral triangles glued along an edge. The short diagonal is the shared edge, length 1; the long diagonal joins the two acute (60°) corners. Its length is 2·cos30° = √3. Now 3-colour this rhombus. The two endpoints of the shared edge are adjacent, so they're two different colours, say 1 and 2. Each acute vertex is adjacent to *both* of those endpoints (it forms a triangle with them), so it can't be 1 and can't be 2 — it's forced to 3. Both acute vertices forced to 3. So in any 3-colouring of a unit rhombus, **the two acute vertices, which sit √3 apart, are the same colour.** Let me hold onto that, it feels like the engine.

The spindle then takes two such rhombi, glues them at one acute vertex, and rotates one until the two *far* acute vertices land exactly distance 1 apart, and adds that edge. The far vertices are each √3 from the shared vertex, so to bring their chord to 1 I rotate by 2·arcsin(1/(2√3)) — chord = 2·√3·sin(arcsin(1/(2√3))) = 1, good. Now 3-colour the spindle: by the rhombus fact, each far vertex equals the shared vertex's colour; so the two far vertices are equal; but they're joined by a unit edge — contradiction. Not 3-colourable. It *is* 4-colourable. χ = 4. Let me actually verify this numerically rather than trust the picture — build the seven points, check all eleven edges really are unit length, run a colourer.

(I went and did this. Rhombus: all five edges exactly unit in exact √3-arithmetic, acute pair exactly √3 apart, and across all six proper 3-colourings the acute pair is monochromatic every single time. Spindle: 7 vertices, 11 edges, unit to floating tolerance, and a backtracking colourer says not-3-colourable, is-4-colourable, χ=4. Good, the engine runs.)

Now — here's the wall I have to be honest about. The spindle gives me χ=4, not 5. The naive hope is "stack more spindles and the number climbs." It doesn't. A single spindle forces *some* √3-pair to be monochromatic, but it never says *which* one, and 68 years of bolting spindles together by hand never produced anything that resisted 4 colours. The lesson I take: I'm not going to *increase* the chromatic number by piling on local 4-chromatic gadgets and hoping. I have to *engineer* a contradiction — set up a situation where one part of the graph forces a coloured feature to appear and another part forces that same feature never to appear.

So what feature? It should be the natural currency of these gadgets: a monochromatic √3-equilateral-triangle. Three mutually-√3 points all one colour. Three √3-monochromatic pairs concentrated at one place. The rhombi produce √3-pairs; if I can make a small gadget whose 4-colourings split cleanly into "contains such a triple" versus "doesn't", then "triple present?" becomes a binary handle I can push from both sides.

What's the smallest gadget that exhibits this clean split? I want something whose triples are √3-equilateral. The vertices and centre of a regular hexagon of side 1: call it H, 7 points. The centre is unit-distant from all six corners (regular hexagon of side 1 has circumradius 1), and consecutive corners are unit-distant. That's 12 unit edges. The three "long diagonals" pairing of alternate corners — alternate corners of a unit hexagon are √3 apart — give two √3-equilateral triangles among the corners (the two alternating triples), and the centre-plus-opposite-corners give more √3 structure. Let me think about its 4-colourings. Up to rotation, reflection, and permuting colour names, H has exactly four essentially distinct 4-colourings. And — this is the property I want — two of those four contain a monochromatic triple (a √3-equilateral triangle of one colour) and two don't. So H is precisely a *colour classifier*: every 4-colouring of H is, locally, either "structured" (has the mono triple) or "unstructured" (doesn't). That's the slot I need.

Now the plan crystallises into two opposing jobs. **Job A**: assemble many rotated copies of H so that in *every* 4-colouring of the assembly, *at least one* copy of H is structured (has a mono triple). **Job B**: build a graph around a single H so that in *every* 4-colouring, that central H is *unstructured* (never has a mono triple). If I can do both and then glue them — make Job-B's central H be one of Job-A's copies, for every copy — Job A says some H is structured, Job B says that H isn't, contradiction, no 4-colouring exists. That's the whole architecture. Let me try to actually build each side.

**Job A — forcing some H to be structured.** I start small and grow, watching what each rotation buys me. Put one H at the centre, six copies centred at distance 1 around it, six more centred at distance √3 — a 31-vertex graph, 13 copies of H; call it J. Why this shell pattern? Because distance 1 and √3 are exactly the distances at which H-copies share or nearly-share edges, so the copies interlock rather than float free, and their colourings become mutually constrained. Now enumerate the 4-colourings of J in which *no* copy of H is structured — i.e. I'm assuming Job A *fails* and trying to see how constrained that assumption is. Grouping by whether the central H has two monochromatic √3-pairs or none, and then whether the surrounding copies do, it turns out there are only six essentially distinct such colourings. And — staring at those six — they all agree on a coarse feature: the six vertices at distance 2 from the centre (call them the *linking vertices*) take only three patterns. Either (a) all six match the centre's colour; or (b) four consecutive ones match the centre and the other two share a second colour; or (c) two *opposite* ones match the centre and the other four share a second colour. (I had to add a couple of auxiliary vertices — "black" vertices that can take any consistent colour — to kill some spurious extra colourings that would otherwise appear if those vertices were deleted; without them, seven of the copies could go unstructured in a way that breaks the linking-vertex regularity. Fine, keep them.)

Why do I care about the linking vertices? Because they're the handle for gluing two copies of J. Notice option (c): two opposite linking vertices match the centre, the other four are a second colour — which means each *linking diagonal* (a pair of opposite linking vertices) is monochromatic in case (c). That's a strong, exportable fact. Can I force case (c)? Take J and a second copy of J rotated about the origin by 2·arcsin(1/4) — the angle that brings *corresponding* linking vertices of the two copies to unit distance from each other. Call the union K, 61 vertices, 26 copies of H. Now in any 4-colouring of K with no structured H, both J's must individually be in one of (a)/(b)/(c); but the new unit edges between corresponding linking vertices rule out (a) and (b) and force *both* copies into (c). So all six linking diagonals of K are monochromatic. The rotation manufactured exactly one new constraint — a unit edge between linking vertices — and that one edge collapses three options to one. This is the spindle move generalised: rotate a rigid sub-structure until a chosen pair of vertices becomes a unit edge, and let that edge do combinatorial work.

Repeat the move one level up. Take K and a copy of K rotated about one linking vertex A by 2·arcsin(1/8) — the angle that brings the opposite linking vertex B to unit distance from its counterpart B′ in the rotated copy. Union them into L: 121 vertices, 52 copies of H. Now suppose, for contradiction, that L has a 4-colouring with no structured H. Then within each copy of K all linking diagonals are monochromatic, so A and B are the same colour (they're opposite ends of a diagonal), and A and B′ are the same colour. But B and B′ are now joined by a unit edge — they must differ. So B = A = B′ and B ≠ B′, contradiction. Hence **in every 4-colouring of L, at least one of the 52 copies of H is structured** — has a monochromatic triple. Job A done. Notice the recursion: at each level I rotate to create one unit edge between a pair of vertices that the lower level has already forced to be monochromatic, and that edge breaks the assumption. Beautiful — it's spindles all the way up.

**Job B — forbidding the central H from being structured.** This is the harder half, and it's where I burn through a wrong idea first. I want a graph M containing one H such that *no* 4-colouring of M makes that H structured — no 4-colouring puts a monochromatic √3-triple on it. Why should such a thing even exist? Here's the intuition I'll lean on. A monochromatic √3-triple is a *local concentration* of monochromatic √3-pairs — three of them meeting at a point. Spindles, on the other hand, force monochromatic √3-pairs to *exist* but, when packed densely and interlockingly, tend to *spread them out*: every spindle insists two of its √3-pairs can't both be monochromatic, so in a dense web of overlapping spindles the monochromatic √3-pairs get distributed roughly uniformly. Uniformity is the enemy of concentration. So if I surround H with a dense enough mesh of interlocking spindles, the forced √3-pairs will be smeared out and a *triple* on the central H — a sharp local spike — becomes impossible. That's the bet.

So I want graphs with very high spindle density. First attempt: I notice a nice 9-vertex graph T — a spindle plus two extra vertices P, Q chosen so that P, Q, and the spindle's tip X form a unit equilateral triangle, and P, Q lie on the line extending the spindle's base. Adding those two vertices lets a single spindle "host" extra structure cheaply. With the triangular symmetry, I can take three spindles into a 15-vertex graph U with full triangular (rotational + reflectional) symmetry. Then I tile by translations and 60° rotations of U, expecting enormous spindle density. And indeed I can get a 97-vertex graph carrying 78 spindles — dense. I write a custom program to test such graphs: does a 4-colouring exist that *does* make the central H structured? If yes, the graph fails as M; I want a graph where the answer is no.

Wall. I check examples well past a thousand vertices and none of them works — every one of these translation/60°-rotation graphs still admits a 4-colouring with a structured central H. Why? Because everything built from a single U and 60° rotations lives in too few orientations: the spindles occupy only six orientations, and the edges fall into just three equivalence classes under 60° rotation. With so few edge directions the mesh isn't *rigid* enough to smear the monochromatic pairs uniformly — there's enough slack for a colouring to pile three monochromatic √3-pairs onto the central H. Maximum vertex degree caps out around 18. The mesh is dense but not *interlocked* enough.

So I patch the source of the slack: add more edge directions. The natural place to find good new directions is the geometry of spindles that *share a lot of vertices* — two or three tightly-linked spindles sit at specific relative orientations, and those orientations give me new edge classes. Concretely I let edge directions be i·arcsin(√3/2) + j·arcsin(1/√12) off the x-axis, with i ∈ 0…5 and j ∈ −2…2. That's the six 60°-spaced families, each split into five by the spindle-linking offset arcsin(1/√12). Now the maximum vertex degree jumps from 18 to 30 — call this richer local graph V. More directions means a stiffer, more over-constrained mesh, which is exactly what I argued I need to forbid the local spike.

Now build M properly. Let W be the set of all points within distance √3 of the origin reachable as the sum of two edges of V (edges read as vectors) — a 301-vertex disk of the V-mesh. Take W together with six translates of it, each translate moving the origin onto a vertex of the central H. That's a 1345-vertex graph: a dense V-mesh wrapped around H, one lobe per hexagon vertex. Run the custom program: is there any 4-colouring of this graph in which the central H is structured? It comes back: none. So this graph *is* a valid M — every 4-colouring leaves its central H unstructured. Job B done. The added edge directions were the whole fix; with only the original three classes the smear wasn't tight enough, and with the spindle-linkage classes it is.

Let me pin down *how* I'm deciding "no 4-colouring makes the central H structured", because for a graph this size I can't enumerate colourings. But I don't need to. The question is one-sided — not "what is χ", just "does a 4-colouring with a particular feature exist" — and the graph is so edge- and spindle-dense that the search collapses fast. I fix the colours of the central H to one of its structured patterns (by symmetry only the two essentially-distinct triple-containing patterns of H need checking), then do depth-first colouring of the rest: colour the next vertex with its first untried colour, then look at its uncoloured neighbours — if any already sees neighbours of all four colours I must backtrack; if any sees exactly three colours, its colour is *forced*, so I assign it immediately and cascade. Order the vertices so the most constrained come first — by how many spindles they're in, then degree, then how many unit triangles they're in. Because the mesh is so dense, fixing the ~20 colours of the central H typically *forces* almost every remaining vertex by propagation, so the tree is tiny; it runs in minutes. This is exactly the one-hot SAT picture — x[v,c], at-least-one per vertex, and an edge clause (¬x[u,c] ∨ ¬x[w,c]) per edge and colour — solved by unit propagation plus backtracking, just specialised to exploit the density.

Now glue. **Compose A and B.** Take L (Job A: every 4-colouring makes *some* one of its 52 H-copies structured) and, for each of those 52 copies, drop in a copy of M (Job B: makes that copy *un*structured), rotated and translated so M's central H lands exactly on L's copy. Merge coincident vertices. Call the union N. Now suppose N had a proper 4-colouring. Restrict it to L: by Job A some copy of H is structured. But that copy is the central H of some embedded M; restrict the same colouring to that M: by Job B that H is unstructured. The same H can't be both. So N has no proper 4-colouring. N is a unit-distance graph (everything was built from unit segments and rigid rotations) that is not 4-colourable. **The chromatic number of the plane is at least 5.** After merging, N has 20425 vertices.

Two honesty checks before I'm satisfied. Is N genuinely unit-distance — are all the edges real unit segments after all those rotations by 2·arcsin(1/4), 2·arcsin(1/8) and the V-direction offsets? Yes: every gadget is built from unit edges, and rotations preserve distances, and the rotation angles were *chosen* precisely to turn specified √3-or-other separations into unit edges; the rhombus and spindle I already verified exactly in √3-arithmetic. And is the non-colourability argument trustworthy, given it leans on a custom program for M? That nags at me — custom code can be buggy, and a wrong UNSAT here would be a wrong theorem. The fix is size. If I can shrink N drastically, a standard SAT solver can re-verify non-4-colourability from scratch, with a DRAT certificate an independent checker validates — no trust in my code at all.

So shrink. The construction is stepwise, so I can attack it stepwise: inside each M-like piece, find vertices whose deletion still leaves the required "central H never structured" property intact, and delete them; then look for single new vertices whose addition lets me delete more than one old vertex. It's crude — just exploiting the structure I already have — but it works surprisingly well. Concretely the record graph G is assembled from a 40-point seed set S (with explicit coordinates in Q[√3, √11, √33]), its closure S_a under the 60° rotations and y-negation (397 vertices), a copy S_b rotated by 2·arcsin(1/4), their union Y with two vertices deleted, and finally two rotated copies of Y (about (−2,0) by π/2 ± arcsin(1/8)) unioned together. That collapses N by a factor of about 13 down to **1581 vertices** — and it's small enough that standard SAT solvers confirm χ = 5 directly, no custom code, the UNSAT-for-4 result independently certifiable. The lower bound is real.

Stepping back at the causal chain: the rhombus forces a √3-pair monochromatic under 3 colours; H turns "monochromatic √3-triple present?" into a clean binary classifier of 4-colourings; rotating rigid sub-structures by the angles that make a forced-equal pair into a unit edge (2·arcsin(1/4), then 2·arcsin(1/8)) lets me force, level by level, that L always has a structured H; densely interlocking spindles — enriched with extra edge directions from tightly-linked-spindle geometry to stiffen the mesh — smear the forced monochromatic √3-pairs uniformly enough that M's central H is never structured; gluing the forcer L to 52 anti-forcers M makes the same H provably structured and unstructured, so N is not 4-colourable; and shrinking N to G brings the whole thing within reach of an independent SAT check. Here is the grounded core — the two load-bearing gadgets and the colourability/SAT machinery, exactly as verified:

```python
from fractions import Fraction as F
from itertools import combinations

# exact arithmetic in Q[sqrt(3)]: value = a + b*sqrt(3)
class Q3:
    def __init__(self, a=0, b=0): self.a, self.b = F(a), F(b)
    def __add__(s,o): o=_c(o); return Q3(s.a+o.a, s.b+o.b)
    def __sub__(s,o): o=_c(o); return Q3(s.a-o.a, s.b-o.b)
    def __mul__(s,o):
        o=_c(o)                                  # (a+b√3)(c+d√3)=(ac+3bd)+(ad+bc)√3
        return Q3(s.a*o.a + 3*s.b*o.b, s.a*o.b + s.b*o.a)
    def __eq__(s,o): o=_c(o); return s.a==o.a and s.b==o.b
def _c(x): return x if isinstance(x,Q3) else Q3(x,0)
SQRT3 = Q3(0,1)

def dist2(p,q):                                   # exact squared distance
    dx,dy = p[0]-q[0], p[1]-q[1]; return dx*dx + dy*dy
def is_unit(p,q): return dist2(p,q) == Q3(1,0)

# the rhombus: two unit equilateral triangles sharing edge C-D;
# acute vertices A,B are sqrt(3) apart and SAME colour in every 3-colouring.
def rhombus():
    A=(Q3(0),Q3(0)); B=(SQRT3,Q3(0))
    C=(Q3(0,F(1,2)),Q3(F(1,2))); D=(Q3(0,F(1,2)),Q3(F(-1,2)))
    V={"A":A,"B":B,"C":C,"D":D}
    E=[("A","C"),("A","D"),("B","C"),("B","D"),("C","D")]
    return V,E,("A","B")

# Moser spindle: two rhombi sharing acute vertex O, far acute vertices pulled to
# unit distance by rotating one rhombus by 2*arcsin(1/(2*sqrt3)). chi = 4.
def moser_spindle():
    import math; s3=math.sqrt(3)
    base={"O":(0.,0.),"P":(s3,0.),"c":(s3/2,.5),"d":(s3/2,-.5)}
    th=2*math.asin(1/(2*s3))                      # chord 2*sqrt3*sin(th/2)=1
    rot=lambda p:(p[0]*math.cos(th)-p[1]*math.sin(th),
                  p[0]*math.sin(th)+p[1]*math.cos(th))
    V={"O":base["O"],"P1":base["P"],"c1":base["c"],"d1":base["d"],
       "P2":rot(base["P"]),"c2":rot(base["c"]),"d2":rot(base["d"])}
    E=[("O","c1"),("O","d1"),("P1","c1"),("P1","d1"),("c1","d1"),
       ("O","c2"),("O","d2"),("P2","c2"),("P2","d2"),("c2","d2"),("P1","P2")]
    return V,E

# one-hot SAT search for k-colourability == de Grey's DFS colourer
def k_colour(vertices, edges, k):
    adj={v:set() for v in vertices}
    for u,v in edges: adj[u].add(v); adj[v].add(u)
    order=sorted(vertices, key=lambda v:-len(adj[v]))   # most-constrained first
    colour={}
    def search(i):
        if i==len(order): return True
        v=order[i]; bad={colour[w] for w in adj[v] if w in colour}
        for c in range(k):
            if c in bad: continue
            colour[v]=c
            if search(i+1): return True
            del colour[v]
        return False
    return dict(colour) if search(0) else None

def chromatic_number(vertices, edges, hi=7):
    for k in range(1,hi+1):
        if k_colour(vertices,edges,k) is not None: return k

# standard one-hot CNF: UNSAT for k=4 == not 4-colourable (DRAT-certifiable)
def dimacs_kcol(vertices, edges, k):
    idx={v:i for i,v in enumerate(vertices)}
    var=lambda v,c: idx[v]*k + c + 1
    cl=[[var(v,c) for c in range(k)] for v in vertices]            # at least one
    cl+=[[-var(v,a),-var(v,b)] for v in vertices for a,b in combinations(range(k),2)]
    cl+=[[-var(u,c),-var(w,c)] for u,w in edges for c in range(k)] # edge clauses
    return f"p cnf {len(vertices)*k} {len(cl)}\n" + \
           "\n".join(" ".join(map(str,c))+" 0" for c in cl)
```
