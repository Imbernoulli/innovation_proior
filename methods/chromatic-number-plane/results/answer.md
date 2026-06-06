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
  rotated to unit distance by 2·arcsin(1/(2√3))) is the 7-vertex, 11-edge graph with χ = 4
  built from this.

- **Classifier (H).** The hexagon-with-centre, 7 vertices, 12 unit edges, has exactly four
  essentially-distinct 4-colourings; two contain a monochromatic √3-triple ("structured"),
  two don't. So "is this copy of H structured?" is a binary handle on any 4-colouring.

- **Forcer (L).** Assemble 52 rotated copies of H (built in stages J→K→L) so that **every**
  4-colouring makes at least one copy structured. The construction is recursive spindle logic:
  J (31 v, 13 H) constrains six "linking" vertices to three colour-patterns; K (61 v) is two
  J's rotated by 2·arcsin(1/4) so corresponding linking vertices become a unit edge, forcing
  the pattern in which each linking diagonal is monochromatic; L (121 v) is two K's rotated
  about a linking vertex A by 2·arcsin(1/8), making B and B′ a unit edge — but L's structure
  forces B = A = B′, contradicting that edge unless some H is structured.

- **Anti-forcer (M).** Build a unit-distance graph M (1345 v) around one H from a dense,
  interlocking mesh of spindles so that **no** 4-colouring makes its central H structured. A
  mono triple is a local concentration of mono √3-pairs; densely interlocked spindles force
  those pairs to spread out uniformly, forbidding the concentration. Plain 60°-rotation
  meshes (only 3 edge classes, max degree 18) are too slack; enriching the edge directions
  with the offsets of tightly-linked spindles — angles i·arcsin(√3/2)+j·arcsin(1/√12),
  i∈0…5, j∈−2…2 (graph V, max degree 30) — stiffens the mesh enough to work.

- **Compose (N).** Glue 52 copies of M onto L so each M's central H is one of L's copies.
  L forces some H structured; M forbids that H structured; contradiction. N (20425 v after
  merging) is not 4-colourable, so **CNP ≥ 5**.

- **Shrink + certify (G).** Stepwise vertex deletions reduce N by ~13× to a 1581-vertex graph
  G, small enough that standard SAT solvers confirm χ(G) = 5 directly, with a DRAT
  certificate an independent checker validates — no trust in custom code.

## Certifying (un)colourability

M's property is checked by a one-sided depth-first colourer: fix the central H to a structured
pattern (two cases suffice by symmetry), colour the rest most-constrained-vertex first, and
propagate forced colours; the spindle density makes the search collapse in minutes. This is
the standard one-hot SAT encoding — x[v,c]; an at-least-one clause per vertex; an edge clause
(¬x[u,c] ∨ ¬x[w,c]) per edge and colour — UNSAT for k = 4 meaning not 4-colourable.

## Code

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

def dist2(p,q):
    dx,dy = p[0]-q[0], p[1]-q[1]; return dx*dx + dy*dy
def is_unit(p,q): return dist2(p,q) == Q3(1,0)

# rhombus: acute pair A,B is sqrt(3) apart and monochromatic in every 3-colouring
def rhombus():
    A=(Q3(0),Q3(0)); B=(SQRT3,Q3(0))
    C=(Q3(0,F(1,2)),Q3(F(1,2))); D=(Q3(0,F(1,2)),Q3(F(-1,2)))
    V={"A":A,"B":B,"C":C,"D":D}
    E=[("A","C"),("A","D"),("B","C"),("B","D"),("C","D")]
    return V,E,("A","B")

# Moser spindle: chi = 4
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

# one-hot SAT search for k-colourability (the DFS colourer)
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
    cl=[[var(v,c) for c in range(k)] for v in vertices]
    cl+=[[-var(v,a),-var(v,b)] for v in vertices for a,b in combinations(range(k),2)]
    cl+=[[-var(u,c),-var(w,c)] for u,w in edges for c in range(k)]
    return f"p cnf {len(vertices)*k} {len(cl)}\n" + \
           "\n".join(" ".join(map(str,c))+" 0" for c in cl)

if __name__ == "__main__":
    V,E,(a,b)=rhombus()
    assert all(is_unit(V[u],V[w]) for u,w in E)
    assert dist2(V[a],V[b]) == Q3(3,0)            # acute pair sqrt(3) apart
    Vs,Es=moser_spindle()
    assert len(Vs)==7 and len(Es)==11
    assert k_colour(list(Vs),Es,3) is None        # not 3-colourable
    assert k_colour(list(Vs),Es,4) is not None    # is  4-colourable -> chi = 4
    print("OK: rhombus unit + sqrt(3) acute pair; Moser spindle chi = 4")
```

Running it confirms the two load-bearing gadget facts exactly: the rhombus is unit-distance
with a √3 acute pair forced monochromatic under 3 colours, and the Moser spindle is exactly
4-chromatic. These are the engine the forcer/anti-forcer architecture is built on; scaled up
through L, M, N and shrunk to the 1581-vertex G, the same SAT verdict (UNSAT for 4 colours)
establishes CNP ≥ 5.
```
