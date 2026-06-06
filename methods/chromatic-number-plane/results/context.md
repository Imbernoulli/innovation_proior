# Context: the chromatic number of the plane, on the eve of a new lower bound

## Research question

Colour every point of the Euclidean plane so that no two points at distance exactly 1
receive the same colour. The minimum number of colours that suffices is the *chromatic
number of the plane* (CNP). The question, posed by Nelson in 1950 and now called the
Hadwiger–Nelson problem, has stood with a frustratingly wide gap: at least 4, at most 7,
and not a single point of movement for nearly seven decades.

The lower bound is, by a compactness theorem, equivalent to a purely finite question: does
there exist a *finite* unit-distance graph — a finite set of points in the plane with an
edge between every pair at distance exactly 1 — whose chromatic number exceeds 4? If one
can exhibit a finite unit-distance graph that admits no proper 4-colouring, then no
4-colouring of the whole plane can exist either (restrict it to those points), and CNP ≥ 5.
So the goal is concrete: **engineer a finite unit-distance graph that is not 4-colourable.**
What makes this hard is that the obstruction must be enforced purely by distances realisable
in the plane — every edge has to be a genuine unit segment — while colourability is a global
combinatorial property that local gadgets do not obviously control.

## Background

A *unit-distance graph* is any graph drawn in the plane with all edges of length 1. Two
facts about distances are load-bearing throughout.

First, the **unit equilateral triangle**: three mutually unit-distant points force three
distinct colours. Two such triangles sharing an edge make a **unit rhombus** (angles
60°/120°, side 1). Its short diagonal has length 1 and splits it into the two triangles; its
long diagonal joins the two acute (60°) vertices and has length 2·cos 30° = √3. The decisive
property: in any *3-colouring* of the rhombus, each acute vertex must differ from both
endpoints of the shared edge, which between them already use two colours — so each acute
vertex is forced to the third colour, and **the two acute vertices, a distance √3 apart, are
the same colour**. Distance-√3 monochromatic pairs are thus produced by 3-colourings of
rhombi, and a monochromatic √3-equilateral-triangle is three such pairs concentrated at one
spot.

Second, the **Moser spindle** (Moser & Moser, 1961): take two unit rhombi, glue them at one
acute vertex, and rotate one so that the two *far* acute vertices come to distance 1 apart.
Those far vertices each lie at distance √3 from the shared vertex, so the rotation that
brings their chord to length 1 is 2·arcsin(1/(2√3)). Adding the edge between them yields a
7-vertex, 11-edge unit-distance graph. In a 3-colouring, each rhombus would force its far
acute vertex to equal the shared vertex's colour — so both far vertices would share that
colour, contradicting the unit edge between them. Hence the spindle is not 3-colourable; it
*is* 4-colourable; its chromatic number is exactly 4. This single gadget is the entire reason
CNP ≥ 4 has been known since 1950, and it is the only known rigid device that *forces*
distance-√3 monochromatic pairs.

The upper bound comes from a colouring, not a graph: tile the plane with regular hexagons of
diameter slightly under 1 and 7-colour the tiles so that any two tiles of the same colour are
more than a diameter apart (Hadwiger 1945; Isbell 1950). No same-colour pair is ever at
distance exactly 1, so 7 colours suffice and CNP ≤ 7. Between the spindle and the hexagons
lies the open gap. The history of all this is documented at length in Soifer's *The
Mathematical Coloring Book* (2008).

On the computational side: deciding whether a graph is k-colourable is the canonical NP
problem, but for a *fixed* small k it reduces cleanly to Boolean satisfiability. The standard
one-hot encoding introduces a variable x[v,c] (vertex v has colour c), a clause per vertex
saying it takes at least one colour, optional at-most-one clauses, and for every edge {u,w}
and colour c a clause (¬x[u,c] ∨ ¬x[w,c]) forbidding both endpoints from colour c. A
satisfying assignment is a proper colouring; an UNSAT verdict for k = 4 is exactly a proof of
non-4-colourability — and a modern solver can emit a DRAT certificate that an independent
checker verifies mechanically, removing trust in any one program.

## Baselines

- **Moser spindle (Moser & Moser 1961).** 7 vertices, 11 edges, χ = 4. Proves CNP ≥ 4. The
  gap it leaves: a single spindle forces only that *some* √3-pair is monochromatic; it does
  not pin down *where*, and it cannot by itself rule out a 4-colouring of any larger
  configuration. Sixty-eight years of attempts to push past 4 by hand-built gadgets found
  nothing larger that helped.

- **Hexagonal 7-colouring (Hadwiger 1945 / Isbell 1950).** A colouring of the whole plane
  giving CNP ≤ 4… no — giving CNP ≤ 7. The gap: it is far from tight, and offers no leverage
  on the lower bound at all.

- **Blind search for a 5-chromatic unit-distance graph.** One could enumerate point sets and
  test 4-colourability. The gap: the search space of planar point configurations with many
  exact unit distances is astronomically structured-yet-sparse; without a principle telling
  you *which* configurations could possibly be non-4-colourable, brute enumeration over
  graphs with hundreds or thousands of vertices is hopeless. A method needs to *engineer* the
  obstruction, not stumble onto it.

## Evaluation settings

The yardstick is a yes/no mathematical fact, not a benchmark with numbers: does the
constructed finite unit-distance graph admit a proper 4-colouring? The natural verification
instruments, all pre-existing:

- **Exact geometric realisability.** Coordinates expressible in a real quadratic/algebraic
  field (here Q[√3] suffices for the rhombus and spindle), so that "two points are at
  distance exactly 1" is decided by exact arithmetic rather than floating point.
- **Colourability decision.** Either a direct backtracking colourer (depth-first search over
  colours with forcing/propagation) or the one-hot CNF handed to a general SAT solver.
- **Independent certification.** A DRAT proof from the SAT solver, checkable by a separate
  verifier, so the conclusion does not rest on any single custom program.

The smallest known 4-chromatic example (the spindle, 7 vertices) is the reference scale for
how compact such obstructions can be.

## Code framework

What already exists before any new construction: exact arithmetic over an algebraic field for
distances, a generic colourability decision procedure, and the standard SAT encoding. The
*specific gadgets* and the *scheme that composes them into a non-4-colourable graph* are the
empty slots.

```python
from fractions import Fraction as F
from itertools import combinations

# --- exact arithmetic in a quadratic field, so distances are decided exactly ---
class Q3:                       # a + b*sqrt(3)
    def __init__(self, a=0, b=0): self.a, self.b = F(a), F(b)
    # __add__/__sub__/__mul__/__eq__  (field ops)  -- known
    ...

def dist2(p, q):                # exact squared distance -- known
    dx, dy = p[0]-q[0], p[1]-q[1]
    return dx*dx + dy*dy

def is_unit(p, q):              # distance == 1 ? -- known
    return dist2(p, q) == Q3(1, 0)

# --- generic colourability decision (backtracking == one-hot SAT search) -- known ---
def k_colour(vertices, edges, k):
    """Return a proper k-colouring or None. DFS over colours with pruning."""
    ...

def chromatic_number(vertices, edges, hi=7):
    ...

def dimacs_kcol(vertices, edges, k):
    """Standard one-hot CNF: UNSAT == not k-colourable (DRAT-certifiable)."""
    ...

# --- the slots the construction will fill ---
def forcing_gadget():
    # TODO: a rigid unit-distance gadget that, under 3 colours, forces a
    #       specific monochromatic distance-sqrt(3) pair.
    pass

def colour_classifier_gadget():
    # TODO: a small unit-distance gadget whose 4-colourings split into a
    #       "structured" class and an "unstructured" class.
    pass

def assemble_to_force(classifier):
    # TODO: rotate/translate many copies so EVERY 4-colouring lands a copy in
    #       the structured class.
    pass

def assemble_to_forbid(classifier):
    # TODO: rotate/translate many copies so NO 4-colouring lets the central
    #       copy land in the structured class.
    pass

def compose(force_part, forbid_part):
    # TODO: glue the two so their shared classifier copies coincide; the
    #       result is not 4-colourable.
    pass
```
