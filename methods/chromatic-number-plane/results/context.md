# Context: the chromatic number of the plane

## Research question

Colour every point of the Euclidean plane so that no two points at distance exactly 1
receive the same colour. The minimum number of colours that suffices is the *chromatic
number of the plane* (CNP). The question, posed by Nelson in 1950 and now called the
Hadwiger–Nelson problem, has stood with a frustratingly wide gap: at least 4, at most 7,
and not a single point of movement for nearly seven decades.

The lower bound is, by a compactness theorem, equivalent to a purely finite question: does
there exist a *finite* unit-distance graph — a finite graph drawn with all listed edges
realised by distance-1 segments in the plane — whose chromatic number exceeds 4? If one
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
distinct colours. Two such triangles sharing an edge make a **unit rhombus graph**: the four
side edges of a 60°/120° rhombus, plus the shared short diagonal of length 1. Its long
diagonal joins the two acute (60°) vertices and has length 2·cos 30° = √3. The decisive
property: in any *3-colouring* of the rhombus graph, each acute vertex must differ from both
endpoints of the shared edge, which between them already use two colours — so each acute
vertex is forced to the third colour, and **the two acute vertices, a distance √3 apart, are
the same colour**. Distance-√3 monochromatic pairs are thus produced by 3-colourings of
rhombi.

Second, the **Moser spindle** (Moser & Moser, 1961): take two unit rhombi, glue them at one
acute vertex, and rotate one so that the two *far* acute vertices come to distance 1 apart.
Those far vertices each lie at distance √3 from the shared vertex, so the rotation that
brings their chord to length 1 is 2·arcsin(1/(2√3)). Equivalently, the full rotation has
cosine 5/6 and sine √11/6. Adding the edge between the far acute vertices yields a 7-vertex,
11-edge unit-distance graph. In a 3-colouring, each rhombus would force its far acute vertex
to equal the shared vertex's colour — so both far vertices would share that colour,
contradicting the unit edge between them. Hence the spindle is not 3-colourable; it *is*
4-colourable; its chromatic number is exactly 4. In a 4-colouring, the same geometry is used
more cautiously: the two distinguished √3-pairs sharing the glued acute vertex cannot both
be monochromatic, because that would make the far acute vertices equal across their unit
edge.

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
  gap it leaves: it blocks 3-colourings, but in the 4-colour setting it supplies only a
  negative constraint on √3-pairs — two specified pairs cannot both be monochromatic. That
  is not yet a mechanism for forcing a whole finite graph to fail 4-colourability.

- **Hexagonal 7-colouring (Hadwiger 1945 / Isbell 1950).** A colouring of the whole plane
  giving CNP ≤ 7. The gap: it is far from tight, and offers no leverage on the lower bound at
  all.

- **Blind search for a 5-chromatic unit-distance graph.** One could enumerate point sets and
  test 4-colourability. The gap: the search space of planar point configurations with many
  exact unit distances is astronomically structured-yet-sparse; without a principle telling
  you *which* configurations could possibly be non-4-colourable, brute enumeration over
  graphs with hundreds or thousands of vertices is hopeless. A method needs to *engineer* the
  obstruction, not stumble onto it.

## Evaluation settings

The yardstick is a yes/no mathematical fact, not a benchmark with numbers: does the
constructed finite unit-distance graph admit a proper 4-colouring? The natural verification
instruments are:

- **Exact geometric realisability.** Coordinates expressible in real algebraic fields (Q[√3]
  for the rhombus; Q[√3,√11] for the rotated spindle; larger square-root extensions for
  compact seed descriptions), so that "two points are at distance exactly 1" can be decided
  by exact arithmetic rather than floating point.
- **Colourability decision.** Either a direct backtracking colourer (depth-first search over
  colours with forcing/propagation) or the one-hot CNF handed to a general SAT solver.
- **Independent certification.** A DRAT proof from the SAT solver, checkable by a separate
  verifier, so the conclusion does not rest on any single custom program.

For a lower bound it is enough to prove that the 4-colour CNF is UNSAT. To certify exact
5-chromaticity of a finite graph, the verification also includes a concrete 5-colouring
witness.

## Code framework

This is a mathematical discovery: the landing artifact is an explicit finite unit-distance
graph together with a proof that it admits no proper 4-colouring. There is no algorithm to
tune to convergence and no benchmark to beat. What exists before the method is instead a set
of fixed primitives — the standing apparatus that any candidate construction will be built
out of and checked against — and a set of empty slots that the discovery itself must fill.
Naming no conclusion, the pre-method scaffold is the following.

The fixed primitives, taken as given. First, **exact-distance arithmetic.** The vertices of
every gadget under consideration have coordinates lying in a real algebraic field obtained by
adjoining a few square roots to the rationals — Q(√3) suffices for the rhombus, and Q(√3,√11)
for the rotated Moser spindle, with larger square-root extensions available for more elaborate
seeds. Because elements of such a field carry an exact representation and a decidable equality,
the predicate "these two points are at distance exactly 1" is settled by exact arithmetic on
squared distances, never by floating point: unit-distance is a *decidable* relation, and the
edge set of any drawn configuration is therefore well defined and trustworthy. Second, a
**proper-k-colouring decision check.** Given a finite graph and a value of k, there is a fixed
way to decide whether a proper k-colouring exists — backtracking search with forcing, or the
standard one-hot CNF encoding handed to a SAT solver — together with the convention that an
UNSAT verdict at k = 4 is exactly a proof of non-4-colourability, and that a SAT solver can
emit an independently checkable DRAT certificate so the verdict rests on no single program.
Third, **rigid unit-distance gadget primitives.** The unit equilateral triangle, the unit
rhombus (whose two acute vertices a 3-colouring forces to share a colour at distance √3), and
the Moser spindle (χ = 4, not 3-colourable) are available as exactly realised, reusable
building blocks, each with its known colouring behaviour, to be placed, rotated by exact
angles, and glued.

The empty slots, which the discovery must fill. **Which gadget assembly forces 5 colours:** a
configuration must be found whose colouring constraints, propagated through copies of these
rigid pieces, leave no consistent assignment of 4 colours — that is, the local √3-pair and
unit-edge constraints supplied by the primitives must be combined into a global obstruction
that the spindle alone does not produce. **How to shrink to a finite certifiable graph:** any
first assembly is likely to be large, so a further slot is the reduction of the candidate to a
finite graph small enough that the 4-colour decision check returns UNSAT within reach of an
independent certificate, while its unit distances remain exactly realisable. Filling these
slots — and only then exhibiting the explicit graph — is the discovery; the primitives above
are what it starts from.
