# Context: the chromatic number of the plane

## Research question

Colour every point of the Euclidean plane so that no two points at distance exactly 1
receive the same colour. The minimum number of colours that suffices is the *chromatic
number of the plane* (CNP). The question, posed by Nelson in 1950 and now called the
Hadwiger–Nelson problem, is known to satisfy 4 ≤ CNP ≤ 7.

The lower bound is, by a compactness theorem, equivalent to a purely finite question: does
there exist a *finite* unit-distance graph — a finite graph drawn with all listed edges
realised by distance-1 segments in the plane — whose chromatic number exceeds 4? If one
can exhibit a finite unit-distance graph that admits no proper 4-colouring, then no
4-colouring of the whole plane can exist either (restrict it to those points), and CNP ≥ 5.
So the goal is concrete: **construct a finite unit-distance graph that is not 4-colourable**,
with the obstruction enforced purely by distances realisable in the plane (every edge a
genuine unit segment).

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
distance exactly 1, so 7 colours suffice and CNP ≤ 7. The history of all this is documented
at length in Soifer's *The Mathematical Coloring Book* (2008).

On the computational side: deciding whether a graph is k-colourable is the canonical NP
problem, but for a *fixed* small k it reduces cleanly to Boolean satisfiability. The standard
one-hot encoding introduces a variable x[v,c] (vertex v has colour c), a clause per vertex
saying it takes at least one colour, optional at-most-one clauses, and for every edge {u,w}
and colour c a clause (¬x[u,c] ∨ ¬x[w,c]) forbidding both endpoints from colour c. A
satisfying assignment is a proper colouring; an UNSAT verdict for k = 4 is exactly a proof of
non-4-colourability — and a modern solver can emit a DRAT certificate that an independent
checker verifies mechanically, removing trust in any one program.

## Baselines

- **Moser spindle (Moser & Moser 1961).** 7 vertices, 11 edges, χ = 4. Proves CNP ≥ 4. It
  blocks 3-colourings, and in the 4-colour setting it supplies a constraint on √3-pairs: two
  specified pairs sharing the glued acute vertex cannot both be monochromatic.

- **Hexagonal 7-colouring (Hadwiger 1945 / Isbell 1950).** A colouring of the whole plane
  giving CNP ≤ 7.

- **Blind search for a 5-chromatic unit-distance graph.** Enumerate planar point sets with
  many exact unit distances and test each for 4-colourability.

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


