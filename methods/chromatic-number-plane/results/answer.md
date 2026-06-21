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

SAT verification of G's 5-colourability and non-4-colourability gives χ(G) = 5 and
establishes CNP ≥ 5.
