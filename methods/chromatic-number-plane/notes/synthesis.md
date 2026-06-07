# Synthesis — 5-chromatic unit-distance graph (CNP >= 5)

## Pain point / goal
CNP = min colours for the plane so no two points at distance exactly 1 share a colour.
Known since 1950: 4 <= CNP <= 7, stuck for 68 years. Lower bound 4 from Moser spindle
(7-vertex 4-chromatic unit-distance graph). Upper bound 7 from hexagonal 7-colouring.
Goal: a FINITE unit-distance graph that is not 4-colourable -> CNP >= 5.

## Load-bearing prior art (ancestors)
- **Moser spindle (Moser & Moser 1961)**: 7 vertices, 11 edges, chi=4. Two unit rhombi
  (each = two unit equilateral triangles sharing an edge) sharing one acute vertex; the
  far acute vertices rotated to unit distance (rotation 2 arcsin(1/(2 sqrt3))). KEY fact:
  in a 3-colouring of a rhombus the two ACUTE (60-deg) vertices are forced equal, and they
  are sqrt(3) apart. So a spindle has two sqrt(3)-pairs that can't both be monochromatic
  under 4 colours -> tension. VERIFIED in code: rhombus acute pair always same in all 6
  proper 3-colourings; spindle chi=4 exactly.
- **Hadwiger 1945 / Isbell 1950**: 7-colouring of hexagonal tiling -> upper bound 7.
- **Soifer, Mathematical Coloring Book (2008)**: history, the whole frame.
- **SAT graph-colouring encoding (standard)**: one-hot x[v,c]; at-least-one per vertex;
  edge clause (~x[u,c] | ~x[v,c]); UNSAT for k=4 <=> not 4-colourable. DRAT certificate ->
  machine-checkable. (Heule 1805.12181 later used clausal-proof minimization to shrink.)

## The methodological leap (target)
Don't search blindly for a 5-chromatic graph. Engineer FORCING:
- **H**: hexagon-with-centre, 7 vertices 12 edges. Exactly 4 essentially-distinct
  4-colourings; 2 contain a monochromatic triple (an equilateral sqrt(3)-triangle),
  2 don't. The unit of "structure" is "does this H have a mono triple?".
- **L** (121 vertices, 52 copies of H): a rigid assembly of rotated H-copies such that in
  EVERY 4-colouring, at least one copy of H contains a monochromatic triple. Built by
  stages: J (31 v, 13 H) constrains the "linking vertices" to 3 colouring-types; K (61 v)
  = two J rotated by 2 arcsin(1/4) so linking vertices fall to unit distance, forcing
  option (c) where linking diagonals are monochromatic; L = two K rotated by 2 arcsin(1/8)
  so that B,B' become unit-distance -> some diagonal must be non-mono -> some H mono-triple.
- **M** (1345 v): a spindle-dense graph around one H, engineered (via vector classes from
  tightly-linked spindles, edge angles i*arcsin(sqrt3/2)+j*arcsin(1/sqrt12)) so that NO
  4-colouring has its central H containing a mono triple. Rationale: dense interlocking
  spindles spread the forced sqrt(3)-mono-pairs uniformly; a mono triple is a local
  CONCENTRATION of such pairs, which uniformity forbids.
- **N** = 52 copies of M glued so their central H's form L (20425 v). L says "some H mono",
  M says "this H never mono" -> contradiction -> N not 4-colourable -> CNP >= 5.
- **Search role**: M's property was certified by a custom DFS colourer (Mathematica),
  fixing the central H's colours and checking no completion exists (the two essentially
  distinct triple-colourings). Then N shrunk to G (1581 v) by vertex deletion; G verified
  by standard SAT solvers (independent, DRAT-checkable).

## Design decisions -> why
- Why H = hexagon+centre? Smallest gadget whose 4-colourings split cleanly into
  "mono-triple / no-mono-triple"; the triple is a sqrt(3) equilateral triangle = the same
  object spindles constrain. Gives a binary handle to force/forbid.
- Why rotations 2 arcsin(1/4), 2 arcsin(1/8)? They are exactly the angles that bring a
  specific pair of vertices (linking vertices; B,B') to UNIT distance, adding a constraining
  edge between copies. arcsin(1/4): two points at distance ... rotate about origin so chord
  =1. arcsin(1/8): about A. These are the spindle-style "rotate to create a unit edge" move.
- Why spindle density for M? Spindles are the only known rigid sqrt(3)-pair enforcers;
  packing them makes the mono-sqrt(3)-pairs as uniform as possible, which is what kills any
  local mono triple. Empirically needed >18-degree vertices -> extra vector classes (V).
- Why SAT at the end? Custom code may be buggy; G small enough that an independent
  one-hot CNF + DRAT proof certifies UNSAT-for-4 mechanically.

## Verification (grounded, in code/verify.py)
- Exact Q[sqrt3] arithmetic: rhombus 5 edges all exactly unit; acute pair exactly sqrt(3).
- All 6 proper 3-colourings of rhombus: acute pair always monochromatic.
- Moser spindle: 7 v / 11 e, unit edges to 1e-9, DPLL colourer gives chi=4 (not 3-, is 4-).
- DPLL colourer = the one-hot SAT encoding searched directly (de Grey's DFS skeleton).

## Uncertainty flags
- Full N/G non-4-colourability is not re-run here (20425 / 1581 vertices, and G's vertex
  set is given only as 40 algebraic seed points + rotations); we verify the GADGET-LEVEL
  facts the construction rests on, plus chi(spindle)=4, exactly. The paper itself reports G
  was confirmed UNSAT-for-4 by independent SAT solvers.
- Exact essentially-distinct-colouring counts of H, J (4 and 6) are taken from the primary
  text; not independently re-enumerated in code (H's full edge set / J geometry not all
  reconstructed). Flagged.
