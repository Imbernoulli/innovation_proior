# Polymath16, first thread: Simplifying de Grey's graph
Source: https://dustingmixon.wordpress.com/2018/04/14/polymath16-first-thread-simplifying-de-greys-graph/ (retrieved this run)

## H/L/M modular construction
- H: the unit-distance gadget building block (the centre+vertices of a unit regular
  hexagon in de Grey's paper).
- L: forces at least one copy of H to contain a monochromatic triple under any 4-coloring.
  Polymath16 later simplified de Grey's L dramatically ("down to just three points, a
  triangle of edges 2,2,1, and H to two points 2 apart").
- M: the critical component. M "prevents vertices of a particular sqrt(3)-regular triangle
  from being monochromatic." The original M required checking the nonexistence of particular
  4-colourings of subgraphs with almost 400 vertices, necessitating computer verification.

## de Grey's original graph
The 1581-vertex configuration required proving no 4-coloring exists; the proof relied on
analyzing a 397-vertex subgraph computationally.

## Polymath16 progress
- Goal: reduce the amount of computer assistance and find smaller 5-chromatic graphs.
- Vertex-removal shrinking: iteratively remove vertices not needed for 5-chromaticity.
- Marijn Heule used SAT solving with proofs of unsatisfiability; found an 874-vertex
  5-chromatic graph by rotating two copies of M so distance between point {-1,0} in M1 and
  its counterpart in M2 is 1, then extracting subgraphs; random-probing later found an
  826-vertex / 4273-edge version.
- Dustin Mixon proposed a weakened M that merely prevents a particular sqrt(3)-regular
  triangle from being monochromatic while strengthening L, aiming for hand-verifiability.
