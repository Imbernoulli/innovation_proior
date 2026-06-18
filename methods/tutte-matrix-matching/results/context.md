## Research Pressure

A perfect matching asks for a global pairing of all vertices by edges. The local constraints are simple: every vertex must be used once, and no chosen edges may touch. The difficulty is that local compatibility does not compose cleanly. A graph can have many plausible partial pairings and still fail at the last few vertices; an odd number of vertices already makes success impossible.

## Existing Combinatorics

Regular graphs gave early structure. Petersen's work studies when a regular graph can be decomposed into lower-degree factors, using alternating color changes along closed polygons. This makes factorization concrete, but it remains tied to special degree patterns and to operations on already-colored decompositions.

## General Obstruction

For arbitrary graphs, the real obstacle is parity after deletion. If a set of removed vertices leaves too many odd components, a perfect pairing cannot cover all those components, because each odd component needs an edge leaving it. A useful theory must explain when this parity obstruction is merely necessary and when it is complete.

## Algebraic Tools

Classical determinant theory already contains special facts about skew-symmetric arrays. Odd-order skew determinants vanish, even-order skew determinants are squares, and the chosen square root expands through pairings of indices. These facts are purely algebraic before any graph is inserted.

## Desired Compression

The goal is a criterion that keeps the whole pairing problem visible at once without committing to a particular partial matching. It has to separate genuinely different pairings, respect the parity obstruction, and avoid a case analysis over all ways a search can get stuck.
