# Moser spindle — construction and chromatic number
Source: https://en.wikipedia.org/wiki/Moser_spindle (retrieved this run)
Primary citation: L. Moser and W. Moser, "Solution to Problem 10", Can. Math. Bull. 4 (1961), 187–189.

## Structure
Undirected graph with 7 vertices and 11 edges; embeddable as a unit-distance graph in the
Euclidean plane.

## Geometric construction
Formed from two rhombi with 60° and 120° angles. Their sides and short diagonals create
equilateral triangles. The two rhombi share one acute-angled vertex; their other two
acute-angled vertices are positioned exactly one unit apart. The 11 edges: eight rhombus
sides, two short diagonals, and the edge joining the unit-distance vertices.

## Why four colors (forcing argument)
In any 3-coloring of one rhombus, its two acute angles must receive identical colors. If
the shared vertex also takes this color, then the two opposite acute vertices match in
color too — violating the unit-distance edge between them. Contradiction ⇒ 3 colors are
insufficient ⇒ χ = 4.

## Alternative (Hajós) construction
Two K4's (each χ=4); remove one edge from each; merge two endpoints into a single shared
vertex; connect the other two endpoints. Preserves the 4-color requirement.

## Hadwiger–Nelson implication
As a subgraph of the infinite plane unit-distance graph, it proves χ(plane) ≥ 4. Until
2018 no finite subgraph required more than four colors.

## Relevance to de Grey
de Grey's §4 builds M by seeking graphs with HIGH SPINDLE DENSITY: a spindle contains two
pairs of vertices √3 apart that cannot both be monochromatic, so interlocking spindles
constrain monochromatic √3-pairs to spread out uniformly — which is what lets some
hexagon H avoid a monochromatic triple (an equilateral triangle of edge √3).
