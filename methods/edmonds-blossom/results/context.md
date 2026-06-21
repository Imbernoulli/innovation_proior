## The Matching Task

A finite undirected graph may contain many edge sets whose members do not share endpoints. The optimization problem is to choose such a set with maximum cardinality. One finite approach is to compare possibilities, but the number of possibilities grows quickly with the size of the graph.

The useful local move is to change a current matching only where there is a path that alternates between edges outside the matching and edges inside it. If such a path starts and ends at exposed vertices, flipping membership along the path increases the matching by exactly one edge.

## The Augmenting-Path Criterion

A pre-existing theorem states that this local move is not merely sufficient. A matching is maximum exactly when there is no alternating path joining two exposed vertices. This turns the optimization problem into a search problem: either find one more improving path or certify that none exists.

## Bipartite Search

In a bipartite graph, an alternating search from exposed roots has a stable parity pattern. Vertices reached after an even number of path edges play one role, and vertices reached after an odd number play the other. A search tree can grow by taking an unmatched edge out and a matched edge back.

Because every cycle has even length, this parity assignment is consistent. If the search reaches another exposed vertex, it has found an improving path. If it exhausts the reachable region, that failure has structural meaning.

## The General-Graph Setting

In a general graph, the same search can encounter an edge joining two vertices with the same even search parity. If those vertices belong to different rooted trees, the two root paths plus this edge form an improving path. The remaining case is when they belong to the same tree: the two paths back toward their common root plus this extra edge close an odd alternating circuit. That circuit is not by itself an improving path.

## Search and Duality

A general-graph search aims to preserve the augmenting-path criterion: any improvement found in the search must correspond to a real alternating path in the original graph, and a failed search must report a genuine certificate that the matching is maximum.

This question also has a dual face. The simple degree constraints for matchings describe the right integer structure on bipartite graphs through the vertex-cover picture, and the search for a matching certificate in a general graph is tied to how odd vertex sets enter that structure.
