## The Matching Task

A finite undirected graph may contain many edge sets whose members do not share endpoints. The optimization problem is to choose such a set with maximum cardinality. The obvious finite approach is to compare possibilities, but the number of possibilities grows too quickly to count as a good algorithmic explanation.

The useful local move is to change a current matching only where there is a path that alternates between edges outside the matching and edges inside it. If such a path starts and ends at exposed vertices, flipping membership along the path increases the matching by exactly one edge.

## The Augmenting-Path Criterion

The decisive pre-existing theorem says that this local move is not merely sufficient. A matching is maximum exactly when there is no alternating path joining two exposed vertices. That turns the whole optimization problem into a search problem: either find one more improving path or certify that none exists.

This criterion is powerful but incomplete as an algorithm. It says what to look for, not how to look without trying exponentially many alternating walks.

## Why Bipartite Search Behaves

In a bipartite graph, an alternating search from exposed roots has a stable parity pattern. Vertices reached after an even number of path edges play one role, and vertices reached after an odd number play the other. A search tree can grow by taking an unmatched edge out and a matched edge back.

Because every cycle has even length, this parity assignment does not contradict itself. If the search reaches another exposed vertex, it has found an improving path. If it exhausts the reachable region, that failure has structural meaning.

## The General-Graph Obstruction

In a general graph, the same search can encounter an edge joining two vertices with the same even search parity. If those vertices belong to different rooted trees, the two root paths plus this edge already form an improving path. The hard case is when they belong to the same tree: the two paths back toward their common root plus this extra edge close an odd alternating circuit. That circuit is not already an improving path, yet it is not irrelevant noise.

Naively following every possible route through such a circuit loses the efficiency target. Ignoring the circuit can also be wrong, because later access to the circuit may be exactly what makes an improving path visible.

## The Required Structural Move

A good general-graph search needs a way to preserve the augmenting-path criterion while controlling the local parity conflict. The move must be reversible in both senses: any improvement found after handling the circuit must correspond to a real alternating path in the original graph, and a failed search must not have hidden an improvement through the circuit.

The same obstacle also has a dual face. Odd vertex sets are where the simple degree constraints for matchings stop describing the right integer structure, so any final certificate of failure has to account for them rather than only replaying the bipartite vertex-cover picture.
