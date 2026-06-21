# Edmonds' Blossom Algorithm

Given a graph `G` and a matching `M`, repeat:

1. Grow an alternating forest from every exposed vertex.
2. Scan edges incident to outer vertices.
3. If an edge reaches an unreached vertex that is matched by `M`, add the scanned edge and that vertex's matched edge to the forest.
4. If an edge reaches an inner vertex already in the forest, mark the edge examined and do not grow the forest.
5. If an edge joins outer vertices, trace their parent paths. Different roots give an augmenting path; replace `M` by the symmetric difference of `M` and that path.
6. If those outer vertices have the same root, the two tree paths plus the scanned edge form an odd alternating circuit with a base. Shrink that circuit into one pseudovertex and continue the same search in the reduced graph.
7. If an augmenting path is found in a reduced graph, expand pseudovertexes in dependency order from outermost to inner stored circuits. Each stored odd circuit has a unique maximum internal matching compatible with the external vertex used by the lifted path; in the tree-path view this is the unique alternating route from that vertex to the circuit base.
8. If the dense forest search exhausts all possible extensions and no augmenting path is found, stop.

The core invariant is:

`G` has an augmenting path with respect to `M` if and only if the graph produced by shrinking a blossom has an augmenting path with respect to the contracted matching.

The forward direction says shrinking does not hide a real improvement. The backward direction says every contracted augmenting path can be expanded through the stored odd circuit by selecting the unique compatible internal matching, or equivalently the correct alternating route to the base in the search tree.

Therefore each successful iteration increases `|M|` by one, and when the search reports no augmenting path, Berge's theorem proves that `M` is maximum. Edmonds's conceptual cardinality algorithm has the stated upper bounds `n^4` time and `n^2` memory for `n` vertices. The current NetworkX implementation is the weighted Galil/Edmonds primal-dual variant, so its dual and slack machinery is extra, but its `scanBlossom`, `addBlossom`, `expandBlossom`, `augmentBlossom`, and `augmentMatching` routines match the search, nested-blossom storage, expansion, and alternating-flip structure above.

Edmonds's paper also gives the matching-duality theorem: the maximum size of a matching equals the minimum capacity-sum of an odd-set cover. A singleton has capacity `1`; a set of `2k + 1` vertices has capacity `k` and covers edges with both endpoints inside it. This is the dual face of the same obstruction that the algorithm handles by odd circuits.
