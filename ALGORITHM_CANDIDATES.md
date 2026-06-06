# Algorithm & Data-Structure Candidates

A working backlog of **classic computer-science algorithms and data structures** to run
through the `paper-to-reasoning` pipeline, plus an `AI-for-algorithms` bridge group that
connects this list to the existing ML corpus (AlphaDev, AlphaTensor, AlphaEvolve…).

This is the "traditional CS" expansion discussed in session — deliberately **not** limited to
what FrontierSmith touches. FrontierSmith is a *data-synthesis pipeline* (competitive-programming
seeds → open-ended problems); it does not catalog algorithms, so it is not a source here.

## How to read each row

| Column | Meaning |
| --- | --- |
| `slug` | Proposed kebab-case folder name under `methods/<slug>/`. |
| Title | Display title for `methods.json`. |
| The discovery prior | *What hurt at the time* + *the non-obvious leap*. This is the trace's spine — the thing `reasoning.md` must re-derive. The strongest candidates have a crisp, surprising leap. |
| Reference | Canonical source (author, year). |

## ⚠️ Before ingesting

- **References are from memory and MUST be verified** (author/year/venue, arxiv/DOI) before they
  become `methods.json` entries. Many predate arxiv — use the `arxiv` field for a DOI or leave a
  citation string; the site only needs `{slug, title, domain, arxiv}`.
- Slugs are proposals — check for collisions against existing `methods/` before creating folders.
- Rows marked **(weak)** have a thin innovation prior (pedagogical, no single paper). Keep them as
  *contrast material* inside another trace rather than standalone methods.
- Suggested new domains for `methods.json`: `Sorting & selection`, `Hashing`, `Heaps`,
  `Search trees`, `Strings & text`, `Graph algorithms`, `Combinatorial optimization`,
  `Computational geometry`, `Numerical & algebraic`, `Randomized & streaming`,
  `Compression & coding`, `Cryptographic algorithms`, `Concurrency & distributed`,
  `Parsing & automata`, `AI for algorithms`. (Collapse/rename to taste.)

---

## 1. Sorting & selection — domain `Sorting & selection`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| quicksort | Quicksort | Sorting in place without a merge buffer hurt; the leap is *partition around a pivot*, recursion falls out for free. | Hoare 1961 |
| mergesort | Merge sort | Stable, worst-case n log n before in-place tricks existed; the leap is divide-and-conquer + linear merge. | von Neumann 1945 |
| heapsort | Heapsort | n log n sort with O(1) extra space; the leap is *sort = repeatedly pop a heap built in place*. | Williams / Floyd 1964 |
| timsort | Timsort | Real data is already partly sorted; the leap is *detect natural runs + galloping merge* to hit near-linear on real inputs. | Peters 2002 |
| introsort | Introsort | Quicksort's O(n²) adversarial worst case; the leap is *watch recursion depth, bail to heapsort*. | Musser 1997 |
| shellsort | Shellsort | Insertion sort moves elements one step; the leap is *gap sequences* that let elements jump far early. | Shell 1959 |
| quickselect | Quickselect | Full sort to find the k-th is wasteful; the leap is *partition but recurse into one side only*. | Hoare 1961 |
| median-of-medians | Median of medians | Quickselect's worst case; the leap is *a provably good pivot in linear time* via recursive grouping. | Blum–Floyd–Pratt–Rivest–Tarjan 1973 |
| radix-sort | Radix sort | The Ω(n log n) comparison bound; the leap is *don't compare — bucket by digit*. | Seward 1954 |
| counting-sort | Counting sort | Same escape from comparison bound for small key ranges; tally then prefix-sum. | Seward 1954 |
| bucket-sort | Bucket sort | Uniform keys; the leap is *scatter into buckets, sort locally, concatenate*. | classic |
| samplesort | Samplesort | Parallel/external partitioning; the leap is *pick many splitters by sampling* to balance buckets. | Frazer–McKellar 1970 |
| pdqsort | Pattern-defeating quicksort | Quicksort degrades on patterns/duplicates; the leap is *detect bad patterns, swap strategy adaptively*. | Peters 2016 |
| dual-pivot-quicksort | Dual-pivot quicksort | One pivot wastes cache passes; the leap is *two pivots, three partitions* (Java's default). | Yaroslavskiy 2009 |

## 2. Hashing & filters — domain `Hashing`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| linear-probing | Linear probing | Chaining wastes pointers/cache; the leap is *store inline, probe the next slot on collision*. | Knuth 1963 |
| cuckoo-hashing | Cuckoo hashing | Worst-case O(1) lookup; the leap is *two tables, kick the occupant out and rehome it*. | Pagh–Rodler 2001 |
| robin-hood-hashing | Robin Hood hashing | Probe-length variance; the leap is *steal from the rich — swap so displacements equalize*. | Celis 1985 |
| hopscotch-hashing | Hopscotch hashing | Concurrent + cache-local open addressing; the leap is *keep keys within a neighborhood window*. | Herlihy–Shavit–Tzafrir 2008 |
| swiss-table | Swiss tables | Probing touches too much memory; the leap is *SIMD-scan a byte of control metadata per group*. | Google / Abseil 2017 |
| fks-perfect-hashing | FKS perfect hashing | Worst-case O(1) static lookup; the leap is *two-level hashing, square the bucket sizes, no collisions*. | Fredman–Komlós–Szemerédi 1984 |
| minimal-perfect-hashing | Minimal perfect hashing | Map n keys to 0..n−1 with no gaps; the leap is *peel a random hypergraph / assign by displacement*. | Czech–Havas–Majewski 1992 |
| bloom-filter | Bloom filter | Membership without storing keys; the leap is *k hashes into a bit array, accept false positives*. | Bloom 1970 |
| counting-bloom-filter | Counting Bloom filter | Bloom can't delete; the leap is *replace bits with small counters*. | Fan–Cao–Almeida–Broder 2000 |
| cuckoo-filter | Cuckoo filter | Bloom's poor locality + no delete; the leap is *store fingerprints in a cuckoo table*. | Fan–Andersen–Kaminsky–Mitzenmacher 2014 |
| quotient-filter | Quotient filter | Cache-unfriendly Bloom; the leap is *store hash quotients in a compact, resizable, mergeable array*. | Bender et al. 2012 |
| xor-filter | XOR filter | Beat Bloom's space at the same FP rate; the leap is *solve a small linear system over fingerprints*. | Graf–Lemire 2019 |
| consistent-hashing | Consistent hashing | Rehashing on resize moves everything; the leap is *map nodes+keys to a ring, move only neighbors*. | Karger et al. 1997 |
| rendezvous-hashing | Rendezvous (HRW) hashing | Same goal, no ring state; the leap is *pick the node maximizing hash(key,node)*. | Thaler–Ravishankar 1996 |

## 3. Heaps & priority queues — domain `Heaps`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| binary-heap | Binary heap | Need a cheap priority queue; the leap is *a complete tree flattened into an array*, parent/child by index. | Williams 1964 |
| binomial-heap | Binomial heap | Mergeable priority queues; the leap is *a forest of binomial trees, merge = binary addition*. | Vuillemin 1978 |
| fibonacci-heap | Fibonacci heap | Dijkstra/Prim need cheap decrease-key; the leap is *lazy melding + cascading cuts → O(1) amortized decrease-key*. | Fredman–Tarjan 1987 |
| pairing-heap | Pairing heap | Fibonacci heap is complex; the leap is *a self-adjusting multiway heap, simple to code, near-optimal in practice*. | Fredman–Sedgewick–Sleator–Tarjan 1986 |
| leftist-heap | Leftist heap | Fast melding; the leap is *keep the null-path short on the right, recurse merges there*. | Crane 1972 |
| skew-heap | Skew heap | Leftist needs stored ranks; the leap is *unconditionally swap children on merge — amortized magic*. | Sleator–Tarjan 1986 |
| brodal-queue | Brodal queue | Worst-case (not amortized) O(1) decrease-key; the leap is intricate guide/structure bookkeeping. | Brodal 1996 |
| dary-heap | d-ary heap | Decrease-key vs extract-min tradeoff; the leap is *tune the branching factor d*. | Johnson 1975 |
| soft-heap | Soft heap | Beat comparison bounds by *corrupting* keys; the leap is *allow bounded error to get O(1) amortized ops*. | Chazelle 2000 |

## 4. Search trees & ordered structures — domain `Search trees`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| avl-tree | AVL tree | Unbalanced BSTs degrade to lists; the leap is *track height balance, rotate to restore it*. | Adelson-Velsky–Landis 1962 |
| red-black-tree | Red-black tree | AVL rotates too eagerly; the leap is *color invariants giving looser, cheaper balance*. | Guibas–Sedgewick 1978 |
| b-tree | B-tree | Disk seeks dominate; the leap is *fat nodes matched to a disk page → shallow trees*. | Bayer–McCreight 1972 |
| b-plus-tree | B+ tree | Range scans on B-trees; the leap is *keys only in leaves, leaves linked for sequential scan*. | Comer 1979 |
| splay-tree | Splay tree | No balance metadata, exploit locality; the leap is *splay accessed node to root → amortized optimal*. | Sleator–Tarjan 1985 |
| treap | Treap | Balance without rebalancing logic; the leap is *assign random priorities, keep a heap on them*. | Seidel–Aragon 1996 |
| scapegoat-tree | Scapegoat tree | Balance with zero per-node overhead; the leap is *rebuild the subtree of the deepest "scapegoat" when too skewed*. | Galperin–Rivest 1993 |
| skip-list | Skip list | Balanced trees are fiddly to code; the leap is *a linked list with random express lanes* — probabilistic balance. | Pugh 1990 |
| van-emde-boas | van Emde Boas tree | O(log n) too slow for integer keys in a universe u; the leap is *recurse on √u → O(log log u)*. | van Emde Boas 1975 |
| y-fast-trie | y-fast trie | vEB's space; the leap is *x-fast trie + bucketing → O(n) space, O(log log u) ops*. | Willard 1983 |
| fusion-tree | Fusion tree | Beat the comparison bound for integers; the leap is *pack many keys into one word, compare in parallel*. | Fredman–Willard 1993 |
| segment-tree | Segment tree | Range queries with point updates; the leap is *a tree over intervals, query = O(log n) canonical pieces*. | Bentley 1977 |
| fenwick-tree | Fenwick tree (BIT) | Segment tree's memory/constant; the leap is *prefix sums indexed by low-bit decomposition*. | Fenwick 1994 |
| sparse-table | Sparse table | Idempotent range queries (min/gcd) in O(1); the leap is *precompute power-of-two intervals*. | Bender–Farach-Colton 2000 |
| link-cut-tree | Link-cut tree | Dynamic tree path queries; the leap is *preferred-path decomposition over splay trees*. | Sleator–Tarjan 1983 |
| kd-tree | k-d tree | Multidimensional nearest-neighbor; the leap is *recursively split space on alternating axes*. | Bentley 1975 |

## 5. Strings & text — domain `Strings & text`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| kmp | Knuth–Morris–Pratt | Naive matching re-scans after a mismatch; the leap is *a failure function — never back up in the text*. | Knuth–Morris–Pratt 1977 |
| boyer-moore | Boyer–Moore | Most positions can't match; the leap is *scan right-to-left, skip via bad-char/good-suffix*. | Boyer–Moore 1977 |
| rabin-karp | Rabin–Karp | Multiple/parallel pattern search; the leap is *rolling hash → O(1) window compare*. | Karp–Rabin 1987 |
| z-algorithm | Z-algorithm | Linear-time matching without failure tables; the leap is *Z-box reuse of prior matches*. | Gusfield 1997 |
| aho-corasick | Aho–Corasick | Many patterns at once; the leap is *a trie + failure links → a string-matching automaton*. | Aho–Corasick 1975 |
| manacher | Manacher's algorithm | All palindromic substrings; the leap is *mirror known radii across the current center*. | Manacher 1975 |
| suffix-tree | Suffix tree | Any substring query in one structure; the leap is *a compressed trie of all suffixes*. | Weiner 1973 |
| ukkonen | Ukkonen's algorithm | Suffix trees were built right-to-left; the leap is *online, left-to-right, linear with suffix links*. | Ukkonen 1995 |
| suffix-array | Suffix array | Suffix trees waste space; the leap is *just sort the suffixes — array + LCP does the same work*. | Manber–Myers 1990 |
| dc3-suffix-array | DC3 / skew algorithm | Linear suffix-array construction; the leap is *sort 2/3 of suffixes recursively, merge in the rest*. | Kärkkäinen–Sanders 2003 |
| sa-is | SA-IS | Simpler linear SA construction; the leap is *induced sorting from LMS substrings*. | Nong–Zhang–Chan 2009 |
| suffix-automaton | Suffix automaton (DAWG) | Smallest automaton recognizing all substrings; the leap is *online construction via suffix links + clones*. | Blumer et al. 1985 |
| bwt | Burrows–Wheeler transform | Compression wants local structure; the leap is *sort rotations → a reversible, clustering permutation*. | Burrows–Wheeler 1994 |
| fm-index | FM-index | Search compressed text without decompressing; the leap is *backward search over BWT + rank*. | Ferragina–Manzini 2000 |
| wavelet-tree | Wavelet tree | rank/select on large alphabets; the leap is *a balanced hierarchy of bitvectors*. | Grossi–Gupta–Vitter 2003 |
| edit-distance | Wagner–Fischer edit distance | Compare sequences by minimal edits; the leap is *DP table over prefixes*. | Wagner–Fischer 1974 |
| hirschberg | Hirschberg's algorithm | Edit distance needs O(nm) space for the path; the leap is *divide-and-conquer → linear space*. | Hirschberg 1975 |
| myers-diff | Myers diff | Practical line diffs; the leap is *shortest edit script = shortest path in an edit graph, O(ND)*. | Myers 1986 |
| bitap | Bitap / shift-or | Approximate matching; the leap is *encode the automaton state in machine-word bits*. | Baeza-Yates–Gonnet 1992 |
| lz-factorization | Lempel–Ziv factorization | Find repeats for compression; the leap is *greedy longest previous factor via suffix structures*. | Lempel–Ziv 1976 |

## 6. Graph algorithms — domain `Graph algorithms`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| dijkstra | Dijkstra's algorithm | Shortest paths with nonneg weights; the leap is *greedily settle the closest unfinished node*. | Dijkstra 1959 |
| bellman-ford | Bellman–Ford | Negative edges break greed; the leap is *relax all edges V−1 times*. | Bellman 1958 |
| floyd-warshall | Floyd–Warshall | All-pairs shortest paths; the leap is *DP over "allowed intermediate vertices"*. | Floyd 1962 |
| johnson | Johnson's algorithm | APSP on sparse graphs with negatives; the leap is *reweight via Bellman–Ford, then run Dijkstra n times*. | Johnson 1977 |
| astar | A* search | Dijkstra explores blindly; the leap is *add an admissible heuristic to guide the frontier*. | Hart–Nilsson–Raphael 1968 |
| prim | Prim's MST | Build a spanning tree cheaply; the leap is *grow one tree, always add the cheapest crossing edge*. | Prim 1957 |
| kruskal | Kruskal's MST | Same goal, different greed; the leap is *sort edges, add if it joins two components*. | Kruskal 1956 |
| boruvka | Borůvka's algorithm | Parallel MST; the leap is *every component picks its cheapest edge simultaneously*. | Borůvka 1926 |
| union-find | Union–Find (DSU) | Dynamic connectivity; the leap is *path compression + union by rank → inverse-Ackermann*. | Tarjan 1975 |
| tarjan-scc | Tarjan's SCC | Strongly connected components in one pass; the leap is *DFS low-link values + a stack*. | Tarjan 1972 |
| kosaraju-scc | Kosaraju's SCC | Same, two passes; the leap is *DFS, then DFS the transpose in finish order*. | Sharir/Kosaraju 1981 |
| articulation-bridges | Articulation points & bridges | Find cut vertices/edges; the leap is *DFS discovery times vs low-link reachability*. | Hopcroft–Tarjan 1973 |
| topological-sort | Topological sort | Order a DAG; the leap is *repeatedly remove in-degree-0 nodes* (or DFS finish order). | Kahn 1962 |
| lca-tarjan | Tarjan offline LCA | Many LCA queries; the leap is *DFS + Union-Find to answer at backtrack time*. | Tarjan 1979 |
| lca-rmq | LCA via Euler tour + RMQ | Online LCA; the leap is *flatten the tree, reduce LCA to range-minimum*. | Bender–Farach-Colton 2000 |
| heavy-light | Heavy-light decomposition | Path queries on trees; the leap is *split into heavy chains → O(log n) segments per path*. | Sleator–Tarjan 1981 |
| centroid-decomposition | Centroid decomposition | Divide-and-conquer on trees; the leap is *recurse on centroids → O(log n) layers*. | Jordan 1869 / folklore |
| pagerank | PageRank | Rank web pages by importance; the leap is *importance = stationary distribution of a random surfer*. | Page–Brin 1998 |
| hits | HITS (hubs & authorities) | Query-dependent ranking; the leap is *mutually reinforcing hub/authority scores*. | Kleinberg 1999 |
| edmonds-arborescence | Chu–Liu/Edmonds | Min spanning arborescence (directed MST); the leap is *contract zero-cost cycles, recurse*. | Chu–Liu 1965 / Edmonds 1967 |
| 2-sat | 2-SAT via implication graph | Satisfy 2-clauses fast; the leap is *clauses → implications → SCCs decide satisfiability*. | Aspvall–Plass–Tarjan 1979 |
| eulerian-path | Hierholzer's algorithm | Trace every edge once; the leap is *splice cycles together greedily*. | Hierholzer 1873 |
| planarity-test | Hopcroft–Tarjan planarity | Is a graph planar? the leap is *linear-time path addition / embedding*. | Hopcroft–Tarjan 1974 |
| yen-k-shortest | Yen's k-shortest paths | Need alternates, not just the best; the leap is *deviate from the best path at each node*. | Yen 1971 |
| thorup-sssp | Thorup linear SSSP | Beat Dijkstra's sorting bottleneck for integer weights; the leap is *hierarchical bucketing, no global heap*. | Thorup 1999 |
| spfa | SPFA | Bellman–Ford is wasteful; the leap is *queue only nodes whose distance changed*. | Duan 1994 |

## 7. Flows, matching & combinatorial optimization — domain `Combinatorial optimization`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| ford-fulkerson | Ford–Fulkerson | Max flow; the leap is *augment along residual paths until none remain*. | Ford–Fulkerson 1956 |
| edmonds-karp | Edmonds–Karp | Ford–Fulkerson can loop on bad paths; the leap is *augment along shortest (BFS) paths → polynomial*. | Edmonds–Karp 1972 |
| dinic | Dinic's algorithm | Speed up max flow; the leap is *level graph + blocking flows*. | Dinic 1970 |
| push-relabel | Push–relabel | Augmenting paths are global; the leap is *push excess locally, relabel heights*. | Goldberg–Tarjan 1988 |
| hopcroft-karp | Hopcroft–Karp | Bipartite matching faster; the leap is *augment many shortest paths per phase → O(E√V)*. | Hopcroft–Karp 1973 |
| hungarian | Hungarian algorithm | Min-cost assignment; the leap is *dual potentials / matrix reductions reveal a zero-cost matching*. | Kuhn 1955 |
| blossom | Edmonds' blossom algorithm | General (non-bipartite) matching; the leap is *contract odd "blossom" cycles to expose augmenting paths*. | Edmonds 1965 |
| stable-matching | Gale–Shapley | Stable marriages; the leap is *deferred acceptance — propose, tentatively hold, bump*. | Gale–Shapley 1962 |
| stoer-wagner | Stoer–Wagner min cut | Global min cut without flows; the leap is *maximum-adjacency ordering merges the last two*. | Stoer–Wagner 1997 |
| karger-min-cut | Karger's min cut | Randomized global min cut; the leap is *random edge contraction survives the min cut often enough*. | Karger 1993 |
| gomory-hu | Gomory–Hu tree | All-pairs min cuts; the leap is *n−1 max-flow computations encode them all in a tree*. | Gomory–Hu 1961 |
| simplex | Simplex method | Linear programming; the leap is *walk vertex-to-vertex along improving edges of the polytope*. | Dantzig 1947 |
| ellipsoid-method | Ellipsoid method | Is LP polynomial? the leap is *shrink an ellipsoid around the feasible region*. | Khachiyan 1979 |
| karmarkar | Karmarkar's algorithm | Practical polynomial LP; the leap is *move through the interior via projective transforms*. | Karmarkar 1984 |
| branch-and-bound | Branch and bound | Exact discrete optimization; the leap is *prune subtrees using bounds*. | Land–Doig 1960 |
| cutting-planes | Gomory cutting planes | Integer programming; the leap is *add valid inequalities that slice off fractional optima*. | Gomory 1958 |
| lin-kernighan | Lin–Kernighan | Near-optimal TSP tours; the leap is *variable-depth sequential edge exchanges*. | Lin–Kernighan 1973 |
| christofides | Christofides algorithm | Approximate metric TSP; the leap is *MST + min matching on odd-degree vertices → 1.5-approx*. | Christofides 1976 |

## 8. Computational geometry — domain `Computational geometry`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| graham-scan | Graham scan | Convex hull; the leap is *sort by angle, walk and pop right turns*. | Graham 1972 |
| jarvis-march | Jarvis march (gift wrapping) | Output-sensitive hull; the leap is *repeatedly pick the most clockwise next point*. | Jarvis 1973 |
| quickhull | QuickHull | Divide-and-conquer hull; the leap is *recurse on the farthest point from a hull edge*. | Barber–Dobkin–Huhdanpaa 1996 |
| chan-hull | Chan's algorithm | Optimal output-sensitive hull; the leap is *combine Graham + Jarvis with guessed group size*. | Chan 1996 |
| closest-pair | Closest pair of points | Avoid O(n²) distance checks; the leap is *divide-and-conquer + a thin strip merge*. | Shamos–Hoey 1975 |
| bentley-ottmann | Bentley–Ottmann | All segment intersections; the leap is *sweep a line, maintain order, test only neighbors*. | Bentley–Ottmann 1979 |
| fortune-voronoi | Fortune's algorithm | Voronoi diagram; the leap is *a sweepline + parabolic beach front*. | Fortune 1987 |
| delaunay-triangulation | Delaunay triangulation | Best-shaped triangulation; the leap is *empty-circumcircle property, dual of Voronoi*. | Delaunay 1934 |
| bowyer-watson | Bowyer–Watson | Incremental Delaunay; the leap is *insert a point, retriangulate its violated cavity*. | Bowyer / Watson 1981 |
| kirkpatrick-location | Kirkpatrick point location | Which region contains a query? the leap is *a hierarchy of coarsening triangulations*. | Kirkpatrick 1983 |
| rotating-calipers | Rotating calipers | Diameter/width of a hull; the leap is *spin two parallel support lines around it*. | Toussaint 1983 |
| welzl-mec | Welzl's smallest enclosing circle | Min enclosing circle; the leap is *randomized incremental — at most 3 points define it*. | Welzl 1991 |
| ear-clipping | Ear clipping triangulation | Triangulate a simple polygon; the leap is *repeatedly snip a convex "ear"*. | Meisters 1975 |
| r-tree | R-tree | Spatial indexing on disk; the leap is *bounding-box hierarchy, B-tree style*. | Guttman 1984 |
| bvh | Bounding volume hierarchy | Ray/collision queries; the leap is *recursively box-group primitives, prune by box test*. | Clark 1976 / Rubin–Whitted 1980 |
| segment-intersection-sweep | Sweep-line framework | Many geometry problems share structure; the leap is *event queue + status structure*. | Shamos–Hoey 1976 |

## 9. Numerical & algebraic — domain `Numerical & algebraic`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| fft | Cooley–Tukey FFT | DFT is O(n²); the leap is *recurse on even/odd indices → O(n log n)*. | Cooley–Tukey 1965 |
| ntt | Number-theoretic transform | FFT with exact integer arithmetic; the leap is *do the FFT in a finite field*. | Pollard 1971 |
| karatsuba | Karatsuba multiplication | Schoolbook multiply is O(n²); the leap is *3 multiplications instead of 4*. | Karatsuba 1962 |
| toom-cook | Toom–Cook multiplication | Generalize Karatsuba; the leap is *evaluate/interpolate polynomials at more points*. | Toom 1963 / Cook 1966 |
| schonhage-strassen | Schönhage–Strassen | Near-linear big multiply; the leap is *multiplication = convolution = FFT over a ring*. | Schönhage–Strassen 1971 |
| strassen | Strassen matrix multiply | n³ matmul; the leap is *7 multiplies for a 2×2 block instead of 8*. | Strassen 1969 |
| conjugate-gradient | Conjugate gradient | Solve SPD systems without inverting; the leap is *conjugate search directions, exact in n steps*. | Hestenes–Stiefel 1952 |
| gmres | GMRES | Nonsymmetric linear systems; the leap is *minimize the residual over a growing Krylov subspace*. | Saad–Schultz 1986 |
| lanczos | Lanczos algorithm | Few eigenvalues of huge sparse matrices; the leap is *tridiagonalize within a Krylov subspace*. | Lanczos 1950 |
| qr-algorithm | QR algorithm | Compute all eigenvalues; the leap is *iterate A = QR → RQ until triangular*. | Francis 1961 |
| power-iteration | Power iteration | Dominant eigenvector; the leap is *repeated multiply amplifies the top eigendirection*. | von Mises 1929 |
| gaussian-elimination | Gaussian elimination | Solve linear systems; the leap is *systematic row reduction to triangular form*. | Gauss / ancient |
| simulated-annealing | Simulated annealing | Escape local optima; the leap is *accept worse moves with a cooling probability*. | Kirkpatrick–Gelatt–Vecchi 1983 |
| newton-method | Newton's method | Root finding; the leap is *follow the tangent line to the next guess*. | Newton 1669 |
| fast-multipole | Fast multipole method | N-body forces are O(N²); the leap is *multipole expansions group far interactions → O(N)*. | Greengard–Rokhlin 1987 |
| barnes-hut | Barnes–Hut | Same N-body pain, simpler; the leap is *an octree treats distant clusters as one mass → O(N log N)*. | Barnes–Hut 1986 |
| remez | Remez exchange | Best polynomial approximation; the leap is *iteratively equalize the error extrema*. | Remez 1934 |
| montgomery-multiplication | Montgomery multiplication | Modular multiply without division; the leap is *work in a residue domain where reduction is a shift*. | Montgomery 1985 |

## 10. Randomized, streaming & probabilistic — domain `Randomized & streaming`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| reservoir-sampling | Reservoir sampling | Sample k from an unknown-length stream; the leap is *replace with probability k/i*. | Vitter 1985 |
| fisher-yates | Fisher–Yates shuffle | Unbiased permutation; the leap is *swap each element with a uniform earlier/later one*. | Fisher–Yates / Durstenfeld 1964 |
| morris-counter | Morris approximate counter | Count to N in log log N bits; the leap is *increment probabilistically, store the exponent*. | Morris 1978 |
| hyperloglog | HyperLogLog | Count distinct in tiny space; the leap is *max leading-zero run estimates cardinality, harmonic-mean the buckets*. | Flajolet et al. 2007 |
| count-min-sketch | Count-Min sketch | Frequency estimates in sublinear space; the leap is *hash into d×w counters, take the min*. | Cormode–Muthukrishnan 2005 |
| count-sketch | Count sketch | Unbiased heavy-hitter estimates; the leap is *signed hashes cancel collisions in expectation*. | Charikar–Chen–Farach-Colton 2002 |
| ams-sketch | AMS sketch | Estimate F2/stream norms; the leap is *random ±1 projections, average of medians*. | Alon–Matias–Szegedy 1996 |
| misra-gries | Misra–Gries | Frequent items in one pass; the leap is *keep k counters, decrement-all on overflow*. | Misra–Gries 1982 |
| space-saving | Space-Saving | Top-k with tight bounds; the leap is *evict the min counter, inherit its count*. | Metwally–Agrawal–Abbadi 2005 |
| minhash | MinHash | Estimate Jaccard similarity; the leap is *P(min hash equal) = Jaccard*. | Broder 1997 |
| simhash | SimHash | Near-duplicate detection; the leap is *random hyperplane signs → similar items share bits*. | Charikar 2002 |
| lsh | Locality-sensitive hashing | Approximate nearest neighbor in high-D; the leap is *hashes that collide more for nearby points*. | Indyk–Motwani 1998 |
| miller-rabin | Miller–Rabin primality | Fast probabilistic primality; the leap is *witnesses to compositeness via Fermat + square roots of 1*. | Miller 1976 / Rabin 1980 |
| t-digest | t-digest | Streaming quantiles; the leap is *variable-size centroids, finer near the tails*. | Dunning 2013 |

## 11. Compression & coding — domain `Compression & coding`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| huffman | Huffman coding | Optimal prefix code; the leap is *merge the two least-frequent symbols bottom-up*. | Huffman 1952 |
| arithmetic-coding | Arithmetic coding | Huffman wastes fractional bits; the leap is *encode the whole message as one interval*. | Witten–Neal–Cleary 1987 |
| ans | Asymmetric numeral systems | Arithmetic speed + Huffman simplicity; the leap is *a single integer state encodes the stream*. | Duda 2009 |
| lz77 | LZ77 | Dictionary compression; the leap is *replace repeats with (distance, length) back-references*. | Ziv–Lempel 1977 |
| lz78 | LZ78 | Streaming dictionary; the leap is *build an explicit phrase dictionary incrementally*. | Ziv–Lempel 1978 |
| lzw | LZW | Practical LZ78; the leap is *grow the dictionary implicitly, no explicit indices transmitted*. | Welch 1984 |
| deflate | DEFLATE | Ship a real codec; the leap is *LZ77 + Huffman in one stream (gzip/zlib)*. | Katz 1991 |
| ppm | PPM | Context modeling; the leap is *predict via variable-length contexts, escape to shorter ones*. | Cleary–Witten 1984 |
| elias-codes | Elias gamma/delta codes | Code integers without a fixed width; the leap is *encode the length, then the value*. | Elias 1975 |
| golomb-rice | Golomb–Rice coding | Geometric-distributed integers; the leap is *quotient in unary, remainder in binary*. | Golomb 1966 |
| shannon-fano | Shannon–Fano coding | First prefix code; the leap is *recursively split symbols into near-equal-probability halves*. | Shannon / Fano 1949 |
| range-coding | Range coding | Arithmetic coding, patent-free + byte-wise; the leap is *renormalize on a wide integer range*. | Martin 1979 |

## 12. Number theory & cryptographic algorithms — domain `Cryptographic algorithms`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| euclid-gcd | Euclidean algorithm | GCD without factoring; the leap is *gcd(a,b) = gcd(b, a mod b)*. | Euclid c.300 BC |
| extended-euclid | Extended Euclidean | Modular inverses; the leap is *track Bézout coefficients alongside the GCD*. | classic |
| sieve-eratosthenes | Sieve of Eratosthenes | List primes fast; the leap is *cross out multiples instead of testing each*. | Eratosthenes c.240 BC |
| sieve-atkin | Sieve of Atkin | Faster prime sieve; the leap is *use quadratic forms mod 60 to mark primes*. | Atkin–Bernstein 2003 |
| pollard-rho | Pollard's rho | Factor without trial division; the leap is *a pseudo-random walk + cycle detection finds a factor*. | Pollard 1975 |
| pollard-p-minus-1 | Pollard's p−1 | Special-form factoring; the leap is *exploit smooth p−1 via Fermat's little theorem*. | Pollard 1974 |
| quadratic-sieve | Quadratic sieve | General factoring; the leap is *find x²≡y² by sieving smooth relations + linear algebra*. | Pomerance 1981 |
| chinese-remainder | Chinese remainder theorem | Reconstruct from residues; the leap is *independent moduli pin down a unique value*. | Sunzi c.400 |
| tonelli-shanks | Tonelli–Shanks | Square roots mod p; the leap is *climb the 2-Sylow subgroup*. | Tonelli 1891 / Shanks 1973 |
| aks-primality | AKS primality test | Deterministic poly-time primality; the leap is *check a polynomial congruence (x+a)ⁿ≡xⁿ+a*. | Agrawal–Kayal–Saxena 2002 |
| rsa | RSA | Public-key encryption; the leap is *trapdoor from the hardness of factoring*. | Rivest–Shamir–Adleman 1977 |
| diffie-hellman | Diffie–Hellman | Key exchange over a public channel; the leap is *shared secret from commutative exponentiation*. | Diffie–Hellman 1976 |
| ecc | Elliptic-curve cryptography | Smaller keys, same security; the leap is *the discrete log on an elliptic-curve group*. | Miller / Koblitz 1985 |
| shor | Shor's algorithm | Factoring on a quantum computer; the leap is *period-finding via the quantum Fourier transform*. | Shor 1994 |

## 13. Concurrency, distributed & systems algorithms — domain `Concurrency & distributed`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| peterson-lock | Peterson's algorithm | Mutual exclusion with only loads/stores; the leap is *"I want in" flags + a turn variable*. | Peterson 1981 |
| lamport-bakery | Lamport's bakery | N-process mutex without atomics; the leap is *take a ticket, serve in order*. | Lamport 1974 |
| lamport-clocks | Lamport timestamps | Order events without a global clock; the leap is *a counter that bumps on send/receive*. | Lamport 1978 |
| vector-clocks | Vector clocks | Detect causality, not just order; the leap is *one counter per process, compare component-wise*. | Fidge / Mattern 1988 |
| chandy-lamport | Chandy–Lamport snapshot | Consistent global state of a running system; the leap is *marker messages flush the channels*. | Chandy–Lamport 1985 |
| two-phase-commit | Two-phase commit | Atomic commit across nodes; the leap is *prepare-vote then commit/abort*. | Gray 1978 |
| paxos | Paxos | Consensus despite failures; the leap is *prepare/promise/accept with majority quorums*. | Lamport 1998 |
| raft | Raft | Paxos is unintelligible; the leap is *leader election + log replication, made understandable*. | Ongaro–Ousterhout 2014 |
| byzantine-generals | Byzantine generals | Agreement with lying nodes; the leap is *3f+1 nodes tolerate f traitors*. | Lamport–Shostak–Pease 1982 |
| pbft | Practical BFT | BFT fast enough to deploy; the leap is *three-phase agreement with a stable leader*. | Castro–Liskov 1999 |
| michael-scott-queue | Michael–Scott queue | Lock-free FIFO; the leap is *CAS on head/tail with helping*. | Michael–Scott 1996 |
| treiber-stack | Treiber stack | Lock-free LIFO; the leap is *CAS the head, retry on conflict*. | Treiber 1986 |
| hazard-pointers | Hazard pointers | Safe memory reclamation lock-free; the leap is *publish what you're about to read, defer frees*. | Michael 2004 |
| rcu | Read-copy-update | Near-free reads under updates; the leap is *update a copy, reclaim after a grace period*. | McKenney–Slingwine 1998 |
| work-stealing | Work-stealing scheduler | Balance parallel load; the leap is *idle workers steal from the tails of busy deques*. | Blumofe–Leiserson 1999 |
| lsm-tree | Log-structured merge tree | Write-heavy storage; the leap is *batch in memory, flush sorted runs, merge in the background*. | O'Neil et al. 1996 |
| aries | ARIES recovery | Crash recovery with WAL; the leap is *redo-then-undo with LSNs + fuzzy checkpoints*. | Mohan et al. 1992 |
| crdt | CRDTs | Conflict-free replicated state; the leap is *merge operations that are commutative/associative/idempotent*. | Shapiro et al. 2011 |
| gossip-protocol | Gossip / epidemic protocols | Disseminate updates at scale; the leap is *each node tells a few random peers → exponential spread*. | Demers et al. 1987 |
| skip-graph | Skip graph | Distributed ordered overlay; the leap is *a distributed skip list for range queries on DHTs*. | Aspnes–Shah 2003 |
| chord-dht | Chord DHT | Decentralized key lookup; the leap is *consistent hashing + finger tables → O(log n) hops*. | Stoica et al. 2001 |
| cache-oblivious | Cache-oblivious algorithms | Tune for memory hierarchy without knowing it; the leap is *recursive divide-and-conquer is optimal at every level*. | Frigo–Leiserson et al. 1999 |

## 14. Parsing & automata — domain `Parsing & automata`

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| thompson-nfa | Thompson NFA construction | Regex matching; the leap is *compile the regex to an NFA, simulate all states at once*. | Thompson 1968 |
| dfa-minimization | Hopcroft DFA minimization | Smallest equivalent DFA; the leap is *partition-refinement of states → O(n log n)*. | Hopcroft 1971 |
| earley-parser | Earley parser | Parse any CFG; the leap is *chart of dotted items advanced by scan/predict/complete*. | Earley 1970 |
| cyk | CYK algorithm | CFG membership in P; the leap is *DP over substrings in Chomsky normal form*. | Cocke–Younger–Kasami 1965 |
| lr-parsing | LR parsing | Deterministic bottom-up parsing; the leap is *shift/reduce driven by an automaton over item sets*. | Knuth 1965 |
| pratt-parsing | Pratt parsing | Expression precedence cleanly; the leap is *bind tokens by left/right binding power*. | Pratt 1973 |
| glr-parsing | GLR parsing | Parse ambiguous/nondeterministic grammars; the leap is *fork the stack on conflicts (graph-structured stack)*. | Tomita 1985 |
| packrat-peg | Packrat parsing / PEG | Linear-time backtracking parsers; the leap is *memoize parse results to kill exponential retries*. | Ford 2002 |

## 15. AI for algorithms — domain `AI for algorithms` (bridge to the ML corpus)

| slug | Title | The discovery prior | Reference |
| --- | --- | --- | --- |
| alphadev | AlphaDev | Hand-tuned libc++ sort had plateaued; the leap is *RL searches assembly directly, finds shorter sort3/4/5 — merged into LLVM*. | Mankowitz et al. 2023 |
| alphatensor | AlphaTensor | Strassen-style matmul stuck for decades; the leap is *frame tensor decomposition as a game, RL finds new low-rank schemes*. | Fawzi et al. 2022 |
| alphaevolve | AlphaEvolve | One-shot LLM coding plateaus; the leap is *an evolutionary loop of LLM mutation + automated evaluation discovers algorithms (matmul, circle packing, datacenter/TPU)*. | Novikov et al. 2025 |
| funsearch | FunSearch | LLMs hallucinate on math; the leap is *evolve programs (not answers) scored by an evaluator → new cap-set / bin-packing results*. | Romera-Paredes et al. 2023 |
| alphageometry | AlphaGeometry | Olympiad geometry needs proofs; the leap is *neural conjecturing of auxiliary constructions + symbolic deduction*. | Trinh et al. 2024 |
| learned-index | Learned index structures | B-trees ignore data distribution; the leap is *a model predicts position — the index is a learned CDF*. | Kraska et al. 2018 |
| pointer-networks | Pointer networks | Outputs that index the input (TSP, convex hull); the leap is *attention as a pointer over input positions*. | Vinyals et al. 2015 |
| neural-comb-opt | Neural combinatorial optimization | Hand-designed heuristics per problem; the leap is *RL learns a TSP/routing heuristic end-to-end*. | Bello et al. 2016 |
| neural-turing-machine | Neural Turing machine | Nets can't do algorithmic memory; the leap is *a differentiable external memory with read/write heads*. | Graves et al. 2014 |
| neural-algorithmic-reasoning | Neural algorithmic reasoning | Nets don't generalize like algorithms; the leap is *train GNNs to imitate classical algorithm steps (CLRS)*. | Veličković et al. 2021 |
| dancing-links | Dancing Links (Algorithm X) | Exact cover / Sudoku backtracking; the leap is *a doubly-linked structure that undoes deletions in O(1)*. | Knuth 2000 |
| satzilla-cdcl | CDCL SAT solving | DPLL backtracks blindly; the leap is *learn a clause from each conflict, backjump non-chronologically*. | Marques-Silva–Sakallah 1996 |

---

## Counts

15 groups, ~205 candidate methods. Strongest "discovery prior" density is in groups 1–6 and 15
(sorting, hashing, heaps, trees, strings, graphs, AI-for-algorithms). Group 15 is the natural
**first pilot wave** — it bridges directly to the existing ML corpus and includes AlphaDev (the
sorting-optimization × ML bullseye discussed in session).

## Suggested first wave (6)

`alphadev`, `alphatensor`, `alphaevolve`, `funsearch` (AI-for-algorithms bridge) +
`quicksort`, `fibonacci-heap` (pure-classic anchors, crisp aha). Build all three artifacts
(`context.md` / `reasoning.md` / `answer.md`) for each, add to `methods.json`, eyeball quality,
then scale per group.
