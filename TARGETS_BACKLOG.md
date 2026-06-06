# Innovation-Prior Targets — Combinatorial Optimization & Classic CS

Wave-based backlog for the "innovation prior" expansion: each target is a **TASK** (combinatorial
optimization / mathematical construction / classic-CS algorithm design), and each subagent
reconstructs the **principled, insight-driven method** for it (NOT a brute-force / evolutionary
search output — that anti-pattern is what we teach the model to transcend). Everything web-grounded
per the `paper-to-reasoning` skill.

Subagents write `methods/<slug>/results/{context,reasoning,answer}.md`; raw materials under
`methods/<slug>/{src,refs,code,notes}/` (gitignored). `methods.json` is updated centrally (not by
subagents) to avoid concurrent-write conflicts. Status: ☐ todo · ◐ launched · ✓ done · ⚠ needs check.

## Wave 1 — launched (AlphaEvolve-core + anchors)

| status | slug | task |
| --- | --- | --- |
| ◐ | circle-packing-in-square | n equal circles in unit square, maximize radius |
| ◐ | kissing-number | max unit spheres touching a central one (LP bound, E8/Leech) |
| ◐ | cap-set | no-3-AP sets in F_3^n (polynomial method / slice rank) |
| ◐ | erdos-minimum-overlap | minimum overlap constant via variational/LP method |
| ◐ | fast-matrix-multiplication | subcubic matmul (Strassen + tensor rank) |
| ◐ | maxcut-sdp | Goemans–Williamson SDP + hyperplane rounding |
| ◐ | heilbronn-triangle | maximize smallest triangle area of n points |
| ◐ | low-autocorrelation-sequences | merit factor, Legendre construction + character sums |
| ◐ | christofides-tsp | metric TSP 3/2-approximation |
| ◐ | hyperloglog | distinct counting in sublinear space |

## Wave 2 — AlphaEvolve math / geometry / packing

| status | slug | task |
| --- | --- | --- |
| ☐ | sphere-packing-lattices | densest lattice/sphere packings; LP bounds (Cohn–Elkies) |
| ☐ | tammes-problem | spread n points on a sphere maximizing min distance |
| ☐ | thomson-problem | min-energy point configurations on a sphere |
| ☐ | sums-and-differences-sets | MSTD sets / Erdős sum-difference constructions |
| ☐ | sum-free-sets | largest sum-free subset; density bounds |
| ☐ | constructive-ramsey | explicit Ramsey-graph lower-bound constructions |
| ☐ | finite-field-kakeya | Kakeya sets over finite fields (polynomial method) |
| ☐ | autocorrelation-inequalities | Erdős autocorrelation / nonnegative-cosine extremal constants |
| ☐ | moser-spindle-chromatic | Hadwiger–Nelson / chromatic number of the plane constructions |
| ☐ | minimum-energy-riesz | Riesz s-energy minimization configurations |

## Wave 3 — fast algebra & numerical methods

| status | slug | task |
| --- | --- | --- |
| ☐ | karatsuba-multiplication | subquadratic integer multiplication |
| ☐ | toom-cook-multiplication | evaluate/interpolate generalization of Karatsuba |
| ☐ | schonhage-strassen | FFT-based near-linear integer multiplication |
| ☐ | harvey-hoeven-multiplication | O(n log n) integer multiplication |
| ☐ | cooley-tukey-fft | O(n log n) DFT |
| ☐ | winograd-convolution | minimal-filtering convolution (used in CNN kernels) |
| ☐ | strassen-tensor-decomposition | matmul tensor rank / border rank framing |
| ☐ | toeplitz-superfast-solver | superfast Toeplitz/structured-matrix solvers |
| ☐ | montgomery-multiplication | division-free modular multiplication |
| ☐ | barrett-reduction | modular reduction via precomputed reciprocal |

## Wave 4 — combinatorial optimization & approximation

| status | slug | task |
| --- | --- | --- |
| ☐ | karlin-klein-tsp | (3/2−ε) metric TSP via randomized rounding |
| ☐ | submodular-greedy | (1−1/e) greedy for monotone submodular maximization |
| ☐ | algorithmic-lll | Moser–Tardos algorithmic Lovász Local Lemma |
| ☐ | jain-iterative-rounding | iterative LP rounding for survivable network design |
| ☐ | karmarkar-karp-binpacking | asymptotic FPTAS for bin packing |
| ☐ | primal-dual-steiner-forest | primal-dual 2-approximation for Steiner forest |
| ☐ | ellipsoid-method | polynomial-time LP via shrinking ellipsoids |
| ☐ | interior-point-lp | Karmarkar / central-path interior-point LP |
| ☐ | lin-kernighan-tsp | variable-depth local search for TSP |
| ☐ | held-karp-lower-bound | Lagrangian 1-tree bound for TSP |
| ☐ | metric-embedding-bourgain | Bourgain embedding into ℓ_p / metric methods |
| ☐ | multiplicative-weights | multiplicative-weights / online learning meta-method |

## Wave 5 — online, streaming & randomized

| status | slug | task |
| --- | --- | --- |
| ☐ | k-server-problem | competitive k-server / work-function algorithm |
| ☐ | online-bipartite-ranking | Karp–Vazirani–Vazirani RANKING (1−1/e) |
| ☐ | adwords-online-matching | online budgeted matching / AdWords |
| ☐ | ski-rental | rent-or-buy competitive analysis |
| ☐ | paging-marking | competitive paging / marking algorithm |
| ☐ | count-min-sketch | frequency estimation in sublinear space |
| ☐ | ams-frequency-moments | AMS sketch for F_2 / frequency moments |
| ☐ | minhash-lsh | Jaccard similarity / locality-sensitive hashing |
| ☐ | karger-min-cut | randomized contraction global min cut |
| ☐ | reservoir-sampling | uniform sampling from a stream |
| ☐ | johnson-lindenstrauss | distance-preserving random projection |
| ☐ | miller-rabin-primality | randomized primality testing |

## Wave 6 — codes, designs, spectral & expanders

| status | slug | task |
| --- | --- | --- |
| ☐ | reed-solomon-codes | algebraic error-correction via polynomial evaluation |
| ☐ | ldpc-belief-propagation | sparse-graph codes + message-passing decoding |
| ☐ | polar-codes | channel polarization (Arıkan) |
| ☐ | expander-ramanujan | explicit expander / Ramanujan graph constructions |
| ☐ | zig-zag-product | combinatorial expander construction |
| ☐ | spectral-sparsification | Spielman–Srivastava graph sparsifiers |
| ☐ | laplacian-solver | near-linear-time SDD/Laplacian solvers |
| ☐ | steiner-systems-construction | combinatorial design / Steiner-system constructions |
| ☐ | fks-perfect-hashing | worst-case O(1) static dictionary |
| ☐ | cuckoo-hashing | worst-case O(1) lookup hashing |

## Wave 7 — sorting networks, search, SAT, scheduling, systems kernels

| status | slug | task |
| --- | --- | --- |
| ☐ | aks-sorting-network | O(log n)-depth sorting network |
| ☐ | batcher-bitonic-sort | bitonic / odd-even merge networks |
| ☐ | cdcl-sat | conflict-driven clause learning |
| ☐ | dpll-backtracking | DPLL with unit propagation |
| ☐ | astar-admissible-heuristics | A* / heuristic-guided search |
| ☐ | pattern-database-heuristics | additive pattern databases (Korf) |
| ☐ | alpha-beta-pruning | minimax with pruning |
| ☐ | monte-carlo-tree-search | UCT / MCTS |
| ☐ | simplex-method | vertex-walking LP |
| ☐ | branch-and-bound | exact discrete optimization via bounding |
| ☐ | gomory-cutting-planes | integer programming via valid inequalities |
| ☐ | dancing-links-exact-cover | Knuth's Algorithm X |
| ☐ | gemm-tiling-autotuning | cache/register tiling + autotuning for GEMM |
| ☐ | register-allocation-coloring | Chaitin graph-coloring register allocation |
| ☐ | list-scheduling | list scheduling for instruction/task scheduling |
| ☐ | polyhedral-loop-optimization | polyhedral model loop transforms |
| ☐ | consistent-hashing | minimal-disruption distributed hashing |
| ☐ | work-stealing-scheduler | provably efficient parallel scheduling |

## Wave 8+ — overflow pool (continue past 80 while tokens remain)

Continuation policy: **after the first 80 are done, keep generating from this pool (and extend it)
as long as token budget remains.** Refill in bounded waves on completion notifications.

| status | slug | task |
| --- | --- | --- |
| ☐ | suffix-automaton | smallest automaton recognizing all substrings |
| ☐ | suffix-array-dc3 | linear-time suffix-array construction (skew/DC3) |
| ☐ | fm-index | search compressed text via BWT + backward search |
| ☐ | aho-corasick | multi-pattern matching automaton |
| ☐ | fibonacci-heap | O(1) amortized decrease-key for Dijkstra/Prim |
| ☐ | link-cut-trees | dynamic-tree path queries |
| ☐ | van-emde-boas | O(log log u) integer priority queue |
| ☐ | fusion-tree | beat the comparison bound for integer keys |
| ☐ | persistent-data-structures | fully/partially persistent structures (fat node / path copying) |
| ☐ | union-find-analysis | inverse-Ackermann amortized analysis |
| ☐ | splay-tree | self-adjusting amortized-optimal BST |
| ☐ | treap-randomized-bst | balance via random priorities |
| ☐ | smawk-algorithm | row minima of totally monotone matrices |
| ☐ | knuth-yao-dp-speedup | quadrangle-inequality DP speedup |
| ☐ | convex-hull-trick | DP optimization via lower envelope of lines |
| ☐ | li-chao-tree | online lower-envelope queries |
| ☐ | mo-algorithm | offline query reordering on sqrt blocks |
| ☐ | heavy-path-fft-string | bitset / FFT string-matching speedups |
| ☐ | edmonds-arborescence | min-cost directed spanning tree |
| ☐ | stoer-wagner-min-cut | global min cut without flows |
| ☐ | gale-shapley-stable-matching | deferred-acceptance stable matching |
| ☐ | hungarian-assignment | min-cost bipartite assignment via dual potentials |
| ☐ | hopcroft-dfa-minimization | partition-refinement DFA minimization |
| ☐ | thompson-nfa | regex → NFA simulation |
| ☐ | earley-parser | parse any context-free grammar |
| ☐ | pratt-parsing | precedence parsing via binding powers |

Extend this pool further (more classic algorithms / open combinatorial problems) if it empties
before the budget does.

## Notes

- Wave 1 launched as background subagents; verify the first completed result end-to-end (output dir,
  in-frame voice, web-grounding, math) BEFORE scaling — a systematic mistake should not be amplified 80×.
- After each method's three files land and pass review, add its `methods.json` entry centrally:
  `{ "slug": ..., "title": ..., "domain": ..., "arxiv": <id|DOI|url> }` and commit in batches.
- Suggested domains: `Combinatorial optimization`, `Computational geometry`, `Extremal combinatorics`,
  `Fast algorithms`, `Approximation algorithms`, `Streaming & sketching`, `Coding & expanders`,
  `Search & SAT`, `Systems & compilers`.
- The earlier `ALGORITHM_CANDIDATES.md` was memory-drafted and pedagogical; this file supersedes it
  for the task-oriented, web-grounded effort.
