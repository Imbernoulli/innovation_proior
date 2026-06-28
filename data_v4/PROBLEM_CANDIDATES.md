# PROBLEM_CANDIDATES.md — Insight-Bearing SFT Problem List (FrontierCS + ALE-Bench)

> Goal: a concrete, diverse, self-contained candidate list for generating **new** Innovation-Prior SFT
> traces that land in the format both target benchmarks reward. Each FrontierCS candidate becomes a
> method dir `{context.md, reasoning.md, train_answer.md, verify/{sol.cpp, brute.py, gen.py}}` shaped
> exactly like the gold trace `data_v4/cp-noadj-commit/`: a **single-file C++17 program that reads
> stdin and writes stdout**, with the trace built around **one non-obvious idea** (and, where natural,
> a foil — the obvious-but-wrong approach — in `verify/`, mirroring `cp-noadj-commit/verify/greedy.cpp`).
>
> Why this exists (per `experiments/DATA_FIX_FCS_LANDING_zh.md`): the model already *reasons* well; it
> loses on FCS because the data taught it to deliver a Python library `class`, not a single-file C++
> stdin solver. So every candidate here is specified to land in the FCS-correct format, and every
> candidate's value is its **INSIGHT** — the project theme is the innovation prior: the trace must hit a
> genuinely non-obvious idea, not "apply Dijkstra/standard DP". Textbook-with-no-twist is explicitly out.
>
> Decontamination note: these are **newly authored** specs in the *spirit* of FrontierSmith/FrontierCS,
> not copies of their problems. Existing repo slugs (`data_v4/_comp_methods.json`,
> `methods/`) were checked to avoid overlap.

---

## 1. FrontierSmith method (summary + what we borrow)

Sourced from the FrontierSmith paper (*Synthesizing Open-Ended Coding Problems at Scale*,
arXiv:2605.14445, frontier-cs.org/blog/frontiersmith) and the FrontierCS benchmark paper
(*Evolving Challenges for Evolving Intelligence*, arXiv:2512.15699):

- **What FrontierSmith does.** It takes ordinary closed-ended contest problems and *mutates* them into
  open-ended ones via three operators: **(1) goal-modification** (turn a yes/no or exact target into a
  continuous optimize-this objective), **(2) output-restriction** (add a constraint that makes the exact
  optimum infeasible at scale, e.g. a degree-2 cap on MST turning it into a bounded-degree TSP variant),
  and **(3) input-generalization** (relax input assumptions so the easy structure disappears).
- **Selection by idea-divergence.** It keeps only problems that *provoke genuinely different solver
  strategies*: a semantic stage (an LLM judge compares whether sampled solutions use meaningfully
  different algorithms) plus a behavioral stage (compare per-test score vectors across solvers; discard
  low-divergence candidates). The thesis: closed-ended problems are dominated by a single "gold idea";
  good problems have many divergent good ideas.
- **Verifiable by construction.** Every kept problem ships a test-case **generator** + a **deterministic
  verifier** returning a normalized score in [0,1] (relative to a trivial baseline and a strong human
  reference); crashes/timeouts/invalid outputs score 0 — giving a continuous, abuse-resistant signal.
- **FrontierCS itself** is two tracks: *algorithmic* (optimization / constructive / interactive, NP-hard
  variants of contest problems with partial scoring) and *research*. No problem has a known optimum;
  any solution is deterministically scored 0–100.

**What we borrow (and what we deliberately diverge on).** We adopt the *spirit* — **insight-requiring,
idea-divergent, verifiable** — and the *generator+verifier+reference+baseline* artifact bundle. We
deliberately diverge on format to match what *our* harness scores: this repo's FCS landing requires a
**single-file C++ exact judge** (`§5.1 EXPERIMENTS_zh.md`), so the §2 FrontierCS list is **exact-judge
insight problems** (the idea-divergence is moved into the *reasoning*: the obvious approach is a
plausible trap, and the trace must find the one correct reformulation/invariant). FrontierSmith's
open-ended partial-scoring style maps naturally onto the **ALE-Bench** list (§3), where score-maximization
*is* the objective. The idea-divergence filter is reused as a generation-time quality gate (§4).

---

## 2. FrontierCS candidates (~50) — single-file C++17, stdin → stdout, exact judge

> Each row: `id | title | area | THE INSIGHT | I/O sketch | constraints/difficulty | oracle`.
> THE INSIGHT is the load-bearing field — the non-obvious idea the solver must discover. Every problem
> is chosen so the *obvious* method (named in the insight) is wrong or too slow, and a reformulation /
> invariant / unusual combination saves it.

### 2.1 Dynamic Programming (with a twist) — 7

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-dp-01 | Aliens-Trick Job Split | DP | You must pick **exactly k** disjoint intervals maximizing value; the per-k DP is O(nk). **Lagrangian/"Aliens" trick**: the answer-as-function-of-k is convex, so binary-search a penalty λ added per interval, drop the k-dimension to O(n), then recover k. | `n k` then `n` ints (interval value array) → max value with exactly k picks | n≤2e5, k≤n; hard | brute O(nk) DP on n≤200, cross-check |
| fcs-dp-02 | Divide-and-Conquer-Optimization Mailboxes | DP | Place p mailboxes on a line to minimize sum of distances; the cost matrix satisfies the **Monge/quadrangle inequality**, so the optimal split point is monotonic → divide-and-conquer optimization turns O(pn²) into O(pn log n). | `n p` then n sorted positions → min total distance | n≤8000, p≤n; medium-hard | O(pn²) DP on n≤500 |
| fcs-dp-03 | Knuth-Opt Optimal BST-ish Merge | DP | Interval DP for optimal merge cost is O(n³); the cost is Monge, so **Knuth's optimal-split monotonicity** `opt[i][j-1] ≤ opt[i][j] ≤ opt[i+1][j]` collapses it to O(n²). | `n` then n weights → min merge cost | n≤2000; medium-hard | O(n³) interval DP on n≤80 |
| fcs-dp-04 | Bitmask-over-Profile Broken Grid | DP | Tile/count over an h×w grid with h≤14: naive 2^(hw) is hopeless. **Broken-profile DP** carries the contour as one (h+1)-bit mask processed cell-by-cell, O(w·h·2^h). | `h w` grid of `.`/`#` → #tilings mod p | h≤14, w≤1000; hard | brute backtracking on ≤5×5 |
| fcs-dp-05 | Connected-Component DP on Trees ("rerooting count") | DP | Count labeled ways under a subtree constraint; the naive is per-root O(n²). **Rerooting**: compute the answer for one root, then push parent-contributions in a second pass for O(n). | `n` then n−1 edges → vector of n answers | n≤2e5; medium-hard | O(n²) recompute-per-root on n≤300 |
| fcs-dp-06 | SOS (Subset-Sum-over-Subsets) Pairing | DP | Count pairs (i,j) with `a[i] AND a[j] == 0` (disjoint bitmasks). Brute is O(n²). **Sum-over-subsets DP** over the 2^B mask lattice computes, for each mask, the count of array values that are submasks, in O(B·2^B). | `n` then n ints <2^20 → count of disjoint pairs | n≤1e6, values<2^20; medium-hard | O(n²) on n≤2000 |
| fcs-dp-07 | Digit DP with Carry State | DP | Count integers in [L,R] whose decimal digits sum to a multiple of their length-dependent modulus — a *non-local* digit property. **Digit DP** carrying (position, running mod, tight-flag) makes it O(digits·mod·2). | `L R` → count | L,R ≤ 1e18; medium | direct count on ranges ≤ 1e6 |

### 2.2 Graph — 7

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-gr-01 | Project Selection (Max-Weight Closure) | graph | "Pick projects to maximize profit, but each project requires prerequisite machines that cost money" reads like a knapsack; it is exactly **max-weight closure → min-cut**: source→profit nodes, cost nodes→sink, prerequisite edges ∞; answer = Σpositive − mincut. | `n m` profits, then prerequisite edges → max profit | n,m≤500 nodes, cap edges; hard | brute over 2^n subsets on n≤18 |
| fcs-gr-02 | Tournament Scheduling as 2-SAT | graph | Each constraint "team i plays home XOR team j plays home in round r" plus implications is a tangle of boolean constraints. **Model as 2-SAT**, build implication graph, SCC; satisfiable iff no var and its negation share an SCC. | clauses (i,a)∨(j,b) → `YES`+assignment / `NO` | vars≤1e5, clauses≤1e5; medium-hard | brute 2^v on v≤20 |
| fcs-gr-03 | Min-Cost Flow Disguised as Assignment-with-Quotas | graph | Workers→tasks with per-worker quotas and convex overtime cost looks like Hungarian; the convex cost breaks Hungarian. **Successive-shortest-path MCMF** with per-unit edges encodes the convex cost as parallel increasing-cost unit edges. | bipartite costs + quotas → min total cost | ≤200 nodes; hard | brute permutation/ILP on ≤8 |
| fcs-gr-04 | Eulerian-Circuit Reconstruction (de Bruijn) | graph | "Reassemble a string from all its overlapping k-mers" is not DP — it is finding an **Eulerian path in the de Bruijn graph** of (k−1)-mers; Hierholzer in O(E). | `k` then list of k-mers → one valid superstring or `IMPOSSIBLE` | E≤2e5; medium-hard | verify output contains all k-mers; brute on tiny |
| fcs-gr-05 | Bridges → Biconnected 2-Edge-Connectivity Augmentation | graph | "Add fewest edges so the network has no single point of failure" — find bridges (Tarjan low-link), contract 2-edge-connected components into a tree, answer = ⌈leaves/2⌉. The ⌈leaves/2⌉ formula is the non-obvious step. | `n m` edges → min #edges to add | n,m≤2e5; medium-hard | brute add-edges + bridge-check on n≤10 |
| fcs-gr-06 | Shortest Path on a State-Product Graph (0-1 BFS) | graph | Grid with "you may flip ≤1 wall" — don't run Dijkstra ad hoc; build the **layered graph** (cell × flips-used) and run **0-1 BFS** (deque) since edge weights ∈{0,1}, giving O(VE)→O(V). | grid + budget → min steps | grid ≤1000×1000; medium | Dijkstra on same state graph as cross-check |
| fcs-gr-07 | Functional-Graph Cycle Arithmetic | graph | Each node has exactly one out-edge `f(i)`; answer queries "where after t steps" with t up to 1e18. **Decompose into ρ-shape (tail + cycle)**, jump the tail, then mod into the cycle — *not* binary lifting (too much memory at this scale). | `n` then f[], q queries (start,t) → endpoints | n,q≤2e5, t≤1e18; medium-hard | direct simulation for small t |

### 2.3 Tree — 6

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-tr-01 | Auxiliary (Virtual) Tree Queries | tree | Many queries each give a subset S of ≤... nodes and ask a tree-DP over them; redoing DP on the full tree per query is O(qn). **Build the virtual tree** of S (sorted by Euler-in time + LCAs) so each query costs O(|S| log n). | `n` edges, q queries each a node-set → per-query answer | n,Σ|S|≤2e5; hard | full-tree DP per query on n≤500 |
| fcs-tr-02 | Centroid Decomposition Path Counting | tree | Count paths with length exactly L; per-pair is O(n²). **Centroid decomposition**: every path passes through some centroid; count cross-centroid pairs with a frequency array, recurse → O(n log n). | `n` weighted edges, `L` → #paths of length L | n≤2e5; hard | brute all-pairs BFS on n≤400 |
| fcs-tr-03 | Small-to-Large (DSU on tree) Color Queries | tree | For each subtree report #distinct colors; naive per-subtree is O(n²). **Small-to-large / DSU-on-tree**: keep heavy child's structure, re-add only light children → O(n log n). | `n` colors+edges → n subtree answers | n≤2e5; medium-hard | brute per-subtree set on n≤500 |
| fcs-tr-04 | Re-rooting Sum-of-Distances | tree | Sum of distances from *every* node; the O(n²) BFS-from-each is too slow. **Two-pass rerooting**: subtree sizes give the root's answer, then `ans[child]=ans[par]+(n−2·size[child])` propagates in O(n). The transfer formula is the insight. | `n` edges → n sums | n≤2e5; medium | O(n²) BFS on n≤500 |
| fcs-tr-05 | Heavy-Light + Segment Tree on Path XOR | tree | Path-update / path-query with XOR-assignment can't be plain LCA prefix because of updates. **HLD** decomposes each path into O(log n) chains, each a segment-tree range; combine. The "XOR-assign is range-affine" observation enables lazy. | `n` edges, q mixed ops → query answers | n,q≤2e5; hard | brute path walk on n≤300 |
| fcs-tr-06 | Tree Isomorphism by Canonical Hashing | tree | "Are two rooted forests structurally equal?" — string-equality on serialization is wrong (sibling order). **Canonical AHU hashing**: hash each node from the *sorted multiset* of children hashes; equal canonical roots ⇔ isomorphic. | two trees → `YES`/`NO` | n≤2e5; medium-hard | brute permutation-matching on n≤8 |

### 2.4 Strings — 5

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-st-01 | Distinct Substrings via Suffix Automaton | strings | Count distinct substrings; the O(n²) enumerate-and-hash dies at 1e6. **Suffix automaton**: #distinct substrings = Σ over states `(len[s]−len[link[s]])`, built online in O(n). | string → count (mod p if huge) | |s|≤1e6; hard | O(n² log) hashset on |s|≤2000 |
| fcs-st-02 | Z-Function "Border-Forced" Compression | strings | "Shortest string whose self-overlap matches a given overlap profile" — reverse-engineer from the **Z/prefix-function**; the constraint that borders compose transitively is the non-obvious feasibility test. | overlap array → reconstructed string / `-1` | n≤2e5; medium-hard | brute construct+check on n≤12 |
| fcs-st-03 | Aho-Corasick Multi-Pattern Weighting | strings | Score a text by total weight of all dictionary-pattern occurrences; running each pattern's KMP is O(text·patterns). **Aho-Corasick** automaton + fail-tree subtree-sum makes it O(text + Σpattern). | dictionary+weights, text → total score | Σlen≤1e6; hard | naive substring scan on small |
| fcs-st-04 | Palindromic Factorization (eertree) | strings | Min #palindromes to partition s; per-position O(n) palindrome check is O(n²). **Palindromic tree (eertree)** gives all distinct palindromic suffixes incrementally → O(n log n) DP via series-links. | string → min palindrome partition | |s|≤1e6; hard | O(n²) Manacher+DP on |s|≤2000 |
| fcs-st-05 | Lyndon / Booth Minimal Rotation | strings | Find the lexicographically least rotation; the O(n²) compare-all-rotations is too slow. **Booth's algorithm** (or Duval/Lyndon factorization) finds it in O(n) by a clever failure-function-like scan — the "skip a whole block on mismatch" step is the insight. | string → least-rotation index | |s|≤1e6; medium-hard | O(n²) on |s|≤2000 |

### 2.5 Number theory — 4

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-nt-01 | CRT Reconstruction with Inconsistency | number-theory | "Find x ≡ rᵢ (mod mᵢ)" with **non-coprime** moduli; textbook CRT assumes coprime. The insight is the **merge rule that checks `(r₁−r₂) divisible by gcd(m₁,m₂)`** and combines via lcm, returning −1 on contradiction. | k congruences → x mod lcm / `-1` | mᵢ≤1e9, k≤1e5; medium | brute search small lcm |
| fcs-nt-02 | Mod-Inverse-Free Combinatorics (Lucas / prime-power) | number-theory | nCr mod m where m is **not prime** (or prime power) — Fermat-inverse fails. **Lucas for primes, Andrew Granville / factor-and-Legendre for prime powers**, then CRT. Recognizing inverses don't exist is the trap. | `n r m` → nCr mod m | n,r≤1e18, m≤1e6; hard | direct nCr%m for small n |
| fcs-nt-03 | Divisor-Sum via Hyperbola Trick | number-theory | Σ_{i=1..n} (n div i) (or d(i)) for n up to 1e12; the O(n) loop is too slow. **Block decomposition / √n hyperbola method**: floor(n/i) takes only O(√n) distinct values, sum each block in O(1). | `n` → Σ divisor-count or Σ floor(n/i) | n≤1e12; medium-hard | O(n) loop for n≤1e7 |
| fcs-nt-04 | Pollard-Rho + Miller-Rabin Factor Queries | number-theory | Factor up to 1e18 fast; trial division to √n=1e9 is too slow. **Miller-Rabin** primality + **Pollard-Rho ρ** with Brent's cycle detection + `__int128` mulmod. The combination + correct deterministic MR witnesses is the insight. | q numbers ≤1e18 → factorizations | q≤500; hard | trial-division cross-check ≤1e12 |

### 2.6 Geometry — 4

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-ge-01 | Rotating Calipers Max-Width Pair | geometry | Farthest pair of points (diameter); all-pairs is O(n²). **Convex hull + rotating calipers**: the diameter is between hull antipodal points, found in one O(n) sweep after the hull. | n points → max pairwise distance² | n≤2e5; medium-hard | O(n²) on n≤2000 |
| fcs-ge-02 | Half-Plane Intersection Feasibility | geometry | "Is there a point satisfying all m linear inequalities?" — don't LP-solve generally. **Half-plane intersection** by angular sort + deque; empty intersection ⇔ infeasible. The deque-pop invariant is the insight. | m half-planes → `YES`+point / `NO` | m≤2e5; hard | random-sample / brute vertex-enum on m≤30 |
| fcs-ge-03 | Closest Pair by Sweepline + Set | geometry | Closest pair; divide-and-conquer is classic but fiddly. **Sweepline with an ordered set window of width = current best d**: only O(1) amortized candidates per point → O(n log n), simpler to get exactly right. | n points → min distance² | n≤2e5; medium-hard | O(n²) on n≤2000 |
| fcs-ge-04 | Pick's-Theorem Lattice Count | geometry | "Count lattice points strictly inside a polygon" — don't scan the bounding box (too big). **Pick's theorem** `A = I + B/2 − 1` with shoelace area A and boundary points B via gcd of edge deltas → I in O(vertices). | polygon vertices → #interior lattice pts | coords≤1e9, n≤1e5; medium-hard | brute box-scan on coords≤300 |

### 2.7 Data structures — 6

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-ds-01 | Offline Range-Distinct (Mo's algorithm) | data-structures | "#distinct in [l,r]" over many queries; per-query rescan is O(qn). **Mo's algorithm**: sort queries by (block of l, r), move pointers with O(1) add/remove of a count array → O((n+q)√n). The √n block ordering is the insight. | array + q ranges → distinct counts | n,q≤2e5; medium-hard | brute per-query set on n≤1000 |
| fcs-ds-02 | Persistent Segment Tree k-th in Range | data-structures | "k-th smallest in [l,r]" without updates; a merge-sort tree is O(log²). **Persistent segment tree** keyed by value, version per prefix; subtract versions r−(l−1) and descend → O(log n) per query. The prefix-version subtraction is the trick. | array + q (l,r,k) → k-th value | n,q≤2e5; hard | brute sort-subarray on n≤500 |
| fcs-ds-03 | Li Chao Tree / CHT on a Non-Obvious Cost | data-structures | A DP `dp[i]=min_j(dp[j]+b[j]·a[i])` looks O(n²); the transition is a set of **lines evaluated at a[i]** → **convex-hull trick / Li Chao tree** gives O(n log n). Recognizing the linear-in-a[i] form is the whole game. | n with a[],b[] → dp[n] | n≤2e5; hard | O(n²) DP on n≤2000 |
| fcs-ds-04 | Wavelet Tree Range-Rank | data-structures | "How many elements in [l,r] are ≤ x" for many queries; an offline BIT works but the problem forbids offline (online, forced by XOR-encoded queries). **Wavelet tree** answers rank/quantile online in O(log σ). | array + online queries → ranks | n,q≤2e5; hard | brute count on n≤1000 |
| fcs-ds-05 | DSU with Rollback + Offline Dynamic Connectivity | data-structures | "Are u,v connected at time t?" with edge add/remove — DSU has no delete. **Segment-tree-on-time + DSU with rollback (union by rank only)**: each edge lives on an interval, push to log nodes, DFS with undo. The rollback (no path compression) is the insight. | timeline of add/remove + queries → yes/no | n,q≤2e5; hard | brute recompute components on small |
| fcs-ds-06 | Fenwick of Sorted-Lists "BIT-of-BIT" 2D | data-structures | 2D dominance counting with updates; a static merge-sort tree fails under updates. **BIT indexed by x, each node a BIT/ordered structure over y** (or offline + BIT-on-compressed-y). The product-structure decomposition is the insight. | points + (l,r,query) → counts | n,q≤1e5; hard | brute 2D scan on n≤1000 |

### 2.8 Greedy / exchange-argument — 5

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-gx-01 | Scheduling by an Exchange-Argument Comparator | greedy-exchange | Order jobs (aᵢ,bᵢ) to minimize a coupled cost; the right order is **sort by the pairwise exchange comparator** (e.g. Johnson's rule / `aᵢbⱼ < aⱼbᵢ`), provably optimal by a swap argument — *not* "sort by aᵢ" or "by bᵢ". | n pairs → min cost | n≤2e5; medium-hard | brute all n! on n≤9 |
| fcs-gx-02 | Lexicographically-Smallest via Monotonic Stack | greedy-exchange | "Delete k chars to make the smallest string" — greedy "delete largest" is wrong. **Monotonic stack**: pop while top > current and budget remains; the stack invariant *is* the proof. | string, k → smallest result | |s|≤1e6; medium | brute choose-k-deletions on |s|≤16 |
| fcs-gx-03 | Binary-Search-the-Answer + Greedy Feasibility | greedy-exchange | "Max-min: place k items so the minimum gap is largest" — direct optimization is hard. **Binary search the answer x**, then a linear greedy checks "can we place k with all gaps ≥ x". Monotonic feasibility is the insight. | positions, k → max achievable min-gap | n≤2e5; medium | brute search threshold on n≤200 |
| fcs-gx-04 | Median-Minimizes-L1 Meeting Point | greedy-exchange | "Choose a point minimizing total Manhattan distance to all" — *not* the mean. The **median per axis** minimizes Σ|x−xᵢ|; the separability of L1 across axes is the non-obvious step. | n points → min total L1 | n≤2e5; easy-medium | brute candidate-grid on small |
| fcs-gx-05 | Hall's-Theorem Greedy Matching Feasibility | greedy-exchange | "Can each request be served given interval availabilities?" — a greedy + **Hall's condition** check (sort, assign earliest-deadline) rather than full bipartite matching; the EDF exchange argument proves optimality. | intervals/requests → `YES`/`NO`+assignment | n≤2e5; medium-hard | Hopcroft-Karp on small as oracle |

### 2.9 Ad-hoc / constructive — 4

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-ac-01 | Parity-Invariant Reachability | ad-hoc-constructive | "Can configuration A reach B under these moves?" — simulation is exponential. The moves preserve an **invariant (e.g. parity of inversions / a coloring sum mod k)**; reachable ⇔ invariants match. Finding the invariant is the entire problem (cf. 15-puzzle). | two configs → `YES`/`NO` | n≤1e6; medium-hard | BFS over states on tiny configs |
| fcs-ac-02 | Constructive: Build a Graph with Exact Property | ad-hoc-constructive | "Output any graph on n nodes with exactly k triangles" — not search. A **constructive recipe** (greedy clique-stacking: each added node into the largest clique adds a known count) hits any feasible k; recognizing the count formula is the insight. | `n k` → edge list / `-1` | n≤1000; medium-hard | verify output has exactly k triangles |
| fcs-ac-03 | XOR-Basis Linear Algebra over GF(2) | ad-hoc-constructive | "Max XOR-subset" / "is x representable" — greedy by value is wrong. Build a **linear basis (Gaussian elimination over GF(2))**; max XOR by greedily including basis vectors high-bit-first. The vector-space reformulation is the insight. | n numbers, queries → max-XOR / yes-no | n≤1e5, ≤60-bit; medium-hard | brute 2^n subset XOR on n≤20 |
| fcs-ac-04 | Game / Sprague-Grundy Nim-Value | ad-hoc-constructive | "Who wins this take-away game on independent piles?" — direct minimax is exponential. Compute **Grundy numbers per pile (mex of reachable)**, XOR them; nonzero ⇔ first player wins. Decomposing into independent games + Sprague-Grundy is the insight. | pile configs → `First`/`Second` | piles≤1e5; medium-hard | brute minimax memo on small |

### 2.10 Interactive — 2

| id | title | area | THE INSIGHT | I/O sketch | constraints / difficulty | oracle |
|---|---|---|---|---|---|---|
| fcs-in-01 | Guess Permutation by ≤ n log n Comparisons | interactive | Recover a hidden permutation using a query oracle (`? i j` returns sign); naive is O(n²) queries. **Divide-and-conquer / merge-sort query schedule** hits O(n log n) — the query-budget reformulation as a sorting lower bound is the insight. | interactive: `? i j` ↔ response; final `! perm` | n≤2e4, queries≤n log n; hard | local judge simulates oracle + counts queries |
| fcs-in-02 | Binary-Search a Hidden Threshold with Lies | interactive | Find a hidden value when up to 1 answer may be a lie. **Ternary/redundant binary search** (Rényi–Ulam): each step encodes a parity check so one lie is detectable/correctable; plain binary search fails. | interactive queries → `! x` | range≤1e9, ≤1 lie; hard | local judge with adversarial-lie simulator |

**FrontierCS subtotal: 50** (DP 7, graph 7, tree 6, strings 5, number-theory 4, geometry 4, data-structures 6, greedy-exchange 5, ad-hoc-constructive 4, interactive 2).

---

## 3. ALE-Bench (heuristic) candidates (~15) — single-file, stdin → stdout, score-maximization

> AtCoder-Heuristic style: no known optimum, continuous score, local seeds + best-of-N. Each row:
> `id | title | objective | I/O sketch | the heuristic-design INNOVATION | how to score locally`.
> These are where FrontierSmith's *open-ended / partial-scoring / idea-divergence* spirit lives.

| id | title | objective | I/O sketch | heuristic-design innovation | local scoring |
|---|---|---|---|---|---|
| ale-01 | Bounded-Degree Spanning Tour | minimize total edge weight of a degree-≤2 spanning structure (FrontierSmith's MST→TSP mutation) | `n` then coords → tour permutation | **Christofides-style init + 2-opt/Or-opt with an O(1) incremental delta** on the changed edges only; don't recompute tour length | local scorer reads perm, sums edges, validates degree; report ratio vs greedy baseline |
| ale-02 | Grid Polyomino Packing | maximize covered-cell fraction packing given pieces | `H W` + piece shapes → placements | **simulated annealing over placement order with a fast bitmask collision test** (rows as bitsets); incremental coverage update on add/remove | scorer recomputes covered cells; ratio to a first-fit baseline |
| ale-03 | Ad-Hoc Routing under Time Windows | maximize #served requests within windows | requests with (loc, window) → route | **insertion heuristic + ruin-and-recreate (LNS)**; a *clever neighborhood* = remove the k most "expensive" stops and reinsert greedily | replay route, count served + check windows |
| ale-04 | Heat-Diffusion Tile Coloring | minimize a roughness/energy functional over a colored grid | grid params → coloring | **problem-specific relaxation**: optimize the continuous relaxation, then threshold/round; local search on boundary cells with O(1) energy delta | recompute functional from output grid |
| ale-05 | Server Placement (k-median-ish) | minimize sum of client→nearest-server distances | clients, k → server set | **PAM/Lloyd swap with incremental "second-nearest" caching** so a single swap costs O(clients) not O(clients·k) | sum nearest distances from output |
| ale-06 | Production Scheduling (AHC flowshop) | minimize makespan | jobs×machines times → schedule | **NEH construction + critical-path-guided neighborhood** (only swap operations on the critical path change makespan) → O(1)-ish moves | simulate schedule, output makespan |
| ale-07 | Maze Treasure Collection (time-budgeted) | maximize collected value within T steps | grid + values → move string | **beam search over (position, collected-mask-summary) with a distance-to-value admissible heuristic** for pruning | replay moves, sum value, check ≤T |
| ale-08 | Cable Layout (rectilinear Steiner-ish) | minimize total wire length connecting terminals | terminals → routing | **Hanan-grid restriction + minimum-spanning-tree on Steiner candidates, then local rerouting**; the Hanan-grid relaxation cuts the search space | sum routed length, verify connectivity |
| ale-09 | Dynamic Bin Packing with Rebalancing | minimize #bins / overflow over a stream | item stream → assignments | **best-fit-decreasing online + periodic "repack the fullest few bins" move** with incremental fill tracking | replay assignments, count bins/overflow |
| ale-10 | Tiling-Color SA (AHC "wall painting") | maximize matched target pattern | target grid + ops budget → op sequence | **SA where the move = a single brush stroke and the score delta is computed only over the stroke's footprint** (O(stroke), not O(grid)) | apply ops to blank grid, count matches |
| ale-11 | Vehicle Dispatch (assignment over time) | maximize fulfilled rides | timed requests + vehicles → dispatch | **rolling-horizon Hungarian on a small window + greedy carry-over**; the horizon relaxation makes a global NP problem tractable | simulate dispatch, count fulfilled |
| ale-12 | Lattice Antenna Coverage | maximize covered demand under power budget | demand grid, budget → antenna set | **greedy submodular-maximization (lazy-greedy / CELF) with marginal-gain caching**; lazy evaluation skips recomputing stale gains | recompute coverage from output |
| ale-13 | String Reassembly (shortest superstring approx) | minimize superstring length covering all fragments | fragments → superstring | **greedy max-overlap merge (Aho-Corisick overlaps) then 3-opt-style fragment reordering**; overlap graph relaxation of an NP problem | verify all fragments are substrings; report length |
| ale-14 | Graph Coloring with Soft Conflicts | minimize weighted conflict edges with ≤k colors | graph, k → coloring | **tabu search (TabuCol) with an incremental conflict-count table** updated in O(degree) per recolor; tabu tenure tuned to escape plateaus | recount conflicts from output coloring |
| ale-15 | Continuous Facility Layout (annealing on geometry) | minimize overlap + dispersion energy | rectangles → positions | **SA with a force-directed move proposal + spatial hash for O(1) neighbor overlap queries**; spatial hashing makes each move's delta cheap | recompute energy from positions |

**ALE subtotal: 15.** Recurring innovation themes (deliberately diverse): incremental/O(1) score deltas (ale-01,06,10,15), large-neighborhood / ruin-and-recreate (ale-03,09), continuous relaxations of NP objectives (ale-04,08,11,13), submodular/lazy-greedy (ale-12), beam/tabu/SA structure-specific neighborhoods (ale-02,05,07,14).

---

## 4. How each candidate is VERIFIED at generation time

The verifier is built **before** the trace is finalized, so the reference solution is provably correct
and the trace's "I traced it / I cross-checked it" claims are *earned* (the disposition `data_v4`
already gets right — see DATA_FIX). Each FCS dir mirrors `cp-noadj-commit/verify/`.

**FrontierCS (exact judge):**
1. **Compile gate.** `sol.cpp` must compile clean with `g++ -O2 -std=c++17` and read stdin / write
   stdout (no class-library landing — this is the P0 fix from DATA_FIX_FCS_LANDING).
2. **Brute oracle on small random cases.** `verify/brute.py` is an *independent*, obviously-correct but
   slow implementation (exponential / O(n²) / direct enumeration). `verify/gen.py` emits random small
   instances (and the named edge cases: empty, n=1, all-negative, overflow-scale, ties). A driver runs
   `sol` vs `brute` on ≥1000 random small cases; **any mismatch fails the candidate**. (Cross-check
   variant where no brute is feasible: two *independent* methods must agree, or a known closed-form.)
3. **Foil where natural.** For greedy-trap / obvious-wrong problems, ship `verify/greedy.cpp` (the
   plausible-but-wrong approach, like `cp-noadj-commit/verify/greedy.cpp`) and *demonstrate* it failing
   on a generated counterexample — this becomes the trace's "why the obvious idea is wrong" moment.
4. **Scale/TLE gate.** Run `sol` on a max-constraint instance; must finish in the stated limit (guards
   the (a) "wrote it brittle/recursive → TLE" failure mode from the autopsy).
5. **Idea-divergence sanity (FrontierSmith-borrowed).** Reject candidates whose only intended solution
   *is* the textbook algorithm with no twist: the INSIGHT field must name a real reformulation/invariant,
   and a quick LLM-judge pass confirms the "obvious" approach and the "insight" approach are genuinely
   different (mirrors FrontierSmith's semantic divergence filter), keeping the list off-textbook.

**ALE-Bench (heuristic, partial score):**
1. **Local scorer.** Each candidate ships a deterministic scorer that reads the output, **validates
   feasibility** (invalid/crash/timeout → 0, matching ALE's real failure floor) and computes the
   continuous objective.
2. **Baseline normalization.** Report score relative to a trivial baseline and a strong reference
   (FrontierSmith's `max((x−y)/max(x,y),0)` style), so improvement is continuous and not gameable.
3. **Best-of-N.** Generate N candidate programs / N seeds; keep the best feasible score per seed,
   average across a fixed seed set — exactly the ALE `best@N` + seed-averaging harness, and the natural
   guard against the ~310 "broken submission" floor (`§5.2 EXPERIMENTS_zh.md`): we only count
   `overall_absolute_score > 0` submissions as real.

Every candidate is self-contained enough that a downstream subagent can: (i) expand the row into a full
`context.md` (research question + I/O contract + C++ code framework, like the gold trace), (ii) write the
first-person `reasoning.md` that *discovers* the INSIGHT (hits the obvious-wrong approach, traces a
counterexample, lands the reformulation), (iii) emit `train_answer.md` with the single-file C++17
solution, and (iv) drop the `verify/{sol.cpp, brute.py, gen.py(, greedy.cpp)}` oracle that proves it.

---

### Sources
- FrontierSmith: *Synthesizing Open-Ended Coding Problems at Scale* — arXiv:2605.14445 ; frontier-cs.org/blog/frontiersmith
- FrontierCS: *Evolving Challenges for Evolving Intelligence* — arXiv:2512.15699
- ALE-Bench: *A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering* — arXiv:2506.09050
- Internal: `experiments/EXPERIMENTS_zh.md` (§5.1 FCS, §5.2 ALE), `experiments/DATA_FIX_FCS_LANDING_zh.md`, gold trace `data_v4/cp-noadj-commit/`.
