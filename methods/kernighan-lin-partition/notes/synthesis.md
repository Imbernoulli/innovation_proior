# KL graph partitioning — synthesis

## Three sources (all read this run)
1. PRIMARY: Kernighan & Lin 1970, BSTJ 49(2):291-307, "An Efficient Heuristic Procedure for Partitioning Graphs". Read full scan pp.291-307 visually (refs/kl1970.pdf, image-only scan, OCR-hostile → read with Read pages=).
2. BACKGROUND/ancestors (cited in KL refs): Ford & Fulkerson *Flows in Networks* (max-flow/min-cut, ref 1); Lin 1965 BSTJ "Computer Solutions of the TSP" (λ-opt / λ-change, ref 2, same author S. Lin — the variable-rearrangement ancestor); Kruskal 1964 MDS (ref 4). Plus NP-hardness framing of balanced min-cut (the exact-solution count, integer-program remark). EDA application = min-cut placement (Breuer 1977) — the balanced bisection underneath physical design.
3. THIRD-PARTY: FM 1982 (refs/fm1982.pdf, clean text) successor — single-cell moves, cell gain, gain BUCKET[-pmax..pmax], locking, hypergraph nets, linear per pass. Berkeley CS267 lecture18 + Columbia handout walkthrough (gain=D(a)+D(b)-2w(a,b), escape local minima, pick prefix j max cumulative gain, O(N^3) per pass).

## EXACT MATH (from primary, Lemma 1 + §2.2, verified)
- External cost E_a = Σ_{y∈B} c_ay ; Internal cost I_a = Σ_{x∈A} c_ax ; D_s = E_s − I_s.
- Lemma 1: interchange a∈A, b∈B → gain (reduction in cost) = D_a + D_b − 2 c_ab.
  Proof: let z = cost of A-B connections not involving a or b. Old T = z + E_a + E_b − c_ab. New T' = z + I_a + I_b + c_ab. gain = T − T' = (E_a−I_a)+(E_b−I_b) − 2c_ab = D_a + D_b − 2c_ab. (the −c_ab/+c_ab: the edge a-b is counted in E_a AND E_b before swap, so subtract once; after swap it is counted in I_a AND I_b, so add once.)
- After swapping a_i, b_i out of contention, D-UPDATE for remaining x∈A−{a_i}, y∈B−{b_i}:
    D'_x = D_x + 2 c_{x a_i} − 2 c_{x b_i}
    D'_y = D_y + 2 c_{y b_i} − 2 c_{y a_i}
  Reason: edge (x,a_i) was internal for x (a_i in A), now a_i is in B so it's external → D_x rises by 2c_{x a_i}. Edge (x,b_i) was external for x (b_i in B), now b_i in A so internal → D_x falls by 2c_{x b_i}. Symmetric for y.
- Sequential pairing: pick (a_1,b_1) max g_1 = D_{a1}+D_{b1}−2c_{a1 b1}; set aside; recompute D over remaining; pick (a_2,b_2) max g_2; … until all n exhausted. Σ_1^n g_i = 0 (swapping all = original up to relabeling).
- Choose k maximizing partial sum G = Σ_{i=1}^k g_i. If G>0, interchange X={a_1..a_k}, Y={b_1..b_k}; new cost = old − G. Treat as new partition, repeat pass. If G=0 → phase-1-optimal (local min).
- Note g_i may be negative for the chosen-max pair even — process does NOT stop at first negative; the cumulative-prefix is what matters (escape local min).

## Complexity (primary §2.4)
- Initial D compute = n² (each of 2n elements vs all). Updates per pass: (n-1)+(n-2)+...+1 ∝ n².
- Selection of next pair: sort D's descending in each set → scan few contenders (if D_ai+D_bj ≤ best-so-far, stop, since c≥0). Sorting per pass ≈ n²log n. Or fast-scan (largest D_a, largest D_b, plus save top-3 to handle large c_ab) → essentially linear per selection, ~n² lower bound per pass, ~30% faster, tiny power loss.
- # passes small: 2-4 on instances up to 360 points. Observed total time ~ n^2.4 (FORTRAN G, IBM 360/65).
- p(n)=prob a phase-1 optimal is global ≈ 2^{-n/30}; p≈0.5 at 30×30, 0.2-0.3 at 60×60, 0.05-0.1 at 120×120.

## Exact / hardness framing (§1.2)
- Exhaustive count for kp=n into k subsets of size p: (1/k!)·C(n,p)C(n-p,p)···C(2p,p)C(p,p); for n=40,p=10,k=4 > 10^20. Could be ILP. "Any direct approach likely inordinate computation" → heuristic.

## Unequal sizes (§2.6) & multiway (§3)
- Unequal sets n1<n2: restrict #pairs exchanged per pass to n1; or add 2n2−n "dummy" elements (zero cost rows) to pad to 2n2, run, discard dummies. (§2.7 elements of unequal size = blow node of size k into k size-1 nodes bound by high-cost edges.)
- k-way: start with k sets of size n, apply 2-way procedure to all C(k,2) pairs repeatedly → pairwise-optimal (necessary but not sufficient for global). Starting partitions: recursive 2-way splits, or sequential break-off (set aside n, repartition rest).

## False starts the paper itself rejects (§1.3) — prior art
- Random solutions: n² but optimal partitions appear with prob <10^-7 on 32×32.
- Max-flow/min-cut (Ford-Fulkerson): finds MIN-cut but UNCONSTRAINED sizes — no way to constrain block sizes; if subsets very different, no benefit. (Does give a lower bound on unconstrained 2-way cut.) THE balanced-size constraint is exactly what breaks flow methods.
- Clustering: find natural clusters in cost matrix; no size control, no systematic handling of "stragglers".
- λ-opting (Lin TSP analogue): 1-change = swap a single pair; 1-opt. Experiments on 32×32: 1-opt reaches apparent optimum ~10% of trials, within 1-2 of optimum ~75%. Extending λ>1 "fruitless" (1-opt already n²). → motivates sequential variable-depth instead of fixed λ.

## EDA / Frontier application
- KL's own intro example: assigning circuit components to PC boards/cards to minimize inter-board connections, max cap per board. This IS min-cut placement / physical design. Breuer 1977 min-cut placement = recursively bisect chip area + netlist with min-cut (KL/FM) → top-down placement. FM 1982 made it linear-time on hypergraph netlists (nets, not edges) → the workhorse under classic min-cut placement and modern multilevel partitioners (hMETIS).

## House-style positioning vs lin-kernighan TSP anchor
- SAME variable-depth idea, same authors (Lin). TSP: edge-exchange (2-opt/3-opt → variable k via sequential edge swaps, broken x_i / added y_i, close-up feasibility, positive-gain prefix). HERE: vertex EXCHANGE across a balanced bipartition; the move is a pair-swap not an edge-replace; the bookkeeping object is the D-value (E−I) not edge lengths; gain = D_a+D_b−2c_ab; lock the swapped pair; accumulate g_1..g_n; pick prefix k max G_k; multi-pass. Position THIS as the graph-bisection ANCESTOR of the TSP variable-depth idea (KL 1970 predates LK 1973). Do NOT duplicate TSP trace; novelty = D-value bookkeeping + exchange move on balanced bipartition + the "swap whole pre-selected sets at once" framing.

## Design-decision → why
- Why D = E − I (not just E)? Because the gain of a swap factors cleanly into D_a+D_b−2c_ab; D is the marginal cost of moving the vertex to the other side. The −2c_ab corrects double-removal of the a-b edge.
- Why exchange a PAIR, not move one vertex? To preserve the balanced n/n split exactly (a single move unbalances). (FM later relaxes this with single-cell moves + an explicit balance constraint — the successor's contribution.)
- Why pre-select the whole sequence (a_1,b_1)...(a_n,b_n) then pick a prefix, instead of stopping at first non-improving swap? Because a locally bad swap can unlock a much bigger later gain → the cumulative-prefix G_k escapes local minima. This is the whole point.
- Why lock swapped vertices? So Σg_i is a clean telescoping of disjoint swaps (each vertex moves at most once per pass) → the prefix-sum reasoning is valid and the pass terminates.
- Why sort D's / fast-scan? Selection is the dominant cost; sorting lets you stop scanning early because c≥0 caps the achievable gain.
- Why multi-pass? One pass reaches a 2-way-exchange local optimum given THIS labeling; re-running from the improved partition can find more. Stop when G≤0.
