# LK synthesis (from retrieved sources)

## Sources retrieved & read
- PRIMARY: Lin & Kernighan 1973, "An Effective Heuristic Algorithm for the Traveling-Salesman Problem", Operations Research 21(2):498-516. Obtained from Kernighan's own Princeton Bell-Labs mirror `cs.princeton.edu/~bwk/btl.mirror/tsp.pdf` (10 pages, the algorithm core). Read in full: abstract, iterative-improvement framework, the general "element-by-element" out-of-place heuristic, the additive-gain idea, the 5 requirements, §1 basic TSP algorithm (Steps 1-7), gain criterion + cyclic-permutation lemma & proof, feasibility/close-up, the worked example, backtracking discussion, 3-opt subsumption.
- BACKGROUND: Croes 1958 (2-opt, k=2), Lin 1965 (3-opt, k=3), the λ-opt notion / λ-optimality. Described both in primary (cites Croes[2], Lin[11]) and in Helsgaun report §2-3. DESCENDANT prior-art: Helsgaun LKH report (LKH_REPORT.pdf) and Helsgaun 2009 "General k-opt submoves" (5-opt sequential submoves, α-nearness from 1-trees / Held-Karp lower bound). Read.
- THIRD-PARTY EXPLAINER: Helsgaun LKH report §3 (clean restatement of original LK: sequential exchange criterion, feasibility criterion, positive gain criterion, disjunctivity criterion, the 13-step basic algorithm, the alternate-x2 figures). Arthur Mahéo blog "Implementing Lin-Kernighan in Python" + its GitLab repo `gitlab.com/Soha/local-tsp` (tsp_local/kopt.py) — the canonical Python implementation used to ground the code.

## The problem
Symmetric TSP: n×n symmetric distance matrix, find min-length tour visiting each city once. Exact is (believed) exponential; want good-to-optimal tours fast (~n^2-ish runtime), up to ~100+ cities. Tour T = n-subset of all n(n-1)/2 links forming a Hamiltonian cycle.

## Background / lineage (the wall)
- Iterative-improvement frame: random start tour, transform to improve, repeat to local opt, multistart.
- k-opt interchange: delete k links from T, add k links from S-T, keep if shorter and feasible. Croes: k=2 (2-opt = reverse a subsegment). Lin: k=3 (3-opt). λ-optimality: tour is λ-opt if no λ-exchange improves; larger λ → more likely optimal, n-opt = optimal. But: testing all λ-exchanges is O(n^λ); no nontrivial bound on number of moves; **k must be fixed in advance**, and effort rises steeply with k, and you cannot know the right k a priori. That fixed-k drawback is the pain point.

## The key intellectual move
Don't fix k. View "T not optimal" as "there are k out-of-place edges x1..xk to swap for y1..yk", with k unknown. Build X and Y **element by element**, greedily choosing the "most out-of-place" pair each step, and let the depth k be decided dynamically by a stopping rule. Variable-depth.

## The chain (sequential exchange) — exact mechanics
- City chain t1, t2, t3, ... . x_i = (t_{2i-1}, t_{2i}) is a tour edge being broken; y_i = (t_{2i}, t_{2i+1}) is a new edge being added. x_i and y_i share endpoint t_{2i}; y_i and x_{i+1} share endpoint t_{2i+1}. (x_{k+1}=x_1, i.e. y_k closes back to t_1.)
- Start: pick t1, x1=(t1,t2) one of two tour edges at t1. Choose y1=(t2,t3), t3 not a tour-neighbor of t2, with g1>0.
- Step i: x_i is FORCED: from t_{2i-1} (=the far end of y_{i-1}), x_i must be the tour edge such that joining t_{2i} back to t1 yields a Hamiltonian path that can close to a tour. For i>=2 (actually the "close-up" feasibility), given y_{i-1}, x_i is uniquely determined to keep the close-up valid. Then y_i = (t_{2i}, t_{2i+1}) is a chosen new edge (nearest-neighbor preference, small |y_i|).

## Gain accounting (exact, load-bearing)
- g_i = |x_i| - |y_i| = c(x_i) - c(y_i)  (length broken minus length added). Gains additive: total = g_1+...+g_k.
- If final tour T' shorter: f(T)-f(T') = sum g_i > 0.
- POSITIVE GAIN CRITERION: only extend while running sum G_i = g_1+...+g_i > 0. Justified by the cyclic-permutation lemma: if a sequence of numbers has positive total sum, SOME cyclic permutation has all partial sums positive. PROOF (from primary): let k be the largest index for which g_1+...+g_{k-1} is minimum. For k<=j<=n: g_k+...+g_j = (g_1+...+g_j) - (g_1+...+g_{k-1}) > 0 (since the prefix to k-1 is the minimum). For 1<=j<k: g_k+...+g_n+g_1+...+g_j >= g_k+...+g_n + g_1+...+g_{k-1} >= 0 wait — careful: it's g_k+...+g_n + g_1+...+g_j, and since g_1+...+g_{k-1} is the minimum prefix, g_1+...+g_j >= g_1+...+g_{k-1}, and g_k+...+g_n = total - (g_1+...+g_{k-1}) > 0... The primary states: if 1<=j<k, g_k+...+g_n+g_1+...+g_j >= g_k+...+g_n+g_1+...+g_{k-1} = total sum > 0. So starting the chain at the right city makes all partial sums positive ⇒ insisting G_i>0 throughout loses no improving move, only re-anchors where it starts. Massive pruning.

## Close-up (feasibility) — the gem
- Before adding y_i, check the close-up: tentatively join t_{2i} to t1 with edge y_i* = (t_{2i}, t1), gain g_i* = |x_i| - |y_i*|. Because the feasibility criterion holds, joining t_{2i} to t1 gives a valid tour. If G_{i-1} + g_i* > G*, record G* = G_{i-1}+g_i*, k=i. G* = best improvement so far, monotone nondecreasing, starts 0.
- So at each depth you evaluate "close now" against the best seen; you keep extending only if the open chain's running gain G_i stays positive and beats G*.
- y_i must be chosen so that x_{i+1} can be broken (ensures next-step feasibility), and disjointness (x's and y's all distinct: a y can't later be broken, an x can't later be added).

## Stopping rule
Terminate building when no x_i, y_i satisfy disjointness/feasibility/gain, OR when G_i <= G*. Then if G*>0, perform the k-exchange (the one achieving G*), set T' = T - G*, restart from step 2. If G*=0, backtrack.

## Backtracking (only levels 1 and 2)
If G*=0 (no improvement from this t1/x1/y1):
- (a) try alternate y2 (in increasing length, while g1+g2>0),
- (b) try the alternate x2 — this one is special: breaking the alternate x2 temporarily violates feasibility (can't close at i=2), allowed ONLY at i=2; gives the non-sequential-ish flexibility. The cases of where t5 lands (between t2-t3 vs t1-t4) restrict t6, t7.
- (c) backtrack to alternate y1 (increasing length),
- (d) alternate x1,
- (e) new t1.
Backtracking only when no gain found, only at i=1,2. Measurements: mean choice number 1.2 (level 1), 1.8 (level 2) ⇒ cap to ~5 candidates each. Deeper backtracking too costly.

## Properties
- LK local optimum is necessarily 3-opt (subsumes Lin 1965), in much less time.
- Average growth ~n^2.2. Improvements per local opt between n/4 and n/3; early moves have large k (depth), settle to small (2-7) with slight overshoot.
- Reduction idea (multistart): edges common to several local optima are likely in the global opt; can fix them. (Secondary, in later part of paper not in my 10-page extract — keep light / omit to avoid fabrication.)

## Descendant (later prior art, for context only — NOT in-frame as the method)
- LKH (Helsgaun): replaces 2/3-opt basic step with sequential 5-opt submoves; uses α-nearness candidate lists (α(i,j) = increase in a minimum 1-tree if edge (i,j) is required), derived from the Held-Karp 1-tree lower bound and subgradient π-penalties, instead of plain nearest-neighbor. Bigger but fewer, better-directed moves. This is a descendant — mention only as "a natural way to push the idea further" framing, never as the target.

## Design decisions → why
- Variable k vs fixed k: fixed k forces a bad time/quality tradeoff you can't tune a priori; variable-depth lets the data pick k. WHY it's tractable: gain criterion prunes.
- Positive-gain partial-sum criterion vs only requiring positive total: cyclic-permutation lemma says no improving sequential move is lost (just re-anchor t1), but pruning is enormous. WHY G_i>0 not "allow temporary negative": you'd explore exponentially more; the lemma guarantees you don't need to.
- Forced x_i / feasibility criterion: keeping the configuration one-step-from-a-tour means you can always "close up", so you can stop at any depth and read off a valid tour — no wasted work assembling infeasible swaps. Applied for i>=2 (close-up), the alternate infeasible x2 allowed only at i=2 for extra power.
- Nearest-neighbor preference for y_i: to make g_i large you want |y_i| small ⇒ try nearest neighbors first; bounds the branching.
- Disjointness of X,Y: avoids undoing within an iteration, simplifies gain bookkeeping, gives a clean stop rule, avoids implementation bugs.
- Backtrack only levels 1,2 + cap ~5: empirically the right candidate is almost always the 1st/2nd (mean 1.2 / 1.8); deeper backtracking is a big time penalty for little gain.
- Multistart from random tours: random starts as fast as constructive ones and give a population of local optima; better heuristic ⇒ fewer, better local optima.
