# Synthesis notes — RANKING for online bipartite matching

## Sources read this run (all in refs/)
- kvv1990_online.pdf/.txt — PRIMARY. Karp, Vazirani, Vazirani, STOC 1990. Full text (7 pp).
- djk2013_ranking.pdf/.txt — Devanur, Jain, Kleinberg, SODA 2013. Clean primal-dual analysis (9 pp).
- birnbaum_mathieu_madesimple.pdf/.txt — Birnbaum, Mathieu, SIGACT News 2008. The combinatorial "made simple" proof; documents the Krohn–Varadarajan bug in the original KVV proof and the perturbation fix.
- wisc_onlinematching.pdf/.txt — Wisconsin CS787 lecture notes: clean statement of deterministic 1/2 lower bound, randomized-greedy still ~1/2, and primal-dual modified RANKING.

## Problem
Bipartite G=(L,R,E) with a perfect matching of size n. L (offline) known; R arrives online, edges revealed on arrival; each arrival matched/dropped irrevocably. Maximize |matching|. Measure = competitive ratio = min over G and arrival order of E[ALG]/OPT.

## Antecedents / walls
- GREEDY (match to any unmatched neighbor): always maximal => >= OPT/2. Adversary forces exactly 1/2: first n/2 columns all-ones; algorithm matches some set S of rows; last n/2 columns each have a single 1 in a row of S. (KVV intro; Wisc notes; B-M intro.)
- Any DETERMINISTIC algorithm <= 1/2 (same adversary, adapt to the algorithm's choices). So to beat 1/2 you MUST randomize.
- RANDOM (each arrival picks a uniformly random unmatched neighbor): still ~ n/2 + O(log n), i.e. essentially 1/2. KVV give the bad matrix B_ij=1 for i=j or (n/2<j<=n and 1<=i<=n/2). Reason: per-step randomness "concentrates on the dense upper half," wasting rows the sparse lower half needs. Lesson: per-step coin flips are not enough; need correlated/global randomness.
- Adaptive online adversary caps ANY randomized algorithm at n/2+O(log n) — but that is the wrong adversary model; against the oblivious (fix-graph-first) adversary, 1-1/e is achievable.

## The leap: RANKING
Fix ONE uniformly random total order (priority/ranking) on L at the start. Each arriving j -> unmatched neighbor of smallest rank. Equivalent threshold form: each i draws Y_i ~ U[0,1]; match to smallest-Y unmatched neighbor. Self-correcting: tends to favor offline vertices that have been "eligible least often," so it doesn't burn the rows a later sparse column will need.

## KVV original analysis (the hard one) — lived in reasoning
- Duality/symmetry: with both sides ordered, RANKING is symmetric in L<->R (Lemma 1).
- Refusal monotonicity (Lemma 2): a refusal variant matches a SUBSET; so RANKING dominates EARLY.
- Reduce to upper-triangular (Lemma 3) via refusal argument; diagonal = perfect matching.
- D-set (Lemma 4): if for each i row-or-column matched, |M| = (n + |D|)/2 where D = both matched. So E|M| = n/2 + (1/2)E|D|.
- EARLY: row i matched iff column i not yet matched on arrival. W(sigma,i) machinery; perturbation lemma (Lemma 6/7): Pr[both i matched] = (1/n) sum_{t<W} ... -> Pr[row i and col i both matched] depends on rank-t match prob.
- Recurrence: with x_t = Pr[rank-t vertex matched], 1 - x_t <= (1/n) sum_{s<=t} x_s. Worst case tight: S_t(1+1/n) >= 1+S_{t-1}, giving x_t -> (1-1/(n+1))^t and ratio 1-(1-1/(n+1))^n -> 1-1/e. (KVV phrase it as w_t=(1-1/n)^{t-1}, sum t w_t, Theorem 1 = n(1-1/e)+o(n).)
- KNOWN BUG (B-M, p.5): the "intuitive proof" uses independence of u and R_{t-1}, which is FALSE. Krohn–Varadarajan found this 17 yrs later; Goel–Mehta and Birnbaum–Mathieu fix it by perturbing sigma (Lemma 4): move v to rank t in a freshly random permutation so u becomes independent of R_t. This self-correction must appear in the trace.
- Optimality (Theorem 2): RANDOM on complete upper-triangular = n(1-1/e); Yao's lemma => no randomized algorithm beats 1-1/e.

## DJK 2013 clean analysis (the slick one) — lived in reasoning
- LP: max sum x_ij s.t. row/col sums <=1. Dual: min sum alpha_i + sum beta_j s.t. alpha_i+beta_j>=1, all >=0.
- Threshold form Y_i~U[0,1]. Monotone g:[0,1]->[0,1], g(1)=1. On matching i to j: alpha_i = g(Y_i)/F, beta_j = (1-g(Y_i))/F. Unmatched -> 0.
- Per match, primal +1, dual +1/F => dual value = (1/F)|matching| exactly. If dual feasible in expectation, weak duality gives E[ALG] >= F * OPT.
- y^c = critical threshold: run on G\{i}; the neighbor matched to j there has Y-value y^c (=1 if j unmatched). 
- Dominance Lemma (Lemma 1): i gets matched if Y_i < y^c.
- Monotonicity Lemma (Lemma 2): beta_j >= beta_j^c = (1-g(y^c))/F for ALL Y_i (removing i only grows the unmatched set; superset induction).
- Feasibility in expectation: E[alpha_i] >= integral_0^{y^c} g(y)dy / F (Dominance); beta_j >= (1-g(y^c))/F (Monotonicity). So E[alpha_i+beta_j] >= (1/F)[ integral_0^{theta} g + 1 - g(theta) ] with theta=y^c.
- Need integral_0^theta g(y)dy + 1 - g(theta) >= F for all theta in [0,1]  ... (1).
- Solve tight: g(y)=e^{y-1}, F=1-1/e. Check: integral_0^theta e^{y-1}dy = e^{theta-1}-e^{-1}; +1-e^{theta-1} = 1-e^{-1}=F. Equality => largest feasible F is 1-1/e.
- Vertex-weighted (Agarwal et al.): offer v_i(1-g(Y_i)); same proof. AdWords/BJN07: water-level y_i = fraction of budget used; alpha_i ~ integral_0^{y_i} g; same eq (1). Unifies the two streams.

## Design decisions -> why
- Random *permutation* not per-step coin: per-step randomness (RANDOM) is ~1/2 because it has no memory of which rows it has been spending; one global random order correlates decisions so a high-priority row is consistently protected/spent coherently. 
- Match to SMALLEST rank (highest priority): the priority is what a later sparse column "expects" to still have available; greedily honoring it is what makes the charging argument close.
- Threshold/[0,1] form: turns a discrete permutation into a continuous variable so the dual can be an integral integral_0^{y^c} g(y)dy and the feasibility condition becomes a clean ODE/integral inequality.
- Dual split alpha=g/F, beta=(1-g)/F: keeps alpha_i+beta_j = 1/F on every match so the primal:dual ratio is exactly F (gives the clean "value = ALG/F"); the SPLIT by g(Y_i) is the only freedom, tuned by eq (1).
- g monotone increasing, g(1)=1: monotonicity makes the Monotonicity Lemma's inequality go the right way; g(1)=1 forces beta=0 when matched at the worst threshold (and in AdWords, stops matching to an exhausted budget).
- Feasible only in EXPECTATION: the novelty. Deterministic online primal-dual keeps duals feasible every step; here a single unmatched neighbor of j can leave alpha=0, beta small => infeasible for that outcome. Averaging over Y_i restores feasibility. Necessary because you cannot maintain integral matching duals feasibly online deterministically (that would beat 1/2 deterministically, impossible).

## Code (code/ranking.py, verified)
permutation form + threshold form (same algorithm); greedy baseline; KVV upper-triangular worst case with columns arriving n..1; Monte-Carlo ratio -> 0.6325 at n=1000 (1-1/e=0.6321). Confirms construction + algorithm.
