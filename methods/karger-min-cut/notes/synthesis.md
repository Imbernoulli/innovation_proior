# Synthesis — Karger / Karger-Stein randomized min cut

## Sources retrieved this run
- Wikipedia "Karger's algorithm" (contract pseudocode, fastmincut pseudocode, m>=nk/2, telescoping, T=(n choose 2) ln n, O(n^2 log^3 n)).
- Princeton COS521 F'13 Lec 2 (Arora; appended Dasgupta CSE103 + Vigoda CS7530 notes) — PDF in src/princeton_lec2.pdf. Full facts 1-4, telescoping product = 2/(n(n-1)), Omega(n^2) repetitions, recursion: contract to n/sqrt2+1, repeat twice, T(n)=2T(n/sqrt2)+O(n^2)=O(n^2 log n), P(n)>=1-(1-P(n/sqrt2+1)/2)^2 = Omega(1/log n), total O(n^2 log^3 n). Also Kruskal/random-weight-MST equivalence; Corollary: <= O(n^2) min cuts.
- cshjin/MinCutAlgo (code/MinCutAlgo/algo/MinCut.py): adjacency-list MinCut(graph,t) contraction + FastMinCut recursion (t=1+int(n/sqrt2), <6 base case). Note: its main/return logic is buggy (returns smaller graph, not min cut); I will write a corrected, clean version.

## The leap to reconstruct: "random contraction works"
Pain: min cut classically = n-1 max-flow computations (Gomory-Hu) or O(mn) flow; deterministic, heavy. Want something elementary.
Key structural fact chain (all pre-method, knowable):
1. sum of degrees = 2|E| (handshake).
2. avg degree = 2|E|/n.
3. min cut size k <= degree(u) for every u (single-vertex cut), so k <= avg degree = 2|E|/n  =>  |E| >= nk/2.
4. random edge is in a fixed min cut with prob k/|E| <= k/(nk/2) = 2/n.

Contraction operation: merge endpoints of an edge into a supernode; drop self-loops; KEEP parallel edges (multigraph). Cuts of contracted graph <-> cuts of G that don't separate the merged set. A fixed min cut C survives a contraction iff the contracted edge is not in C.

Survival telescoping: at the step where the graph has i vertices, the same fact 3/4 applies to the *current* multigraph (min cut still >= k since contracting non-cut edges can't reduce the global min cut below k — every cut of contracted graph is a cut of G), so P[contract a non-C edge | i vertices] >= 1 - 2/i = (i-2)/i. Product from i=n down to i=3:
prod_{i=3}^{n} (i-2)/i = [(n-2)/n][(n-3)/(n-1)]...[1/3] = 2/(n(n-1)) = 1/C(n,2).
=> P[one run returns C] >= 2/(n(n-1)).

Boosting: run T = C(n,2) ln n ~ (n^2/2) ln n independent times; failure <= (1 - 2/(n(n-1)))^T <= e^{-2T/(n(n-1))} = e^{-ln n} = 1/n. Each run O(n^2) (n-2 contractions, each O(n) with adjacency-list / or O(m)). Total O(n^4 log n) dense.

Corollary (free): number of distinct min cuts <= C(n,2), since each has survival prob >= 1/C(n,2) and the survival events for distinct min cuts on a single run are disjoint outcomes => sum of probs <= 1.

## Karger-Stein leap
Waste insight: the telescoping product is front-loaded with near-1 factors and the danger is concentrated at the END (small i). Early contractions almost never hit C; the per-step kill prob 2/i only becomes significant when i is small. So sharing the expensive early work across many runs and only multiplying effort late is the win.
Threshold: survival down to t vertices >= C(t,2)/C(n,2) ~ t(t-1)/(n(n-1)) ~ (t/n)^2. Set this = 1/2 => t ~ n/sqrt(2). So contracting from n to ~n/sqrt2 preserves C with prob >= 1/2.
Recursion: contract G down to t = 1 + ceil(n/sqrt2) (so survival >= ~1/2), make TWO independent contracted copies, recurse on each, return the min cut found. Base case: small graph (<= 6 vertices) -> run plain contraction to 2.
Time: T(n) = 2 T(n/sqrt2) + O(n^2). Master theorem: a=2, b=sqrt2, n^{log_b a}=n^{log_{sqrt2}2}=n^2; work per level O(n^2) matches => critical case => T(n)=O(n^2 log n).
Success: P(n) >= 1 - (1 - (1/2) P(n/sqrt2))^2. With p=P at depth, substitution p_{k+1}=p_k - p_k^2/4 ~ recurrence whose solution is P(n)=Omega(1/log n). (Princeton: solves to ~1/(log n) with corrections.)
Boost: repeat fastmincut O(log n / P(n)) = O(log^2 n) times to push failure to 1/poly(n). Total O(n^2 log n) * O(log^2 n) = O(n^2 log^3 n).

## Design decisions -> why
- Keep parallel edges, drop self-loops: parallel edges encode multiplicity = how many original edges cross; dropping them would lose cut weights; self-loops are internal to a supernode (never crossing) so they're irrelevant. The final two supernodes' edge count = a real cut of G.
- Pick edge UNIFORM over edges (not vertices): so P[pick a C edge] = k/|E|, which the m>=nk/2 bound controls. Picking a random vertex then random neighbor would bias toward high-degree vertices (this is exactly the heuristic value, but the clean analysis needs uniform-over-edges).
- Why contraction not deletion: deleting an edge could disconnect / change which cut is min; contraction commits "these two are on the same side", which is the actual decision a min-cut search makes.
- t = n/sqrt2 specifically: the unique threshold where survival hits 1/2, balancing branch factor 2 against per-branch failure.
- Branch factor 2 (not 3+): expected number of surviving sub-instances = 2 * 1/2 = 1, the critical branching process; >1 would blow up time, the recursion is tuned to keep it ~1 while the log-depth gives the Omega(1/log n).
- MST/Kruskal equivalence (aside, real): assign iid random weights, run Kruskal, the heaviest MST edge's removal = a contraction-equivalent random cut. Lets one reuse union-find / MST machinery.

## Grounding / refs read this run (HARD GATE)
- PRIMARY: Karger 1993 "Global Min-cuts in RNC..." (SODA) — refs/karger-1993-global-mincuts-rnc.pdf/.txt. Read full. Source of: Contraction Algorithm, Theorem 2.1 survival 2/(n(n-1)), m>=nk/2 via min-degree, telescoping, Cor 2.1 O(n^2 log n) trials, Kruskal/random-rank MST view, Thm 6.1 <=C(n,2) min cuts, weighted reduction.
- PRIMARY: Karger-Stein "A New Approach to the Minimum Cut Problem" JACM journal version — refs/karger-stein-contract.pdf/.txt (37pp, full) + conference/short version refs/karger-stein-1996-new-approach-fastcut.pdf. Read full §2-4. Source of: Cor 2.3 survival to k vertices >=C(k,2)/C(n,2), Recursive-Contract (Fig 6) with t=ceil(n/sqrt2+1) and <6 base case, T(n)=2(n^2+T(n/sqrt2))=O(n^2 log n), P(n) recurrence p_{k+1}=p_k - p_k^2/4, z_k=4/p_k-1 -> Theta(k), P=Omega(1/log n), depth 2log2 n, Thm 4.4 O(n^2 log^3 n).
- AUTHOR RETROSPECTIVE: Karger "Random Sampling in Cut, Flow, and Network Design Problems" (Math of OR) — refs/karger-random-sampling-cut-flow-network-design-survey.pdf/.txt. Author-voice framing of the unifying random-sampling research program ("the representative random sample is a central concept of statistics"). NOT a genesis-of-contraction self-account.
- ANALYSIS/EXPLAINERS: Princeton COS521 F'13 Lec2 (refs/princeton-cos521-lec2-karger.pdf, also src/), Toronto CSC473 Karger-Stein notes (refs/toronto-csc473-karger-stein.pdf/.txt).
- CODE: code/MinCutAlgo (cshjin) cloned; code/karger_clean.py is the corrected runnable version (its FastMinCut returns min of both branches; ran -> min cut 1 on K4-bridge-K4).

## SELF-ACCOUNT GAP
No discrete, citable Karger first-person account of how the random-contraction idea arose (the prompt's "MIT 6.046 lecture commentary") is available/indexed. Web search surfaced only third-party lecture notes; the Wikipedia article cites only his published papers; no interview/oral-history transcript found. Closest author-voice source is the Math-of-OR random-sampling survey (program-level intuition, not contraction genesis). The 1993 paper's own "Open Questions" (smarter edge-choice rules; deterministic contraction) is the only in-paper trace of his forward thinking. Reasoning reconstructed from primaries + antecedents per skill 1.2b fallback.
