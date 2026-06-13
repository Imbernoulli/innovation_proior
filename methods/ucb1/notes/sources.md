# Sources for ucb1 (Auer, Cesa-Bianchi, Fischer 2002)

## Method identification
MLS-Bench task `optimization-online-bandit`, baseline `ucb1`.
- task_description.md: "UCB1 (Auer, Cesa-Bianchi, and Fischer, 'Finite-time Analysis of the
  Multiarmed Bandit Problem', Machine Learning 47, 2002): plays the arm with the highest upper
  confidence bound mu_hat + sqrt(2 log(t)/n_a)". Paper-default exploration constant c=2 in
  sqrt(c log t / n_a).
- edits/ucb1.edit.py: implements mu_hat + sqrt(2 log(t+1)/N_a) (initial round-robin t<K),
  the edit explicitly disables an SW-UCB sliding-window fallback and declares vanilla UCB1 (full
  history) the correct version. So the canonical paper method to reconstruct is plain UCB1.

## (1) PRIMARY — read in full
Auer, P., Cesa-Bianchi, N., Fischer, P. "Finite-time Analysis of the Multiarmed Bandit
Problem." Machine Learning 47(2-3):235-256, 2002. Kluwer.
Preliminary version: Proc. 15th ICML, pp. 100-108, Morgan Kaufmann, 1998.
PRE-ARXIV CLASSIC (no e-print). NOT on arXiv. arxiv id = "".
- refs/acbf2002.pdf (Springer A:1013689704352, 10 pp), identical copy also at
  homes.di.unimi.it/cesa-bianchi/Pubblicazioni/ml-02.pdf (Cesa-Bianchi's own page).
- refs/acbf2002.txt (full pdfplumber/pdftotext extraction).
- Read visually pp. 2-9 (Read tool, pages=) to get every equation: regret def (Eq. 5), UCB1
  Figure 1 (index x_bar_j + sqrt(2 ln n / n_j)), Theorem 1, Eq. 2, the D>=2Delta^2 remark,
  UCB2 (Figure 2, Eq. 3), eps_n-greedy (Figure 3), UCB1-NORMAL (Figure 4, Theorem 4),
  Fact 1 (Chernoff-Hoeffding), Fact 2 (Bernstein), full Proof of Theorem 1 (Eq. 6-9 + sum).
  All math in reasoning.md/answer.md cross-checked against these pages.

## (2) BACKGROUND / load-bearing ancestors
- Lai, T.L., Robbins, H. "Asymptotically efficient adaptive allocation rules." Adv. Appl.
  Math. 6(1):4-22, 1985. PRE-ARXIV CLASSIC. PRIMARY PDF PAYWALLED (ScienceDirect
  0196885885900028 returns HTML login wall) and only on aggregators (scribd, semanticscholar,
  oa.mg). Per skill's pre-arXiv-classic rule, its load-bearing content is grounded via a
  high-quality secondary that reproduces its equations: the ACBF paper's own Section 1
  (Eq. 1: E[T_j(n)] <= (1/D(p_j||p*) + o(1)) ln n; the matching lower bound
  E[T_j(n)] >= (ln n)/D(p_j||p*); the "upper confidence index" framing; D = KL divergence),
  plus the explainer + WebSearch confirmations (change-of-measure lower bound; first to use the
  term "upper confidence bound", interpreted as significance level 1/t). GAP FLAGGED: original
  Lai-Robbins full text not obtained this run (paywall). Its role here is only as an antecedent
  (the asymptotic floor + hard-to-compute index), all of which is reproduced verbatim in the
  primary's Section 1.
- Agrawal, R. 1995 — index expressible as a simple function of total reward so far; cheaper
  than Lai-Robbins; keeps optimal log order (larger leading constant in some cases). Grounded
  via ACBF Section 1 (which states this and says UCB1's index "is derived from the index-based
  policy of Agrawal (1995)").
- Chernoff-Hoeffding bound: stated as Fact 1 in the primary itself (cites Pollard 1984
  appendix; classic Hoeffding 1963). Bernstein/empirical-variance = Fact 2.

## (3) THIRD-PARTY EXPLAINER
Jeremy Kun, "Optimism in the Face of Uncertainty: the UCB1 Algorithm" (jeremykun.com, 2013).
WebFetch captured -> notes/explainer-jeremy-kun.md. Confirms regret def, optimism principle,
radius = inverted Hoeffding tail at delta=t^{-4} -> sqrt(2 log t/n_j), the regret theorem
(8 sum log T/Delta_i + (1+pi^2/3) sum Delta_j), and O(sqrt(KT log T)) worst case. Independent
corroboration of every load-bearing formula.
Additional WebSearch corroboration: CSE599i lecture notes (login-walled, not used),
multiple survey arXiv abstracts confirming the Lai-Robbins / optimism / Hoeffding story.

## (4) CANONICAL CODE
SMPyBandits (Lilian Besson), as cited in edit.py
(vendor/external_packages/SMPyBandits/SMPyBandits/Policies/UCB.py). Read in full + snapshotted
to code/:
- SMPyBandits_UCB.py: computeIndex = rewards/pulls + sqrt(2 log(t)/pulls), +inf if pulls<1.
- SMPyBandits_UCBalpha.py: index = mean + sqrt(alpha log t/(2 N_k)); ALPHA default 4 in file
  (alpha=4 -> sqrt(2 log t/N), i.e. UCB1). Ref [Auer et al. 02].
- SMPyBandits_IndexPolicy.py: choice() = argmax index (random tie-break), computeAllIndex loop.
- SMPyBandits_BasePolicy.py: holds t, pulls (N_k), rewards (cumulative); getReward increments.
answer.md code is faithful to this structure (running counts/rewards/clock + index argmax +
inf-init round-robin) and to the task harness interface (select_arm/update/reset).

## Self-account
NO first-person discovery memoir / award lecture for UCB1 located this run. The 1998 ICML
preliminary + 2002 journal versions are the record. Path reconstructed from primary +
antecedents + explainer. (Not added to SELF_ACCOUNT_SOURCES.md since none was found.)
