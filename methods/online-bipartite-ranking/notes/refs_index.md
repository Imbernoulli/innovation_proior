# refs/ index — online-bipartite-ranking

## Primary source (HARD GATE, read in full)
- `kvv1990_online.pdf` / `.txt` — Karp, U. Vazirani, V. Vazirani, "An Optimal Algorithm for On-line Bipartite Matching", STOC 1990, pp. 352-358. The RANKING algorithm, the original combinatorial 1-1/e proof (Lemmas 1-12 + EARLY + the w_t recurrence), and the matching upper bound (Theorem 2, via RANDOM on the upper-triangular matrix + Yao's lemma). Source: people.eecs.berkeley.edu/~vazirani/pubs/online.pdf.

## Required analysis (read in full)
- `djk2013_ranking.pdf` / `.txt` — Devanur, Jain, Kleinberg, "Randomized Primal-Dual analysis of RANKING for Online Bipartite Matching", SODA 2013, pp. 101-107. The clean randomized primal-dual proof: dual split alpha=g/F, beta=(1-g)/F; Dominance and Monotonicity Lemmas; integral inequality int_0^theta g + 1 - g(theta) >= F; g(y)=e^{y-1}, F=1-1/e. Plus vertex-weighted and AdWords unification. Source: cs.cornell.edu/courses/cs6820/2012fa/handouts/djk.pdf.

## Supporting analyses / explainers (read in full)
- `birnbaum_mathieu_madesimple.pdf` / `.txt` — Birnbaum, Mathieu, "On-line Bipartite Matching Made Simple", SIGACT News 2008. The "made simple" combinatorial proof; documents the Krohn-Varadarajan bug in the original KVV independence step and the permutation-perturbation fix (Lemma 4). Source: cs.brown.edu/people/claire/Publis/sigactnews08.pdf.
- `wisc_onlinematching.pdf` / `.txt` — UW-Madison CS787 lecture notes (Lv, 2019). Clean statement of deterministic 1/2 lower bound, randomized-greedy ~1/2, and primal-dual modified RANKING. Source: pages.cs.wisc.edu/~blv/OnlineMatchingNote.pdf.

## Antecedents / analysis status
- Antecedent (deterministic greedy 1/2-competitive + adversary forcing exactly 1/2; per-step RANDOM also ~1/2): covered in KVV intro, Birnbaum-Mathieu intro, and Wisc notes. No gap.
- Analysis (>=1): two independent analyses obtained and read in full (DJK primal-dual; Birnbaum-Mathieu combinatorial). No gap.
- Canonical code: RANKING has no single canonical repo; a faithful self-contained implementation grounded in the algorithm as stated in KVV/DJK is in code/ranking.py, Monte-Carlo-verified to converge to 1-1/e on the KVV upper-triangular worst case.
