# Synthesis — Johnson–Lindenstrauss

## Task
Map n points in R^d into R^k, k = O(log n / eps^2), preserving all C(n,2) pairwise
squared distances within (1±eps). d should NOT appear in k.

## Pain point / why it matters
Curse of dimensionality: many geometric algorithms (nearest neighbor, clustering, low-rank
approx) scale exponentially or at least polynomially in d. Want a single linear map that
compresses d -> k cheaply while certifying ALL pairwise distances are preserved. SVD/PCA
gives a global (Frobenius) optimum but NO local pairwise guarantee — a pair can collapse.

## Key reduction (the central trick)
- A pairwise-distance statement over C(n,2) pairs reduces to a SINGLE-VECTOR statement via
  linearity: for a linear map f, f(u)-f(v) = f(u-v). So preserving ||u-v|| for all pairs =
  preserving ||x|| for the m = C(n,2) difference vectors x.
- Distributional JL (DJL): there is a distribution over linear maps f: R^d -> R^k such that
  for ANY fixed x, Pr[ |  ||f(x)||^2 - ||x||^2 | > eps ||x||^2 ] <= 2 exp(-k(eps^2/2 - eps^3/3)).
  Failure prob decays like e^{-Omega(k eps^2)} per vector.
- Union bound: choose k so the per-vector failure prob <= 1/n^2 (actually <= 2/n^2 split over
  two tails). Then over C(n,2) < n^2/2 pairs the total failure prob < 1 (DG: success >= 1/n;
  can boost). The map exists by the probabilistic method.

## Construction (Gaussian / IM form — what becomes the code)
- Build R as k x d with i.i.d. N(0,1) entries; f(x) = (1/sqrt(k)) R x.  (Indyk–Motwani form;
  scaling 1/sqrt(k) makes E||f(x)||^2 = ||x||^2.)
- Equivalent normalization in DG: project to first k coords of a random rotation, scale sqrt(d/k);
  analytically cleaner.

## DG concentration proof (exact constants — VERIFIED against dasgupta_gupta_jl.txt)
- WLOG unit vector. L = sum of k squared coords of a random unit vector in R^d, projected onto
  fixed k-subspace. E[L] = mu = k/d. Y = X/||X|| uniform on sphere; Z = first k coords; L=||Z||^2.
- Lemma (DG 2.2a): for beta<1, Pr[L <= beta k/d] <= exp( (k/2)(1 - beta + ln beta) ).
  (DG 2.2b): for beta>1, Pr[L >= beta k/d] <= exp( (k/2)(1 - beta + ln beta) ).
- MGF method: E[e^{sX^2}] = 1/sqrt(1-2s), s<1/2. Chernoff on
  d(X_1^2+...+X_k^2) - k beta (X_1^2+...+X_d^2) >= 0, optimize t; t0=(1-beta)/(2 beta (d-k beta)).
- Lower tail beta = 1-eps: use ln(1-x) <= -x - x^2/2 (0<=x<1):
    (k/2)(1-(1-eps)+ln(1-eps)) = (k/2)(eps + ln(1-eps)) <= (k/2)(eps - eps - eps^2/2)
    = -k eps^2/4.  So Pr[L <= (1-eps)mu] <= exp(-k eps^2/4).
- Upper tail beta = 1+eps: use ln(1+x) <= x - x^2/2 + x^3/3 (x>=0):
    (k/2)(1-(1+eps)+ln(1+eps)) = (k/2)(-eps + ln(1+eps))
    <= (k/2)(-eps + eps - eps^2/2 + eps^3/3) = -(k/2)(eps^2/2 - eps^3/3).
    So Pr[L >= (1+eps)mu] <= exp( -(k/2)(eps^2/2 - eps^3/3) ).
- Set k >= 4 (eps^2/2 - eps^3/3)^{-1} ln n. Then upper tail = exp(-(k/2)(eps^2/2-eps^3/3))
  <= exp(-2 ln n) = 1/n^2; lower tail <= exp(-k eps^2/4) <= 1/n^2 too (since eps^2/4 >=
  eps^2/2-eps^3/3 for eps in (0,1)). Both tails 2/n^2. Union over C(n,2): failure <= 1-1/n.
  (DG Theorem 2.1 exact: k >= 4(eps^2/2 - eps^3/3)^{-1} ln n.) Upper tail is the BINDING one.

## Achlioptas sign / sparse variant (VERIFIED against achlioptas_dbfriendly.txt)
- Replace N(0,1) entries by simpler mean-0 var-1 distributions:
  (a) r = +1/-1 each w.p. 1/2;  (b) r in {+sqrt3, 0, -sqrt3} w.p. {1/6, 2/3, 1/6} (density 1/3,
  3x sparsity). f(x) = sqrt(1/k) R x (with 1/sqrt(d) scaling absorbed into R as rij/sqrt(d)).
- Why it still works: E||f(x)||^2 = ||x||^2 needs only mean 0, var 1 + independence (eq (2)).
  The game is upper-tail concentration via E[exp(h Q^2)], Q = <r, x>. Achlioptas proves the even
  moments E[Q(x)^{2k}] are MAXIMIZED at worst-case w = (1/sqrt d)(1,...,1) (Lemma 7), and there
  are DOMINATED by the Gaussian moments E[T^{2k}], T~N(0,1/d) (Lemma 8). Hence the MGF and tail
  are at least as good as Gaussian => SAME k bound, k0 = (4/(eps^2/2-eps^3/3) + ...) ln n; as
  gamma->0 it's exactly DG's 4/(eps^2/2-eps^3/3) ln n. Spherical symmetry NOT essential;
  concentration is. Strictly BETTER for each fixed d.
- Lemma 6: E[exp(h Q^2)] <= 1/sqrt(1 - 2h/d), h in [0,d/2); E[Q^4] <= 3/d^2. Same MGF as scaled
  chi-square => same Chernoff. 2/3 zero-mass is TIGHT for all-ones worst case.
- Insight: each COLUMN of R is an independent unbiased bounded-variance estimator of ||x|| (inner
  product); summing k of them = consensus / CLT; estimator variance sets how many are "enough".
  Orthonormal columns = "greatest efficiency" but not required.

## Code grounding (scikit-learn sklearn/random_projection.py, VERIFIED)
- johnson_lindenstrauss_min_dim (line 145-146): denominator = eps**2/2 - eps**3/3;
  return 4 log(n_samples)/denominator.
- _gaussian_random_matrix (line 204): N(0, 1/n_components) entries, shape (n_components,
  n_features). i.e. R with 1/sqrt(k) folded in. transform: X @ R^T.
- _sparse_random_matrix (line 271-305): density = 1/sqrt(n_features) (Li et al). nonzero entries
  +- sqrt(1/density)/sqrt(n_components) = +-sqrt(s)/sqrt(k); density=1/3 reproduces Achlioptas
  {+-sqrt3, 0}. values +-1 each w.p 1/(2s), 0 w.p 1-1/s, then scaled.

## Design decisions -> why
- Linear map: f(u)-f(v)=f(u-v) collapses pairwise->single-vector for the union bound. Nonlinear
  loses that.
- Random oblivious (not deterministic SVD): SVD optimizes global Frobenius, no per-pair floor;
  a data-oblivious random map preserves EVERY direction w.h.p. Gives d-independence: k depends
  only on n, eps — impossible for any data-dependent top-k subspace in the worst case.
- 1/sqrt(k) scaling: unbiased estimator, E||f(x)||^2 = ||x||^2.
- Gaussian entries: 2-stability => <r,x> ~ N(0,||x||^2), so ||f(x)||^2 = ||x||^2 chi^2_k/k
  exactly, SAME law for every x. Clean chi^2 concentration.
- per-vector prob target 1/n^2: union over <n^2 pairs leaves constant success; yields factor 4
  and log n.
- keep eps^3/3 term: sharper ln(1+x) expansion on the binding upper tail; dropping it weakens
  the constant. Keep it -> tight 4/(eps^2/2-eps^3/3).
- sign entries: only mean 0/var 1 needed for unbiasedness; moment domination => concentration no
  worse than Gaussian. Buys integer arithmetic, fewer random bits.
- density 1/3 (or 1/sqrt d): more zeros = faster projection; 2/3 zero-mass is the tightness limit
  for the all-ones worst case to remain worst-case.
