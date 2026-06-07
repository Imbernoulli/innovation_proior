# Synthesis — CMA-ES

## Sources retrieved & read this run
1. PRIMARY: Hansen & Ostermeier 2001 "Completely Derandomized Self-Adaptation in Evolution Strategies", Evolutionary Computation 9(2):159-195. PDF at refs/hansen2001.pdf (cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf). Read pp.1-9 (intro, ES adaptation history, derandomization motivation). The conceptual origin: MSC shortcomings → derandomization → CMA → cumulation/evolution path.
2. PRIMARY (companion): Hansen 2016 "The CMA Evolution Strategy: A Tutorial" arXiv:1604.00772. Full LaTeX source in src/tutorial.tex. Read in full: Preliminaries, Sampling, Mean update, Cov update (estimate-from-scratch / rank-mu / rank-one / cumulation / combine), Step-size control (CSA), Discussion (invariance/stationarity), Algorithm Summary appendix + default parameters table.
3. CODE: pycma purecma.py (CMA-ES/pycma master) at code/purecma.py. Canonical minimal implementation. ask/tell read in full.
4. EXPLAINERS: Wikipedia CMA-ES (isotropic failure, ML/natural-gradient interpretation, invariances); WebSearch on natural-gradient connection (Akimoto 2010, Glasmachers 2010: rank-mu with equal learning rates = natural gradient update; IGO Ollivier et al. 2011).

## Domain
Evolutionary Computation (also Global / black-box continuous optimization). Title for answer: "CMA-ES — Covariance Matrix Adaptation Evolution Strategy".

## The problem (research question)
Minimize a black-box f: R^n -> R, only function values available (no gradients), on non-linear, non-convex, ILL-CONDITIONED (very different scales/curvatures along directions) and NON-SEPARABLE (variables coupled; can't solve coordinate-wise) landscapes. Search cost = number of f-evaluations. Want a method that is invariant to rotation+scaling so performance is uniform over the whole class of affine-transformed problems.

## Background / lineage (load-bearing ancestors, all PRE-method)
- **Evolution Strategy (Rechenberg 1973, Schwefel 1981).** Sample lambda offspring by adding N(0,sigma^2 I) mutation to parent(s), select best mu, recombine. (mu/mu,lambda) loop. The comma = non-elitist truncation selection.
- **Isotropic Gaussian mutation N(0, sigma^2 I).** One scalar step-size. Surfaces of equal density = spheres. Fails on ill-conditioned: on f(x)=sum h_i x_i^2 with hugely different h_i, a single sigma must be small enough for the steepest direction yet that makes progress in the flat direction glacial. Fails on non-separable because the favorable directions are rotated off the coordinate axes and a diagonal (axis-parallel) covariance can't represent them.
- **Individual step-sizes (axis-parallel, diag covariance).** n variances, ellipsoids axis-parallel. Helps scaling but is coordinate-system dependent → NOT rotation invariant. On a rotated ellipsoid it's back to square one.
- **Schwefel 1981: mutative self-adaptation of full normal distribution.** Mutate the strategy parameters too, let selection pick them. The MSC (mutative strategy parameter control) concept.
- **MSC shortcomings (2001 paper §2, the core motivation):**
  (a) Selection of a strategy-parameter setting is INDIRECT — it's only selected via the object-parameter individual it happened to produce, so the signal is highly noisy/disturbed.
  (b) The mutation strength on the strategy level faces a conflict: large enough to give a significant selection DIFFERENCE between settings, yet small enough to give an optimal CHANGE RATE — these two demands disagree, and the gap widens with dimension and with the number of strategy parameters.
  (c) To get a reliable change rate you must crank up parent number mu / population — change rate can only be tuned down via recombination — so MSC's success scales population linearly with #strategy-params. Expensive.
  → Three goals of "derandomization": (i) reduce the disturbance/indirectness; (ii) UNLINK change rate from mutation strength; (iii) make adaptation reliable & fast independent of population size, even for small populations.
- **The derandomization idea:** instead of mutating strategy params and hoping selection finds good ones, DIRECTLY increase the probability of reproducing the mutation steps that were just selected. This is the seed of the covariance update.
- **Hessian/covariance link:** on convex-quadratic f(x)=½x^T H x, setting C = H^{-1} turns the ellipsoid into a sphere. So the optimal C ≈ inverse Hessian (up to scalar). CMA = learning the metric ≈ quasi-Newton without gradients.
- **EMNA / cross-entropy EDA (Larranaga 2001/2002) — a baseline that gets the reference mean WRONG:** estimates covariance of the SELECTED POINTS around their own mean x^{(g+1)}. This shrinks variance in the gradient direction → premature convergence. CMA's C_mu uses the OLD mean x^{(g)} as reference → estimates SELECTED STEPS → increases variance in the successful direction. Crucial distinction.
- **Natural-gradient / information-geometry view (post-hoc understanding only — Akimoto 2010, Glasmachers 2010, Ollivier IGO 2011):** the rank-mu update with equal learning rates equals a natural-gradient ascent on E[f] under the Fisher metric. Use to UNDERSTAND why rank-mu is principled; do not cite in-frame.

## The derivation chain (reasoning.md spine)
1. Sampling: x_k = m + sigma * B D z_k, z_k ~ N(0,I), so x_k ~ N(m, sigma^2 C), C = B D^2 B^T. Three knobs: m (where), C (shape/orientation = the metric), sigma (overall scale). WHY split sigma out of C: scale must adapt on a much faster timescale (∝1/n) than the full matrix (∝n^2/mueff); coupling them would bottleneck.
2. Mean update (move the search): m <- m + cm * sum_{i=1}^mu w_i (x_{i:lambda} - m). Weighted recombination = truncation selection (mu<lambda) + weighting. w_1>=...>=w_mu>0, sum=1. mueff = 1/sum w_i^2 = effective sample size (1<=mueff<=mu). cm=1 default.
3. Covariance — estimate from scratch: C_mu = sum w_i (x_{i:lambda}-m)/sigma ((...))^T  = covariance of SELECTED STEPS (reference = OLD mean). Contrast EMNA (reference = new mean) which shrinks. But C_mu alone needs mueff ≈ 10n to be reliable (cond<10). Too expensive for fast (small-population) search.
4. Rank-mu update: average C_mu over generations with exponential smoothing: C <- (1-cmu) C + cmu * C_mu/sigma^2 = (1-cmu)C + cmu sum w_i y_i y_i^T, y_i = (x_{i:lambda}-m)/sigma. Backward time horizon ~1/cmu. cmu ≈ mueff/n^2. Rank min(mu,n). (Active CMA: extend to lambda weights, ~half negative, sum to 0, no decay; Jastrebski & Arnold 2006.)
5. Rank-one update + evolution path: with mu=1, C <- (1-c1)C + c1 y y^T adds the max-likelihood line distribution for y. Problem: y y^T = (-y)(-y)^T → SIGN of the step is lost. Reintroduce sign via the EVOLUTION PATH = cumulated sum of consecutive mean-steps: p_c <- (1-cc) p_c + sqrt(cc(2-cc) mueff) (m_new - m_old)/(cm sigma). Normalization sqrt(cc(2-cc)mueff) makes p_c ~ N(0,C) under random selection (since (1-cc)^2 + sqrt(cc(2-cc))^2 = 1 and the weighted average of mueff steps has covariance C/mueff). Then C <- (1-c1)C + c1 p_c p_c^T. c1 ≈ 2/n^2. Correlated consecutive steps lengthen p_c by up to 1/sqrt(cc) → effectively boosts learning rate → O(n) to learn a long axis (cigar). This is the key win for SMALL populations.
6. Combine: C <- (1 - c1 - cmu*sum_w) C + c1 p_c p_c^T + cmu sum w_i y_i y_i^T.
7. Step-size control (CSA) — needed because: (a) optimal sigma scales with mu/mueff, which C-update can't capture; (b) C's learning rate (~1/n^2) is too slow to track the needed sigma change (timescale ~n). Conjugate evolution path p_sigma <- (1-cs)p_sigma + sqrt(cs(2-cs)mueff) C^{-1/2} (m_new-m_old)/(cm sigma). The C^{-1/2} = B D^{-1} B^T whitens so that under random selection p_sigma ~ N(0,I), expected length = chiN ≈ sqrt(n), INDEPENDENT of direction (that's why "conjugate"/whitened, unlike p_c which is N(0,C)). Then compare ||p_sigma|| to chiN: longer than expected ⇒ steps correlated ⇒ could've gone further ⇒ increase sigma; shorter ⇒ steps cancel/anti-correlated ⇒ overshooting ⇒ decrease. sigma <- sigma exp( (cs/ds)(||p_sigma||/chiN - 1) ). Unbiased on log scale under random selection. (purecma uses the squared-norm variant: exp( (cs/ds)/2 (||p_sigma||^2/n - 1) ).)
8. Invariances: rank-only use of f ⇒ invariant to strictly monotonic transforms of f-value. Affine invariance of search space (rotation+scaling) given matching C^{(0)} and m^{(0)} — this is the payoff and the reason CMA beats isotropic ES on ill-conditioned/non-separable. Stationarity/unbiasedness under random selection is a design criterion: m, C, ln(sigma) unbiased.

## Default parameters (2016 tutorial Table 1 / purecma)
- lambda = 4 + floor(3 ln n); mu = floor(lambda/2)
- w_i' = ln((lambda+1)/2) - ln i ; positive ones normalized to sum 1; negative ones (active) scaled
- mueff = (sum_{1..mu} w_i)^2 / sum_{1..mu} w_i^2
- cm = 1
- cc = (4 + mueff/n)/(n + 4 + 2 mueff/n)
- cs = (mueff + 2)/(n + mueff + 5)
- c1 = 2/((n+1.3)^2 + mueff)   (alpha_cov=2)
- cmu = min(1-c1, 2 (mueff - 2 + 1/mueff)/((n+2)^2 + 2 mueff/2))   ~ alpha_cov (mueff-2+1/mueff)/((n+2)^2+alpha_cov mueff/2)
- damps ds = 1 + 2 max(0, sqrt((mueff-1)/(n+1)) - 1) + cs   (purecma: 2 mueff/lambda + 0.3 + cs)
- chiN = sqrt(n)(1 - 1/(4n) + 1/(21 n^2))
- hsig (Heaviside): 1 if ||p_sigma||/sqrt(1-(1-cs)^{2(g+1)}) < (1.4 + 2/(n+1)) chiN else 0; stalls p_c when sigma rising fast.

## Design-decision → why
- Multivariate Gaussian search dist: max entropy given (co)variances; no coordinate distinguished. ✓
- Split (m, sigma, C): different optimal timescales; sigma must move ∝n, C ∝n^2/mueff. ✓
- Weighted recombination (vs best-1): reduces variance, uses more info; mueff quantifies it. ✓
- Reference = OLD mean in C_mu (vs EMNA new mean): estimates steps not points → grows variance in good direction, avoids premature convergence. ✓
- Exponential smoothing for C: single-gen estimate unreliable for small pop; accumulate. ✓
- Evolution path (rank-one): restores lost sign; exploits inter-step correlation → O(n) on cigar. ✓
- p_c normalization sqrt(cc(2-cc)mueff): keep p_c ~ N(0,C) (stationary) under random selection. ✓
- CONJUGATE (whitened) p_sigma via C^{-1/2}: make expected path length direction-independent so a single threshold chiN works for sigma. ✓
- Compare to chiN under random selection: principled neutral reference; unbiased on log-sigma. ✓
- exp() multiplicative sigma update: keeps sigma>0, symmetric in log. ✓
- Defaults c1~2/n^2, cmu~mueff/n^2, lambda=4+3ln n: empirically tuned on sphere, transfer broadly. ✓

## In-frame cautions
- Never name "CMA-ES paper"/Hansen-Ostermeier as artifact; the method NAME CMA-ES is fine (mainly answer.md).
- Cite ancestors (Rechenberg 1973, Schwefel 1981, Larranaga 2001) freely.
- Natural-gradient framing: derive in-frame ("this is exactly a natural-gradient step") as an aha, don't cite Akimoto/IGO.
- No proposed-method benchmark numbers. The "orders of magnitude speedup on ill-conditioned" is the 2001 ABSTRACT claim about the method itself → treat as forward-looking intent / motivation, not as reported result; safer to phrase as the GOAL.
