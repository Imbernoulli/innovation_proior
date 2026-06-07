# Chinchilla synthesis (Phase 1.5)

## Verified
- arXiv 2203.15556, "Training Compute-Optimal Large Language Models". Title verified from source.
- No single canonical code repo (DeepMind, closed). Canonical artifact = the parametric fit + the closed-form allocation. Code deliverable = a faithful reimplementation of the Approach-3 fitting (LSE/Huber/LBFGS) and the closed-form N_opt/D_opt, which is the actual reproducible method. Cross-check the LSE-Huber-LBFGS recipe against the paper's appendix eq. (the well-known epfml/chinchilla and cloneofsimo reproductions match this).

## Pain point / question
- LLMs at 500B+ params (GPT-3 175B, Gopher 280B, MT-NLG 530B, Jurassic 178B). Training is a one-shot, expensive event: you know the compute budget C in advance (# accelerators × time), can only train once, so picking the right (N,D) for that C is critical.
- Prevailing wisdom from Kaplan et al. 2020: power law between params N and loss; "don't train to lowest loss"; and crucially their allocation says when C goes up 10×, scale N by ~5.5× and D by only ~1.8× (a≈0.73, b≈0.27). So everyone scaled N hard and trained on ~300B tokens regardless of size.
- Central question: GIVEN fixed FLOPs budget C, how to trade off N vs D to minimize final pretraining loss L(N,D)? min L(N,D) s.t. FLOPs(N,D)=C.
- FLOPs approximation C ≈ 6ND (Kaplan): ~2ND for forward (each param ~1 multiply-add = 2 FLOPs per token), ~4ND backward → 6ND per pass over D tokens. So C is essentially a constant times N·D.

## Why Kaplan got a≈0.73
- (motivating/diagnostic, allowed) Kaplan's runs: majority < 100M params, and they used a FIXED cosine schedule / didn't re-tune the LR schedule per token horizon. Chinchilla's key methodological fix: the cosine cycle length must MATCH the number of training tokens. If the cosine cycle overshoots the actual training steps by >25%, performance is noticeably degraded — the LR hasn't decayed enough at the point you stop. Using a single long schedule and reading off intermediate points (as Kaplan did) systematically mis-measures the loss at intermediate token counts and biases the frontier toward big-N/small-D. Also Chinchilla uses models up to 16B and mostly >500M.

## The three approaches (all estimate N_opt ∝ C^a, D_opt ∝ C^b)

### Approach 1: minimum over training curves (fix N, vary D)
- Family of models 70M–10B. For each N, train 4 runs with cosine cycle decaying 10× over token-horizons spanning 16×.
- Smooth + interpolate each loss-vs-FLOPs curve → continuous map FLOPs→loss per run.
- For each of 1500 log-spaced FLOP values C, find the run/model achieving lowest loss → gives (N,D) on the FLOP=C envelope (the lower envelope / "training curve envelope").
- Fit power laws N_opt∝C^a, D_opt∝C^b → a=0.50, b=0.50.
- Footnote: all selected points within last 15% of training → confirms cosine cycle ≈ D is right.

### Approach 2: IsoFLOP profiles (fix C, vary N)
- 9 fixed FLOP budgets (6e18 to 3e21). For each, vary N (D determined by D=C/(6N)), cosine cycle matched to target FLOPs.
- Plot final loss vs N for each FLOP budget → a clear valley/minimum (the optimal N for that C). Fit a PARABOLA to each IsoFLOP curve to locate the minimum N.
- Then power-law fit of those minima → a=0.49, b=0.51.

### Approach 3: parametric loss fit
- Functional form (classical risk decomposition): L̂(N,D) = E + A/N^α + B/D^β.
  - E = Bayes risk / entropy of natural text (irreducible).
  - A/N^α = function-approximation gap: finite N-dim hypothesis space H_N can't reach f*. (For 2-layer nets ∝ N^{-1/2}, Siegel 2020.)
  - B/D^β = stochastic-approximation / not-trained-to-convergence gap: single epoch over D points, SGD convergence rate lower-bounded by D^{-1/2} (Robbins-Monro 1951), dimension-free.
- Decomposition: L(N,D) = L(f*) + (L(f_N)-L(f*)) + (L(f̄_{N,D})-L(f_N)). Three terms = Bayes, approximation, stochastic.
- Fit (A,B,E,α,β) by minimizing Huber_δ(log L̂ - log L) over runs, δ=1e-3 (robust to outliers; downweights low-FLOP runs). L-BFGS from a grid of inits. Implemented numerically stably as:
  min over (a,b,e,α,β) Σ Huber_δ( LSE(a-α log N_i, b-β log D_i, e) - log L_i ), then A,B,E = exp(a),exp(b),exp(e). LSE = log-sum-exp (= log(e^{a-α logN}+e^{b-β logD}+e^e) = log L̂).
- Fitted: L(N,D) = E + A/N^0.34 + B/D^0.28, E=1.69, A=406.4, B=410.7. (α=0.34, β=0.28.)

### Closed-form efficient frontier (the heart — DERIVE this)
- Minimize L̂ = E + A/N^α + B/D^β s.t. C = 6ND. Substitute D = C/(6N):
  g(N) = E + A N^{-α} + B (6N/C)^β.
  g'(N) = -α A N^{-α-1} + β B (6/C)^β N^{β-1} = 0
  ⇒ α A N^{-α-1} = β B (6/C)^β N^{β-1}
  ⇒ N^{α+β} = (αA/βB)(C/6)^β
  ⇒ N_opt = (αA/βB)^{1/(α+β)} (C/6)^{β/(α+β)} = G (C/6)^a,  G=(αA/βB)^{1/(α+β)}, a=β/(α+β).
  And D = C/(6N) = (C/6)/N_opt = G^{-1}(C/6)^{1-a} = G^{-1}(C/6)^b, b=α/(α+β).
- Check a+b=1 (since a=β/(α+β), b=α/(α+β)). Good — equal-ish split.
- Plug α=0.34, β=0.28: a=β/(α+β)=0.28/0.62=0.4516, b=α/(α+β)=0.34/0.62=0.5484 → a≈0.46, b≈0.54. Matches Approach 3 row.
- Equivalently: at the optimum, balancing — the marginal loss reduction per FLOP from adding params equals that from adding tokens; the two reducible terms are kept in fixed ratio.

## Results summary (Table tab:comparison)
- Approach 1: a=0.50, b=0.50. Approach 2: a=0.49, b=0.51. Approach 3: a=0.46, b=0.54. Kaplan: a=0.73, b=0.27.
- All three: scale N and D in ~equal proportion. Approach 3 predicts slightly smaller N at large C (negative curvature in C→N_opt, Huber downweights low-FLOP outliers).
- Gopher prediction: at Gopher's compute (5.76e23 FLOPs), optimal model is ~4× smaller (→ ~70B) trained on ~4× more tokens (→ ~1.4T). Validated by training Chinchilla 70B / 1.4T. (Chinchilla result itself = proposed-method eval, OUT of scope for reasoning/context.)

## Scaffold ↔ final code correspondence
- chinchilla_loss(N,D,params) — the L̂ = E + A/N^α + B/D^β ← context stub: parametric loss function
- fit_parametric(runs) — LSE+Huber+LBFGS grid-init fit ← context stub: fit routine
- optimal_allocation(C, params) — closed form N_opt,D_opt ← context stub: constrained-minimization solver
- isoflop_minimum / envelope helpers ← context stub: the empirical estimators

## OUT of scope (proposed-method eval)
- Chinchilla 70B benchmark wins vs Gopher (MMLU, etc.), model+training details of Chinchilla itself, downstream eval. Stop at the scaling-law estimation methodology.
