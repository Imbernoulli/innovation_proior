# SAM — synthesis notes (Phase 1.5)

## Pain point / research question
Overparameterized nets can drive training loss to ~0 with many distinct global minima that generalize very differently. Minimizing only L_S(w) gives no control over *which* minimum you land in. Prevailing wisdom (Hochreiter-Schmidhuber; Keskar et al.): **flat** minima generalize better than **sharp** ones; large-batch SGD lands in sharp minima and generalizes worse. Jiang et al. 2019 ("Fantastic Generalization Measures") empirically: sharpness-based measures correlate best with the generalization gap among 40 measures. So: can we make the optimizer *directly* seek flat regions, scalably (no Hessian materialization), as a drop-in for SGD/Adam?

## Load-bearing ancestors (with the gap each leaves)
- **SGD / Robbins-Monro (1951), minibatch.** Minimizes L_S(w) only; geometry-blind; lands in whatever basin gradient flow leads to.
- **Hochreiter-Schmidhuber flat minima (1995 "Simplifying nets by minimizing description length", 1997 "Flat Minima").** Argued flat minima = short description length (MDL) = better generalization. Gap: their "flat minimum search" requires expensive Hessian-related quantities (second derivatives, box around minimum); not scalable to deep nets, hard to differentiate through.
- **Keskar et al. 2016 (large-batch generalization gap).** Empirical: large-batch training converges to sharp minima (large Hessian eigenvalues) and generalizes worse than small-batch (flat). Defines a sharpness metric = max increase of loss in a small box. Gap: diagnostic only — a *measure*, not a training procedure; metric is expensive/non-differentiable.
- **PAC-Bayes (McAllester 1999; Dziugaite-Roy 2017; Neyshabur et al. 2017).** Bound on population loss of a *stochastic* predictor (posterior Q over weights): E_Q[L_D] ≤ E_Q[L_S] + sqrt((KL(Q‖P)+log(n/δ))/(2(n-1))). With Gaussian P,Q this turns the expected-loss-under-perturbation into a bound. Gap: gives a bound, not an algorithm; the expected loss under Gaussian perturbation is what to control — but how to optimize it cheaply?
- **Adversarial perturbation framing (FGSM, Goodfellow et al. 2015).** Worst-case input perturbation via first-order Taylor + sign; the inner max over a norm ball has a closed-form first-order solution. SAM transplants this from *inputs* to *weights*.

## The derivation (the spine of reasoning.md)
1. Object: instead of min_w L_S(w), control the *worst case in a neighborhood*: min_w max_{‖ε‖_p≤ρ} L_S(w+ε). Motivate it from PAC-Bayes: the bound has the form L_D(w) ≤ max_{‖ε‖_2≤ρ} L_S(w+ε) + h(‖w‖²/ρ²). The max term = (sharpness) + L_S(w): rewrite max L_S(w+ε) = [max L_S(w+ε) − L_S(w)] + L_S(w). Bracket = sharpness. h → replace by λ‖w‖² (standard L2). So objective: min_w max_{‖ε‖_p≤ρ} L_S(w+ε) + λ‖w‖².
2. Inner max is intractable exactly → first-order Taylor of L_S(w+ε) in ε at 0: L_S(w+ε) ≈ L_S(w) + εᵀ∇L_S(w). Drop constant L_S(w): ε̂ = argmax_{‖ε‖_p≤ρ} εᵀg, g=∇L_S(w).
3. Solve via Hölder/dual norm: εᵀg ≤ ‖ε‖_p‖g‖_q ≤ ρ‖g‖_q, 1/p+1/q=1. Equality at ε̂_i = ρ·sign(g_i)|g_i|^{q-1}/(‖g‖_q^q)^{1/p}. Verify ‖ε̂‖_p=ρ (uses (q-1)p=q). p=2: ε̂=ρ g/‖g‖₂. p=∞: ε̂=ρ sign(g). [The appendix denominator ‖g‖_2^2 is a typo; correct is ‖g‖_2.]
4. SAM gradient: ∇_w L_S^{SAM}(w) ≈ ∇_w L_S(w+ε̂(w)). Chain rule: = (d(w+ε̂)/dw)ᵀ ∇L|_{w+ε̂} = ∇L|_{w+ε̂} + (dε̂/dw)ᵀ ∇L|_{w+ε̂}. Second term contains dε̂/dw which depends on the Hessian (ε̂ ∝ g(w)). It enters only via Hessian-vector products (tractable) — but drop it for a 2× cheaper update. Final: g_SAM = ∇L|_{w+ε̂}. Empirically dropping helps (keeping the 2nd-order term degraded perf in their ablation).
5. Two passes/step: (a) forward-backward at w → g → ε̂; ascend to w+ε̂; (b) forward-backward at w+ε̂ → g_SAM; restore w; apply descent w ← w − η g_SAM (with the base optimizer). 2 grad evals/step → run SGD 2× epochs for fair compute comparison.
6. m-sharpness: the max is computed per-batch (or per-shard of size m), not over full S. Averaging per-shard SAM updates = summing independent ε-maximizations over disjoint m-subsets. Smaller m → better generalization empirically and better correlation with gen-gap. Not synced across accelerators on purpose.

## Design-decision → why
- **Worst-case (max) over a ball, not expected/random perturbation.** PAC-Bayes controls E_ε[L]; worst-case upper-bounds it and is what the bound's max term is. Empirically (appendix) adversarial ε beats random ε of same norm. Random Gaussian ε of fixed norm is the rejected alternative.
- **First-order Taylor for inner max.** Exact max intractable; one PGA step. Appendix: multi-step PGA gives slightly higher sharpness estimate but ~no test-acc gain → one step is enough.
- **p=2 default.** Theorem derived for L2; empirically p=2 beats p=∞ and beats random. p=1 would perturb a single weight (degenerate). So ε̂=ρ g/‖g‖₂.
- **Drop second-order term.** 2× speed; HVP is tractable but unnecessary; keeping it *hurt* test error in their ablation; cosine sim between 1st/2nd-order updates ≈1 for most of training. Rejected alt: keep full gradient.
- **ρ=0.05 default.** Single hyperparam; grid {0.01,0.02,0.05,0.1,0.2,0.5}; 0.05 robust default.
- **λ‖w‖² instead of h(‖w‖²/ρ²).** h is proof-artifact-dependent and ugly; standard L2/weight-decay is the practical stand-in.
- **Per-batch / m-sharpness (don't sync perturbations across accelerators).** Smaller m generalizes better and aligns with data-parallel scaling; surprising free win.
- **Adaptive SAM (davda54 impl extra, ASAM idea).** scale ε per-weight by w² to be scale-invariant; off by default.

## Canonical code (davda54/sam)
SAM is an Optimizer wrapper around a base optimizer. first_step: grad_norm over all params, scale=ρ/(‖g‖+eps), e_w = g*scale (or *w² if adaptive), save old_p, p += e_w (ascend). second_step: restore p=old_p, base_optimizer.step() (descend using grad computed at w+ε̂). step(closure): first_step(zero_grad)→closure (fwd-bwd at w+ε̂)→second_step. grad_norm = L2 norm over the stacked per-param grad norms. Note p.add_(e_w) ascends (the sign is correct: we MAXIMIZE inner, so move +ρg/‖g‖).
