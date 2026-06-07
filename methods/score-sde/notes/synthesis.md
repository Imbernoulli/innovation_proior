# Synthesis — score-based generative modeling through SDEs

## The pain point
Both SMLD/NCSN and DDPM are "perturb data with a ladder of noise, learn to reverse it" models, but
they use a *finite* set of discrete noise scales chosen by hand, and each has its own bespoke training
loss and bespoke sampler with no shared theory. Open questions:
1. The discrete ladder is a hyperparameter (how many scales? what spacing?). Is there a principled,
   parameter-free way to bridge data → prior?
2. NCSN's sampler (annealed Langevin) and DDPM's sampler (ancestral) look unrelated. Are they?
3. Score-based models give no exact likelihood (unlike flows). Can we get one?
4. Can a single trained model do conditional tasks (inpainting, colorization, class-conditional)
   without retraining?

## The central object
A *continuous* diffusion {x(t)}_{t∈[0,T]} from x(0)~p_data to x(T)~tractable prior, an Itô SDE
dx = f(x,t)dt + g(t)dw. Reverse it with Anderson 1982:
dx = [f - g² ∇log p_t(x)] dt + g dw̄. Everything you need to reverse is the time-dependent score
∇log p_t(x). Estimate it with a network s_θ(x,t) by (continuous) denoising score matching, then
plug into a numerical SDE solver.

## Design-decision → why table

| Decision | Why this, why not the alternative |
|---|---|
| Model the *score* ∇log p, not the density | EBM density needs intractable Z_θ; the score kills ∇log Z=0, freeing the network architecture. |
| Continuum of noise scales (SDE) not a finite ladder | The ladder is an unprincipled hyperparameter and forces ad-hoc interpolation if you want more steps; a continuous-time SDE has a single schedule function and any solver can choose its own step count. It also exposes the shared structure NCSN/DDPM were hiding. |
| Forward SDE has no trainable parameters / doesn't depend on data | Makes the encoding *uniquely identifiable* (same trajectory given true score, independent of architecture), and lets us train the score by simple denoising. |
| Reverse-time SDE (Anderson) for sampling | It is the exact time reversal of the forward diffusion; its only unknown is ∇log p_t, exactly what score matching gives. The -g²∇log p drift term is what "pushes uphill" toward data. |
| Denoising score matching target ∇log p_{0t}(x(t)|x(0)) | The marginal score ∇log p_t is intractable, but Vincent 2011's identity makes the *conditional* score the regression target, and for affine drift that conditional is Gaussian → target = -z/σ_t, trivial. |
| λ(t) ∝ 1/E||∇log p_{0t}||² (≈ σ_t²) | Equalizes the loss magnitude across noise scales so no scale dominates; gives the clean ||σ_t s_θ + z||² form. Same weighting both NCSN (σ_i²) and DDPM (1-α_i) used, now unified. |
| VE SDE (= NCSN limit), no drift, variance explodes | The continuous limit of NCSN's additive-noise chain. σ(t) geometric → g(t)=σ(t)sqrt(2 log(σ_max/σ_min)). Good sample quality, but variance grows unboundedly so prior is N(0,σ_max²I). |
| VP SDE (= DDPM limit), -½βx drift, variance bounded | Continuous limit of DDPM's sqrt(1-β) scaling chain; variance stays ≤1, ≡1 if started at 1, so prior is N(0,I). Better likelihoods than VE. |
| sub-VP SDE (new): diffusion ×sqrt(1-e^{-2∫β}) | Built to have provably *smaller* variance than VP at every t (so less noise injected) while still →N(0,I); empirically the best likelihoods. The 1-e^{-2∫β} factor vanishes at t=0 and →1 for large t, matching VP asymptotically. |
| Probability flow ODE dx=[f - ½g²∇log p]dt | Rewriting Fokker-Planck as a continuity equation shows a deterministic ODE shares the SDE's marginals. The ½ (vs the SDE's full g²) is exactly what the FP-to-Liouville algebra produces. Gives: exact likelihood (neural-ODE change of variables), fast adaptive sampling (black-box RK45), invertible identifiable latents. |
| Hutchinson trace estimator for the divergence | Exact divergence ∇·f̃ is O(d) backprops; E_v[vᵀ∇f̃ v] is one vjp, unbiased, arbitrarily accurate by averaging. |
| Predictor-Corrector sampler | A pure numerical SDE step (predictor) accumulates discretization error; since we *also* have the score, run score-based MCMC (Langevin) as a corrector to re-project the sample onto the correct marginal p_t. NCSN-sampling = corrector-only, DDPM-sampling = predictor-only; PC strictly generalizes both. |
| Langevin step size ε = 2α(r||z||/||s||)² | Sets a target signal-to-noise ratio r between the score step and the noise so the corrector neither stalls nor diverges across scales; α=1-β rescales for VP. |
| Reverse diffusion predictor (mirror forward discretization) | Deriving ancestral sampling per new SDE is tedious; discretizing the reverse SDE the same way as the forward one is automatic, and ancestral sampling falls out as a special (β→0) case of it. |
| One denoising step (Tweedie) at the end | Samples carry imperceptible noise that wrecks FID; a single denoising predictor step without added noise removes it. This is partly why NCSN had lagged DDPM on FID. |
| Conditional reverse SDE via ∇log p_t(x|y)=∇log p_t(x)+∇log p_t(y|x) | Bayes on the marginals: only the unconditional score (already trained) plus a cheap conditional term (a time-dependent classifier, or a heuristic) is needed; no retraining of the generator. |
| Restrict integration to [ε,1] | VE σ(t) is discontinuous at 0; VP variance →0 at 0 causes numerical blow-up. Small ε (1e-5 train, 1e-3 sample for VP) avoids both. |
| VP score parameterization: net predicts noise, score=-net/std | Matches DDPM's ε-prediction; the network targets a unit-variance quantity (z), and the -1/std converts it to the score. |

## What to validate (forward-looking, not results)
Whether the continuous framework recovers (or beats) NCSN/DDPM sample quality; whether PC beats
predictor-only/corrector-only under equal compute; whether the probability-flow ODE gives competitive
exact bits/dim; whether one unconditional model handles inpainting/colorization/class-conditioning.
</content>
