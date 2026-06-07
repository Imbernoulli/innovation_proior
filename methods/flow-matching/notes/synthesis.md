# Synthesis — Flow Matching (arXiv 2210.02747)

## Pain point at the time
Diffusion models (DDPM, score SDE) are scalable and stable, but the data→noise process is restricted to a *simple* SDE (VE/VP). That fixes the family of probability paths you can use; the paths are curved, sampling needs many NFEs, and you must reason indirectly through a stochastic process and then time-reverse it.

Continuous Normalizing Flows (CNFs; Chen et al. 2018) are strictly more general: a neural vector field v_t(x;θ) defines a flow φ_t via dφ_t/dt = v_t(φ_t), and the model density is the push-forward [φ_t]_* p_0. They can model *arbitrary* probability paths (they encompass the diffusion probability-flow ODE as a special case). BUT the standard training is maximum likelihood: you compute log p_1(x) via the instantaneous change of variables
  d/dt log p_t(φ_t(x)) = -div(v_t(φ_t(x))),
which requires numerically *simulating* the ODE (and its divergence) forward/backward every step. Sequential, expensive, doesn't scale past ~32×32.

Existing simulation-free CNF attempts:
- Moser Flow (Rozen et al. 2021): linear interpolation of densities, but involves an intractable integral (works with densities not log-densities → cannot scale to high dim) and needs divergence of the learned field.
- Ben-Hamu et al. 2022 ("matching"): general probability paths from conditional paths (very close in spirit), but uses a *log* form of the continuity equation → must approximate log of the marginal density → **biased gradients** in minibatch; also needs divergence computation.

So the goal: train a CNF *simulation-free*, with *unbiased* gradients, on *general* (not only diffusion) probability paths.

## Load-bearing ancestors
1. **CNF / Neural ODE (Chen et al. 2018; FFJORD Grathwohl et al. 2018).** Flow as ODE, instantaneous change of variables, Hutchinson trace estimator for divergence. Limitation: ML training needs ODE simulation.
2. **Continuity equation (transport/Liouville PDE; Villani 2009).** ∂_t p_t + div(p_t v_t) = 0 is the necessary-and-sufficient test that v_t generates p_t. This is the central tool of every proof.
3. **Score-based SDE / probability-flow ODE (Song et al. 2020/2021).** Every diffusion SDE dy=f_t dt+g_t dw has a deterministic probability-flow ODE with the same marginals, vector field w_t = f_t − (g_t²/2)∇log p_t (via rewriting Fokker–Planck as a continuity equation, Maoutsa et al. 2020). So diffusion *is* a CNF in disguise. The VP/VE paths are Gaussian. This is what FM will subsume.
4. **Denoising score matching (Vincent 2011) / its use in diffusion training.** You can't regress the marginal score ∇log p_t(x) (intractable), but regressing the *conditional* score ∇log p_t(x|x_1) gives the same minimizer/gradient. This is the exact template CFM generalizes — from scores to vector fields.
5. **OT displacement interpolation (McCann 1997).** Between two Gaussians (one standard), the W2 geodesic / displacement map is the affine interpolation; particles move on straight lines at constant speed.

## The derivation skeleton (everything worked inline in reasoning.md)
1. FM objective: L_FM = E_{t,p_t(x)} ||v_t(x) − u_t(x)||². Clean regression, but u_t (the field generating the desired marginal p_t) is unknown/intractable. Wall.
2. Build p_t as a **mixture of conditional paths**: p_t(x)=∫ p_t(x|x_1) q(x_1) dx_1, with p_t(x|x_1) easy (e.g. concentrate at x_1 at t=1, standard normal at t=0).
3. Define the **marginal VF** as a posterior-weighted average of conditional VFs:
   u_t(x) = ∫ u_t(x|x_1) [p_t(x|x_1)q(x_1)/p_t(x)] dx_1 = E_{q(x_1|x)}[u_t(x|x_1)].
   THEOREM 1: this u_t generates p_t. Proof via continuity equation: d/dt p_t = ∫ (d/dt p_t(x|x_1)) q dx_1 = -∫ div(u_t(x|x_1)p_t(x|x_1)) q dx_1 = -div(∫ u_t p_t(·|x_1) q dx_1) = -div(u_t p_t). The numerator-times-1/p_t is exactly what makes the divergence collapse.
4. Still intractable (integral). So regress the **conditional** field instead: CFM = E_{t,q(x_1),p_t(x|x_1)} ||v_t(x) − u_t(x|x_1)||².
   THEOREM 2 (gradient equivalence): ∇_θ L_FM = ∇_θ L_CFM. Proof: both losses expand to ||v||² − 2⟨v,target⟩ + ||target||²; ||target||² is θ-independent. The ||v||² term: E_{p_t(x)}||v||² = E_{q(x_1),p_t(x|x_1)}||v||² (just unfold the marginal). The cross term: E_{p_t(x)}⟨v,u_t(x)⟩ = ∫⟨v, ∫u_t(x|x_1)p_t(x|x_1)q dx_1 / p_t⟩ p_t dx = the p_t cancels = ∫⟨v,u_t(x|x_1)⟩ p_t(x|x_1)q dx_1 dx = E_{q(x_1),p_t(x|x_1)}⟨v,u_t(x|x_1)⟩. So L_FM and L_CFM differ only by a θ-independent constant. KEY: the cancellation of p_t in the cross term is exactly the same algebra that defined the marginal VF in step 3.
5. Pick conditional path: Gaussian p_t(x|x_1)=N(μ_t(x_1), σ_t(x_1)²I), with μ_0=0,σ_0=1 (→ standard normal) and μ_1=x_1, σ_1=σ_min (→ concentrated at x_1).
   Pick the canonical (simplest) generating flow ψ_t(x)=σ_t x + μ_t.
   THEOREM 3: its VF is u_t(x|x_1) = (σ_t'/σ_t)(x − μ_t) + μ_t'. Proof: from dψ_t/dt=u_t(ψ_t), set y=ψ_t(x), x=(y−μ_t)/σ_t, ψ_t'(x)=σ_t' x+μ_t', substitute.
6. Special cases by choosing μ_t,σ_t:
   - **VE**: μ_t=x_1, σ_t=σ_{1-t} → u_t=-(σ'_{1-t}/σ_{1-t})(x−x_1).
   - **VP**: μ_t=α_{1-t}x_1, σ_t=√(1−α²_{1-t}), α_t=e^{-T(t)/2} → u_t = -(T'(1-t)/2)·[(e^{-T(1-t)}x − e^{-T(1-t)/2}x_1)/(1−e^{-T(1-t)})]. Coincides with Song's probability-flow ODE field (derived independently in appendix via Fokker-Planck w_t=f_t−(g²/2)∇log p_t plus the time-reversal lemma ũ_t=−u_{1-t}).
   - **OT (linear interpolant)**: μ_t=t x_1, σ_t=1−(1−σ_min)t → u_t=(x_1−(1−σ_min)x)/(1−(1−σ_min)t). Constant-direction field; ψ_t(x_0)=(1−(1−σ_min)t)x_0 + t x_1; the CFM regression target reduces to x_1−(1−σ_min)x_0 (a constant in t per pair) — straight-line constant-speed trajectory = McCann OT displacement between the two Gaussians.

## Design-decision → why table
| Decision | Why this and not the alternative |
|---|---|
| Parameterize the **vector field** v_t directly, regress on it | Diffusion regresses the *score* and then needs reweighting λ(t) and a separate ODE field at sampling; regressing the velocity directly gives an L2 objective with no weighting, and the same net is the sampling field. More stable, larger LR. |
| Build p_t as **mixture of conditional paths** | The marginal path/field have no closed form; per-sample conditionals do. Mixture marginal hits p_1≈q automatically. |
| Marginal VF = **posterior average** (eq for u_t) | It's the unique aggregation that makes the continuity-equation divergence collapse so u_t actually generates the mixture p_t (Thm 1). |
| Train with **CFM** not FM | Marginal u_t is an intractable integral; conditional u_t(x|x_1) is closed-form. Thm 2 says the gradient is identical, so we lose nothing. |
| **Gaussian** conditional paths | Closed form for sampling x_t = μ_t+σ_t ε and for u_t; the boundary conditions (std normal at 0, point mass at 1) are met by choosing μ,σ. |
| **Canonical** flow ψ_t=σ_t x+μ_t (simplest VF) | A path has infinitely many generating fields (add any divergence-free component); the affine map gives the minimal-motion one — extra rotational components just waste compute. |
| **OT/linear** μ_t=tx_1, σ_t=1−(1−σ_min)t | Mean & std change linearly → field has constant direction in t (u=g(t)h(x)), straight constant-speed trajectories = simpler regression target + fewer NFE; also it is literally the W2-optimal displacement map between the two endpoint Gaussians (McCann). Diffusion paths are curved and can overshoot. |
| σ_min small but >0 (e.g. 1e-4..1e-5) instead of 0 | keeps p_1(·|x_1) a proper (nondegenerate) Gaussian so the path/field are well-defined; σ_min=0 recovers the pure interpolation (Liu/Albergo rectified-flow style). |
| Diffusion as a **special case** (don't reason about SDEs) | Once you have the Gaussian-path field formula, VP/VE drop out by plugging μ,σ. You can train diffusion *paths* with the FM loss (more robust) and never construct/time-reverse an SDE. |
| Sample by solving the ODE with off-the-shelf adaptive solver (dopri5); likelihood via instantaneous change of variables + Hutchinson trace | deterministic, fewer NFE than SDE; unbiased log-likelihood estimator. |

## Canonical code grounding
- torchcfm `TargetConditionalFlowMatcher` = exactly FM-OT: compute_mu_t = t*x1, compute_sigma_t = 1-(1-σ)t, compute_conditional_flow = (x1-(1-σ)xt)/(1-(1-σ)t); base `ConditionalFlowMatcher.sample_xt` = μ_t+σ_t·ε.
- facebookresearch/flow_matching `AffineProbPath`/`CondOTProbPath`: X_t=α_t X_1+σ_t X_0, conditional velocity dX_t=α̇_t X_1+σ̇_t X_0; CondOT scheduler α_t=t, σ_t=1−t. Loss = MSE(model(x_t,t), dx_t).
- Training loop: x1~data, x0~N(0,I), t~U[0,1], x_t=ψ_t(x0), target=dψ_t/dt, minimize ||v_θ(x_t,t)−target||². Sampling: integrate dx/dt=v_θ(x,t) from x0~N(0,I) over [0,1].
