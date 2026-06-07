# DPM-Solver synthesis notes

## Pain point / research question
DPMs (DDPM, score-SDE) generate high-quality samples but need hundreds–thousands of sequential
neural-network evaluations (NFE). This is the dominant cost. Goal: a *training-free* sampler
(plug into any pre-trained noise-prediction model ε_θ) that produces high-quality samples in
~10 NFE — the "few-step" regime — without retraining/distillation.

## Background / what exists
- Forward process: q_{0t}(x_t|x_0) = N(α_t x_0, σ_t² I). Noise schedule = (α_t, σ_t). SNR α_t²/σ_t²
  strictly decreasing. Equivalent forward SDE dx = f(t)x dt + g(t)dw with
  f(t) = d log α_t / dt,  g²(t) = dσ_t²/dt − 2 (d log α_t/dt) σ_t².
- Network ε_θ(x_t,t) trained to predict the noise; ground truth = −σ_t ∇_x log q_t(x_t).
- Reverse SDE (song2020score): dx = [f(t)x + (g²/σ_t) ε_θ] dt + g dw̄.  Ancestral sampling (ho2020denoising)
  = first-order SDE solver. SDE step size limited by Wiener randomness (kloeden1992) → needs many steps.
- Probability-flow ODE (song2020score): same marginals, deterministic:
  dx/dt = h_θ(x,t) := f(t)x + (g²(t)/(2σ_t)) ε_θ(x,t),  solved from T→0.
- song2020score used RK45 (dormand1980): ~60 NFE on CIFAR. General black-box solvers fail < ~10 steps.

## Baselines (prior art to elaborate)
- **DDPM ancestral sampling (ho2020denoising)**: discrete Markov chain, first-order SDE solver, ~1000 steps.
- **score-SDE / probability-flow ODE + RK45 (song2020score)**: treats whole RHS h_θ as a black box →
  discretizes BOTH the linear term f(t)x and nonlinear ε_θ term; linear-term error grows exponentially.
- **DDIM (song2020denoising)**: deterministic, non-Markovian. One step:
  x_{t_i} = (α_{t_i}/α_{t_{i-1}}) x_{t_{i-1}} − α_{t_i}(σ_{t_{i-1}}/α_{t_{i-1}} − σ_{t_i}/α_{t_i}) ε_θ.
  Fast (~50 steps) but motivated by non-Markovian inference, no convergence-order theory, only first order.
- **Analytic-DPM (bao2022analytic)**, learned-trajectory / distillation (salimans2022progressive): still need
  training or ~50 NFE.
- **Adaptive ODE solver (jolicoeur2021gotta)**: adaptive step size for diffusion SDE/ODE; basis for our adaptive schedule.
- ODE literature: **exponential integrators / exponential Runge-Kutta (hochbruck2005,2010)** for semi-linear ODEs
  dx/dt = αx + N(x,t): solve linear part exactly, approximate the integral of the nonlinear part. φ-functions.

## Core derivation (the heart)
1. **Semi-linear structure**: RHS = linear f(t)x + nonlinear (g²/2σ_t)ε_θ. Black-box solvers waste error
   on the linear part, which is solvable exactly.
2. **Variation of constants** for x' = f(t)x + b(t):
   x_t = e^{∫_s^t f dτ} x_s + ∫_s^t e^{∫_τ^t f dr} (g²(τ)/2σ_τ) ε_θ(x_τ,τ) dτ.
   ∫_s^t f dτ = log α_t − log α_s, so e^{∫_s^t f} = α_t/α_s. → exact linear part = (α_t/α_s)x_s.
3. **Rewrite g²**: g²(t) = dσ_t²/dt − 2(d log α_t/dt)σ_t² = 2σ_t²(d log σ_t/dt − d log α_t/dt) = −2σ_t² dλ_t/dt,
   where **λ_t := log(α_t/σ_t)** = half-log-SNR (strictly decreasing in t).
   Plug in: coefficient of ε_θ inside integral becomes
   e^{∫_τ^t f} (g²(τ)/2σ_τ) = (α_t/α_τ)(−σ_τ dλ_τ/dτ) = −α_t (σ_τ/α_τ)(dλ_τ/dτ).
   So x_t = (α_t/α_s)x_s − α_t ∫_s^t (dλ_τ/dτ)(σ_τ/α_τ) ε_θ dτ.
4. **Change of variable** τ→λ: σ_τ/α_τ = e^{−λ}; dλ = (dλ_τ/dτ)dτ. With x̂_λ := x_{t_λ(λ)}, ε̂_θ(x̂_λ,λ):
   **x_t = (α_t/α_s)x_s − α_t ∫_{λ_s}^{λ_t} e^{−λ} ε̂_θ(x̂_λ,λ) dλ.**  (Proposition: exact solution.)
   "Exponentially weighted integral." Only approximation needed = this integral of the network.
5. **Taylor-expand** ε̂_θ around λ_{t_{i-1}}: ε̂_θ(x̂_λ,λ) = Σ_{n=0}^{k-1} (λ−λ_s)^n/n! ε̂^{(n)} + O((λ−λ_s)^k).
   Substituting and integrating ∫ e^{−λ}(λ−λ_s)^n/n! dλ by parts n times → φ-functions.
   φ_k(z) := ∫_0^1 e^{(1−δ)z} δ^{k-1}/(k-1)! dδ; φ_0=e^z; recurrence φ_{k+1}(z)=(φ_k(z)−φ_k(0))/z; φ_k(0)=1/k!.
   φ_1(h)=(e^h−1)/h, φ_2(h)=(e^h−h−1)/h², φ_3(h)=(e^h−h²/2−h−1)/h³.
   Expansion: x_t = (α_t/α_s)x_s − σ_t Σ_{k=0}^n h^{k+1} φ_{k+1}(h) ε̂^{(k)}(x̂_{λ_s},λ_s) + O(h^{n+2}),
   using α_t e^{−λ_t} = α_t (σ_t/α_t) = σ_t to convert α_t·(integral) into σ_t·(...).
6. **DPM-Solver-1** (k=1, n=0): drop O(h²):
   x̃_{t_i} = (α_{t_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − σ_{t_i}(e^{h_i}−1) ε_θ(x̃_{t_{i-1}},t_{i-1}),  h_i = λ_{t_i}−λ_{t_{i-1}}.
   (used ∫_{λ_s}^{λ_t} e^{−λ}dλ = e^{−λ_s}−e^{−λ_t}; α_t(e^{−λ_s}−e^{−λ_t}) = σ_t(e^{h}−1) since e^{−λ_s}=e^{−λ_t}e^{h}, α_t e^{−λ_t}=σ_t.)
7. **DDIM = DPM-Solver-1**: in DDIM use σ_{t_{i-1}}/α_{t_{i-1}}=e^{−λ_{t_{i-1}}}, σ_{t_i}/α_{t_i}=e^{−λ_{t_i}}:
   −α_{t_i}(e^{−λ_{t_{i-1}}} − e^{−λ_{t_i}}) = −α_{t_i} e^{−λ_{t_i}}(e^{h_i}−1) = −σ_{t_i}(e^{h_i}−1). Identical. DDIM
   was implicitly exploiting the semi-linearity (exact linear part) — explains its edge over plain Euler.
8. **DPM-Solver-2** (k=2): needs an intermediate point at λ_s + r₁h (r₁=1/2 default). With Δ = ε_θ(u,s_i)−ε_θ(x̃,t_{i-1}):
   u_i = (α_{s_i}/α_{t_{i-1}})x̃ − σ_{s_i}(e^{r₁h_i}−1)ε_θ(x̃,t_{i-1});
   x̃_{t_i} = (α_{t_i}/α_{t_{i-1}})x̃ − σ_{t_i}(e^{h_i}−1)ε_θ(x̃,t_{i-1}) − (σ_{t_i}/(2r₁))(e^{h_i}−1)Δ.
   (r₁=1/2 → coefficient (1/(2r₁))=1.) Order condition: h²φ₂(h) − (e^h−1)(r₁h)/(2r₁) = (2e^h−h−2−he^h)/2 = O(h³).
9. **DPM-Solver-3** (k=3): r₁=1/3, r₂=2/3. Two intermediate points; uses φ_22-like correction
   ((e^{r₂h}−1)/(r₂h) − 1) and ((e^h−1)/h − 1). See Alg 3.
10. **Order theorem**: DPM-Solver-k is k-th order: x̃_{t_M} − x_0 = O(h_max^k). Assumptions: total derivs of ε̂
    continuous to order k+1; ε_θ Lipschitz in x; h_max = O(1/M).
11. **vs expRK**: same φ-function technique on the *same* integral with α=1, but expRK's linear factor is e^{αh}x_t
    whereas DPM-Solver's is (α_{t+h}/α_t)x_t — customized to diffusion ODE. Note the integral is exactly the
    exponential-integrator form with constant "1".

## Design decisions → why
- **Solve ODE not SDE**: SDE step limited by Wiener randomness; ODE deterministic → larger steps. Why ODE.
- **Variation of constants (exact linear part)**: black-box RK discretizes the linear term whose exact
  solution is exponential → error can blow up; solving it exactly removes that error entirely.
- **λ = half-log-SNR as integration variable**: (a) collapses f,g into the analytic factor e^{−λ}; (b) λ is
  monotone in t so the change of variable is valid and invertible (t_λ); (c) makes the solution invariant to
  the noise schedule — only λ_s, λ_t and ε̂ matter. So nearly all schedule complexity becomes analytic.
- **Taylor in λ (not t)**: because the integral is naturally in λ and the only remaining unknown is ε̂(λ).
- **expm1(h)** instead of exp(h)−1: numerical stability for small h (kingma2021variational).
- **Uniform steps in λ** (not in t): the solution is invariant to the schedule between λ_s,λ_t; uniform-λ is a
  natural simple choice and empirically good.
- **r₁=1/2 (order2), r₁=1/3,r₂=2/3 (order3)**: satisfy the stiff order conditions for the φ-expansion.
- **k ≤ 3 only**: k≥4 exponential RK needs many more intermediate stages.
- **"DPM-Solver-fast" combination**: to use a fixed NFE budget K, do ⌊K/3⌋(-ish) order-3 steps then a 2 or 1 step
  to fill the remainder exactly.
- **Discrete-time DPMs**: wrap discrete model as continuous via t→ scaled index; smooth time embeddings make
  non-integer inputs OK.
- **No final denoising step**: solve T→ε only, performs well enough.
- **Classifier guidance**: ε_θ(x,t,y) := ε_θ(x,t) − s·σ_t ∇_x log p_t(y|x); plug into same solver.

## Canonical code structure (LuChengTHU/dpm-solver)
- `NoiseScheduleVP`: marginal_log_mean_coeff (log α_t), marginal_alpha, marginal_std (σ_t=√(1−α_t²)),
  marginal_lambda (λ=log α − log σ), inverse_lambda (t_λ). linear & discrete schedules.
- `model_wrapper`: converts noise/x_start/v/score models + guidance to a unified noise-pred model_fn(x,t).
- `DPM_Solver`: get_time_steps (logSNR uniform / time_uniform / time_quadratic), dpm_solver_first_update,
  singlestep_dpm_solver_second_update (r1=0.5), singlestep_dpm_solver_third_update (r1=1/3,r2=2/3),
  get_orders_and_timesteps_for_singlestep_solver (fast combination), sample.
  Original solver = `dpmsolver` branch (x/ε space, phi_1=expm1(h)); dpmsolver++ is a later data-prediction variant — use the original.
