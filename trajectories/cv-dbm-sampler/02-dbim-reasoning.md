The reference run told me what I needed and it told me in numbers that are almost embarrassing for the
floor. DDBM at 50 NFE landed FID 11.139 on Edges→Handbags, 10.556 on ImageNet inpainting, 15.811 on
DIODE — real images, the framework and the trained model are sound — but it spent **2149 seconds** on
DIODE to do it, against the ≈270 seconds the budgeted samplers are allotted. And the absolute numbers
are not even good: 11 FID on a 64×64 translation is mediocre. So the diagnosis is sharp and it is *not*
"the model is weak." It is "I am paying for genericity." DDBM treats the reverse bridge dynamics as one
opaque vector field and takes many small Heun-plus-churn steps because that is all a black-box solver
can safely do; it pays small-step discretization error on every part of the drift, including the linear
part, and it needs the churn to recover diversity. Fifty calls is the price of that ignorance. The
whole point of the task is that five calls have to get *closer* to the truth than fifty did — which
means I have to stop discretizing blindly and start exploiting the one thing the black-box solver
ignored: the bridge has a known analytic structure, and the trained network only cares about its
marginals.

That last clause is the lever, and it is worth stating precisely because everything rests on it. The
denoising-bridge-score-matching loss the network was trained with depends on the model *only through
the per-time marginals* `q(x_t | x_0, x_T)`, never through the full joint over the trajectory. The
network never saw a trajectory; it saw `(x_0, x_T, t)` triples, sampled `x_t` from the marginal kernel,
and learned to invert that one marginal. So any inference process I write down that *agrees with these
same marginals* is a process the network is already optimal for. The training never committed me to the
particular joint that DDBM's reverse SDE happens to define. I have complete freedom in the joint as
long as I respect the marginals — and a different joint can take far bigger steps. This is exactly the
move that made fast sampling work for ordinary diffusion: the DDPM loss only sees `q(x_t | x_0)`, so one
is free to replace the Markovian forward chain with a whole family of non-Markovian inference processes
that share the same marginal, each engineered so the reverse step is "predicted clean data, plus a
direction pointing back toward the current state, plus fresh noise," collapsing at zero noise to a
deterministic implicit map that takes big jumps. I want to build the bridge analogue of that family.

But I cannot copy the diffusion formula, because the diffusion construction is welded to its single
Gaussian endpoint — mean `√α_t x_0`, variance `1 − α_t`, no second endpoint — and my bridge has two:
mean `a_t x_T + b_t x_0`, with that extra `a_t x_T` term, and noise scale `c_t`, not `1 − α_t`. The
diffusion update has nowhere to put `x_T` and nothing that matches the `c_t → 0` pin at the source. So I
redo the construction for the bridge kernel and *insist* it preserve the bridge marginals. Set up the
sampling timesteps `0 = t_0 < t_1 < … < t_{N-1} < t_N = t_max`, and posit a family of reverse
conditionals indexed by a per-step injected standard deviation `ρ_n`, factorized backward in time the
way I will actually sample — each `x_{t_n}` conditioned on the next-later state `x_{t_{n+1}}` and on
`x_0`. The conditional I try is a Gaussian whose mean is the bridge mean at `t_n` plus a deterministic
direction, and whose injected variance is `ρ_n²`: the mean is
`a_{t_n} x_T + b_{t_n} x_0 + √(c_{t_n}² − ρ_n²) · (x_{t_{n+1}} − a_{t_{n+1}} x_T − b_{t_{n+1}} x_0)/c_{t_{n+1}}`.
Read that last factor: `(x_{t_{n+1}} − a_{t_{n+1}} x_T − b_{t_{n+1}} x_0)/c_{t_{n+1}}` is precisely the
standardized Gaussian `ε̂` that generated `x_{t_{n+1}}`. So I am recycling a fraction of the later
step's realized noise as a deterministic "direction," scaled by `√(c_{t_n}² − ρ_n²)`, and adding fresh
noise of variance `ρ_n²`. The total noise at `t_n` is `(c_{t_n}² − ρ_n²) + ρ_n² = c_{t_n}²`, which is
the marginal variance the network expects. The intuition is DDIM's, but every coefficient is the
bridge's.

I have to *prove* the marginals are preserved, not hope, because the licence to reuse the network rests
entirely on it. Claim: this family's marginal `q^ρ(x_{t_n} | x_0, x_T)` equals the bridge kernel
`N(a_{t_n} x_T + b_{t_n} x_0, c_{t_n}² I)` for every `n` and every admissible `ρ`. Induct downward from
`n = N-1`. The base case needs the boundary `ρ_{N-1} = c_{t_{N-1}}`: at `n = N-1`, `t_{n+1} = t_max`,
the recycled coefficient `√(c² − ρ²) = √(c_{t_{N-1}}² − c_{t_{N-1}}²) = 0` kills the borrowed-noise
term, and the conditional collapses to exactly the bridge kernel. So the boundary value is *forced*, not
chosen. Inductive step: assume it holds at `n = k`; then `q^ρ(x_{t_{k-1}} | x_0, x_T)` is a Gaussian
marginalized over a Gaussian — the standard linear-Gaussian identity, "if `y | x ~ N(Mx + m, L)` and
`x ~ N(μ, Σ)` then `y ~ N(Mμ + m, L + MΣMᵀ)`." Substituting the mean of `x_{t_k}` into the
recycled-direction term, its argument becomes `(a_{t_k} x_T + b_{t_k} x_0) − a_{t_k} x_T − b_{t_k} x_0
= 0` — the deterministic direction averages to zero, because in expectation `x_{t_k}` sits exactly at
its bridge mean, with no leftover noise to point along. The mean collapses to `a_{t_{k-1}} x_T +
b_{t_{k-1}} x_0`. The variance is `ρ_{k-1}² + (√(c_{t_{k-1}}² − ρ_{k-1}²)/c_{t_k})² · c_{t_k}²`; the
`c_{t_k}²` cancels and it sums to `c_{t_{k-1}}²`. Induction closes. Every `ρ` preserves every bridge
marginal, given the forced boundary. The network is the rigorous optimum for the whole family — and the
ELBO confirms it: the variational objective collapses to a weighted sum of data-prediction errors,
which converts via `s = −(x − a x_T − b x_θ)/c²` to the score-matching loss with a different per-time
weighting, and the weighting does not move the minimizer when the network is not parameter-shared across
`t`. No retraining; I just get to choose `ρ`.

So I have a one-parameter dial. The generative step replaces the unknown `x_0` with the network's data
prediction `x̂_0 = denoiser(x_{t_{n+1}}, t_{n+1})`, and I parameterize the per-step noise by a scalar
`η ∈ [0, 1]`: `ρ_n = η · σ_{t_n} √(1 − SNR_{t_{n+1}}/SNR_{t_n})`. At `η = 1` the induced forward
process becomes Markovian — the `x_T` term cancels and the update reduces to a DDPM-like ancestral
sampler. At `η = 0` there is no fresh noise, the update is a deterministic implicit map, and the process
is maximally non-Markovian — the bridge analogue of DDIM, the one I expect to take clean sharp jumps in
few steps. I expose the dial because the two ends genuinely trade off: the deterministic map is sharp
and invertible, ideal when source and target are tightly correlated; injected noise behaves like a
Langevin correction that washes out accumulated discretization error and helps on diverse tasks. There
is no universally best stochasticity, so it is a knob. One implementation note matters for grounding:
in the harness, the variables `get_alpha_rho` names `rho` are the schedule's `σ_t/α_t`, *not* my
injected `ρ_n` — so the injected std is `η · (α_t · ρ_t) · √(1 − ρ_t²/ρ_s²)`, with `α_t · ρ_t = σ_t`.

Now the wall DDBM never had to face, and it is sharp. Take the deterministic case and look at the very
first step, where `t_{n+1} = t_max`: the recycled term divides by `c_{t_max}`, which is *zero*, because
the bridge is pinned exactly at `x_T` with no spread. The deterministic first step is singular — and
this is the same fact that forced DDBM to inject noise: under a fixed `x_T`, the state `x_t` for `t <
t_max` is genuinely stochastic, because one source admits many targets, so `p(x_t | x_T)` is not a
Dirac. The math refuses to let me erase the bridge's intrinsic stochasticity at the start. The fix is
already in hand: the boundary `ρ_{N-1} = c_{t_{N-1}}` I needed for the marginal proof is exactly the
Markovian boundary at step one, which zeros the recycled coefficient and annihilates the singular
`c_{t_max}` denominator. What is left is a single injection of fresh Gaussian noise of scale
`c_{t_{N-1}}` — the **booting noise** — which places the initial state on the bridge via
`x = a x_T + b x̂_0 + c · noise` and accounts for the spread of `x_0` given `x_T`. This is where DDBM
spent its many churn injections; I spend exactly one, and I return it as the sixth tuple value, the
latent that controls diversity. The mirror concern is at the end: on the final step, fresh injected
noise would land straight on the output with nothing left to denoise it, blurring the result, so I drop
the fresh-noise term on the last step and keep only the deterministic part — the same reasoning that
makes one take a no-churn final step in diffusion samplers.

That is the full first-order DBIM sampler that fills the transition slot: predict at `t_max` and seed
the first interior state with the booting noise; then a loop that, at each step, evaluates the predictor
once at the current larger time `s`, forms the recycled-direction coefficient `√(c_t² − ρ_n²)/c_s` and
the two endpoint coefficients, and writes the next state as a closed-form linear combination of `x̂_0`,
`x_T`, the current state, and fresh noise (dropped on the final step). One denoiser call per step, so
five calls buy four interior transitions plus the booting prediction — a genuine 5-NFE sampler. For the
inpainting workload I keep the observed pixels fixed each call by re-blending `x̂_0` with `x_T` under
the mask, preserving the `mask` semantics the harness requires. The harness's editable `sample_dbim`
honors the caller's `eta` (default 1.0) and `ts` directly, unlike the reference rung which overrode
them. The full module is in the answer.

So the falsifiable expectations against the floor. First and most important: **at one-tenth the NFE,
DBIM should not just match but *beat* DDBM's FID**, because the analytic large-jump update spends its
five calls on clean transitions instead of fifty calls fighting its own churn. I expect Edges→Handbags
to drop from DDBM's 11.139 into the 5-ish range and ImageNet inpainting from 10.556 into the 6-ish
range — roughly halving the FID while cutting calls by 10×. Second, DIODE is the place I am least sure:
the deterministic-leaning update may struggle there, and I would not be shocked if DBIM lands around
DDBM's DIODE number (≈15) rather than beating it, because DIODE's translation may want more
stochasticity than a near-deterministic five-step run provides. Third, the DIODE wall-clock should
crater to ≈270 seconds, the visible payoff of 5 versus 50 calls. If DBIM beats DDBM on the two easier
workloads but stalls near it on DIODE, that is the signal that first-order Euler is leaving accuracy on
the table at five steps — and the next move is to keep the same marginal-preserving family but
integrate the deterministic ODE to *higher order*, reusing past predictions so the extra accuracy costs
no extra calls.
