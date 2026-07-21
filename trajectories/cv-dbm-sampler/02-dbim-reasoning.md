The reference run told me what I needed and it told me in numbers. DDBM at 50 NFE landed FID 11.139 on
Edges→Handbags, 10.556 on ImageNet inpainting, 15.811 on DIODE — real images, so the framework and
trained model are sound — but it spent 2149 seconds on DIODE to do it. Read those four numbers
properly, because they set both target and budget. The FIDs are mediocre, not good: 11.139 on a 64×64
translation is a number a well-tuned bridge should beat, so the floor is genuinely low and beating it is
not a stretch. Across workloads, DIODE at 15.811 is `15.811/11.139 = 1.42` times worse than
Edges→Handbags *even at fifty calls where NFE is no object*, so DIODE's difficulty is structural, baked
into the task — a fact to hold onto, because if a cheap sampler cracks the two easy workloads but stalls
on DIODE, this ratio says the stall is DIODE being intrinsically hard, not the sampler being broken. And
the wall-clock hands me my own budget: `2149/50 ≈ 43` seconds per denoiser call, and at 256×256 the
U-Net forward passes dominate, so a five-call sampler on DIODE should cost roughly `5 × 43 ≈ 215`
seconds of compute plus fixed overhead — I can predict my own DIODE wall-clock in the low-few-hundreds
before running anything, and coming in there confirms I am spending calls, not hiding compute elsewhere.

So the diagnosis is sharp and it is *not* "the model is weak." It is "I am paying for genericity." DDBM
takes many small Heun-plus-churn steps because that is all a black-box solver can safely do; it pays
small-step discretization error on every part of the drift, including the linear part it could integrate
exactly. Fifty calls is the price of that ignorance, and the task demands that five calls get *closer*
to the truth than fifty did — which means exploiting the one thing the black-box solver ignored: the
bridge has a known analytic structure, and the trained network only cares about its marginals.

First kill the tempting shortcut, because if it worked I would not need a new sampler. Run DDBM's own
sampler at five calls instead of fifty: each churn-Heun iteration costs three calls and the terminal
Euler costs two, so five calls is one Heun iteration plus one terminal iteration — two grid intervals
covering the whole trajectory. Each step is then `Δt ~ 0.5` of the range, and Heun's global error scales
like `Δt²`: at two steps that is `~0.25`, against the reference's `(1/17)² ≈ 0.0035` at seventeen steps.
The quality the reference bought by keeping steps small evaporates the instant I shrink the step count,
because the black-box solver has no way to take a big step accurately. So "run the floor cheaper" is a
non-starter; the five-call regime demands a transition that is *analytically exact for large jumps*. The
pure deterministic bridge ODE I already ruled out on the blur argument (and it will fail at the first
step for a singular reason I get to below). The surviving option is a bridge-specific large-jump update
with a stochasticity dial, and I have to build it.

The lever is the marginals-only property. The denoising-bridge-score-matching loss depends on the model
*only through the per-time marginals* `q(x_t | x_0, x_T)`, never through the full joint. The network
never saw a trajectory; it saw `(x_0, x_T, t)` triples, sampled a single `x_t` from the marginal kernel,
and learned to invert that one marginal. So any inference process that *agrees with these same
marginals* is one the network is already optimal for. The training never committed me to the joint that
DDBM's reverse SDE happens to define; I have complete freedom in the joint as long as I respect the
marginals — and a different joint can take far bigger steps. This is the move that made fast sampling
work for ordinary diffusion: the DDPM loss only sees `q(x_t | x_0)`, so one is free to replace the
Markovian forward chain with a family of non-Markovian processes sharing the same marginal, each with a
reverse step of "predicted clean data, plus a direction pointing back toward the current state, plus
fresh noise," collapsing at zero noise to a deterministic implicit map that jumps. I want the bridge
analogue.

I cannot copy the diffusion formula, because that construction is welded to its single Gaussian endpoint
— mean `√α_t x_0`, variance `1 − α_t`, no second endpoint — and my bridge has two: mean
`a_t x_T + b_t x_0` with the extra `a_t x_T`, and noise scale `c_t`. So I redo the construction for the
bridge kernel and insist it preserve the bridge marginals. Set timesteps `0 = t_0 < … < t_N = t_max`,
and posit reverse conditionals indexed by a per-step injected std `ρ_n`, factorized backward in time —
each `x_{t_n}` conditioned on the next-later state `x_{t_{n+1}}` and on `x_0`. The conditional is a
Gaussian whose mean is the bridge mean at `t_n` plus a deterministic direction, with injected variance
`ρ_n²`:
`a_{t_n} x_T + b_{t_n} x_0 + √(c_{t_n}² − ρ_n²) · (x_{t_{n+1}} − a_{t_{n+1}} x_T − b_{t_{n+1}} x_0)/c_{t_{n+1}}`.
That last factor is precisely the standardized noise `ε̂` that generated `x_{t_{n+1}}`, so I am
recycling a fraction of the later step's realized noise as a deterministic "direction," scaled by
`√(c_{t_n}² − ρ_n²)`, and adding fresh noise of variance `ρ_n²`. The total noise at `t_n` is
`(c_{t_n}² − ρ_n²) + ρ_n² = c_{t_n}²`, the marginal variance the network expects. The intuition is
DDIM's, every coefficient the bridge's. One domain check that matters later: `√(c_{t_n}² − ρ_n²)` is
real only if `ρ_n² ≤ c_{t_n}²`, so `ρ_n = c_{t_n}` is the ceiling — a structural cap on how much fresh
noise this coupled construction can carry, and I will bump into it hard much later.

I have to *prove* the marginals are preserved, because the licence to reuse the network rests on it.
Claim: `q^ρ(x_{t_n} | x_0, x_T) = N(a_{t_n} x_T + b_{t_n} x_0, c_{t_n}² I)` for every `n` and every
admissible `ρ`. Induct downward from `n = N-1`. The base case forces the boundary `ρ_{N-1} =
c_{t_{N-1}}`: at `n = N-1`, `t_{n+1} = t_max`, so `√(c² − ρ²) = 0` kills the borrowed-noise term and the
conditional collapses to the bridge kernel. So the boundary value is *forced*, not chosen — the same
`ρ = c` ceiling, sitting at the top of the schedule. Inductive step: with the linear-Gaussian identity,
substituting the mean of `x_{t_k}` into the recycled-direction term makes its argument
`(a_{t_k} x_T + b_{t_k} x_0) − a_{t_k} x_T − b_{t_k} x_0 = 0` — the direction averages to zero because
in expectation `x_{t_k}` sits at its bridge mean — and the variance sums to `c_{t_{k-1}}²`. Induction
closes: every `ρ` preserves every bridge marginal given the forced boundary, and the ELBO agrees (it
reduces to score-matching with a per-time reweighting that does not move the minimizer when the network
is not parameter-shared across `t`). No retraining; I just get to choose `ρ`.

So I have a one-parameter dial. The step replaces `x_0` with the network's `x̂_0 = denoiser(x_{t_{n+1}},
t_{n+1})`, and I parameterize the per-step noise by a scalar `η ∈ [0, 1]`:
`ρ_n = η · σ_{t_n} √(1 − SNR_{t_{n+1}}/SNR_{t_n})`. `η` is the split of the fixed marginal variance `c²`
between two jobs. At `η = 0` all of `c²` goes into the recycled deterministic direction — a deterministic
implicit map, maximally non-Markovian, the bridge analogue of DDIM that should take clean sharp jumps in
few steps. At `η = 1` the injected variance is maximal, the `x_T` term cancels, and the update reduces to
a DDPM-like ancestral sampler. The two ends genuinely trade off — the deterministic map is sharp and
invertible, ideal when source and target are tightly correlated; injected noise behaves like a Langevin
correction that washes out accumulated discretization error and helps on diverse tasks — so there is no
universally best stochasticity and I hand the caller the knob. One grounding note: in the harness the
variables `get_alpha_rho` names `rho` are the schedule's `σ_t/α_t`, *not* my injected `ρ_n`, so the
injected std is `η · (α_t · ρ_t) · √(1 − ρ_t²/ρ_s²)` with `α_t · ρ_t = σ_t`, and since the smaller
time carries the larger SNR that square root stays real across the interior.

Now the wall DDBM never had to face. Take the deterministic case at the very first step, `t_{n+1} =
t_max`: the recycled term divides by `c_{t_max} = 0`, because the bridge is pinned exactly at `x_T` with
no spread. The deterministic first step is singular — the same fact that forced DDBM to inject noise:
under a fixed `x_T`, `x_t` for `t < t_max` is genuinely stochastic because one source admits many
targets, so `p(x_t | x_T)` is not a Dirac. The math refuses to let me erase the bridge's intrinsic
stochasticity at the start. The fix is already in hand: the forced boundary `ρ_{N-1} = c_{t_{N-1}}`
zeros the recycled coefficient and annihilates the singular `c_{t_max}` denominator — the indeterminate
`0 · (1/0)` resolves to a finite update. What is left is a single injection of fresh Gaussian noise of
scale `c_{t_{N-1}}` — the **booting noise** — placing the initial state on the bridge via
`x = a x_T + b x̂_0 + c · noise` and accounting for the spread of `x_0` given `x_T`. This is where DDBM
spent its many churn injections and its crude `t_max − 1e-4` offset; I spend one deliberate draw at the
endpoint and return it as the sixth tuple value, the diversity latent — the "one principled shot instead
of many churned ones" I flagged as wasted in the reference. Mirror concern at the end: on the final step
fresh injected noise would land straight on the output with nothing left to denoise it, blurring the
result, so I drop the fresh-noise term on the last step and keep only the deterministic part — the same
reasoning that makes one take a no-churn final step in diffusion samplers.

In the harness's coefficients, writing `tmp_var = √(c_t² − ρ_n²)/c_s` for the recycled fraction (`s` the
current larger time, `t` the next smaller), the next state is
`x = coeff_x0_hat · x̂_0 + coeff_xT · x_T + coeff_xs · x + ρ_n · noise` with `coeff_xs = tmp_var`,
`coeff_x0_hat = b_t − tmp_var · b_s`, `coeff_xT = a_t − tmp_var · a_s`. Each endpoint coefficient is the
*target*-time bridge weight minus the recycled fraction of the *source*-time weight — the algebra of
"strip out the bridge mean the state should have at time `s`, keep `tmp_var` of the leftover as a
direction, re-attach the bridge mean it should have at `t`." Nothing here is a small step along a vector
field; it is a *re-projection* of the state from one time's marginal onto the next, in one shot, which is
why it can jump. At `η = 0`, `tmp_var = c_t/c_s`, and the parenthesized residual is `c_s` times the
standardized noise that made `x_{t_{n+1}}`, so multiplying by `c_t/c_s` rescales that same realized noise
to the new marginal scale and adds it to the new bridge mean — the state stays exactly one standard draw
off the mean, at the correct scale, with no fresh randomness. Only the single boot draw seeds the
randomness; every interior `η = 0` step is a rescaling of it. That is why one denoiser call per step
suffices: the predictor is evaluated once at `s`, reused in all three coefficients, with no corrector
evaluation to pay for, unlike Heun. At `η = 1` the recycled coefficient hits its boundary, the
dependence on the *specific* realized `x_{t_{n+1}}` drops out, and the step becomes a memoryless
ancestral transition — the same two ends of the dial, now visible in the closed form, both licensed on
the frozen network by the marginal proof.

That is the full first-order DBIM sampler that fills the transition slot: predict at `t_max`, seed the
first interior state with the booting noise; then a loop that evaluates the predictor once at the current
`s`, forms the recycled-direction coefficient and the two endpoint coefficients, and writes the next
state as a closed-form combination of `x̂_0`, `x_T`, the current state, and fresh noise (dropped on the
final step). One call per step, so five calls buy the booting prediction plus four interior transitions
— a genuine 5-NFE sampler. For inpainting the mask handling has to be exact: the observed region is data
I already have, and the network must not hallucinate over it. So on every predictor call I re-blend
`x̂_0 ← x̂_0 · mask + x_T · (1 − mask)` — the network's completion inside the unknown region, the
observed `x_T` outside — and because that blended `x̂_0` flows through the same update, the known pixels
are re-anchored to the observation at every step rather than drifting. One line on the prediction, and
the `mask` semantics fall out with no special case. The full module is in the answer.

I default the dial to the caller's `eta = 1.0`, the fully Markovian maximal-noise end, defensible
precisely because I do not know in advance which regime a given workload wants. Too little noise gives
the blur I already argued from a pinned start — a near-deterministic run collapses toward the
conditional mean and smears multimodal targets. Too much noise is milder — a slightly rougher image
that still samples the right distribution, since the marginal proof guarantees `η = 1` is as valid as
`η = 0`. Between a knob that can *blur* and one that can *roughen*, defaulting to roughening is the safe
choice for a general translation task, though I expect tightly-correlated workloads would prefer it
turned down. The editable body honors the caller's `eta` and `ts` directly, unlike the reference.

So the expectations against the floor's specific numbers. First and most important: at one-tenth the NFE,
DBIM should not just match but *beat* DDBM's FID, because the analytic large-jump update spends five
clean transitions instead of fifty calls fighting its own churn — I expect the tightly-correlated
workloads (Edges→Handbags, ImageNet inpainting) to fall substantially, plausibly into single digits.
DIODE is where I am least sure, and the `1.42` structural-hardness ratio is why: a deterministic-leaning
update may struggle where the targets are less correlated with the source, so DBIM may barely move DIODE
off the floor if that workload wants more stochasticity than the default `η` provides at five steps. The
DIODE wall-clock should crater to the low-few-hundred seconds I back-computed from 43 seconds per call.
If DBIM beats DDBM on the two easier workloads but stalls near it on DIODE, that points at first-order
Euler leaving accuracy on the table at five steps — the same marginal-preserving family and booting
noise would then want the deterministic ODE integrated to higher order, reusing past predictions so the
accuracy costs no extra calls.
