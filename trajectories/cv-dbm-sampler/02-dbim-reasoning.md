The reference run told me what I needed and it told me in numbers that are almost embarrassing for the
floor. DDBM at 50 NFE landed FID 11.139 on Edges→Handbags, 10.556 on ImageNet inpainting, 15.811 on
DIODE — real images, so the framework and the trained model are sound — but it spent 2149 seconds on
DIODE to do it. Before I even think about a new sampler I want to read those four numbers properly,
because they set both the target and the budget. First the FIDs themselves: 11.139 on a 64×64
translation is mediocre, not good — a well-tuned bridge should be doing far better than that — so the
floor is genuinely low, and beating it is not a stretch goal. Look across workloads: DIODE at 15.811 is
the worst, `15.811 / 11.139 = 1.42`, so DIODE is 42% harder than Edges→Handbags *even at fifty calls
where NFE is no object*. That tells me DIODE's difficulty is structural, baked into the task, not merely
a budget artifact — a fact I should hold onto, because if a cheap sampler cracks the two easy workloads
but stalls on DIODE, this ratio says the stall is DIODE being intrinsically hard, not the sampler being
broken. Now the wall-clock, which quietly hands me my own budget. 2149 seconds across 50 denoiser calls
is `2149 / 50 ≈ 43` seconds per call; if the U-Net forward passes dominate the runtime — and at 256×256
they do — then a five-call sampler on DIODE should cost roughly `5 × 43 ≈ 215` seconds of compute plus
some fixed overhead for data and FID bookkeeping, landing in the low-few-hundreds. So I can predict my
own DIODE wall-clock at around 250–270 seconds before I have run anything, and if it comes in there that
confirms the 5-vs-50 story and that I am spending calls, not hiding compute elsewhere.

So the diagnosis is sharp and it is *not* "the model is weak." It is "I am paying for genericity." DDBM
treats the reverse bridge dynamics as one opaque vector field and takes many small Heun-plus-churn steps
because that is all a black-box solver can safely do; it pays small-step discretization error on every
part of the drift, including the linear part it could integrate exactly, and it needs the churn to
recover diversity it never scheduled. Fifty calls is the price of that ignorance. The whole point of the
task is that five calls have to get *closer* to the truth than fifty did — which means I have to stop
discretizing blindly and start exploiting the one thing the black-box solver ignored: the bridge has a
known analytic structure, and the trained network only cares about its marginals.

Before I build on that, let me kill the tempting shortcut, because if it worked I would not need a new
sampler at all. The shortcut is: just run DDBM's own sampler at five calls instead of fifty. Count what
that buys. Each churn-Heun iteration costs three calls and the terminal Euler costs two, so five calls
is exactly one Heun iteration plus one terminal iteration — two grid intervals covering the entire
trajectory from `t_max` to `0`. Two intervals means each step is enormous, `Δt ~ 0.5` of the whole
range, and Heun's local error scales like `Δt³` while its global error scales like `Δt²`: at two steps
that global error is `~0.25`, against the reference's `(1/17)² ≈ 0.0035` at seventeen steps — roughly
seventy times larger per unit, on a tenth as many steps. The quality the reference bought by keeping
steps small evaporates the instant I shrink the step count, because the black-box solver has no way to
take a big step accurately. So "run the floor cheaper" is a non-starter; the five-call regime demands a
transition that is *analytically exact for large jumps*, not a fine-grained discretizer starved of
steps. That eliminates option one. Option two — a pure deterministic bridge ODE from the pinned start —
I already know fails on the blur argument from the last rung, and I will find it also fails at the very
first step for a singular reason, which I will get to. So the surviving option is a bridge-specific
large-jump update with a stochasticity dial, and I have to build it.

The lever is the marginals-only property, and it is worth stating precisely because everything rests on
it. The denoising-bridge-score-matching loss the network was trained with depends on the model *only
through the per-time marginals* `q(x_t | x_0, x_T)`, never through the full joint over the trajectory.
The network never saw a trajectory; it saw `(x_0, x_T, t)` triples, sampled a single `x_t` from the
marginal kernel, and learned to invert that one marginal. So any inference process I write down that
*agrees with these same marginals* is a process the network is already optimal for. The training never
committed me to the particular joint that DDBM's reverse SDE happens to define; I have complete freedom
in the joint as long as I respect the marginals — and a different joint can take far bigger steps. This
is exactly the move that made fast sampling work for ordinary diffusion: the DDPM loss only sees
`q(x_t | x_0)`, so one is free to replace the Markovian forward chain with a whole family of
non-Markovian inference processes that share the same marginal, each engineered so the reverse step is
"predicted clean data, plus a direction pointing back toward the current state, plus fresh noise,"
collapsing at zero noise to a deterministic implicit map that takes big jumps. I want to build the bridge
analogue of that family.

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
bridge's. One immediate domain check: the recycled coefficient `√(c_{t_n}² − ρ_n²)` is real only if
`ρ_n² ≤ c_{t_n}²`, so the family is defined only up to injected noise equal to the full marginal scale —
`ρ_n = c_{t_n}` is the ceiling. That ceiling is not a nuisance; it is a structural cap on how much fresh
noise this coupled construction can carry, and I will bump into it hard much later. For now it is exactly
the constraint that forces the boundary value I need for the proof.

I have to *prove* the marginals are preserved, not hope, because the licence to reuse the network rests
entirely on it. Claim: this family's marginal `q^ρ(x_{t_n} | x_0, x_T)` equals the bridge kernel
`N(a_{t_n} x_T + b_{t_n} x_0, c_{t_n}² I)` for every `n` and every admissible `ρ`. Induct downward from
`n = N-1`. The base case needs the boundary `ρ_{N-1} = c_{t_{N-1}}`: at `n = N-1`, `t_{n+1} = t_max`,
the recycled coefficient `√(c² − ρ²) = √(c_{t_{N-1}}² − c_{t_{N-1}}²) = 0` kills the borrowed-noise
term, and the conditional collapses to exactly the bridge kernel. So the boundary value is *forced*, not
chosen — it is the same `ρ = c` ceiling I just found, sitting at the top of the schedule. Inductive step:
assume it holds at `n = k`; then `q^ρ(x_{t_{k-1}} | x_0, x_T)` is a Gaussian marginalized over a
Gaussian — the standard linear-Gaussian identity, "if `y | x ~ N(Mx + m, L)` and `x ~ N(μ, Σ)` then
`y ~ N(Mμ + m, L + MΣMᵀ)`." Substituting the mean of `x_{t_k}` into the recycled-direction term, its
argument becomes `(a_{t_k} x_T + b_{t_k} x_0) − a_{t_k} x_T − b_{t_k} x_0 = 0` — the deterministic
direction averages to zero, because in expectation `x_{t_k}` sits exactly at its bridge mean, with no
leftover noise to point along. The mean collapses to `a_{t_{k-1}} x_T + b_{t_{k-1}} x_0`. The variance is
`ρ_{k-1}² + (√(c_{t_{k-1}}² − ρ_{k-1}²)/c_{t_k})² · c_{t_k}²`; the `c_{t_k}²` cancels and the whole thing
sums to `(c_{t_{k-1}}² − ρ_{k-1}²) + ρ_{k-1}² = c_{t_{k-1}}²`. Induction closes. Every `ρ` preserves
every bridge marginal, given the forced boundary. The network is the rigorous optimum for the whole
family — and the ELBO confirms it: the variational objective collapses to a weighted sum of
data-prediction errors, which converts via `s = −(x − a x_T − b x_θ)/c²` to the score-matching loss with
a different per-time weighting, and the weighting does not move the minimizer when the network is not
parameter-shared across `t`. No retraining; I just get to choose `ρ`.

So I have a one-parameter dial. The generative step replaces the unknown `x_0` with the network's data
prediction `x̂_0 = denoiser(x_{t_{n+1}}, t_{n+1})`, and I parameterize the per-step noise by a scalar
`η ∈ [0, 1]`: `ρ_n = η · σ_{t_n} √(1 − SNR_{t_{n+1}}/SNR_{t_n})`. The `η` is not a mood; it is the split
of the fixed marginal variance `c²` between two jobs. At `η = 0` all of `c²` goes into the recycled
deterministic direction — the update is a deterministic implicit map, the process is maximally
non-Markovian, and this is the bridge analogue of DDIM that I expect to take clean sharp jumps in few
steps. At `η = 1` the injected variance is maximal, the induced forward process becomes Markovian — the
`x_T` term cancels and the update reduces to a DDPM-like ancestral sampler. I expose the dial because the
two ends genuinely trade off: the deterministic map is sharp and invertible, ideal when source and
target are tightly correlated; injected noise behaves like a Langevin correction that washes out
accumulated discretization error and helps on diverse, multimodal tasks. There is no universally best
stochasticity, so it is a knob and I hand the caller control of it. One implementation note matters for
grounding: in the harness, the variables `get_alpha_rho` names `rho` are the schedule's `σ_t/α_t`, *not*
my injected `ρ_n` — so the injected std is `η · (α_t · ρ_t) · √(1 − ρ_t²/ρ_s²)`, with `α_t · ρ_t = σ_t`.
And I should check that square root is real on my grid: `ρ_t = σ_t/α_t = 1/√SNR_t`, so
`ρ_t²/ρ_s² = SNR_s/SNR_t`, and since the smaller time `t` has the *larger* SNR, `SNR_s/SNR_t < 1` and
the argument `1 − SNR_s/SNR_t` stays in `(0, 1)`. The dial is well-defined everywhere in the interior.

Now the wall DDBM never had to face, and it is sharp. Take the deterministic case and look at the very
first step, where `t_{n+1} = t_max`: the recycled term divides by `c_{t_max}`, which is *zero*, because
the bridge is pinned exactly at `x_T` with no spread. The deterministic first step is singular — and
this is the same fact that forced DDBM to inject noise: under a fixed `x_T`, the state `x_t` for
`t < t_max` is genuinely stochastic, because one source admits many targets, so `p(x_t | x_T)` is not a
Dirac. The math refuses to let me erase the bridge's intrinsic stochasticity at the start. The fix is
already in hand: the boundary `ρ_{N-1} = c_{t_{N-1}}` I needed for the marginal proof is exactly the
Markovian boundary at step one, which zeros the recycled coefficient and annihilates the singular
`c_{t_max}` denominator — the indeterminate `0 · (1/0)` resolves, through the forced boundary, to a
finite update. What is left is a single injection of fresh Gaussian noise of scale `c_{t_{N-1}}` — the
**booting noise** — which places the initial state on the bridge via `x = a x_T + b x̂_0 + c · noise`
and accounts for the spread of `x_0` given `x_T`. This is where DDBM spent its many churn injections and
its crude `t_max − 1e-4` grid offset; I spend exactly one deliberate draw at the endpoint, and I return
it as the sixth tuple value, the latent that controls diversity — which is precisely the "one principled
shot instead of many churned ones" I filed away as wasted in the reference. The mirror concern is at the
end: on the final step, fresh injected noise would land straight on the output with nothing left to
denoise it, blurring the result, so I drop the fresh-noise term on the last step and keep only the
deterministic part — the same reasoning that makes one take a no-churn final step in diffusion samplers.

It is worth reading the closed-form update once in the harness's own coefficients, because the shape of
it is the whole payoff and I want to see it land. Writing `tmp_var = √(c_t² − ρ_n²)/c_s` for the
recycled fraction (with `s` the current larger time, `t` the next smaller one), the next state is
`x = coeff_x0_hat · x̂_0 + coeff_xT · x_T + coeff_xs · x + ρ_n · noise`, where `coeff_xs = tmp_var`,
`coeff_x0_hat = b_t − tmp_var · b_s`, and `coeff_xT = a_t − tmp_var · a_s`. Each endpoint coefficient
reads the same way: it is the *target*-time bridge weight minus the recycled fraction of the *source*-
time bridge weight — `b_t − tmp_var · b_s` for the clean estimate, `a_t − tmp_var · a_s` for the source.
That is exactly the algebra of "take the current state, strip out the bridge mean it should have at
time `s`, keep `tmp_var` of the leftover as a direction, and re-attach the bridge mean it should have at
the new time `t`." Nothing here is a small step along a vector field; it is a *re-projection* of the
state from one time's marginal onto the next, in one shot, and that is why it can jump.

Set `η = 0` and check the deterministic limit explicitly, because it is the cleanest member of the family
and the one whose accuracy I would most want to push further if first order turns out to leave error on
the table. Then `ρ_n = 0`, `tmp_var = c_t/c_s`, and the update becomes
`x_{t_n} = a_{t_n} x_T + b_{t_n} x̂_0 + (c_{t_n}/c_{t_{n+1}})·(x_{t_{n+1}} − a_{t_{n+1}} x_T − b_{t_{n+1}} x̂_0)`.
I can sanity-check that this respects the marginal by construction: the parenthesized residual is
`c_{t_{n+1}}` times the standardized noise `ε̂` that made `x_{t_{n+1}}`, so multiplying by
`c_{t_n}/c_{t_{n+1}}` rescales that same realized noise from the old marginal scale to the new one,
`c_{t_n} ε̂`, and adds it to the new bridge mean — the state stays exactly one standard draw off the
mean, at the correct scale, with no fresh randomness. It is a deterministic map, but it is the *right*
deterministic map, the one whose pushforward is the bridge marginal. Only the single boot draw seeds the
randomness; every interior step at `η = 0` is a rescaling of that one draw. That is the cleanest possible
statement of what "implicit bridge model" means, and it is why one denoiser call per step suffices — the
predictor is evaluated once at `s`, reused in all three coefficients, and there is no corrector
evaluation to pay for, unlike Heun.

The other end of the dial deserves the same explicit check, because "Markovian at `η = 1`" is a claim I
want to trust and not just assert. At `η = 1` the injected `ρ_n` takes its maximal admissible value, so
`√(c_t² − ρ_n²)` shrinks toward its smallest — the recycled deterministic direction carries the *least*
weight, and the fresh noise the *most*. When the recycled coefficient is at that boundary the dependence
of `x_{t_n}` on the *specific* realized `x_{t_{n+1}}` (beyond its bridge mean) drops out, leaving a step
that depends only on `x̂_0`, `x_T`, and a fresh draw — which is exactly the structure of an ancestral,
Markovian transition: the past enters only through the current prediction, not through the accumulated
noise history. So the two ends of the dial are genuinely different *processes* sharing one marginal:
`η = 0` is a fully non-Markovian implicit map that threads a single noise draw through the whole
trajectory, and `η = 1` is a memoryless ancestral chain that re-randomizes each step. Everything in
between is a valid interpolation, and the marginal proof is what licenses calling any of them by the same
frozen network. That spectrum is the real object I built; the scalar `η` is just where I read it off.

That is the full first-order DBIM sampler that fills the transition slot: predict at `t_max` and seed
the first interior state with the booting noise; then a loop that, at each step, evaluates the predictor
once at the current larger time `s`, forms the recycled-direction coefficient `√(c_t² − ρ_n²)/c_s` and
the two endpoint coefficients, and writes the next state as a closed-form linear combination of `x̂_0`,
`x_T`, the current state, and fresh noise (dropped on the final step). One denoiser call per step, so
five calls buy the booting prediction plus four interior transitions — a genuine 5-NFE sampler, exactly
matching the count I costed against the reference. For the inpainting workload the mask handling has to
be exactly right or the whole thing corrupts: the observed region is data I already have, and the network
must not hallucinate over it. So on every predictor call I re-blend `x̂_0 ← x̂_0 · mask + x_T · (1 − mask)`
— keep the network's completion inside the unknown region, overwrite the prediction with the observed
`x_T` outside it — and because that blended `x̂_0` then flows through the same closed-form update, the
known pixels are re-anchored to the observation at every step rather than drifting. It is the minimal
correct thing: one line on the prediction, before the linear combination touches it, and the `mask`
semantics the harness requires fall out with no special-case in the transition rule.

I also have to pick a default for the dial, and the caller's default is `eta = 1.0`, the fully Markovian,
maximal-fresh-noise end — defensible precisely because I do not know in advance which regime a given
workload wants. The failure mode of too little noise is the blur from last rung: a near-deterministic run
from a pinned start collapses toward the conditional mean and smears multimodal targets. The failure mode
of too much noise is milder — a slightly rougher image that still samples the right distribution, since
the marginal proof guarantees `η = 1` is as valid as `η = 0`. Between a knob that can *blur* and a knob
that can *roughen*, defaulting to the roughening end is the safe choice for a general translation task,
even though I expect tightly-correlated workloads would prefer to turn it down. The point of exposing `η`
is that I am not committing to one answer; I hand the trade-off to whoever knows the workload. The harness's
editable `sample_dbim` honors the caller's `eta` and `ts` directly, unlike the reference rung which
overrode them. The full module is in the answer.

So the falsifiable expectations against the floor, made against the specific numbers it posted. First and
most important: **at one-tenth the NFE, DBIM should not just match but *beat* DDBM's FID**, because the
analytic large-jump update spends its five calls on clean transitions instead of fifty calls fighting its
own churn. I expect Edges→Handbags to drop from DDBM's 11.139 into the 5-ish range and ImageNet
inpainting from 10.556 into the 6-ish range — roughly halving the FID while cutting calls by 10×. Second,
DIODE is the place I am least sure, and the `1.42` structural-hardness ratio is why: the
deterministic-leaning update may struggle there, and I would not be shocked if DBIM lands around DDBM's
DIODE number (≈15) rather than beating it, because DIODE's translation may want more stochasticity than a
near-deterministic five-step run at the default `η` provides. Third, the DIODE wall-clock should crater
to the ≈250–270 seconds I back-computed from the reference's 43-seconds-per-call, the visible payoff of 5
versus 50 calls. If DBIM beats DDBM on the two easier workloads but stalls near it on DIODE, that is the
signal that first-order Euler is leaving accuracy on the table at five steps — and the next move is
already visible: keep the same marginal-preserving family and the same booting noise, but integrate the
deterministic ODE to *higher order*, reusing past predictions so the extra accuracy costs no extra calls.
