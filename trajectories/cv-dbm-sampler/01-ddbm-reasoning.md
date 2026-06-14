The sampler is the whole point, but a sampler steps *something*, and before I can be clever about the
transition rule I need a process to march and a reference for what "good but slow" even looks like. So
the rung to start from is the original bridge sampler — the one that gets the quality right by brute
force, paying many denoiser calls, with no attempt to be frugal. It is the floor in exactly the sense
that matters here: it answers "what FID can I reach if NFE is no object," and everything I build later
has to approach that number under five calls instead of fifty.

Let me get the object straight first, because the transition rule lives on top of it. I have a trained
diffusion bridge: a stochastic process pinned to start at the clean target `x_0` and arrive almost
surely at the informative endpoint `x_T = y` — a sketch, a degraded image, a masked photo. It was
built by Doob's h-transform of an ordinary diffusion `dx = f x dt + g dw`: adding the drift
`g² ∇log p(x_T = y | x_t)` forces the process to hit `y`, and pinning the other end too gives a bridge
whose doubly-conditioned forward kernel is an analytic Gaussian. Completing the square on the product
of base-diffusion Gaussians gives me the coefficients I will use everywhere:
`q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t² I)` with `a_t = (α_t/α_T)(SNR_T/SNR_t)`,
`b_t = α_t(1 − SNR_T/SNR_t)`, `c_t² = σ_t²(1 − SNR_T/SNR_t)`, `SNR_t = α_t²/σ_t²`. Sanity-check the
ends. At `t → t_max`, `SNR_T/SNR_t → 1`, so `b_t, c_t → 0`, `a_t → 1`, and the kernel collapses to a
Dirac at `x_T` — the pin. At `t → 0`, `SNR_T/SNR_t → 0`, so `a_t → 0`, `b_t → α_t`, `c_t → σ_t`, and
the kernel is the ordinary diffusion marginal `N(α_t x_0, σ_t² I)` around the target. So the bridge
mean is, at every time, a weighted interpolation between the two endpoints, tight at both ends and fat
in the middle — exactly what I want for translation, where the source and target are already close in
pixel space. The harness hands me this kernel through `get_abc(t) → (a_t, b_t, c_t)` and a data
predictor `denoiser(x, t) → x̂_0` that estimates the clean target `x_0` from a noisy bridge state.

How do I generate from it? The score-based recipe says: reverse the bridge's own dynamics. The bridge
is itself a diffusion with drift `f + g² h`, where `h = ∇log p(x_T | x_t)` is the h-transform term I
can write in closed form. Anderson's reversal gives a reverse SDE, and Song's continuity identity
gives a probability-flow ODE with the same marginals. Writing both through the learned bridge score
`s = ∇log q(x_t | x_T)` and the analytic `h`, the reverse SDE drift is `f − g²(s − h)` and the PF-ODE
drift is `f − g²(½ s − h)`. The one place this is easy to get wrong, and it is load-bearing for the
sampler, is the factor of one-half: only the *learned score* `s` gets halved in the ODE — the
h-transform term `h` is part of the bridge's defining forward drift, not the thing the SDE→ODE
conversion splits, so it stays at full strength in both. I keep that as a single switch: the per-step
drift is `f x − g²(κ s − h)` with `κ = 1` for a stochastic step and `κ = ½` for a deterministic step.
The score itself comes from the predictor: `s = −(x_t − a_t x̂_0 − b_t x_T)/c_t²` (the code stores the
standard-deviation coefficient `c_t`, so the division is by `c_t²`), and the h-term reads
`h = −(x_t − (α_t/α_T) x_T)/(α_t² ρ̄_t²)` with `ρ̄_t² = ρ_T² − ρ_t²` from `get_alpha_rho`.

Now the actual sampling decision, and here is where this rung is honest about being a brute-force
floor. The obvious thing — integrate the PF-ODE backward from `x_T` with a fast deterministic solver,
the way EDM does for generation — is *wrong* for a bridge, and the reason teaches me the central
tension of this whole task. A bridge has a fixed, given starting point `x_T = y`, real data, not a
fresh noise draw. Integrating a deterministic ODE backward from one fixed point produces exactly *one*
trajectory: the conditional-mean path. But translation is genuinely one-to-many — one edge map
corresponds to many plausible handbags, one mask to many completions — so the conditional `p(x_0 | x_T)`
is not a point mass, and the conditional mean of a multi-modal distribution is a blurry average. A pure
ODE from a pinned start hands me that blur. So I *need* stochasticity to recover diversity and
sharpness; determinism plus a pinned start equals a washed-out image. This is the fact every later rung
will have to respect under a tight budget: noise is not optional here, it is what makes the output a
real sample instead of an average.

So I put noise back, predictor-corrector style. Pure Euler-Maruyama on the reverse SDE is both slow and
inaccurate per step, but I want the SDE's diversity and the ODE's per-step accuracy at once. EDM's clean
engineering of this is "churn": each step, briefly add a controlled bit of noise (a short stochastic
Euler move that bumps the noise level up), then take an accurate deterministic step back down (a
second-order Heun move on the ODE). The churn supplies the stochasticity the bridge needs and corrects
accumulated discretization error; the Heun step does the heavy lifting cheaply. I discretize time on
EDM's power-law grid `t_i = (t_max^{1/ρ} + (i/N)(t_min^{1/ρ} − t_max^{1/ρ}))^ρ` with `ρ = 7`, which
equalizes truncation error across the trajectory and is image-friendly, and I end the array with a
trailing zero so the last iteration lands exactly on `t = 0`.

Let me count calls, because that count is the entire reason this is the floor and not the endpoint. At
each interior step I spend one denoiser call on the churn-Euler move and two on the Heun
predictor-corrector — three calls per Heun iteration. On the final interval, where `t_{i+1} = 0`, there
is no valid second evaluation at `t = 0`, so I take a single churn-Euler step (one churn call plus one
Euler call = two), which also saves a call. With the reference setting — `churn_step_ratio = 0.33`,
`ρ = 7`, and 17 iterations — sixteen Heun iterations at three calls plus one terminal iteration at two
calls total **50 NFE**. That is ten times the agent's budget. The structure of the cost is the point:
this sampler treats the whole reverse drift as one opaque vector field and pays small-step
discretization error on *every* part of it, including the linear part that a smarter sampler could
integrate in closed form. It is a generic ODE/SDE discretizer borrowed wholesale from diffusion models,
not derived from the bridge's own structure, so it carries no bridge-specific large-jump update and no
way to take the big steps a five-call budget demands. It buys quality with calls, full stop.

One more grounding detail, because the harness exposes this rung differently from the agent rungs. The
editable `sample_dbim` body for this reference does *not* honor the caller's `ts` or `eta`: it
overrides `churn_step_ratio = 0.33` internally, builds its own EDM ρ=7 17-step grid with the trailing
zero, and routes every transition through a shared `ddbm_simulate(denoiser, noise_schedule, x, x_T,
t_cur, t_next, stochastic, second_order)` helper that computes the one-line drift
`f x − g²(κ s − h)` (κ from the `stochastic` flag) and applies either an Euler or a Heun update. It
returns `None` in the sixth tuple slot — there is no booting-noise latent here, the diversity comes
from the per-step churn injections rather than a single seeded draw. So when I read this row's numbers I
am reading a 50-NFE, churn-0.33, Heun-on-EDM-ρ=7 sampler, and the budget it used is *not* available to
anything I write next. The full reference module is in the answer.

So what do I expect this floor to show, and why does its result point straight at the next rung? Because
it is unconstrained in NFE, I expect its FID to be *good in absolute terms but not the best on this
leaderboard* — and that second half is the surprise that motivates everything. A generic high-order
discretizer with churn is accurate per unit step, but it is solving the bridge dynamics the hard way,
and on these workloads the dedicated fast samplers that come later will actually *beat* it on FID while
using a tenth of the calls, because they exploit the bridge's analytic structure to take clean large
jumps instead of many small noisy ones. So I expect three things. First, the per-workload FIDs should
all be in a single-to-low-double-digit range — real images, not garbage — confirming the framework and
the trained model are sound. Second, the DIODE wall-clock should be enormous relative to the budgeted
samplers (≈2000+ seconds versus ≈270), the visible price of 50 versus 5 calls. Third, and decisively,
the FIDs should be *worse* than what a five-call bridge-specific sampler can reach, because the churn
that this rung needs for diversity is also injecting noise the few-step samplers will learn to schedule
far more carefully. If that holds, the diagnosis is sharp: the problem is not the model and not the
need for stochasticity — it is that I am paying for genericity. The next move is forced — stop treating
the reverse dynamics as a black box, exploit the marginals-only property to build a *bridge-specific*
transition that takes large analytic jumps, and bring the call count down by an order of magnitude
without giving up the quality. That is the first thing I can actually run inside the five-call budget.
