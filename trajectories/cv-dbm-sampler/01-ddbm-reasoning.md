The sampler is the whole point, but a sampler steps *something*, and before I can be clever about the
transition rule I need a process to march and a reference for what "good but slow" even looks like. So
the place to start is the original bridge sampler — the one that gets the quality right by brute
force, paying many denoiser calls, with no attempt to be frugal. It is the floor in exactly the sense
that matters here: it answers "what FID can I reach if NFE is no object," and everything I build later
has to approach that number under five calls instead of fifty. I will throw this sampler away — it
cannot even run inside the budget — but without a measured "quality is achievable on this model at
all," I cannot tell whether a five-call FID is a triumph or a disaster. So the job here is a
trustworthy ceiling, and, in producing it, exposing which part of the cost is essential and which part
is the price of ignorance I will later refuse to pay.

Let me get the object straight first, because the transition rule lives on top of it. I have a trained
diffusion bridge: a stochastic process pinned to start at the clean target `x_0` and arrive almost
surely at the informative endpoint `x_T = y` — a sketch, a degraded image, a masked photo. It was
built by Doob's h-transform of an ordinary diffusion `dx = f x dt + g dw`: adding the drift
`g² ∇log p(x_T = y | x_t)` forces the process to hit `y`, and pinning the other end too gives a bridge
whose doubly-conditioned forward kernel is an analytic Gaussian. Completing the square on the product
of base-diffusion Gaussians gives the coefficients I will use everywhere:
`q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t² I)` with `a_t = (α_t/α_T)(SNR_T/SNR_t)`,
`b_t = α_t(1 − SNR_T/SNR_t)`, `c_t² = σ_t²(1 − SNR_T/SNR_t)`, `SNR_t = α_t²/σ_t²`. The ends are worth
checking because one of them bites later. At `t → t_max`, `SNR_T/SNR_t → 1`, so `b_t, c_t → 0`,
`a_t → 1`, and the kernel collapses to `δ(x_T)` — the pin. That is a genuine boundary singularity in
the *variance*: `c_t → 0` means no spread at the endpoint, and I will collide with that zero the
moment I try to divide by it. At `t → 0`, `SNR_T/SNR_t → 0`, so `a_t → 0`, `b_t → α_t`, `c_t → σ_t`,
and the kernel is the ordinary diffusion marginal `N(α_t x_0, σ_t² I)` around the target. So the bridge
mean is, at every time, a weighted interpolation between the two endpoints — weight `a_t` on the
source, `b_t` on the target — tight at both ends (`c_t → 0` at each) and fattest in the middle where
`c_t²` peaks. Exactly the right shape for translation, where source and target already live close in
pixel space and the honest uncertainty is concentrated in the interior. The three coefficients are a
single completed square — one Gaussian with braided coefficients, all functions of the same `α_t, σ_t`
— which tells me the object I am sampling has exactly this much structure and no more, and that
structure is what a black-box solver is about to throw away. The harness hands me the kernel through
`get_abc(t) → (a_t, b_t, c_t)` and a data predictor `denoiser(x, t) → x̂_0`, the single NFE-counted
resource.

How do I generate from it? The score-based recipe says reverse the bridge's own dynamics. The bridge
is itself a diffusion with forward drift `μ = f x + g² h`, where `h = ∇log p(x_T | x_t)` is the
h-transform term I can write in closed form. Anderson's reversal turns `dx = μ dt + g dw` into
`dx̄ = [μ − g² ∇log q_t] dt + g dw̄`, and Song's continuity identity gives a PF-ODE
`dx = [μ − ½ g² ∇log q_t] dt` with the same marginals. Writing both through the learned bridge score
`s = ∇log q(x_t | x_T)` and substituting `μ = f x + g² h`, the reverse-SDE drift becomes
`f x − g²(s − h)` and the PF-ODE drift becomes `f x − g²(½ s − h)`. The one place this is easy to get
wrong, and it is load-bearing for the sampler, is that factor of one-half: only the *learned score* `s`
picks it up. The h-transform term sits inside the bridge's defining forward drift `μ`, not inside the
`−g² ∇log q_t` correction that the SDE→ODE conversion halves, so `h` survives at full strength in both.
I fold this into a single switch: the per-step drift is `d = f x − g²(κ s − h)` with `κ = 1` for a
stochastic step and `κ = ½` for a deterministic one, which lets a single helper serve both moves. The
score reads off the kernel as a Gaussian, `s = −(x_t − a_t x_T − b_t x̂_0)/c_t²` (the code stores the
std coefficient `c_t`, so the division is by `c_t²`), and `h = −(x_t − (α_t/α_T) x_T)/(α_t² ρ̄_t²)` with
`ρ̄_t² = ρ_T² − ρ_t²` from `get_alpha_rho`. As `t → t_max`, `ρ̄_t² → 0`, so `h` diverges — the pin is
enforced by an infinitely strong restoring drift toward `x_T`. That is the analytic face of the same
`c_t → 0` singularity, and it is why a naive fast solver started right at `x_T` hits a wall.

Now the sampling decision. I have three honest options for integrating these reverse dynamics: integrate
the PF-ODE backward with a fast deterministic high-order solver, EDM-style; run raw Euler–Maruyama on
the reverse SDE; or a churn predictor–corrector that adds controlled noise then takes an accurate
deterministic step. Eliminate the first, because its failure teaches me the central tension of the
whole task. A bridge has a *fixed, given* start `x_T = y` — real data, not a fresh noise draw.
Integrating a deterministic ODE backward from one fixed point produces exactly *one* trajectory, the
conditional-mean path `t ↦ E[x_0 | x_T]`. But translation is genuinely one-to-many — one edge map
corresponds to many plausible handbags, one mask to many completions — so `p(x_0 | x_T)` is not a point
mass, and the conditional mean of a multimodal distribution is a blurry average of its modes. If
`p(x_0 | x_T)` put half its mass on handbag A and half on B, the deterministic ODE lands me at
`½(A + B)`, a pixel-wise ghost that is neither. So determinism plus a pinned start equals a washed-out
image: noise here is not decoration, it is the thing that turns the output from an average into a real
sample. Option one is out on quality grounds, before any question of cost.

The churn step, concretely, splits every interval `[t_i, t_{i+1}]` (`t_{i+1} < t_i`) at
`t_hat = t_i + churn_step_ratio·(t_{i+1} − t_i)`. With `churn_step_ratio = 0.33` the first third is
integrated with the *stochastic* branch (`κ = 1`, the reverse SDE with its live `g dw̄` injecting fresh
Gaussian noise of scale `g √Δt`) and the remaining two thirds with the *deterministic* Heun branch
(`κ = ½`). A noisy sub-step supplies diversity; a second-order deterministic sub-step supplies accuracy.
The Heun branch is the standard two-evaluation predictor–corrector — evaluate the drift, take a
provisional Euler step, re-evaluate, advance by the average — which is why the deterministic move costs
two denoiser calls and the stochastic sub-step costs one.

Raw Euler–Maruyama, option two, has the diversity but fails on both cost and per-step accuracy at once.
The reverse-SDE drift carries that `h` term whose magnitude blows up near `t_max` and whose stiffness
demands tiny steps; a first-order Euler–Maruyama step commits `O(Δt)` error on a stiff, near-singular
field and needs a punishingly fine grid. That is many calls *and* a noisy low-order estimate each —
the worst of both worlds. What I want is the SDE's diversity with the ODE's per-step accuracy, and that
is option three: a short stochastic move (`κ = 1`) that briefly bumps the noise level up and injects
diversity, then an accurate deterministic Heun move (`κ = ½`) that carries the state back down, at
`O(Δt³)` local error rather than Euler's `O(Δt²)`. Option three wins on every axis, so that is the
floor's transition rule.

One boundary subtlety, and how the reference handles it is telling. At exactly `t_max` the score's
`1/c_t²` and the h-term's `1/ρ̄_t²` both blow up: the first denoiser evaluation cannot sit on the pin.
The reference sidesteps this the crudest way — it builds its grid from `t_max − 1e-4` rather than
`t_max`, nudging the top a hair inside the boundary so every division stays finite. That works, but the
one-to-many stochasticity that lives at `t_max` — many targets behind one source — is a real
singularity, not a numerical accident, and a crude offset is the reference's way of not dealing with
it. A sampler that took it seriously would spend a *deliberate* draw of noise there to represent the
spread of `x_0 | x_T`, rather than nudging the grid and letting churn accumulate diversity step by
step. I file that away: the endpoint is where diversity has to be injected, and doing it in one
principled shot instead of many churned ones is a call the black box is wasting.

The time grid follows the same borrowed-from-diffusion logic: EDM's power-law grid
`t_i = (t_max^{1/ρ} + (i/N)(t_min^{1/ρ} − t_max^{1/ρ}))^ρ` with `ρ = 7`. The map `t ↦ t^{1/7}`
compresses large `t` and stretches small `t`, so the grid is dense near small `t` and coarse near
`t_max` — the right instinct for image generation, where small `t` decides the sharp high-frequency
structure and I want resolution there. I append a trailing zero so the final iteration lands exactly on
`t = 0`.

Now the call count, because that count is the whole reason this is the floor. The grid uses `_n = 17`
power-law nodes; the trailing zero makes 18 points and 17 iterations. Each interior step spends one
churn-Euler call plus two Heun calls — three per Heun iteration. On the final interval, where
`t_{i+1} = 0`, there is no valid second evaluation at `t = 0` (the predictor is undefined there and the
kernel is a Dirac), so I take a single churn-Euler step: one churn call plus one Euler call, two total.
So sixteen interior Heun iterations at three calls, `16 × 3 = 48`, plus one terminal at two,
`= 50` NFE — ten times the agent's budget, exactly as a reference should be. And 50 not 500 because Heun
is second order: global error `O((1/N)²)` is about `0.0035` at `N = 17`, already small relative to the
model's own fixed `x̂_0` approximation error that no extra stepping can reduce, so beyond this many
steps the FID stops moving. That saturation is what makes 50 a fair *ceiling* rather than an arbitrary
large number.

The structure of that cost is the diagnosis I carry forward. The drift `f x − g²(κ s − h)` splits into
a part *linear* in `x` — the `f x` term plus the affine-in-`x` pieces inside `s` and `h`, both affine
because the kernel and endpoint likelihood are Gaussian — and a *nonlinear* part that enters only
through the network's `x̂_0`. The linear part of a semi-linear ODE is exactly integrable in closed form
by an integrating factor; there is no reason to discretize it. But Heun cannot tell the two parts
apart: it evaluates the whole right-hand side and averages, committing `O(Δt³)` local error on the
linear part it could solve exactly, and it must keep `Δt` small to control the stiff linear piece even
though that piece has an analytic answer. Multiply that avoidable per-step error by the many steps churn
needs to stay diverse, and the fifty calls decompose into a handful doing genuine irreducible work —
approximating the one nonlinear integral of the network output — and the rest paying interest on the
black box's ignorance of its own linear structure. That decomposition is what I exploit next.

One grounding detail, because the harness exposes this reference differently from the budgeted samplers. The
editable `sample_dbim` body for this reference does *not* honor the caller's `ts` or `eta`: it overrides
`churn_step_ratio = 0.33`, builds its own EDM ρ=7 17-step grid with the trailing zero, and routes every
transition through a shared `ddbm_simulate(...)` helper that computes the one-line drift `f x − g²(κ s −
h)` and applies an Euler or Heun update. It returns `None` in the sixth tuple slot — no booting-noise
latent, because diversity comes from the many per-step churn injections rather than a single seeded draw
at the start. So this row is a 50-NFE, churn-0.33, Heun-on-EDM-ρ=7 sampler, and the budget it used is
*not* available to anything I write next.

Since it is unconstrained in NFE, I expect its FIDs to be good in absolute terms — real images in the
single-to-low-double-digit range, confirming the framework and trained model are sound so any later
disaster is the sampler's fault — with a DIODE wall-clock of order thousands of seconds, the visible
price of 50 versus 5 calls at the roughly 8× the context already flags. The claim I cannot prove from
here, and hold as the hypothesis to test next, is that these FIDs are *beatable* by a five-call
bridge-specific sampler. If one at a tenth the calls reaches a *lower* FID, the diagnosis is confirmed —
the problem was never the model or the need for stochasticity, only the price of genericity — and the
move is to stop treating the reverse dynamics as a black box and build a transition that takes large
analytic jumps off the marginals-only loss.
