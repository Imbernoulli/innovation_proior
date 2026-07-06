The sampler is the whole point, but a sampler steps *something*, and before I can be clever about the
transition rule I need a process to march and a reference for what "good but slow" even looks like. So
the rung to start from is the original bridge sampler — the one that gets the quality right by brute
force, paying many denoiser calls, with no attempt to be frugal. It is the floor in exactly the sense
that matters here: it answers "what FID can I reach if NFE is no object," and everything I build later
has to approach that number under five calls instead of fifty. The value of pinning this down first is
not the sampler — I will throw it away, it cannot even run inside the budget — it is the *anchor*.
Without a measured "quality is achievable on this model at all" I cannot tell, when a five-call sampler
gives me some FID, whether that number is good or a disaster. So this rung's job is to produce a
trustworthy ceiling and, in producing it, to expose precisely which part of the cost is essential and
which part is the price of ignorance I will later refuse to pay.

Let me get the object straight first, because the transition rule lives on top of it. I have a trained
diffusion bridge: a stochastic process pinned to start at the clean target `x_0` and arrive almost
surely at the informative endpoint `x_T = y` — a sketch, a degraded image, a masked photo. It was
built by Doob's h-transform of an ordinary diffusion `dx = f x dt + g dw`: adding the drift
`g² ∇log p(x_T = y | x_t)` forces the process to hit `y`, and pinning the other end too gives a bridge
whose doubly-conditioned forward kernel is an analytic Gaussian. Completing the square on the product
of base-diffusion Gaussians gives me the coefficients I will use everywhere:
`q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t² I)` with `a_t = (α_t/α_T)(SNR_T/SNR_t)`,
`b_t = α_t(1 − SNR_T/SNR_t)`, `c_t² = σ_t²(1 − SNR_T/SNR_t)`, `SNR_t = α_t²/σ_t²`. I do not want to take
those on faith, so I check both ends by hand. At `t → t_max` the ratio `SNR_T/SNR_t → 1`, so
`b_t = α_t·(1−1) = 0`, `c_t² = σ_t²·(1−1) = 0`, and `a_t = (α_t/α_T)·1 → (α_T/α_T)·1 = 1`. The kernel
collapses to `N(1·x_T + 0·x_0, 0) = δ(x_T)` — a Dirac exactly at the source. That is the pin, and it
is a genuine boundary singularity in the *variance*, not a cosmetic one: `c_t → 0` means the process
has no spread whatsoever at the endpoint, and I will collide with that zero the moment I try to divide
by it. At `t → 0` the other limit: `σ_t → 0` makes `SNR_t → ∞`, so `SNR_T/SNR_t → 0`, giving `a_t → 0`,
`b_t → α_t`, `c_t → σ_t`, and the kernel becomes `N(α_t x_0, σ_t² I)` — the ordinary diffusion marginal
around the *target*. So the bridge mean `a_t x_T + b_t x_0` is at every time a convex-ish interpolation
between the two endpoints, weight `a_t` on the source and `b_t` on the target, tight at both ends
(`c_t → 0` at each) and fattest somewhere in the middle where `c_t²` peaks — exactly the right shape for
translation, where the source and target already live close in pixel space and the honest uncertainty
is concentrated in the interior of the trajectory. The harness hands me this kernel through
`get_abc(t) → (a_t, b_t, c_t)` and a data predictor `denoiser(x, t) → x̂_0` that estimates the clean
target `x_0` from a noisy bridge state; that predictor is the single NFE-counted resource.

It is worth seeing where those three coefficients actually come from, because the whole later ladder
leans on their analytic form. The doubly-conditioned kernel is `q(x_t | x_0, x_T) ∝ q_base(x_t | x_0) ·
p(x_T | x_t)`, a product of two Gaussians in `x_t`: the base diffusion marginal `q_base(x_t | x_0) =
N(α_t x_0, σ_t² I)` and the endpoint likelihood `p(x_T | x_t) = N(α_{T|t} x_t, σ_{T|t}² I)` from the
base transition, both quadratic in `x_t` in the exponent. Multiplying two Gaussians and completing the
square in `x_t` gives a Gaussian whose precision is the sum of the two precisions and whose mean is the
precision-weighted average of the two means. Grinding that through, with `SNR_t = α_t²/σ_t²`, the
`x_0`-weight lands at `b_t = α_t(1 − SNR_T/SNR_t)`, the `x_T`-weight at `a_t = (α_t/α_T)(SNR_T/SNR_t)`,
and the combined variance at `c_t² = σ_t²(1 − SNR_T/SNR_t)`. I do not need to carry the algebra further
because the harness hands me `a_t, b_t, c_t` numerically through `get_abc`, but knowing they are a
completed square — a *single* Gaussian with these three braided coefficients, all functions of the same
underlying `α_t, σ_t` — tells me the object I am sampling has exactly this much structure and no more,
and that structure is what a black-box solver is about to throw away.

How do I generate from it? The score-based recipe says: reverse the bridge's own dynamics. The bridge
is itself a diffusion, with forward drift `μ = f x + g² h`, where `h = ∇log p(x_T | x_t)` is the
h-transform term I can write in closed form and `f x` is the base drift. Anderson's time-reversal turns
a forward SDE `dx = μ dt + g dw` into `dx̄ = [μ − g² ∇log q_t] dt + g dw̄`, and Song's continuity
identity gives a probability-flow ODE `dx = [μ − ½ g² ∇log q_t] dt` sharing the same marginals. Writing
both through the learned bridge score `s = ∇log q(x_t | x_T)` and substituting `μ = f x + g² h`, the
reverse-SDE drift becomes `f x + g² h − g² s = f x − g²(s − h)`, and the PF-ODE drift becomes
`f x + g² h − ½ g² s = f x − g²(½ s − h)`. The one place this is easy to get wrong, and it is
load-bearing for the sampler, is that factor of one-half: only the *learned score* `s` picks it up. The
h-transform term `h` sits inside the bridge's defining forward drift `μ`, not inside the
`−g² ∇log q_t` correction that the SDE→ODE conversion halves, so `h` survives at full strength in both
the SDE and the ODE. I fold this into a single switch: the per-step drift is `d = f x − g²(κ s − h)`
with `κ = 1` for a stochastic step and `κ = ½` for a deterministic one. That collapse to one switched
object is what lets a single helper serve both moves. The score itself comes straight from the
predictor by reading the kernel as a Gaussian: `s = −(x_t − a_t x_T − b_t x̂_0)/c_t²` (the code stores
the standard-deviation coefficient `c_t`, so the division is by `c_t²`), and the h-term reads
`h = −(x_t − (α_t/α_T) x_T)/(α_t² ρ̄_t²)` with `ρ̄_t² = ρ_T² − ρ_t²` from `get_alpha_rho`. Sanity check
on `h`: as `t → t_max`, `ρ̄_t² = ρ_T² − ρ_t² → 0`, so the denominator `α_t² ρ̄_t²` collapses and `h`
diverges — the pin is enforced by an *infinitely strong* restoring drift toward `x_T` at the endpoint.
That is the analytic face of the same `c_t → 0` singularity I flagged above, and it is the reason a
naive fast solver started right at `x_T` is going to hit a wall.

Now the actual sampling decision, and here the design space is small enough to walk in full. I have
three honest options for how to integrate these reverse dynamics, and I want to pick by argument, not
taste. Option one: integrate the PF-ODE backward from `x_T` with a fast deterministic high-order solver,
the way EDM does for unconditional generation — cheap, accurate per step, no noise to schedule. Option
two: run raw Euler–Maruyama on the reverse SDE. Option three: a churn predictor–corrector that adds a
controlled bit of noise then takes an accurate deterministic step. Let me eliminate the first, because
its failure teaches me the central tension of the whole task. A bridge has a *fixed, given* starting
point `x_T = y` — real data, not a fresh noise draw. Integrating a deterministic ODE backward from one
fixed point produces exactly *one* trajectory, the conditional-mean path `t ↦ E[x_0 | x_T]`. But
translation is genuinely one-to-many — one edge map corresponds to many plausible handbags, one mask to
many completions — so `p(x_0 | x_T)` is not a point mass, and the conditional mean of a multimodal
distribution is a blurry average of its modes. Make it concrete: if `p(x_0 | x_T)` put half its mass on
handbag A and half on a different handbag B, the deterministic ODE lands me at `½(A + B)`, a
pixel-wise ghost of two different bags that is *neither* of them and is not a valid sample of anything.
A pure ODE from a pinned start hands me exactly that ghost. So determinism plus a pinned start equals a
washed-out image, and this is the fact every later rung will have to respect under a tight budget: noise
here is not optional decoration, it is the thing that turns the output from an average into a real
sample. Option one is out on quality grounds, before any question of cost.

I should be precise about what a "churn step" concretely does here, because I will contrast it with a
cleaner mechanism later and I want the contrast exact. The reference splits every interval `[t_i,
t_{i+1}]` (with `t_{i+1} < t_i`) at an interior point `t_hat = t_i + churn_step_ratio·(t_{i+1} − t_i)`.
With `churn_step_ratio = 0.33`, `t_hat` sits 33% of the way from the larger time toward the smaller one,
so the first third of each interval is integrated with the *stochastic* branch (`κ = 1`, the reverse
SDE with its live diffusion term `g dw̄`, which injects fresh Gaussian noise of scale `g √Δt`), and the
remaining two thirds are integrated with the *deterministic* Heun branch (`κ = ½`, no diffusion term).
That is the churn: not a decorative jitter but a deliberate 33/67 split of every step into a noisy
sub-step that supplies diversity and a second-order deterministic sub-step that supplies accuracy.
Inside `ddbm_simulate` the Heun branch is the standard two-evaluation predictor–corrector — evaluate
the drift at the current time, take a provisional Euler step to `t_{i+1}`, evaluate the drift again
there, and advance by the *average* of the two drifts — which is why the deterministic move costs two
denoiser calls and the stochastic sub-step costs one.

Option two, raw Euler–Maruyama, has the diversity but fails on both cost and per-step accuracy at once.
The reverse-SDE drift `f x − g²(s − h)` carries that `h` term whose magnitude blows up near `t_max` and
whose stiffness demands tiny steps to integrate stably; a first-order Euler–Maruyama step commits an
`O(Δt)` error on a stiff, near-singular field and would need a punishingly fine grid to control it. I
would be paying many calls *and* getting a noisy, low-order estimate for each one — the worst of both
worlds. What I actually want is the SDE's diversity together with the ODE's per-step accuracy, and that
is option three, EDM's churn, which is engineered to give exactly that split. Each step I do two things:
first a short stochastic move (`κ = 1`) that briefly bumps the noise level *up* by a controlled amount,
injecting the diversity the bridge needs and jittering off any accumulated bias; then an accurate
deterministic move (`κ = ½`, second-order Heun) that carries the state back *down* to the next, smaller
time. The churn supplies the stochasticity; the Heun step does the heavy lifting cheaply and at second
order, so the deterministic error per step is `O(Δt³)` locally rather than Euler's `O(Δt²)`. Option
three wins on every axis I can measure it by, so that is the floor's transition rule.

There is one boundary subtlety the reference has to handle, and how it handles it is quietly telling. I
argued above that at exactly `t_max` the variance `c_t → 0` and the h-drift `h → ∞`, so the score's
`1/c_t²` and the h-term's `1/ρ̄_t²` both blow up: the first denoiser evaluation cannot sit on the pin.
The reference sidesteps this the crudest possible way — it builds its grid from `t_max − 1e-4` rather
than `t_max`, nudging the top of the schedule a hair inside the boundary so that `c_t` is small but
strictly positive and every division stays finite. That works, but notice what it reveals: even the
brute-force floor cannot start cleanly at the source; it has to fudge the endpoint. The one-to-many
stochasticity that lives at `t_max` — many targets behind one source — is a real singularity in the
sampler, not a numerical accident, and a crude offset is the reference's way of not dealing with it. A
sampler that took the singularity seriously would spend a *deliberate* draw of noise there to represent
the spread of `x_0 | x_T`, rather than nudging the grid and letting the churn accumulate diversity step
by step. I file that away: the endpoint is where diversity has to be injected, and doing it in one
principled shot instead of many churned ones is another call the black box is wasting.

The time grid follows the same borrowed-from-diffusion logic. With `N` steps I discretize on EDM's
power-law grid `t_i = (t_max^{1/ρ} + (i/N)(t_min^{1/ρ} − t_max^{1/ρ}))^ρ` with `ρ = 7`, and I want to
see where that puts the steps rather than assert it. Take `t_max = 1`, a small `t_min`, and step through:
the map `t ↦ t^{1/7}` compresses large `t` and stretches small `t`, so equal increments in the warped
coordinate become *tiny* increments in real `t` near `t_min` and *large* ones near `t_max`. Numerically,
for a nine-point ρ=7 ramp from 1 down to `0.01` the successive gaps come out roughly
`0.35, 0.24, 0.16, 0.10, 0.06, 0.04, 0.02, 0.01` — they shrink by more than an order of magnitude as
`t` falls, so the grid is dense (fine) near small `t` and coarse near `t_max`. That is the right instinct
for image generation: small `t` is where the sharp, high-frequency structure of the output is decided,
so I want my resolution there. I append a trailing zero to the array so the final iteration lands
exactly on `t = 0` and produces a clean endpoint rather than stopping short at some small positive time.

Now the call count, because that count is the entire reason this is the floor and not the endpoint. The
grid uses `_n = 17` power-law nodes; appending the trailing zero makes 18 time points, hence
`len(ts) − 1 = 17` iterations. At each interior step I spend one denoiser call on the churn-Euler move
and two on the Heun predictor–corrector — three calls per Heun iteration. On the final interval, where
`t_{i+1} = 0`, there is no valid second evaluation at `t = 0` (the predictor is undefined there and the
kernel is a Dirac), so I take a single churn-Euler step: one churn call plus one Euler call, two total,
which also happens to save a call. So the arithmetic is sixteen interior Heun iterations at three calls
each, `16 × 3 = 48`, plus one terminal iteration at two calls, `48 + 2 = 50` NFE. That is ten times the
agent's budget of five, exactly as the reference is meant to be. And I should be honest about *why* 50
and not 500: Heun is second order, so the global discretization error scales like `O((1/N)²)`. At
`N = 17` that is about `0.0035`; doubling to `N = 34` only buys `0.0009`, a fourfold shrink of a number
that is already small relative to the model's own fixed `x̂_0` approximation error, which no amount of
extra stepping can reduce. So beyond roughly this many steps the FID stops moving — the reference has
saturated the model's quality, which is precisely what makes 50 a fair *ceiling* rather than an
arbitrary large number. The structure of that cost is the diagnosis I carry forward: this sampler treats
the whole reverse drift as one opaque vector field and pays small-step discretization error on *every*
part of it, including the linear-in-`x` part `f x` that a smarter sampler could integrate in closed form,
and it needs the churn to recover diversity it never got to schedule deliberately. It is a generic
ODE/SDE discretizer lifted wholesale from diffusion models, not derived from the bridge's own analytic
structure, so it carries no bridge-specific large-jump update and no way to take the big steps a
five-call budget demands. It buys quality with calls, full stop.

Let me name the specific waste, because it is the exact thing I will refuse to pay for next. The drift
`f x − g²(κ s − h)` splits into a part that is *linear* in the state `x` — the `f x` term, plus the
linear-in-`x` pieces hidden inside `s` and `h`, both of which are affine in `x` since the kernel and
the endpoint likelihood are Gaussian — and a *nonlinear* part that enters only through the network's
`x̂_0`. The linear part of a semi-linear ODE is exactly integrable in closed form by an integrating
factor; there is no reason on earth to discretize it. But Heun cannot tell the two parts apart: it
evaluates the entire right-hand side at a couple of times and averages, so it commits `O(Δt³)` local
error on the linear part *and* the nonlinear part alike, and it must keep `Δt` small enough to control
the stiff linear piece even though that piece has an analytic answer. Multiply that avoidable per-step
error by the many steps the churn needs to stay diverse, and I have paid roughly ten times the calls a
bridge-aware sampler would need. When I count it that way the fifty calls decompose into maybe a handful
that are doing genuine, irreducible work — approximating the one nonlinear integral of the network
output — and the rest paying interest on the black box's ignorance of its own linear structure. That
decomposition is the whole thesis of the next rung.

One more grounding detail, because the harness exposes this rung differently from the agent rungs. The
editable `sample_dbim` body for this reference does *not* honor the caller's `ts` or `eta`: it overrides
`churn_step_ratio = 0.33` internally, builds its own EDM ρ=7 17-step grid with the trailing zero, and
routes every transition through a shared `ddbm_simulate(denoiser, noise_schedule, x, x_T, t_cur, t_next,
stochastic, second_order)` helper that computes the one-line drift `f x − g²(κ s − h)` — `κ` read off
the `stochastic` flag — and applies either an Euler or a Heun update. It returns `None` in the sixth
tuple slot: there is no booting-noise latent here, because the diversity comes from the many per-step
churn injections rather than from a single seeded draw at the start. So when I read this row's numbers I
am reading a 50-NFE, churn-0.33, Heun-on-EDM-ρ=7 sampler, and the budget it used is *not* available to
anything I write next; the caller's `ts`/`eta` contract that the later rungs must respect simply does
not bind this reference.

So what do I expect this floor to show, and why does its result point straight at the next rung? Because
it is unconstrained in NFE, I expect its FID to be good in absolute terms but *not* the best number this
leaderboard will eventually see — and that second half is the surprise that motivates everything. A
generic high-order discretizer with churn is accurate per unit step, but it is solving the bridge
dynamics the hard way. Concretely, three falsifiable predictions. First, the per-workload FIDs —
`best_fid_edges2handbags`, `best_fid_Imagenet`, `best_fid_DIODE` — should all land in a
single-to-low-double-digit range: real images, not garbage, which would confirm the framework and the
trained model are sound and that any later disaster is the sampler's fault, not the substrate's. Second,
the DIODE wall-clock should be enormous relative to the budgeted samplers — of order thousands of
seconds, since the context already flags this reference at roughly 8× the wall-clock of a five-call run
— the visible price of 50 versus 5 calls. Third, and decisively, I predict these FIDs will be
*beatable* by a five-call bridge-specific sampler, because the churn this rung needs for diversity is
also injecting noise a few-step sampler can schedule far more carefully, and the linear part of the drift
it discretizes blindly can be integrated exactly. I cannot prove that third claim from here — I have no
five-call sampler yet — so I hold it as the hypothesis the next rung exists to test: if a bridge-specific
sampler at a tenth the calls reaches a *lower* FID than this floor, the diagnosis is confirmed and the
problem was never the model or the need for stochasticity; it was that I am paying for genericity. That
outcome would force the next move rather than merely suggest it — stop treating the reverse dynamics as
a black box, exploit the marginals-only property of the training loss to build a bridge-specific
transition that takes large analytic jumps, and bring the call count down by an order of magnitude
without giving up the quality. That is the first thing I can actually run inside the five-call budget,
and this floor is what will let me recognize it as a win.
