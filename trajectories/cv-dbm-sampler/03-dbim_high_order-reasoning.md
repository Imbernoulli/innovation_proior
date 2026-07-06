DBIM did exactly what I predicted on the two easy workloads and stalled exactly where I feared on the
third, and the split tells me precisely what to fix. Let me put real numbers on it before I move, because
the *shape* of the improvement is the diagnosis. Edges→Handbags fell from DDBM's 11.139 to 5.180, a
`(11.139 − 5.180)/11.139 = 53.5%` cut — better than halved. ImageNet inpainting fell from 10.556 to
6.070, a 42.5% cut. And DIODE went from 15.811 to 15.002, a mere `0.809/15.811 = 5.1%` cut — essentially
a tie with the 50-NFE reference. So a five-call bridge-specific sampler *matched* the fifty-call floor on
DIODE while beating it by half on the easy workloads. The marginals-only lever paid off handsomely where
the targets are tightly correlated with the source, and paid off almost nothing where they are not. The
wall-clock also came in as forecast: 271 seconds on DIODE against the ≈250–270 I back-computed from the
reference's 43-seconds-per-call, confirming I really am spending five calls and not hiding compute. So
the budget arithmetic is sound and the family is right; the failure is localized. It is not about
stochasticity — the default `η` gave DBIM whatever noise it had and DIODE still stalled — and it is not
about the model, which reaches 5.180 on e2h. On the hard translation, *first-order Euler is leaving
accuracy on the table at five steps*.

I can say why mechanically. Each DBIM step at `η = 0` is the re-projection I derived last rung: it holds
the predictor's output `x̂_0` constant across the whole interval and re-attaches the bridge mean at the
new time. That is first-order — it treats `x̂_0` as if it did not change between the two ends of a big
step. On Edges→Handbags and ImageNet the predicted clean image barely moves from one step to the next, so
freezing it is nearly exact and the re-projection is nearly perfect; that is why those FIDs are already
low. On DIODE the predicted `x̂_0` swings hard between the large steps a five-call budget forces, so
freezing it commits a real error that the re-projection then faithfully carries to the endpoint. The
5.1% DIODE cut is exactly the residue of a first-order approximation that the two easy workloads never
felt. Before I commit to "more accurate steps," I have to rule out the more obvious reading of the DIODE
stall, because DIODE *is* the one-to-many workload and the reflex is to blame stochasticity. The reflex
says: DIODE stalled because a near-deterministic run cannot diversify, so crank `η` up and let noise
carve the conditional. I do not buy it, and I can argue why without running it. DBIM already carried the
default `η` noise, and DIODE still tied the fifty-call floor — so whatever noise the coupled family
allows was already in play and did not move the number. And the coupled family caps that noise hard: the
marginal proof forces `ρ ≤ c`, so `√(c² − ρ²)` stays real, and I cannot inject more than the full
marginal scale at any step. Even at that ceiling I would be *roughening* the sample, not correcting the
systematic error, because the 5.1% DIODE residue has the signature of a *bias*, not a variance: it is the
same direction every seed, the frozen-`x̂_0` error carried faithfully to the endpoint, and no amount of
symmetric fresh noise cancels a bias. The two easy workloads make the point by contrast — they got the
same `η` and improved by half, because their `x̂_0` barely moves so the freeze is nearly exact. So the
DIODE problem is not too little noise; it is a first-order approximation of a fast-changing integrand,
and I fix approximations by raising the order, not by adding noise. That eliminates the stochastic-crank
option and points at the deterministic accuracy of the step.

The fix has to keep everything that worked — same marginal-preserving family, same booting noise,
five calls — and only make each step *more accurate*, ideally for free. That means understanding what
continuous object the deterministic DBIM step is discretizing, then integrating it to higher order.

So set the deterministic case `η = 0`, write `t_{n+1} = t`, `t_n = t − Δt`, and stare at the update:
`x_{t-Δt} = a_{t-Δt} x_T + b_{t-Δt} x̂_0 + (c_{t-Δt}/c_t)(x_t − a_t x_T − b_t x̂_0)`. This is full of
`x_T` and `x̂_0` terms that do not obviously want to become a clean derivative. The troublesome factor
is that `c_t` ratio, so I divide the whole thing through by `c_{t-Δt}` and regroup:
`x_{t-Δt}/c_{t-Δt} = x_t/c_t + (a_{t-Δt}/c_{t-Δt} − a_t/c_t) x_T + (b_{t-Δt}/c_{t-Δt} − b_t/c_t) x̂_0`.
That is beautiful — it is a finite difference of `x_t/c_t` equal to finite differences of `a_t/c_t` and
`b_t/c_t` weighting the two endpoints. Since `a, b, c` are smooth functions of `t`, this is the Euler
discretization of `d(x_t/c_t) = x_T d(a_t/c_t) + x̂_θ(x_t, t, x_T) d(b_t/c_t)`. So the natural state
variable is not `x_t` but `x_t/c_t`, and the natural coordinates are `a_t/c_t` and `b_t/c_t`, not `t`.
The noise scale `c_t` is exactly what was blowing up the linear part of the dynamics, and dividing it
out gives a clean, low-curvature ODE — the bridge analogue of how DDIM became an Euler step on `x/√α`.

I should sanity-check this is the *same* solution as the bridge probability-flow ODE, because if it
were a different deterministic process I would be sampling the wrong distribution and every FID I quoted
would be measuring the wrong target. The PF-ODE with the score replaced by the data predictor is
`dx_t = [(f + g²/σ² − g²/2c²) x_t + g²a/2c² x_T − g²b/2c² x̂_θ] dt`. Expanding my clean ODE back into
`dx_t` via the product rule (`d(x/c) = dx/c − x c'/c² dt`) gives
`dx_t = [(c'/c) x_t + (a' − a c'/c) x_T + (b' − b c'/c) x̂_θ] dt`. Now I compute the three log-derivatives
from the schedule identities `f = (log α)'`, `g² = (σ²)' − 2(log α)' σ²` and `(1/SNR_t)' = g²/α²`, and
grinding them out gives `c'/c = f + g²/σ² − g²/2c²`, `a' − a c'/c = g²a/2c²`, `b' − b c'/c = −g²b/2c²` —
all three coefficients match the PF-ODE term for term. So my ODE *is* the bridge PF-ODE, just written in
coordinates where it is trivial to integrate. The deterministic DBIM step is not an approximation to the
PF-ODE; it is an exact reparameterization, and a far friendlier one. That cleanness is the doorway to
high order.

Why is it the doorway? Because the ODE is *semi-linear* — a linear-in-`x` part plus a nonlinear network
part — and I have already isolated the linear part. This is the exact waste I named at the floor: DDBM's
Heun sampler treated the entire right-hand side as one opaque field and paid discretization error on the
easy linear part too. The lesson from fast diffusion solvers is to never let a generic integrator chew on
the linear part: cancel it analytically with variation-of-constants and only approximate the integral of
the *smooth* network output. Apply that here. Writing `dx_t = [A(t) x_t + B_T(t) x_T + B_θ(t) x̂_θ] dt`
with `A = c'/c`, the integrating factor is `e^{∫A} = e^{∫(log c)'} = c_t/c_s`. The `x_T` integral closes
in elementary form because `x_T` is constant, contributing `(a_t − (c_t/c_s) a_t) x_T`. The only
genuinely hard integral is the one with the network output, and to make it as simple as possible I change
variable to the bridge log-SNR `λ_t = log(b_t/c_t)` — since `b_t/c_t = √(SNR_t − SNR_T)`, this is the
half-log of the excess signal-to-noise over the endpoint, playing exactly the role `log(α/σ)` played for
ordinary diffusion. It is worth being sure this is a *good* coordinate and not just a slick one. As
`t → t_max` the excess `SNR_t − SNR_T → 0`, so `λ_t → −∞`; as `t → 0`, `SNR_t → ∞` and `λ_t → +∞`. So
`λ` stretches the whole trajectory across the full real line, monotonically, with the endpoint pin pushed
out to `−∞` where its singularity can do no harm to an interior step. And because the integrand's only
remaining `t`-dependence is through the smooth network output `x̂_θ`, a Taylor expansion in `λ` is an
expansion in a variable where the exponential weight `e^λ` is the *entire* explicit `λ`-dependence of the
integral — which is exactly why the φ-functions come out as clean closed forms and the derivatives I need
are just derivatives of `x̂_θ` in `λ`. Choosing `log(α/σ)` did this for ordinary diffusion; `log(b/c)` is
its bridge-faithful analogue, and getting the *excess*-SNR form rather than the raw SNR is what makes the
`t_max` end behave. In this variable the exact solution from time `t` to `s < t` is
`x_s = (c_s/c_t) x_t + (a_s − (c_s/c_t) a_t) x_T + c_s ∫_{λ_t}^{λ_s} e^λ x̂_θ(x_{t_λ}, t_λ, x_T) dλ`.
The linear and endpoint parts are now *exact*; all the discretization error lives in that one
exponentially-weighted integral of the smooth network output — and *that* is the error DBIM's first
order was paying on DIODE.

Let me verify the first-order case reduces to the DBIM step I already have, because that reduction is the
proof that the reparameterization is faithful, not a wish. Freeze `x̂_θ = x̂_0` (constant, its value at
the current node) and integrate: `c_s ∫_{λ_t}^{λ_s} e^λ dλ = c_s(e^{λ_s} − e^{λ_t})`. Since `e^λ = b/c`,
this is `c_s(b_s/c_s − b_t/c_t) x̂_0 = (b_s − (c_s/c_t) b_t) x̂_0`. Adding the exact linear and endpoint
parts, `x_s = (c_s/c_t) x_t + (a_s − (c_s/c_t) a_t) x_T + (b_s − (c_s/c_t) b_t) x̂_0`, which is *exactly*
the DBIM `η = 0` re-projection with `s` and `t` in their bridge roles. Equivalently, in the harness's
step orientation the leading integral term `c_t · e^{λ_t}(1 − e^{-h})` equals `c_t · (b_t/c_t)(1 − e^{-h})
= b_t(1 − e^{-h})`, and since `e^{-h} = e^{λ_s}/e^{λ_t} = (b_s/c_s)(c_t/b_t)`, this collapses to
`b_t − (c_t/c_s) b_s` — precisely the `coeff_x0_hat` of the first-order branch. So the exponential
integrator with only its leading term *is* DBIM, coefficient for coefficient. The higher-order terms are
pure additions on top of a step I already trust.

To go higher order I Taylor-expand `x̂_θ` as a function of `λ` about the current node and integrate each
Taylor term against `e^λ` exactly. With `h` the step in `λ` (note `λ` increases as time decreases, so
stepping toward the smaller time means moving forward in `λ`, `h > 0`), the scalar integrals are
analytic by repeated integration by parts. Do the second one by hand to be sure of the pattern: for
`∫ (λ − λ_t) e^λ dλ` set `u = λ − λ_t`, `dv = e^λ dλ`, giving `(λ − λ_t)e^λ − ∫ e^λ dλ` evaluated over
the interval `= h e^{λ_s} − (e^{λ_s} − e^{λ_t}) = e^{λ_s}(h − 1 + e^{-h})`. The full set comes out
`∫ e^λ dλ = e^{λ_s}(1 − e^{-h})`, `∫ (λ − λ_t) e^λ dλ = e^{λ_s}(h − 1 + e^{-h})`,
`∫ ½(λ − λ_t)² e^λ dλ = e^{λ_s}(h²/2 − h + 1 − e^{-h})`. I checked these three against numerical
quadrature and they agree to six digits, so I am not fooling myself with a sign. These are the φ-functions
of exponential integrators — I did not pick them, the integrals handed them to me. Keeping just the first
bracket term is first order and reduces exactly to the DBIM Euler step I just verified; keeping two terms
is second order, three is third.

Now the decision that actually controls the cost under a five-call budget, and it is a genuine fork with
a computable answer. I need the `λ`-derivatives of the network output, and I have two roads. The
single-step road inserts an extra intermediate timestep, evaluates the network there, and
finite-differences — but that is an *extra* denoiser call per step, so a `k`-th order single-step method
costs `k` calls per step. Cost it out against five: with one booting prediction spent, an order-2
single-step sampler gets `boot + 2 steps × 2 calls = 5`, so *two* transitions; order-3 single-step gets
`boot + 1 step × 3 + …` — it cannot even complete a second full step. That is catastrophic here: it halves
or quarters my already-tiny step count. The multistep road, Adams–Bashforth style, finite-differences
against the predictor outputs I *already computed* at previous timesteps — those are sitting in a buffer,
free. Multistep costs exactly one new call per step, so `boot + 4 steps × 1 = 5` gives *four* transitions.
Four steps versus two is not just more steps; each step's `λ`-increment `h` is roughly half as large, and
the order-2 method's global error scales like `h²`, so halving `h` shrinks the dropped error by about a
factor of four — at *identical* total cost. Multistep wins on step count and on per-step accuracy
simultaneously; under a tight budget it is the only sane choice. With one previous time `u`, the first
derivative is a backward difference `x̂_t^{(1)} ≈ (x̂_t − x̂_u)/h_1`, `h_1 = λ_t − λ_u`. For third order I
keep two previous times and fit the unique quadratic through the three most recent outputs, reading off
its first and second derivatives at the current node. The buffer that makes this free is a small
ring of past predictions and their `λ`-nodes: each step pushes the freshly computed `x̂_0` and its `λ`
onto the buffer and pops the oldest, so the derivative estimate at step `n` reads the outputs from steps
`n−1` and `n−2` that were already paid for. There is no extra denoiser call anywhere in the loop — the
one call per step is the single new prediction, and the whole high-order machinery is arithmetic on
tensors already in memory. That is the entire trick: the accuracy is bought with finite differences of
history, and history is free.

Reading the second-order step in the harness's own form makes the "free accuracy" concrete. The update
is `x = x_old·(c_t/c_s) + x_T·(a_t − a_s·(c_t/c_s)) + c_t · integral`, where the first two terms are the
exact linear and endpoint parts — byte-for-byte the DBIM re-projection — and `integral = e^{λ_t}·[(1 −
e^{-h})·x̂_0 + (e^{-h} + h − 1)·(x̂_0 − x̂_u)/h_2]`. The first bracket is the frozen-predictor term I
verified equals the DBIM `coeff_x0_hat`; the second bracket is the *entire* new content of second order,
the φ_1-weighted backward difference `(x̂_0 − x̂_u)/h_2` that estimates how fast the prediction is
changing in `λ` and corrects the freeze. So the second-order solver is literally "DBIM plus one
correction term built from the previous prediction," and that term vanishes as `x̂_0 → x̂_u` — i.e. it
does nothing on the easy workloads where the prediction is stable, and does the most on DIODE where the
prediction swings. The math is telling me in advance that the gain will be concentrated exactly on the
workload that stalled, which is precisely the falsifiable claim I want to make.

The unequal spacing of my grid forces the divided-difference weights to look ugly, and I want to be sure
they are right rather than plausible, so I check them against the textbook case. The third-order formulas
are `dx̂ = ((x̂ − x̂_{u1})(2h_1 + h_2)/h_1 − (x̂_{u1} − x̂_{u2}) h_1/h_2)/(h_1 + h_2)` and
`d²x̂ = 2((x̂ − x̂_{u1})/h_1 − (x̂_{u1} − x̂_{u2})/h_2)/(h_1 + h_2)`. Set `h_1 = h_2 = η` (a uniform grid)
and they collapse: `dx̂ → (3x̂ − 4x̂_{u1} + x̂_{u2})/(2η)`, the standard second-order backward difference
for a first derivative, and `d²x̂ → (x̂ − 2x̂_{u1} + x̂_{u2})/η²`, the standard central second difference.
So the unequal-spacing weights are the correct generalization — they reduce to the known formulas exactly
when the spacing is equal — and I am not silently introducing a consistency error on the non-uniform
schedule I am forced into. That check cost me a minute and it is the difference between a solver that is
third order and one that quietly is not.

The non-uniformity, and a couple of edge cases, fall out of the same booting-noise logic that DBIM
already needed — this is where the reflection on the previous rung becomes concrete machinery. The very
first step is still singular (`c_{t_max} = 0`), so before the loop I take the same stochastic booting
sample DBIM used: predict at `t_max`, draw the booting noise, seed the first interior state with
`a x_T + b x̂_0 + c · noise`. That is the one shot of stochasticity, and it is the same latent, returned
in the sixth slot. Then the first ordinary loop transition has no trustworthy history in `λ` yet — only
the boot prediction at the pinned endpoint, where `λ(t_max)` is effectively singular — so that step must
drop to first order. Note this also means that even if I ask for order 3, the *second* transition can
only be second order, because after boot plus one first-order step the buffer holds a single usable past
prediction, not two; the third-order branch cannot fire until step index two. And at the very last step
it is prudent to drop back to first order too ("lower-order final"): the derivative estimates get noisy
as `h` shrinks near the sharp endpoint and the predictor's outputs cluster, and a clean low-order finish
avoids amplifying that noise into the final image. So the schedule is: boot sample, order-1 first
transition, order-`k` multistep in the middle, order-1 finish, with the buffer of past outputs filling in
between. This is exactly the lower-order-final logic that keeps DBIM's endpoint sharp, lifted to the
high-order solver, and it spends the *same* five calls — one booting prediction plus four transitions —
so any gain is pure accuracy.

Two grounding details where the harness's edit differs from the clean derivation, and I should be honest
about them. First, the canonical high-order solver *raises* on an unsupported order, but the harness's
editable `sample_dbim` silently *coerces* `order ∉ {2, 3}` to `order = 2` — a defensive default so a bad
caller still runs at second order rather than crashing the evaluation. Second, the edit prints a
"Step order N" line on rank 0 each iteration, which is pure diagnostics and changes nothing
numerically. The default is `order = 2`, `lower_order_final = True`, and the signature drops the `eta`
argument entirely (this variant is the deterministic `η = 0` solver — there is no stochasticity dial,
the only noise is the boot). Everything else — the booting prediction, the mask re-blend each call, the
six-value return with the booting noise in the sixth slot — is identical to DBIM. The full module is in
the answer.

So the falsifiable expectations against DBIM, sharpened by exactly where DBIM failed. The high-order
solver spends the *same* five calls as DBIM — one booting prediction plus four transitions — so any
gain is pure accuracy, not extra compute, and I should predict on each metric column separately. On the
two easy workloads, where DBIM was already within a percent or two of the model's floor, I expect only a
modest but real improvement: `best_fid_edges2handbags` should edge below 5.180 (I would guess ≈5.0) and
`best_fid_Imagenet` below 6.070 (≈5.5), because the second-order correction sharpens transitions that
first order was already getting mostly right — there is not much room left there. The decisive test is
**`best_fid_DIODE`**, where first-order Euler stalled at 15.002, tied with the 50-NFE reference: if the
diagnosis is right — that DIODE's error was the flat-predictor approximation, not the family or the
budget — then the high-order solver should *move* DIODE for the first time, dropping it below 15 (I would
expect ≈14.3). The DIODE wall-clock should sit essentially where DBIM's did, ≈270 seconds, since the call
count is unchanged. If DIODE improves while the easy workloads hold or edge down, that confirms the
bottleneck on hard translation was solver order, and the high-order multistep solver is the right
strongest *deterministic* sampler. But it also sets up the next question I cannot answer from inside this
family: this whole construction is built on DBIM's coupled bridge kernel, where the injectable
stochasticity is capped by the marginals (`√(c² − ρ²)` must stay real, `ρ ≤ c`), so it can take big
accurate *deterministic* steps but cannot reach the strong-noise regime — and on a one-to-many
translation, strong scheduled noise may be exactly what sharpens the conditional sample past where a
deterministic high-order solver can go.
