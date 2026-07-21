DBIM did what I predicted on the two easy workloads and stalled where I feared on the third, and the
split is the diagnosis. Edges→Handbags fell from DDBM's 11.139 to 5.180, a 53.5% cut — better than
halved. ImageNet inpainting fell from 10.556 to 6.070, 42.5%. And DIODE went from 15.811 to 15.002, a
mere `0.809/15.811 = 5.1%` cut — essentially a tie with the 50-NFE reference. So a five-call
bridge-specific sampler *matched* the fifty-call floor on DIODE while beating it by half on the easy
workloads. The marginals-only lever paid off handsomely where targets are tightly correlated with the
source, and almost nothing where they are not. The wall-clock came in as forecast — 271 seconds on
DIODE against the low-few-hundred I back-computed — so the budget arithmetic is sound and the family is
right; the failure is localized. It is not stochasticity (the default `η` gave DBIM whatever noise it
had and DIODE still stalled) and not the model (which reaches 5.180 on e2h). On the hard translation,
first-order Euler is leaving accuracy on the table at five steps.

Mechanically: each DBIM step at `η = 0` is the re-projection I derived for it — it holds the
predictor's output `x̂_0` constant across the whole interval and re-attaches the bridge mean at the new
time. That is first-order: it treats `x̂_0` as if it did not change between the two ends of a big step.
On Edges→Handbags and ImageNet the predicted clean image barely moves from step to step, so freezing it
is nearly exact; that is why those FIDs are already low. On DIODE the predicted `x̂_0` swings hard
between the large steps a five-call budget forces, so freezing it commits a real error the re-projection
carries faithfully to the endpoint. Before committing to "more accurate steps," I have to rule out the
reflex reading — DIODE *is* the one-to-many workload, so blame stochasticity, crank `η` up. I do not buy
it. DBIM already carried the default `η` noise and DIODE still tied the floor, and the coupled family
caps that noise hard: the marginal proof forces `ρ ≤ c`, so I cannot inject more than the full marginal
scale. Even at the ceiling I would be *roughening* the sample, not correcting the systematic error,
because the DIODE residue has the signature of a *bias*, not a variance — the same frozen-`x̂_0` error
in the same direction every seed, which no amount of symmetric fresh noise cancels. So the DIODE problem
is a first-order approximation of a fast-changing integrand, and I fix approximations by raising the
order, not by adding noise. That eliminates the stochastic-crank and points at the deterministic
accuracy of the step.

The fix has to keep everything that worked — same marginal-preserving family, same booting noise, five
calls — and only make each step *more accurate*, ideally for free. That means finding what continuous
object the deterministic DBIM step is discretizing, then integrating it to higher order.

Set the deterministic case `η = 0`, write `t_{n+1} = t`, `t_n = t − Δt`, and stare at the update:
`x_{t-Δt} = a_{t-Δt} x_T + b_{t-Δt} x̂_0 + (c_{t-Δt}/c_t)(x_t − a_t x_T − b_t x̂_0)`. The troublesome
factor is that `c_t` ratio, so divide through by `c_{t-Δt}` and regroup:
`x_{t-Δt}/c_{t-Δt} = x_t/c_t + (a_{t-Δt}/c_{t-Δt} − a_t/c_t) x_T + (b_{t-Δt}/c_{t-Δt} − b_t/c_t) x̂_0`.
That is a finite difference of `x_t/c_t` equal to finite differences of `a_t/c_t` and `b_t/c_t` weighting
the two endpoints — the Euler discretization of `d(x_t/c_t) = x_T d(a_t/c_t) + x̂_θ d(b_t/c_t)`. So the
natural state variable is `x_t/c_t`, not `x_t`, and the natural coordinates are `a_t/c_t`, `b_t/c_t`.
The noise scale `c_t` is exactly what was blowing up the linear part of the dynamics, and dividing it out
gives a clean low-curvature ODE — the bridge analogue of how DDIM became an Euler step on `x/√α`.

I check this is the *same* solution as the bridge PF-ODE, because a different deterministic process would
mean I am sampling the wrong distribution and every FID I quoted measures the wrong target. The PF-ODE
with the score replaced by the data predictor is `dx_t = [(f + g²/σ² − g²/2c²) x_t + g²a/2c² x_T −
g²b/2c² x̂_θ] dt`. Expanding my clean ODE back into `dx_t` via `d(x/c) = dx/c − x c'/c² dt` and computing
the log-derivatives from `f = (log α)'`, `g² = (σ²)' − 2(log α)' σ²`, `(1/SNR_t)' = g²/α²` gives
`c'/c = f + g²/σ² − g²/2c²`, `a' − a c'/c = g²a/2c²`, `b' − b c'/c = −g²b/2c²` — all three match the
PF-ODE term for term. So my ODE *is* the bridge PF-ODE, written in coordinates where it is trivial to
integrate. The deterministic DBIM step is not an approximation to the PF-ODE; it is an exact
reparameterization, and a far friendlier one. That cleanness is the doorway to high order.

Why the doorway? The ODE is *semi-linear* — a linear-in-`x` part plus a nonlinear network part — and I
have isolated the linear part, the exact waste I named at the floor. The lesson from fast diffusion
solvers is never to let a generic integrator chew on the linear part: cancel it analytically with
variation-of-constants and only approximate the integral of the *smooth* network output. Writing
`dx_t = [A(t) x_t + B_T(t) x_T + B_θ(t) x̂_θ] dt` with `A = c'/c`, the integrating factor is
`e^{∫A} = c_t/c_s`. The `x_T` integral closes in elementary form because `x_T` is constant. The only
genuinely hard integral is the network output, and I change variable to the bridge log-SNR
`λ_t = log(b_t/c_t)` — since `b_t/c_t = √(SNR_t − SNR_T)`, this is the half-log of the *excess*
signal-to-noise over the endpoint, the bridge analogue of `log(α/σ)` for ordinary diffusion. It is a
good coordinate, not just a slick one: as `t → t_max` the excess `SNR_t − SNR_T → 0` so `λ_t → −∞`, and
as `t → 0`, `λ_t → +∞`, so `λ` stretches the trajectory across the full real line, monotonically, with
the endpoint pin pushed out to `−∞` where its singularity can do no harm to an interior step. Getting the
*excess*-SNR form rather than the raw SNR is what makes the `t_max` end behave. In this variable the
exact solution from time `t` to `s < t` is
`x_s = (c_s/c_t) x_t + (a_s − (c_s/c_t) a_t) x_T + c_s ∫_{λ_t}^{λ_s} e^λ x̂_θ dλ`. The linear and
endpoint parts are now exact; all the discretization error lives in that one exponentially-weighted
integral of the smooth network output — and that is the error DBIM's first order was paying on DIODE.

The first-order case must reduce to the DBIM step I already have, or the reparameterization is not
faithful. Freeze `x̂_θ = x̂_0` and integrate: `c_s ∫ e^λ dλ = c_s(e^{λ_s} − e^{λ_t})`, and since
`e^λ = b/c` this is `(b_s − (c_s/c_t) b_t) x̂_0`. Adding the exact linear and endpoint parts recovers
exactly the DBIM `η = 0` re-projection. So the exponential integrator with only its leading term *is*
DBIM, coefficient for coefficient; the higher-order terms are pure additions on top of a step I already
trust.

To go higher order I Taylor-expand `x̂_θ` in `λ` about the current node and integrate each term against
`e^λ` exactly. With `h` the step in `λ` (`λ` increases as time decreases, so stepping toward the smaller
time means `h > 0`), repeated integration by parts gives `∫ e^λ dλ = e^{λ_s}(1 − e^{-h})`,
`∫ (λ − λ_t) e^λ dλ = e^{λ_s}(h − 1 + e^{-h})`, `∫ ½(λ − λ_t)² e^λ dλ = e^{λ_s}(h²/2 − h + 1 − e^{-h})`.
These are the φ-functions of exponential integrators — the integrals handed them to me. Keeping one
bracket is first order and reduces to the DBIM Euler step; two terms is second order, three is third.

Now the decision that controls the cost under five calls, and it is a genuine fork with a computable
answer. I need the `λ`-derivatives of the network output, and I have two roads. The single-step road
inserts an extra intermediate timestep, evaluates the network there, and finite-differences — an *extra*
denoiser call per step, so a `k`-th order single-step method costs `k` calls per step. With one booting
prediction spent, order-2 single-step gets `boot + 2 steps × 2 = 5` — only *two* transitions; order-3
cannot even complete a second full step. That is catastrophic: it halves or quarters my already-tiny
step count. The multistep road, Adams–Bashforth style, finite-differences against the predictor outputs
I *already computed* at previous timesteps — sitting in a buffer, free. Multistep costs one new call per
step, so `boot + 4 steps × 1 = 5` gives *four* transitions. Four versus two is not just more steps: each
step's `λ`-increment `h` is roughly half as large, and the order-2 global error scales like `h²`, so
halving `h` shrinks the dropped error about fourfold — at identical total cost. Multistep wins on step
count and per-step accuracy simultaneously; under a tight budget it is the only sane choice. With one
previous time `u`, the first derivative is the backward difference `x̂_t^{(1)} ≈ (x̂_t − x̂_u)/h_1`,
`h_1 = λ_t − λ_u`; for third order I keep two previous times and fit the unique quadratic through the
three most recent outputs. A small ring buffer of past predictions and their `λ`-nodes makes this free:
each step pushes the freshly computed `x̂_0` and pops the oldest, so the derivative estimate reads
outputs already paid for. No extra denoiser call anywhere — the one call per step is the single new
prediction, and the whole high-order machinery is arithmetic on tensors already in memory.

The second-order step in the harness's form makes the free accuracy concrete: `x = x_old·(c_t/c_s) +
x_T·(a_t − a_s·(c_t/c_s)) + c_t · integral` with `integral = e^{λ_t}·[(1 − e^{-h})·x̂_0 + (e^{-h} + h −
1)·(x̂_0 − x̂_u)/h_2]`. The first two terms are the exact linear and endpoint parts — the DBIM
re-projection — and the second bracket is the entire new content of second order: the φ_1-weighted
backward difference `(x̂_0 − x̂_u)/h_2` estimating how fast the prediction changes in `λ` and correcting
the freeze. It vanishes as `x̂_0 → x̂_u` — nothing on the easy workloads where the prediction is stable,
most on DIODE where it swings. The math says in advance the gain concentrates exactly on the workload
that stalled. The third-order divided-difference weights on the unequal grid are
`dx̂ = ((x̂ − x̂_{u1})(2h_1 + h_2)/h_1 − (x̂_{u1} − x̂_{u2}) h_1/h_2)/(h_1 + h_2)` and
`d²x̂ = 2((x̂ − x̂_{u1})/h_1 − (x̂_{u1} − x̂_{u2})/h_2)/(h_1 + h_2)`; on a uniform grid `h_1 = h_2` these
collapse to the standard second-order backward difference and central second difference, so they are the
correct generalization and I am not silently introducing a consistency error on the non-uniform schedule.

The non-uniformity and the edge cases fall out of the same booting-noise logic DBIM already needed. The
first step is still singular (`c_{t_max} = 0`), so before the loop I take DBIM's stochastic booting
sample: predict at `t_max`, draw the booting noise, seed the first interior state with
`a x_T + b x̂_0 + c · noise`, returned in the sixth slot. The first ordinary transition then has no
trustworthy `λ`-history yet — only the boot prediction at the singular endpoint — so it drops to first
order; and even with order 3 requested, the *second* transition can only be second order, because after
boot plus one first-order step the buffer holds a single usable past prediction, not two. At the very
last step it is prudent to drop back to first order too: the derivative estimates get noisy as `h`
shrinks near the sharp endpoint and the predictor outputs cluster, so a clean low-order finish avoids
amplifying that noise into the final image. So the schedule is: boot sample, order-1 first transition,
order-`k` multistep in the middle, order-1 finish — the same lower-order-final logic that keeps DBIM's
endpoint sharp, lifted to the high-order solver, spending the same five calls.

Two grounding details where what I ship differs from the clean derivation. First, the code silently
*coerces* `order ∉ {2, 3}` to `order = 2` — a defensive default so a bad caller
still runs at second order rather than crashing. Second, it prints a "Step order N" line on rank 0 each
iteration, pure diagnostics with no numeric effect. The default is `order = 2`, `lower_order_final =
True`, and the signature drops the `eta` argument entirely — this variant is the deterministic `η = 0`
solver, the only noise being the boot. Everything else — the booting prediction, the mask re-blend each
call, the six-value return with the booting noise in the sixth slot — is identical to DBIM. The full
module is in the answer.

The high-order solver spends the *same* five calls as DBIM, so any gain is pure accuracy, and I should
predict per column. On the two easy workloads, where DBIM was already within a percent or two of the
model's floor, I expect only a modest improvement — the second-order correction sharpens transitions
first order was already getting mostly right, and there is little room left. The decisive test is DIODE,
where first-order stalled at 15.002 tied with the reference: if the diagnosis is right — DIODE's error is
the flat-predictor approximation, not the family or the budget — the high-order solver should *move*
DIODE for the first time. The DIODE wall-clock should sit essentially where DBIM's
did, since the call count is unchanged. If DIODE improves while the easy workloads hold or edge down,
that confirms the bottleneck on hard translation was solver order. But it also sets up the question I
cannot answer from inside this family: the whole construction is built on DBIM's coupled kernel, where
injectable stochasticity is capped by the marginals (`√(c² − ρ²)` must stay real, `ρ ≤ c`), so it takes
big accurate *deterministic* steps but cannot reach the strong-noise regime — and on a one-to-many
translation, strong scheduled noise may be exactly what a deterministic high-order solver cannot go far
enough to give.
