DBIM did exactly what I predicted on the two easy workloads and stalled exactly where I feared on the
third, and the split tells me precisely what to fix. Edges→Handbags fell from DDBM's 11.139 to **5.180**
and ImageNet inpainting from 10.556 to **6.070** — roughly halving the FID at one-tenth the calls, and
the DIODE wall-clock cratered from 2149 seconds to 271. The marginals-only lever paid off: clean
analytic large jumps beat fifty churned Heun steps. But DIODE came in at **15.002**, essentially tied
with DDBM's 15.811 — the deterministic-leaning first-order update bought almost nothing there. So the
diagnosis is sharp and it is not about stochasticity or the model: on the hard translation, *first-order
Euler is leaving accuracy on the table at five steps*. Each DBIM step approximates the true transition
by holding the predictor's output constant over the step, and on DIODE — where the predicted `x̂_0`
swings hard between large steps — that flat approximation accumulates real error. The fix has to keep
everything that worked (same marginal-preserving family, same booting noise, five calls) and only make
each step *more accurate*, ideally for free. That means understanding what continuous object the
deterministic DBIM step is discretizing, then integrating it to higher order.

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
were a different deterministic process I would be sampling the wrong distribution. The PF-ODE with the
score replaced by the data predictor is
`dx_t = [(f + g²/σ² − g²/2c²) x_t + g²a/2c² x_T − g²b/2c² x̂_θ] dt`. Expanding my clean ODE back into
`dx_t` via the product rule gives
`dx_t = [(c'/c) x_t + (a' − a c'/c) x_T + (b' − b c'/c) x̂_θ] dt`. Computing the three log-derivatives
from `f = (log α)'`, `g² = (σ²)' − 2(log α)' σ²` and `(1/SNR_t)' = g²/α²` gives `c'/c = f + g²/σ² −
g²/2c²`, `a' − a c'/c = g²a/2c²`, `b' − b c'/c = −g²b/2c²` — all three coefficients match the PF-ODE
term for term. So my ODE *is* the bridge PF-ODE, just written in coordinates where it is trivial to
integrate. The deterministic DBIM step is not an approximation to the PF-ODE; it is an exact
reparameterization, and a far friendlier one. That cleanness is the doorway to high order.

Why is it the doorway? Because the ODE is *semi-linear* — a linear-in-`x` part plus a nonlinear network
part — and I have already isolated the linear part. The whole disease of DDBM's Heun sampler was that it
treated the entire right-hand side as one opaque field and paid discretization error on the easy linear
part too. The lesson from fast diffusion solvers is to never let a generic integrator chew on the linear
part: cancel it analytically with variation-of-constants and only approximate the integral of the
*smooth* network output. Apply that here. Writing `dx_t = [A(t) x_t + B_T(t) x_T + B_θ(t) x̂_θ] dt` with
`A = c'/c`, the integrating factor is `e^{∫A} = e^{∫(log c)'} = c_t/c_s`. The `x_T` integral closes in
elementary form because `x_T` is constant, contributing `(a_t − (c_t/c_s) a_t) x_T`. The only genuinely
hard integral is the one with the network output, and to make it as simple as possible I change variable
to the bridge log-SNR `λ_t = log(b_t/c_t)` — since `b_t/c_t = √(SNR_t − SNR_T)`, this is the half-log
of the excess signal-to-noise over the endpoint, playing exactly the role `log(α/σ)` played for ordinary
diffusion. In this variable the exact solution from time `t` to `s < t` is
`x_s = (c_s/c_t) x_t + (a_s − (c_s/c_t) a_t) x_T + c_s ∫_{λ_t}^{λ_s} e^λ x̂_θ(x_{t_λ}, t_λ, x_T) dλ`.
The linear and endpoint parts are now *exact*; all the discretization error lives in that one
exponentially-weighted integral of the smooth network output — and *that* is the error DBIM's first
order was paying on DIODE.

To go higher order I Taylor-expand `x̂_θ` as a function of `λ` about the current node and integrate each
Taylor term against `e^λ` exactly. With `h` the step in `λ` (note `λ` increases as time decreases, so
stepping toward the smaller time means moving forward in `λ`, `h > 0`), the scalar integrals are
analytic by repeated integration by parts: `∫ e^λ dλ = e^{λ_s}(1 − e^{-h})`, `∫ (λ − λ_t) e^λ dλ =
e^{λ_s}(h − 1 + e^{-h})`, `∫ ½(λ − λ_t)² e^λ dλ = e^{λ_s}(h²/2 − h + 1 − e^{-h})`. These three
coefficients are the φ-functions of exponential integrators — I did not pick them, the integrals handed
them to me. Keeping just the first bracket term is first order and reduces exactly to the DBIM Euler
step I already have; keeping two terms is second order, three is third.

Now the decision that actually controls the cost under a five-call budget. I need the `λ`-derivatives of
the network output, and I have two roads. The single-step road inserts an extra intermediate timestep,
evaluates the network there, and finite-differences — but that is an *extra* denoiser call per step, so
a `k`-th order single-step method costs `k` calls per step and five calls buy only ⌊5/k⌋ steps. That is
catastrophic here: I cannot afford to halve my already-tiny step count. The multistep road,
Adams-Bashforth style, finite-differences against the predictor outputs I *already computed* at previous
timesteps — those are sitting in a buffer, free. Multistep costs exactly one new call per step, so five
calls buy five steps and each step's `h` is smaller, which shrinks the dropped `O(h^{k+1})` error too.
Under a tight budget multistep is the only sane choice. With one previous time `u`, the first derivative
is a backward difference `x̂_t^{(1)} ≈ (x̂_t − x̂_u)/h_1`, `h_1 = λ_t − λ_u`. For third order I keep two
previous times and fit the unique quadratic through the three most recent outputs, reading off its first
and second derivatives at the current node; the unequal-spacing divided-difference weights are what keep
the estimate consistent on the non-uniform schedule I am forced into.

That non-uniformity, and a couple of edge cases, fall out of the same booting-noise logic that DBIM
already needed — this is where the reflection on the previous rung becomes concrete machinery. The very
first step is still singular (`c_{t_max} = 0`), so before the loop I take the same stochastic booting
sample DBIM used: predict at `t_max`, draw the booting noise, seed the first interior state with
`a x_T + b x̂_0 + c · noise`. That is the one shot of stochasticity, and it is the same latent. Then the
first ordinary loop transition has no trustworthy history in `λ` yet — only the boot prediction at the
pinned endpoint, where `λ(t_max)` is effectively singular — so that step must drop to first order. And
at the very last step it is prudent to drop back to first order too ("lower-order final"): the
derivative estimates get noisy as `h` shrinks near the sharp endpoint and the predictor's outputs
cluster, and a clean low-order finish avoids amplifying that noise into the final image. So the schedule
is: boot sample, order-1 first transition, order-`k` multistep in the middle, order-1 finish, with the
buffer of past outputs filling in between. This is exactly the lower-order-final logic that keeps DBIM's
endpoint sharp, lifted to the high-order solver.

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
gain is pure accuracy, not extra compute. On the two easy workloads, where DBIM was already strong, I
expect a modest but real improvement: Edges→Handbags should edge below 5.180 (I would guess ≈5.0) and
ImageNet inpainting below 6.070 (≈5.5), because the second-order correction sharpens transitions that
first order was already getting mostly right. The decisive test is **DIODE**, where first-order Euler
stalled at 15.002 essentially tied with the 50-NFE reference: if the diagnosis is right — that DIODE's
error was the flat-predictor approximation, not the family or the budget — then the high-order solver
should *move* DIODE for the first time, dropping it below 15 (I would expect ≈14.3). If DIODE improves
while the easy workloads hold or edge down, that confirms the bottleneck on hard translation was solver
order, and the high-order multistep solver is the right strongest *deterministic* sampler. But it also
sets up the next question I cannot answer from inside this family: this whole construction is built on
DBIM's coupled bridge kernel, where the injectable stochasticity is capped by the marginals
(`√(c² − ρ²)` must stay real), so it can take big accurate *deterministic* steps but cannot reach the
strong-noise regime — and on a one-to-many translation, strong scheduled noise may be exactly what
sharpens the conditional sample past where a deterministic high-order solver can go.
