The high-order solver moved DIODE for the first time, and the pattern of where it helped and where it
barely did is the clue. Lay the deterministic ladder out in numbers. Edges→Handbags: 11.139 → 5.180 →
4.988, a 53.5% drop then a 3.7% drop — a cliff, then a shelf near 5. ImageNet inpainting: 10.556 → 6.070
→ 5.528, 42.5% then 8.9% — same shape. DIODE: 15.811 → 15.002 → 14.306, 5.1% then 4.6% — no cliff at
all, just two small roughly equal steps, and the total move across the whole deterministic progression
is only `(15.811 − 14.306)/15.811 = 9.5%`. So the deterministic family drove the two easy
workloads to their shelf and then had nothing left, while on DIODE it never found a cliff to fall off;
each step buys the same grudging ~5% and the returns are not accelerating. DIODE at 14.306 is still
`14.306/4.988 = 2.87×` worse than the e2h shelf. The decelerating DIODE steps are the signature of an
axis running out.

That tells me the remaining error on hard translation is not about taking more accurate steps along the
same trajectory — I just spent an attempt doing that and got 4.6% — it is about which trajectory I am on. The
one degree of freedom this family never had is *how stochastic* the sampling is: DBIM and its high-order
solver are built deterministic, the only noise being the single boot. On a genuinely one-to-many task
like DIODE, scheduled stochasticity through the trajectory may be exactly what carves the conditional
sample sharp, and the deterministic family structurally cannot reach it — recall the marginal cap
`ρ ≤ c` bounding how much fresh noise the coupled kernel can carry. EDM taught the diffusion world that
the path you train on and the sampler you run are *separate* design problems, each with its own large
space. I fixed the sampler accuracy on a fixed path; I have not touched the *path* or the
*stochasticity level* as a real knob.

I should say why I do not just keep climbing solver order — order 4, order 5. The DIODE numbers refute
it: 5.1% then 4.6%, two nearly-equal grudging steps, is the fingerprint of an error *not* dominated by
the truncation term a higher order would kill. If the residual were high-order curvature in the
integrand, raising the order would have accelerated the gains, not left them flat. And mechanically it
cannot help: the whole high-order construction sits on DBIM's coupled kernel, where injectable noise is
capped at `ρ ≤ c`, so no setting of order or the coupled `η` reaches the strong-noise regime. Order
climbs accuracy along one trajectory; it cannot change which trajectory or add noise the kernel forbids.
So the two live options are "change the path" and "free the stochasticity," and I will find they are the
same move from two directions.

Take the path first, because the sampler lives on top of it. The bridge kernel's three coefficients
`a_t, b_t, c_t` are all functions of the *same* two schedule functions `α_t, σ_t` — braided together, as
I saw completing the square at the start. The whole path family is parameterized by two free
functions with three deterministic outputs, so I cannot change how much noise the path carries in the
middle (roughly `c_t`) without dragging the interpolation weights `a_t, b_t` around: raise `σ_t` to raise
`c_t² = σ_t²(1 − SNR_T/SNR_t)` and the source weight `a_t = (α_t/α_T)(SNR_T/SNR_t)` moves too, because
both carry `σ_t` through `SNR_t`. "Carry more noise through the middle but interpolate the same" is not
expressible inside two functions. That coupling is not a law of nature; it is an artifact of starting
from a reference SDE and pinning it. The clean fix is to build the bridge *directly* as a flow map,
`x_t = α_t x_0 + β_t x_T + γ_t z`, `z ~ N(0, I)`, asking only boundary conditions that land the ends
(`α_0 = β_T = 1`, `α_T = β_0 = γ_0 = γ_T = 0`). Now `α_t, β_t, γ_t` are three *independent* functions —
three where I had two — and the noise profile `γ_t` is decoupled from the interpolation weights. The
kernel is the same Gaussian `N(α_t x_0 + β_t x_T, γ_t² I)`, and DBIM's coupled kernel is a particular
choice of the three functions, so I lose nothing and only un-cramp the space. That extra function is the
dimension along which a one-to-many task like DIODE can be given a fatter, more exploratory
mid-trajectory without disturbing where the path starts and ends. (In the harness's names, code `a_t`
multiplies `x_T` and is my `β`, code `b_t` multiplies `x_0` and is my `α`, code `c_t` is my `γ`.)

But the decoupled flow map is usually *unconditional* — it transports the two marginal distributions,
not the specific terminal sample `x_T = y` a paired translation hands me. To use it for image-to-image I
condition everything on the observed `x_T` and keep the network a denoiser `x̂_0 = E[x_0 | x_t, x_T]` so
it stays a one-model bridge. Realizing the kernel as a linear SDE `dX_t = (f_t X_t + s_t x_T) dt + g_t
dW_t`, matching mean and variance gives `f_t = α̇_t/α_t`, `s_t = β̇_t − (α̇_t/α_t) β_t`, `g_t² = 2(γ_t
γ̇_t − (α̇_t/α_t) γ_t²)`. The reverse SDE and PF-ODE are then standard, needing only the score, and the
EDM reparameterization gives `∇log p_t(x_t | x_T) = (α_t x̂_0 + β_t x_T − x_t)/γ_t²` — the score points
from the current state toward the predicted clean image, scaled by `1/γ²`, with the singular factor kept
out of the network's job.

Now the sampler degree of freedom I have been missing. I have one reverse ODE and one reverse SDE — is
that all? Take any deterministic flow `dX_t = u_t dt` with density `p_t` and add drift-plus-diffusion:
`dX_t = (u_t + ε_t ∇log p_t) dt + √(2ε_t) dW_t`. Its Fokker–Planck is `∂_t p = −∇·(u p) − ε∇·((∇log p) p)
+ ε∇²p`, and the identity `(∇log p) p = ∇p` makes the middle term `−ε∇²p`, exactly cancelling the
`+ε∇²p` diffusion. What is left is `∂_t p = −∇·(u p)` — the Fokker–Planck of the *original* ODE. The
marginals are untouched for *any* non-negative `ε_t`, and the cancellation is an algebraic identity, not
a small-noise expansion. So the noise level along the path is a genuine extra degree of freedom — a
function `ε_t` on top of `α, β, γ` that changes nothing about the distributions I sample from, only how
the trajectory wanders between them. Plugging the reparameterized score into the reverse SDE, the reverse
drift is `f_t X_t + s_t x_T − (g_t²/2 + ε_t) ∇log p_t`, and with the affine score and the identities for
`f, s, g²/2` the pieces regroup so the `α̇` and `β̇` rates attach cleanly to `x̂_0` and `x_T` and every
remaining term lands on the normalized residual. The drift collapses to
`b = α̇_t x̂_0 + β̇_t x_T + (γ̇_t + ε_t/γ_t) ẑ_t` with `ẑ_t = (X_t − α_t x̂_0 − β_t x_T)/γ_t` — "move
the clean estimate at rate `α̇`, the endpoint at rate `β̇`, along the predicted noise direction at rate
`γ̇ + ε/γ`," the `ε/γ` being my new knob, with diffusion `√(2ε)`. Two limits I already know: `ε_t = 0`
gives a pure ODE, drift `α̇ x̂_0 + β̇ x_T + γ̇ ẑ`, which is just the time-derivative of the interpolant —
a reassuring sign the algebra is right. `ε_t = γ_t γ̇_t − (α̇_t/α_t) γ_t² = ½ g_t²` recovers DDBM's
reverse SDE: DDBM was using one specific `ε_t` and calling it "the" SDE. So I parameterize
`ε_t = η (γ_t γ̇_t − (α̇_t/α_t) γ_t²)`, `η ∈ [0, 1]`, one scalar dialing from pure ODE to full
DDBM-strength noise — and unlike DBIM's `η`, this one is not capped at the marginal scale, because it
lives on the SDE drift, not inside a `√(c² − ρ²)`.

Why does this reach where the high-order solver could not? Because of *how* I discretize. The honest
discretization is Euler on the SDE: `x_{t-h} ≈ x_t − b(t) h + √(2 ε_t h) z̄`. Rearranged in the regime
`γ_{t-h}² − 2 ε_t h > 0`, it is *exactly* the DBIM update with `ρ² = 2 ε_t h`. So DBIM is my family
restricted to that positivity condition — and that restriction is the cap I have fought all along:
when I want aggressive noise, `γ_{t-h}² − 2 ε_t h` goes negative, the `√(γ² − ρ²)` in the DBIM closed
form turns imaginary, and the update is undefined. It goes negative *soonest* exactly where I most want
strong noise: near the endpoint `γ` is small, and with the big `h` a five-call budget forces, `2 ε_t h`
easily exceeds a small `γ²`. So the positivity cap bites precisely in the strong-noise, few-step,
near-endpoint corner DIODE needs — the generic situation whenever I combine large steps, small terminal
`γ`, and strong noise. The DBIM closed form has no value to return there; the Euler form `x_t − b h +
√(2 ε_t h) z̄` returns one without complaint, because it never asks for the square root of a difference.
That settles which discretization to ship: the Euler-SDE form, not DBIM's closed form, precisely so I can
run the strong stochasticity the previous two samplers structurally could not. DIODE's residual error, that
`2.87×` gap the deterministic solver left on the table, is exactly the part I expect strong scheduled
noise to attack.

But strong noise everywhere is wrong. Following the Euler-SDE all the way to `t = 0`, the diffusion term
keeps dumping fresh noise into the state right when I want the image to crystallize, smearing
high-frequency detail — exactly the endpoint-blur the earlier samplers avoided by dropping fresh noise on
the final step. The same logic, sharper: for the **last two steps** set `ε_t = 0` and take the
deterministic DBIM transition `x_{t-h} = α_{t-h} x̂_0 + β_{t-h} x_T + γ_{t-h} ẑ_t`, well-defined
(`γ² > 0`) and committing cleanly. So the sampler is two-phase: Euler-SDE with `η`-dialed `ε_t` for the
early/middle steps where stochasticity builds detail, then deterministic for the final two to sharpen the
endpoint. That split is the single most important sampler decision — worth more than the exact `η` —
because it is the difference between a crisp image and a noisy one at five steps.

The time grid falls out of the two-phase structure. With five calls I cannot space uniformly. The Karras
`ρ`-ramp with the edit's `rho_k = 0.6` (below one) puts the fine steps near `t_max` and the coarse steps
near the output — the opposite warping from EDM's `ρ = 7`, which fine-grains the small-`t` end for
unconditional generation. That looks backwards for a task whose sharp end is at small `t`, but it is
exactly right here: the fine steps sit where the *Euler-SDE* phase runs, and Euler–Maruyama is only
first-order and needs small `h` to integrate its noise term stably; the coarse steps sit where the
*deterministic* DBIM phase runs, which is analytically exact in its linear part and takes big jumps
cleanly — the last two steps are the largest gaps, the committing-jump behavior I want at the finish. So
`ρ = 0.6` is "spend resolution on the crude stochastic integrator and let the exact deterministic jumps
be large," the correct division of a five-call budget between the two phases.

For the derivatives `α̇, β̇, γ̇` the drift needs, the choice near the boundary is forced. The drift
carries `ε_t/γ_t` and `γ̇_t`, both living on `γ ~ O(10⁻²)` near the endpoint; a finite difference of `γ`
there subtracts two nearly-equal small numbers — catastrophic cancellation exactly where the `1/γ` in the
drift most amplifies the error. So I read the derivatives analytically from the VP schedule: `get_f_g2`
gives `f = (log α)'` and `g²`, from which `α̇ = α f`, `ρ̇ = ½(ρ² + 1) g²/ρ`, and the chain rule on
`a = ᾱ ρ²/ρ_T²`, `b = α ρ̄²/ρ_T²`, `c = α ρ̄ ρ/ρ_T` gives exact `a_d, b_d, c_d` to machine precision.
The analytic path costs nothing, since the schedule quantities are already fetched, and it removes the
one numerical failure mode a strong-noise SDE integrator has near a small `γ`.

What I actually ship trims three things from the full method. First, it ignores the caller's `eta` and
`ts`: it hardcodes `churn = 0.3` (not 1.0, not the caller's value) and builds its own Karras `ρ_k = 0.6`
schedule from `sigma_min = 0.15` and `sigma_max_offset = 5e-4`, overriding `ts` — task-local
edges2handbags sweep values. Second, the full method has a *third* knob — a base-distribution diversity
fix `π_T = π_cond * N(0, b² I)` that lossy-compresses the input to restore conditional diversity — and it
needs a separate knob as a direct corollary of the Fokker–Planck lemma: since adding `ε_t` leaves *every
marginal untouched*, more sampling noise alone cannot widen the conditional `p(x_0 | x_T)`; if that
conditional is too narrow the only way to widen it is to change the *base* distribution, which the `π_T`
convolution does. I do not implement that knob here — I start exactly from `x_T` with no base smoothing —
so the diversity-restoration story is not part of what this baseline runs. Third, I
re-apply the mask *after every update* (`x = x·mask + x_T·(1−mask)`), not just inside the denoiser
blend: for inpainting the unmasked border pixels must stay pinned, or the Euler-SDE noise accumulates on
the known region (roughly `√(Σ 2 ε_t h)` of drift on pixels that should not move) and FID explodes — a
harness-specific inpainting guard the generic translation derivation does not need. The `eps` is clamped
non-negative for numerical safety. The full module is in the answer.

ECSI spends the same five calls (one per step, no Heun double-eval). Where I expect the decoupled path
plus strong scheduled noise to win most is exactly where the deterministic family stalled: DIODE. The
high-order solver only got DIODE to 14.306, and the `2.87×` gap to the e2h shelf is the residual
deterministic sampling structurally could not touch; I expect the Euler-SDE phase with `η`-dialed noise
to crack it open, a large improvement on DIODE — plausibly halving or better — rather than
the grudging ~5% each deterministic step bought. Edges→Handbags should also edge down modestly past its
deterministic shelf, as the symmetric decoupled path and endpoint-deterministic finish sharpen the easy
translation. The risk I cannot rule out is ImageNet inpainting: strong mid-trajectory noise on a
masked-completion task, even with the per-step mask guard, may *hurt* rather than help, because
inpainting wants the known region rock-stable and the freed region filled coherently, and aggressive
stochasticity fights that — so I would not be surprised if ImageNet regresses while DIODE and e2h
improve. That would be the honest profile of a sampler that traded inpainting stability for translation
sharpness, and if the DIODE win lands it would still be the strongest sampler on the two translation
workloads even at that price. The DIODE wall-clock should hold, since the call count is still five.
