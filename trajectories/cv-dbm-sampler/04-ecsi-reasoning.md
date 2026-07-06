The high-order solver did move DIODE for the first time, and the pattern of where it helped and where it
barely did is the clue I need. Let me lay the whole deterministic ladder out in numbers, because the
trajectory of the numbers is the diagnosis. Edges→Handbags: 11.139 (DDBM) → 5.180 (DBIM) → 4.988
(high-order), a 53.5% drop then a 3.7% drop — it fell off a cliff and then flattened onto a shelf near 5.
ImageNet inpainting: 10.556 → 6.070 → 5.528, 42.5% then 8.9% — same shape, cliff then shelf. DIODE:
15.811 → 15.002 → 14.306, `5.1%` then `4.6%` — no cliff at all, just two small, roughly equal steps, and
the total move across the entire three-rung deterministic ladder is only `(15.811 − 14.306)/15.811 =
9.5%`. So the deterministic family drove the two easy workloads down to their shelf and then had nothing
left, while on DIODE it never found a cliff to fall off in the first place; each rung buys the same
grudging ~5% and the returns are not accelerating. And the gap that remains is stark: DIODE at 14.306 is
`14.306/4.988 = 2.87×` worse than the e2h shelf. The high-order solver squeezed the deterministic family
about as far as it goes — the decelerating DIODE steps are the signature of an axis running out — and
DIODE is still stuck almost three times worse than the easy translation.

That tells me the remaining error on hard translation is *not* about taking more accurate steps along the
same trajectory — I just spent a whole rung doing that and got 4.6% — it is about which trajectory I am on
at all. And the one degree of freedom this whole family never had is *how stochastic* the sampling is,
because DBIM and its high-order solver are built deterministic, with the only noise being the single boot.
On a genuinely one-to-many task like DIODE, scheduled stochasticity through the trajectory may be exactly
what carves the conditional sample sharp, and the deterministic family structurally cannot reach it —
recall the marginal cap `ρ ≤ c` that bounds how much fresh noise the coupled kernel can carry. So I want
to step back and ask the structural question EDM taught the diffusion world: the path you train on and
the sampler you run are *separate* design problems, each with its own large space — have I explored both?
The deterministic family fixed the sampler accuracy on a fixed path. I have not touched the *path*, and I
have not touched the *stochasticity level* as a real knob. I should count the degrees of freedom I have
actually been using, because I suspect it is fewer than I think.

The obvious next move inside the current family is to keep climbing solver order — order 4, order 5
multistep — and I should say why I do not. The DIODE numbers already refute it: order 1 bought 5.1%,
order 2 bought 4.6%, and those two nearly-equal grudging steps are the fingerprint of an error that is
*not* dominated by the truncation term a higher order would kill. If the residual were high-order
curvature in the integrand, raising the order would have accelerated the gains, not left them flat; a
flat ~5% per rung says the remaining DIODE error lives somewhere the deterministic exponential integrator
cannot see at all. And there is a mechanical reason it cannot: the entire high-order construction sits on
DBIM's coupled kernel, where injectable noise is capped at `ρ ≤ c` for the marginal proof to hold, so no
setting of order *or* the coupled `η` reaches the strong-noise regime. Order climbs accuracy along one
trajectory; it cannot change which trajectory, and it cannot add noise the kernel forbids. So the two
live options are really "change the path" and "free the stochasticity," and I will find they are the same
move made from two directions.

Take the path first, because the sampler lives on top of it. The bridge kernel I have been using comes
from Doob's h-transform, and its three coefficients `a_t, b_t, c_t` are all functions of the *same* two
schedule functions `α_t, σ_t` — they are braided together, as I saw when I completed the square to get
them at the very first rung. Count it: the whole path family is parameterized by two free functions, and
`a, b, c` are three deterministic outputs of those two. So I cannot change how much noise the path carries
in the middle (that is roughly `c_t`) without dragging the interpolation weights `a_t, b_t` around, because
all three move together when `α, σ` move. That coupling is not a law of nature; it is an artifact of
starting from a reference SDE and pinning it, and it means the family of paths I can build is much smaller
than it looks. The clean fix is to build the bridge *directly* as a flow map,
`x_t = α_t x_0 + β_t x_T + γ_t z`, `z ~ N(0, I)`, and ask only for boundary conditions that land the ends
correctly (`α_0 = β_T = 1`, `α_T = β_0 = γ_0 = γ_T = 0`). Now `α_t, β_t, γ_t` are three *independent*
functions — three free functions where I had two, and crucially the noise profile `γ_t` is now
decoupled from the interpolation weights `α_t, β_t`. The kernel is the same Gaussian
`N(α_t x_0 + β_t x_T, γ_t² I)`, and the DBIM family's coupled kernel is a particular choice of the three
functions, so I lose nothing — I only un-cramp the space. To see how cramped it was, look at the coupled
coefficients: `c_t² = σ_t²(1 − SNR_T/SNR_t)` and `a_t = (α_t/α_T)(SNR_T/SNR_t)` both carry `σ_t` through
`SNR_t = α_t²/σ_t²`, so the moment I try to raise the mid-path noise `c_t` by raising `σ_t`, the
interpolation weight `a_t` on the source moves too, and the endpoint weight `b_t` with it. There is no
way, inside two functions, to say "carry more noise through the middle but interpolate the same" — the
statement is not expressible. With three independent functions it becomes a single free choice of `γ_t`
holding `α_t, β_t` fixed. That extra function is the whole point: it is the dimension along which a
one-to-many task like DIODE can be given a fatter, more exploratory mid-trajectory without disturbing
where the path starts and ends. (In the harness's coefficient names, code `a_t`
multiplies `x_T` and is my `β`, code `b_t` multiplies `x_0` and is my `α`, code `c_t` is my `γ`.)

But the decoupled flow map, as usually formulated, is *unconditional* — it transports the two marginal
distributions, not the specific terminal sample `x_T = y` a paired translation hands me. To use it for
image-to-image I have to put endpoint conditioning *into* it: condition everything on the observed
`x_T`, and keep the network a denoiser `x̂_0 = E[x_0 | x_t, x_T]` so it stays a one-model bridge. Then I
need the generative SDE for the conditional density expressed through that denoiser. Realize the kernel
as a linear SDE `dX_t = (f_t X_t + s_t x_T) dt + g_t dW_t`: matching the mean evolution gives
`f_t = α̇_t/α_t`, `s_t = β̇_t − (α̇_t/α_t) β_t`, and matching the variance gives
`g_t² = 2(γ_t γ̇_t − (α̇_t/α_t) γ_t²)`. The reverse SDE and PF-ODE are then standard, needing only the
score, and the EDM reparameterization gives the clean affine relation
`∇log p_t(x_t | x_T) = (α_t x̂_0 + β_t x_T − x_t)/γ_t²` — the score just points from the current state
toward the predicted clean image, scaled by `1/γ²`, with the singular factor kept out of the network's
job.

Now the sampler degree of freedom I have been missing, and this is the heart of the move. I have one
reverse ODE and one reverse SDE — is that all? Take any deterministic flow `dX_t = u_t dt` with density
`p_t` and consider adding drift-plus-diffusion: `dX_t = (u_t + ε_t ∇log p_t) dt + √(2ε_t) dW_t`. Write
its Fokker–Planck equation and track what the two added pieces do. The Fokker–Planck of the augmented
process is `∂_t p = −∇·((u + ε∇log p) p) + ε∇²p = −∇·(u p) − ε∇·((∇log p) p) + ε∇²p`. The middle term is
where the identity bites: `(∇log p) p = (∇p/p)·p = ∇p`, so `∇·((∇log p) p) = ∇·(∇p) = ∇²p`, and the
middle term is `−ε∇²p`, which exactly cancels the `+ε∇²p` from the diffusion. What is left is
`∂_t p = −∇·(u p)` — the Fokker–Planck of the *original* ODE. The marginals are untouched, for *any*
non-negative `ε_t`, and the cancellation is exact rather than approximate because it is an algebraic
identity, not a small-noise expansion. That is the thing the deterministic family never used: the noise
level along the path is a genuine extra degree of freedom, a function `ε_t` that sits on top of `α, β, γ`
and changes nothing about the distributions I sample from, only how the trajectory wanders between them.
Plugging the reparameterized score into the reverse SDE is worth doing rather than quoting, because the
cancellations are the point. The reverse drift is the forward drift `f_t X_t + s_t x_T` minus
`(g_t²/2 + ε_t) ∇log p_t`, and with the affine score `∇log p_t = (α_t x̂_0 + β_t x_T − X_t)/γ_t² = ẑ_t/γ_t`
this becomes a sum of terms in `X_t`, `x̂_0`, `x_T`, and `ẑ_t`. Using `f_t = α̇/α`,
`s_t = β̇ − (α̇/α)β`, and `g_t²/2 = γγ̇ − (α̇/α)γ²`, the `X_t` and `x_T` pieces regroup so that the `α̇`
and `β̇` rates attach cleanly to `x̂_0` and `x_T`, and every remaining term lands on the normalized
residual `ẑ_t`. The drift collapses to the
beautiful form `b = α̇_t x̂_0 + β̇_t x_T + (γ̇_t + ε_t/γ_t) ẑ_t`, with
`ẑ_t = (X_t − α_t x̂_0 − β_t x_T)/γ_t` the normalized residual — "move the clean estimate at rate `α̇`,
move the endpoint at rate `β̇`, move along the predicted noise direction at rate `γ̇ + ε/γ`," the `ε/γ`
being exactly my new knob, with diffusion `√(2ε)`. Check the two limits I already know: set `ε_t = 0` and
it is a pure ODE, drift `α̇ x̂_0 + β̇ x_T + γ̇ ẑ` with no diffusion — which is just the time-derivative of
the interpolant `x_t = α x_0 + β x_T + γ z` with `x̂_0` in for `x_0` and `ẑ` in for `z`, a reassuring
sign I did the algebra right, since the deterministic flow should be nothing but "differentiate the
interpolant." Set `ε_t = γ_t γ̇_t − (α̇_t/α_t) γ_t² = ½ g_t²` and I recover DDBM's reverse SDE. DDBM
was using one specific `ε_t` and calling it "the" SDE. So I parameterize
`ε_t = η (γ_t γ̇_t − (α̇_t/α_t) γ_t²)`, `η ∈ [0, 1]`, one scalar dialing from pure ODE to full
DDBM-strength noise — and unlike DBIM's `η`, this one is not capped at the marginal scale, because it
lives on the SDE drift, not inside a `√(c² − ρ²)`.

Why does this reach where the high-order solver could not? Because of *how* I discretize, and this is
where the free knob meets a hard wall in the old family. The honest discretization is Euler on the SDE:
`x_{t-h} ≈ x_t − b(t) h + √(2 ε_t h) z̄`. Rearrange it in the regime `γ_{t-h}² − 2 ε_t h > 0` and it is
*exactly* the DBIM update with `ρ² = 2 ε_t h`. So DBIM is my family restricted to that positivity
condition — and that restriction is the cap I have been fighting all three rungs: when I want aggressive
noise, `γ_{t-h}² − 2 ε_t h` goes negative, the `√(γ² − ρ²)` in the DBIM closed form turns imaginary, and
the update is undefined. Worse, it goes negative *soonest* exactly where I most want strong noise: near
the endpoint `γ` is small, and with the big `h` a five-call budget forces, `2 ε_t h` easily exceeds a
small `γ²`. So the positivity cap bites precisely in the strong-noise, few-step, near-endpoint corner —
the corner DIODE needs. Put a number on it from my own grid: near the output `γ` is around the schedule's
small values, and at the last stochastic step the `λ`- or `t`-gap `h` is the largest, roughly `0.3` on
the `rho_k = 0.6` grid I will compute below. If `γ ≈ 0.15` there, then `γ² ≈ 0.0225`, and with `ε_t` near
its `η → 1` value the product `2 ε_t h` is of order the step size times a variance comparable to `γ²`
itself — it does not take much before `2 ε_t h > γ²` and `√(γ² − 2 ε_t h)` is the root of a negative
number. The DBIM closed form simply has no value to return there; the Euler form returns `x_t − b h +
√(2 ε_t h) z̄` without complaint. This is not a corner case I am inventing to justify the choice — it is
the generic situation whenever I combine large steps, small terminal `γ`, and the strong noise DIODE
wants, and it is why the closed form was never going to reach this regime no matter how I tuned it. But the Euler form `x_t − b h + √(2 ε_t h) z̄` has no positivity requirement — it
is well-defined for any `ε_t ≥ 0`, because it never asks for the square root of a difference. That settles
which discretization to ship: the Euler-SDE form, not DBIM's closed form, precisely so I can run the
strong stochasticity the previous two rungs structurally could not. DIODE's residual error, the part the
high-order deterministic solver left on the table — that `2.87×` gap — is exactly the part I expect strong
scheduled noise to attack.

But strong noise everywhere is wrong, and here the reflection on the previous rungs pays off directly. If
I follow the Euler-SDE all the way to `t = 0`, the diffusion term `√(2 ε_t h) z̄` keeps dumping fresh
noise into the state right when I want the image to crystallize, smearing high-frequency detail — exactly
the endpoint-blur the previous rungs avoided by *dropping fresh noise on the final step*. The same logic
applies, only sharper: for the **last two steps** set `ε_t = 0` and take the deterministic DBIM
transition `x_{t-h} = α_{t-h} x̂_0 + β_{t-h} x_T + γ_{t-h} ẑ_t`, which is well-defined (`γ² > 0` always)
and commits cleanly. So the sampler is two-phase: Euler-SDE with the `η`-dialed `ε_t` for the
early/middle steps where stochasticity builds detail, then deterministic for the final two to sharpen
the endpoint. That two-phase split is the single most important sampler decision — worth more than the
exact `η` — because it is the difference between a crisp image and a noisy one at five steps.

The time grid then falls out of that two-phase structure, and here I want to actually compute where a
given `ρ` puts the steps rather than assert it, because the previous rungs taught me EDM's `ρ = 7` warps
one way and I need the other. With five calls I cannot space uniformly. Try the Karras `ρ`-ramp with the
edit's `rho_k = 0.6`, `t_hi ≈ 1.0`, `t_lo = 0.15`, five nodes: the grid comes out approximately
`[1.00, 0.85, 0.68, 0.47, 0.15]`, with successive gaps `0.15, 0.17, 0.21, 0.32` that *grow* as `t`
falls. So `rho_k = 0.6` (below one) makes the steps fine near `t_max` and coarse near the output — the
opposite warping from EDM's `ρ = 7`, which fine-grains the small-`t` end for unconditional generation. At
first this looks backwards for a task whose sharp end is at small `t`, but it is exactly right for the
two-phase sampler: the fine steps sit where the *Euler-SDE* phase runs, and Euler–Maruyama is only
first-order and needs small `h` to integrate its noise term stably; the coarse steps sit where the
*deterministic* DBIM phase runs, and that transition is analytically exact in its linear part and takes
big jumps cleanly — indeed the last two steps here are the two largest gaps (`0.32` and the drop from
`0.15` to `t_min`), which is precisely the committing-jump behavior I want at the finish. So `ρ = 0.6`
is not "bunch near the sharp endpoint"; it is "spend resolution on the crude stochastic integrator and
let the exact deterministic jumps be large," which is the correct division of a five-call budget between
the two phases. For the derivatives `α̇, β̇, γ̇` the sampler needs, I have to choose between finite-differencing them and
reading them off in closed form, and near the boundary the choice is forced. The drift carries `ε_t/γ_t`
and `γ̇_t`, both living on `γ ~ O(10⁻²)` near the endpoint; a finite difference of `γ` there estimates a
small number by subtracting two nearly-equal small numbers — catastrophic cancellation exactly where the
`1/γ` in the drift most amplifies the resulting error. So I read the derivatives *analytically* from the
VP schedule: the harness exposes `get_f_g2` giving `f = (log α)'` and `g²`, from which `α̇ = α f`,
`ρ̇ = ½(ρ² + 1) g²/ρ`, and then the chain rule on `a = ᾱ ρ²/ρ_T²`, `b = α ρ̄²/ρ_T²`, `c = α ρ̄ ρ/ρ_T`
gives exact `a_d, b_d, c_d` for the kernel coefficients — the true `α̇, β̇, γ̇` to machine precision, with
no truncation term for the `1/γ` to blow up. The analytic path costs nothing, since the schedule
quantities are already being fetched, and it removes the one numerical failure mode a strong-noise SDE
integrator has near a small `γ`; that is what keeps the strong-noise drift stable near the boundary.

Now I have to be precise about how the harness's edit differs from the full method, because the authority
is the edit, not the generic construction — and there are three substantive trims. First, the editable
`sample_dbim` *ignores the caller's `eta` and `ts`*: it hardcodes the stochasticity at `churn = 0.3`
(not 1.0, not the caller's value) and builds its own Karras `ρ_k = 0.6` schedule from `sigma_min = 0.15`
and a `sigma_max_offset = 5e-4`, overriding the passed `ts` entirely — these are task-local
edges2handbags sweep values baked in. Second, and importantly, the full method has a *third* knob — a
base-distribution diversity fix, `π_T = π_cond * N(0, b² I)`, that lossy-compresses the input to restore
conditional diversity. The reason it needs a separate knob is a direct corollary of the Fokker–Planck
lemma I just proved: since adding `ε_t` leaves *every marginal untouched*, more sampling noise alone
*cannot* widen the conditional `p(x_0 | x_T)` — the lemma guarantees I am still sampling the same
distribution no matter how large `ε_t` gets, so if that conditional is too narrow the only way to widen
it is to change the *base* distribution I condition on, which the `π_T` convolution does. The harness edit
**does not implement that knob at all**: it starts exactly from the source `x_T` with no base smoothing,
so the diversity-restoration story is not part of what this baseline runs. Third, the edit re-applies the
mask *after every update* (`x = x·mask + x_T·(1−mask)`), not just inside the denoiser blend — for
inpainting the unmasked border pixels must stay pinned, or the Euler-SDE noise `√(2 ε_t h) z̄` accumulates
on the known region over the stochastic steps (roughly `√(Σ 2 ε_t h)` of drift on pixels that should not
move at all) and FID explodes; this is a harness-specific inpainting guard that the generic translation
derivation does not need. The `eps` is also clamped to be non-negative for numerical safety. The full
module is in the answer.

So the falsifiable expectations against the high-order solver, the strongest deterministic rung. ECSI
spends the same five calls (one per step, no Heun double-eval, the budget-honoring discretization). Where
I expect the decoupled path plus strong scheduled noise to win most is exactly where the deterministic
family stalled: **`best_fid_DIODE`**. The high-order solver only got DIODE to 14.306, and the `2.87×` gap
to the e2h shelf is the residual deterministic sampling cannot reach; I expect the Euler-SDE phase with
`η`-dialed noise to crack it open, dropping DIODE into the single digits — I would predict ≈6, a roughly
2× improvement, because that residual was the part deterministic sampling structurally could not touch.
`best_fid_edges2handbags` should also improve modestly, edging below 4.988 toward ≈4.2, as the symmetric
decoupled path and the endpoint-deterministic finish sharpen the easy translation past its deterministic
shelf. The risk I cannot rule out is **`best_fid_Imagenet`**: strong mid-trajectory noise on a
masked-completion task, even with the per-step mask guard, may *hurt* rather than help, because inpainting
wants the known region rock-stable and the freed region filled coherently, and aggressive stochasticity
can fight that — so I would not be surprised if ImageNet regresses badly, plausibly into the double digits
or worse, while DIODE and e2h improve. That would be the honest profile of a sampler that traded
inpainting stability for translation sharpness — and even paying that price it is the strongest baseline
on the two translation workloads the deterministic family could not match. The DIODE wall-clock should
hold near ≈270 seconds, unchanged, since the call count is still five.
