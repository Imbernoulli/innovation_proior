The high-order solver did move DIODE for the first time, and the pattern of where it helped and where it
barely did is the clue I need. Edges→Handbags edged from 5.180 to **4.988**, ImageNet inpainting from
6.070 to **5.528**, and DIODE — the one that had been frozen at 15 — finally dropped from 15.002 to
**14.306**. So the diagnosis from the last rung held: the DIODE bottleneck was solver order, the flat
predictor approximation, and the second-order exponential integrator paid it down. But look at the
*magnitude* of the DIODE gain: 15.002 to 14.306 is real but small, and it is still an order of magnitude
worse than the easy translation. The high-order solver squeezed the deterministic family about as far as
it goes, and DIODE is still stuck in the double digits. That tells me the remaining error on hard
translation is *not* about taking more accurate steps along the same trajectory — it is about which
trajectory I am on at all. And the one degree of freedom this whole family never had is *how stochastic*
the sampling is, because DBIM and its high-order solver are built deterministic, with the only noise
being the single boot. On a genuinely one-to-many task like DIODE, scheduled stochasticity through the
trajectory may be exactly what carves the conditional sample sharp, and the deterministic family
structurally cannot reach it. So I want to step back and ask the structural question EDM taught the
diffusion world: the path you train on and the sampler you run are *separate* design problems, each with
its own large space — have I explored both? The deterministic family fixed the sampler accuracy on a
fixed path. I have not touched the *path*, and I have not touched the *stochasticity level* as a real
knob.

Take the path first, because the sampler lives on top of it. The bridge kernel I have been using comes
from Doob's h-transform, and its three coefficients `a_t, b_t, c_t` are all functions of the *same* two
schedule functions `α_t, σ_t` — they are braided together. I cannot change how much noise the path
carries in the middle without dragging the interpolation weights around, because both are functions of
`α, σ`. That coupling is not a law of nature; it is an artifact of starting from a reference SDE and
pinning it, and it means the family of paths I can build is much smaller than it looks. The clean fix is
to build the bridge *directly* as a flow map, `x_t = α_t x_0 + β_t x_T + γ_t z`, `z ~ N(0, I)`, and ask
only for boundary conditions that land the ends correctly (`α_0 = β_T = 1`, `α_T = β_0 = γ_0 = γ_T = 0`).
Now `α_t, β_t, γ_t` are three *independent* functions: I can set how much noise the path carries in the
middle without touching how it interpolates. The kernel is the same Gaussian
`N(α_t x_0 + β_t x_T, γ_t² I)`, and the DBIM family's coupled kernel is a particular choice of the three
functions, so I lose nothing — I only un-cramp the space. (In the harness's coefficient names, code `a_t`
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
its Fokker–Planck equation. The added drift contributes `−ε_t ∇·[(∇log p_t) p]`, and since
`(∇log p_t) p = ∇p_t`, that is `−ε_t ∇²p_t`, which exactly cancels the `+ε_t ∇²p` from the diffusion
term. What is left is the Fokker–Planck of the *original* ODE — the marginals are untouched, for *any*
non-negative `ε_t`. That is the thing the deterministic family never used: the noise level along the
path is a genuine extra degree of freedom, a function `ε_t` that sits on top of `α, β, γ` and changes
nothing about the distributions I sample from, only how the trajectory wanders between them. Plugging the
reparameterized score into the reverse SDE and simplifying, the drift collapses to a beautiful form:
`b = α̇_t x̂_0 + β̇_t x_T + (γ̇_t + ε_t/γ_t) ẑ_t`, with `ẑ_t = (X_t − α_t x̂_0 − β_t x_T)/γ_t` the
normalized residual — "move the clean estimate at rate `α̇`, move the endpoint at rate `β̇`, move along
the predicted noise direction at rate `γ̇ + ε/γ`," the `ε/γ` being exactly my new knob, with diffusion
`√(2ε)`. Set `ε_t = 0` and it is a pure ODE; set `ε_t = γ_t γ̇_t − (α̇_t/α_t) γ_t² = ½ g_t²` and I
recover DDBM's reverse SDE. DDBM was using one specific `ε_t` and calling it "the" SDE. So I parameterize
`ε_t = η (γ_t γ̇_t − (α̇_t/α_t) γ_t²)`, `η ∈ [0, 1]`, one scalar dialing from pure ODE to full
DDBM-strength noise.

Why does this reach where the high-order solver could not? Because of *how* I discretize. The honest
discretization is Euler on the SDE: `x_{t-h} ≈ x_t − b(t) h + √(2 ε_t h) z̄`. Rearrange it in the regime
`γ_{t-h}² − 2 ε_t h > 0` and it is *exactly* the DBIM update with `ρ² = 2 ε_t h`. So DBIM is my family
restricted to that positivity condition — and that restriction is the cap I have been fighting: when I
want aggressive noise, `γ_{t-h}² − 2 ε_t h` goes negative, the `√(γ² − ρ²)` in the DBIM form turns
imaginary, and the closed-form update is undefined. But the Euler form `x_t − b h + √(2 ε_t h) z̄` has no
positivity requirement — it is well-defined for any `ε_t ≥ 0`. That settles which discretization to ship:
the Euler-SDE form, not DBIM's closed form, precisely so I can run the strong stochasticity the previous
two rungs structurally could not. DIODE's residual error, the part the high-order deterministic solver
left on the table, is exactly the part I expect strong scheduled noise to attack.

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

The schedule choices fall out of arguments, not sweeps. For the time grid, with only five calls I cannot
space uniformly; I want steps bunched where the trajectory changes fastest and where sharpness is
decided, near the small-`t` endpoint. EDM's `ρ`-ramp does this, but here I take `ρ` *below* one (the edit
uses `rho_k = 0.6`) — the opposite of EDM's `ρ = 7` for unconditional generation, because there the hard
part is the noisy end while here the hard part is the sharp endpoint. For the derivatives `α̇, β̇, γ̇`
the sampler needs, finite-differencing near the boundaries where `γ ~ O(10⁻²)` would lose accuracy, so I
read them off *analytically* from the VP schedule: the harness exposes `get_f_g2` giving `f = (log α)'`
and `g²`, from which `α̇ = α f`, `ρ̇ = ½(ρ² + 1) g²/ρ`, and then the chain rule gives exact `a_d, b_d,
c_d` for the kernel coefficients. That analytic-derivative path is what keeps the strong-noise drift
stable near the boundary.

Now I have to be precise about how the harness's edit differs from the full method, because the authority
is the edit, not the generic construction — and there are three substantive trims. First, the editable
`sample_dbim` *ignores the caller's `eta` and `ts`*: it hardcodes the stochasticity at `churn = 0.3`
(not 1.0, not the caller's value) and builds its own Karras `ρ_k = 0.6` schedule from `sigma_min = 0.15`
and a `sigma_max_offset = 5e-4`, overriding the passed `ts` entirely — these are task-local
edges2handbags sweep values baked in. Second, and importantly, the full method has a *third* knob — a
base-distribution diversity fix, `π_T = π_cond * N(0, b² I)`, that lossy-compresses the input to restore
conditional diversity, since the stochasticity lemma proves more sampling noise alone cannot widen the
conditional. The harness edit **does not implement that knob at all**: it starts exactly from the source
`x_T` with no base smoothing, so the diversity-restoration story is not part of what this baseline runs.
Third, the edit re-applies the mask *after every update* (`x = x·mask + x_T·(1−mask)`), not just inside
the denoiser blend — for inpainting the unmasked border pixels must stay pinned, or the SDE noise
accumulates on the known region and FID explodes; this is a harness-specific inpainting guard that the
generic translation derivation does not need. The `eps` is also clamped to be non-negative for numerical
safety. The full module is in the answer.

So the falsifiable expectations against the high-order solver, the strongest deterministic rung. ECSI
spends the same five calls (one per step, no Heun double-eval, the budget-honoring discretization). Where
I expect the decoupled path plus strong scheduled noise to win most is exactly where the deterministic
family stalled: **DIODE**. The high-order solver only got DIODE to 14.306; I expect the Euler-SDE phase
with `η`-dialed noise to crack it open, dropping DIODE into the single digits — I would predict ≈6, a
roughly 2× improvement, because DIODE's residual was the part deterministic sampling cannot reach.
Edges→Handbags should also improve modestly, edging below 4.988 toward ≈4.2, as the symmetric path and
endpoint-deterministic finish sharpen the easy translation. The risk I cannot rule out is **ImageNet
inpainting**: strong mid-trajectory noise on a masked-completion task, even with the per-step mask guard,
may *hurt* rather than help, because inpainting wants the known region rock-stable and the freed region
filled coherently, and aggressive stochasticity can fight that — so I would not be surprised if ImageNet
regresses badly, even into the double digits, while DIODE and e2h improve. If ECSI wins decisively on
DIODE and e2h but pays for it on ImageNet, that is the honest profile of a sampler that traded inpainting
stability for translation sharpness — and it is the strongest baseline on the two translation workloads
that the deterministic family could not match.
