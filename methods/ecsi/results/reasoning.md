Let me start from what actually hurts when I try to translate one image into another with a diffusion bridge. The premise is good: instead of pushing a target image out to Gaussian noise and crawling all the way back, I build a stochastic path whose two ends are the source `x_T` and the target `x_0`. A sketch and the handbag it depicts are already close in pixel space — much closer than the handbag is to white noise — so the path is short, and a short path should be cheap to walk and easy to learn. Zhou and colleagues made this concrete with Doob's `h`-transform: take a reference diffusion `dX_t = f̄_t X_t dt + ḡ_t dW_t` and pin it so it arrives almost surely at a fixed endpoint, which adds a drift `ḡ_t² ∇log p_{T|t}(x_T | X_t)`. The pinned process is a valid bridge between the two distributions, and conditioned on both ends its transition kernel is a clean Gaussian, `p_{t|0,T}(x_t|x_0,x_T) = N(x_t; α_t x_0 + β_t x_T, γ_t² I)`. To generate, I solve the reverse SDE or its probability-flow ODE from `t = T` down to `0`, both needing a score I estimate with a network.

So why am I not done? Stare at where `α_t, β_t, γ_t` come from in the `h`-transform construction. For the VP bridge they are `α_t = a_t(1 − SNR_T/SNR_t)`, `β_t = (a_t/a_T)(SNR_T/SNR_t)`, `γ_t² = σ_t²(1 − SNR_T/SNR_t)`, with `SNR_t = a_t²/σ_t²`. There are really only two free functions here, `a_t` and `σ_t`, and they are *convolved* — they appear braided together inside all three coefficients. I cannot change how much noise the path carries in the middle without simultaneously dragging the interpolation weights around, because both are functions of the same `a_t, σ_t`. That coupling is not a law of nature; it is an artifact of starting from a reference SDE and pinning it. It means the family of paths I can actually build is much smaller than it looks. And DDBM ships exactly one reverse SDE and its one matching ODE — there is no knob for how stochastic the *sampling* is, and the vanilla SDE sampler wants on the order of a hundred-plus denoiser calls. Under a five-call budget the FID is astronomical. Two separate pains: the path space is artificially cramped, and the only good sampler is far too slow.

Zheng and colleagues attacked the speed with DBIM, the bridge version of the DDIM trick. Their observation is sharp: the bridge score depends only on the marginals, so I can replace the Markovian bridge with a *non-Markovian* one that shares the very same marginals at the discretization points yet allows big jumps between them. That gives a closed-form update,

  x_{t_n} = α_{t_n} x̂_0 + β_{t_n} x_T + √(γ_{t_n}² − ρ_{t_n}²) · (x_{t_{n+1}} − α_{t_{n+1}} x̂_0 − β_{t_{n+1}} x_T)/γ_{t_{n+1}} + ρ_{t_n} ε,

with `ρ_{t_n}` a per-step stochasticity level that interpolates from deterministic (`ρ = 0`) up to noisy. This is genuinely fast, twenty-something times faster than the vanilla sampler. But two things still bother me. It is derived *inside* DDBM's coupled kernel, so it inherits exactly the cramped path family I was complaining about. And look at that `√(γ_{t_n}² − ρ_{t_n}²)`: the amount of stochasticity I can inject is capped by the marginals themselves. Equivalently, the discretization that produces this update needs `γ_{t-Δt}² − 2 ε_t Δt > 0`, where `ε_t` is the noise I add. The moment I want a strong noise schedule — one that injects as much as the full reverse SDE does — that argument can go negative, the square root turns imaginary, and the sampler is simply undefined. So DBIM cannot reach the settings I suspect are best. There is a third pain on top of these: when I hold a source image fixed and resample with different seeds, these bridges hand me back near-identical images. The injected diffusion noise barely moves the output. For a one-to-many task — one edge map, many plausible handbags in different colors — that is a real failure, and nobody has named it.

I want to step back and ask the structural question. EDM taught the unconditional-diffusion world a lesson I should steal: the path you train on and the sampler you run are *separate* design problems, each with its own large space, and you get big wins by decoupling them and searching each freely. So two questions for bridges. Have the *paths* been fully explored? And have the *samplers* been fully explored? Let me take them in that order, because the path space is what the sampler lives on top of.

For paths, the coupling is the disease, so I want a construction where the three coefficients are free. Albergo, Boffi and Vanden-Eijnden's stochastic interpolants give me exactly that: build the bridge directly as a flow map,

  x_t = α_t x_0 + β_t x_T + γ_t z,   z ~ N(0, I),

and ask only for boundary conditions that make the ends land where they should — `α_0 = β_T = 1`, `α_T = β_0 = γ_0 = γ_T = 0`, and positivity in between. Now `α_t, β_t, γ_t` are three *independent* functions of time. The transition kernel is the same Gaussian `N(α_t x_0 + β_t x_T, γ_t² I)`, but I am no longer forced to braid the interpolation with the noise: I can set how much noise the path carries in the middle without touching how it interpolates. That is a strictly larger and cleaner path family than the `h`-transform gives, and DDBM-VP and DDBM-VE fall out of it as particular choices of the three functions, so I haven't lost anything — I've only un-cramped the space.

There is a catch, and it is the reason I cannot just adopt stochastic interpolants wholesale: as formulated for generation they are *unconditional*. They transport `π_0` to `π_T` as marginal distributions, but they do not condition on the specific terminal sample `x_T = y` that a paired translation hands me. To use them for I2I I'd have to train two separate models and lose the single-denoiser convenience that makes bridges practical. So the move I need is to put endpoint conditioning *into* the interpolant: condition everything on the observed `x_T`, and make the network a denoiser `x̂_0 = E[x_0 | x_t, x_T]` that predicts the clean target. Then I need the generative SDE for the conditional density `q_t(X_t | x_T)`, expressed through that denoiser.

Let me derive it rather than guess it. I have the kernel `N(x_t; α_t x_0 + β_t x_T, γ_t² I)` and I want a forward SDE whose marginal *is* this Gaussian, in the linear Itô form `dX_t = (f_t X_t + s_t x_T) dt + g_t dW_t`. The mean of such an SDE obeys `dm_t/dt = f_t m_t + s_t x_T`. The kernel's mean is `m_t = α_t x_0 + β_t x_T`, so its time derivative is `α̇_t x_0 + β̇_t x_T`, and matching,

  α̇_t x_0 + β̇_t x_T = f_t(α_t x_0 + β_t x_T) + s_t x_T.

The `x_0` coefficients give `α̇_t = f_t α_t`, i.e. `f_t = α̇_t/α_t`. The `x_T` coefficients give `β̇_t = f_t β_t + s_t`, i.e. `s_t = β̇_t − (α̇_t/α_t) β_t`. Good. Now the variance: a linear SDE's variance `P_t = γ_t²` obeys `dP_t/dt = 2 f_t P_t + g_t²`, so `g_t² = d(γ_t²)/dt − 2 f_t γ_t² = 2 γ_t γ̇_t − 2(α̇_t/α_t) γ_t²`, that is

  g_t = √(2(γ_t γ̇_t − (α̇_t/α_t) γ_t²)).

So any decoupled `(α, β, γ)` is realized by a concrete linear SDE with these `f, s, g`. The reverse-time generative SDE and its probability-flow ODE are then the standard ones,

  dX_t = [f_t X_t + s_t x_T − g_t² ∇log p_t(X_t|x_T)] dt + g_t dW_t,
  dX_t = [f_t X_t + s_t x_T − ½ g_t² ∇log p_t(X_t|x_T)] dt,

and I just need the score. Here I take EDM's reparameterization, because the bare score has a `1/γ²` that detonates at the endpoints where `γ → 0`, and training through that is unstable. The L2-optimal denoiser is the conditional mean, `x̂_0(x_t, x_T, t) = E[x_0 | x_t, x_T]`; differentiating the Gaussian kernel and using that the marginal score is the posterior-averaged conditional score, I get the clean affine relation

  ∇_{x_t} log p_t(x_t | x_T) = (α_t x̂_0 + β_t x_T − x_t)/γ_t².

The score just points from the current state toward the predicted clean image, scaled by `1/γ²`. Parameterizing `x̂_0` directly with the network keeps the singular factor out of the network's job.

Now the sampler question, and this is where I think the real room is. I have one reverse ODE and one reverse SDE. Is that all? Let me think about what freedom I actually have once the *marginals are fixed*. Take any deterministic flow `dX_t = u_t dt` with density `p_t`. I claim I can add noise to it without changing a single marginal. Consider `dX_t = (u_t + ε_t ∇log p_t) dt + √(2ε_t) dW_t` for some `ε_t ≥ 0`. Write its Fokker–Planck equation:

  ∂_t p = −∇·[(u_t + ε_t ∇log p_t) p] + ε_t ∇²p
        = −∇·[u_t p] − ε_t ∇·[(∇log p_t) p] + ε_t ∇²p.

The middle term is `−ε_t ∇·[(∇log p_t) p]`, and `(∇log p_t) p = (∇p_t/p_t) p_t = ∇p_t`, so it is `−ε_t ∇·[∇p_t] = −ε_t ∇²p_t`, which exactly cancels the `+ε_t ∇²p` from the diffusion. What's left is `∂_t p = −∇·[u_t p]` — the Fokker–Planck of the *original* ODE. The marginals are untouched. The same cancellation runs the other way for the backward sign. So for *any* non-negative function `ε_t`, there is an SDE with that diffusion strength that produces identical marginals to the ODE. That is the thing I was missing. The noise level along the path is a genuine extra degree of freedom, a knob `ε_t` that sits on top of `α, β, γ` and changes nothing about the distributions I sample from — only how the trajectory wanders between them. The interpolants framework hinted at this ("the law of the interpolant can be realized by an ODE and by forward/backward SDEs"), but as a *sampler* knob, decoupled from the marginals, it has been left on the table.

So my sampling family is: pick `α, β, γ` (the path), then pick `ε_t` (the stochasticity), independently. Let me write the SDE in terms of the denoiser by plugging the reparameterized score into the reverse SDE, with the diffusion strength set by `ε_t`:

  dX_t = [ (α̇_t/α_t) X_t + (β̇_t − (α̇_t/α_t) β_t) x_T − (γ_t γ̇_t − (α̇_t/α_t) γ_t² + ε_t) · (α_t x̂_0 + β_t x_T − X_t)/γ_t² ] dt + √(2ε_t) dW_t.

This looks ugly, but it collapses. Let me push the algebra. The piece `(α̇_t/α_t) γ_t² · (α_t x̂_0 + β_t x_T − X_t)/γ_t²` is just `(α̇_t/α_t)(α_t x̂_0 + β_t x_T − X_t) = α̇_t x̂_0 + (α̇_t/α_t) β_t x_T − (α̇_t/α_t) X_t`. When I distribute the minus sign in the drift, that last `−(α̇_t/α_t) X_t` from inside cancels the leading `+(α̇_t/α_t) X_t`, and the `+(α̇_t/α_t) β_t x_T` from inside cancels the `−(α̇_t/α_t) β_t x_T` in `s_t`. The `X_t` term is gone entirely, the `x_T` coefficient reduces to plain `β̇_t`, and an `+α̇_t x̂_0` appears. What remains of the score term is `−(γ_t γ̇_t + ε_t)(α_t x̂_0 + β_t x_T − X_t)/γ_t²`. Factor `1/γ_t` out: `(γ_t γ̇_t + ε_t)/γ_t² = (γ̇_t + ε_t/γ_t)/γ_t`, and define `ẑ_t = (X_t − α_t x̂_0 − β_t x_T)/γ_t`, the normalized residual — the bridge's own estimate of the latent noise `z`. Then `−(γ_t γ̇_t + ε_t)(α_t x̂_0 + β_t x_T − X_t)/γ_t² = +(γ̇_t + ε_t/γ_t) ẑ_t`. The whole drift becomes

  b(t, X_t, x_T) = α̇_t x̂_0 + β̇_t x_T + (γ̇_t + ε_t/γ_t) ẑ_t,   dX_t = b dt + √(2ε_t) dW_t.

That is beautiful, and it is exactly the structure I want: the drift is "move the clean estimate at rate `α̇`, move the endpoint at rate `β̇`, and move along the predicted noise direction at rate `γ̇ + ε/γ`," with the extra `ε/γ` precisely the effect of my new stochasticity knob, and the diffusion `√(2ε)` adding fresh noise to match. Set `ε_t = 0` and it's a pure ODE — `b = α̇ x̂_0 + β̇ x_T + γ̇ ẑ`, which is just the time-derivative of the interpolant `α x̂_0 + β x_T + γ ẑ`, sanity-check passed. Crank `ε_t` up and I inject more noise. And here's the unification I should test, because if my family is the right one the old methods must be special cases, not competitors. If I set `ε_t = γ_t γ̇_t − (α̇_t/α_t) γ_t²` — which is exactly `½ g_t²`, the noise the original reverse SDE injects — I recover DDBM's reverse SDE. DDBM was using one specific `ε_t` and calling it "the" SDE; it never saw `ε_t` as a free function. And EDM falls out too: take `α_t = 1, β_t = 0, γ_t = σ_t`, so `ẑ_t = −σ_t ∇log p_t`; the drift is `−(σ_t σ̇_t + ε_t) ∇log p_t` with `√(2ε_t)` diffusion, the same stochastic correction that EDM uses when it adds churn on top of the probability-flow path. So the family genuinely contains the things I came from.

Now to actually *use* this I have to discretize. The honest discretization is Euler on the SDE. Let `h = t_i − t_{i-1} > 0` be the positive step size while I sample from `t_i` down to `t_{i-1}`. Since the drift `b` is written with respect to increasing time but the sampler moves backward,

  x_{t-h} ≈ x_t − b(t, x_t, x_T) h + √(2 ε_t h) z̄,   z̄ ~ N(0, I).

Let me check this connects to DBIM, because that tells me whether DBIM is inside my family. Suppose I'm at the regime where `γ_{t-h}² − 2 ε_t h > 0`. Then the Euler step can be rearranged into

  x_{t-h} ≈ α_{t-h} x̂_0 + β_{t-h} x_T + z̃,   z̃ = √(γ_{t-h}² − 2 ε_t h) · ẑ_t + √(2 ε_t h) · z̄.

The first two terms reconstruct the interpolant at the new time from the clean estimate and the endpoint, and `z̃` is a mixture of the *carried-over* noise direction `ẑ_t` (reweighted) and *fresh* noise `z̄`. To first order in `h`, this is the DBIM update with `ρ² = 2 ε_t h`. So DBIM is my family restricted to the case `γ_{t-h}² − 2 ε_t h > 0` — and that restriction is exactly the constraint I complained about. When I choose an aggressive `ε_t`, that quantity can go negative, the `√(γ_{t-h}² − 2 ε_t h)` becomes imaginary, and the DBIM-form update is undefined; but the Euler form `x_t − b h + √(2 ε_t h) z̄` has no such positivity requirement — it is well-defined for any `ε_t ≥ 0`. That settles which discretization to ship: use the Euler-SDE form, not the DBIM closed form, so I can run the strong stochasticity DBIM can't. (And as a bonus, I2SB falls out as the further special case where the `x_T` coefficient in the DBIM form vanishes, `2 ε_t h = γ_{t-h}² − β_{t-h}² γ_t²/β_t²` — so the whole zoo, DDBM, DBIM, I2SB, EDM, is a strict subset of this one sampler. That's the confirmation that I generalized correctly rather than inventing a fifth thing.)

What should `ε_t` actually be? I want a single dial that goes from pure ODE to "as stochastic as DDBM," because DDBM's total injected noise is a sensible reference point — it's the amount the principled reverse SDE uses. So set

  ε_t = η (γ_t γ̇_t − (α̇_t/α_t) γ_t²),   η ∈ [0, 1],

which is just `η` times the DDBM injection. `η = 0` is the deterministic ODE, `η = 1` is the full DDBM-strength SDE, and everything in between is reachable — including settings past where DBIM's positivity holds. One scalar, the whole stochasticity axis.

If I follow the Euler-SDE with this `ε_t` all the way to the end, I hit a wall. As `t → 0`, `γ_t → 0`. The drift has a `(γ̇_t + ε_t/γ_t) ẑ_t` term and the diffusion is `√(2 ε_t h)`; with `ε_t ∝ γ_t γ̇_t` the `ε_t/γ_t` stays finite, but the freshly injected `√(2 ε_t h) z̄` is still dumping noise into the state right when I want the image to *crystallize*. A crude Euler step near `t = 0` smears the final picture — high-frequency detail gets buried under last-minute noise, the endpoint is not sharp. I can see this without running anything: the last step is supposed to land me on a clean target, but an SDE step always adds a noise term, and at the smallest times that noise is no longer being "denoised away" by enough subsequent steps. So the tail of the trajectory wants to be *deterministic*. The fix is to stop injecting noise near the end: for the last couple of steps, set `ε_t = 0`. With `ε_t = 0` the DBIM-form update is not only well-defined (`γ_{t-h}² > 0` always holds), it's exact and cheap:

  x_{t-h} = α_{t-h} x̂_0 + β_{t-h} x_T + γ_{t-h} ẑ_t,

which is just "reconstruct the interpolant at the new time using the same carried noise direction `ẑ_t`, no fresh noise." So the sampler is two-phase: Euler-SDE with `ε_t = η(γ_t γ̇_t − (α̇_t/α_t) γ_t²)` for the early/middle steps where stochasticity helps build detail, then the deterministic DBIM transition for the final two steps to sharpen the endpoint. The early noise changes the trajectory and the local detail-building dynamics; it does not by itself widen the fixed-endpoint conditional marginal. The final deterministic steps commit cleanly. That two-phase split is the single most important sampler decision — it's worth more than which exact `η` I pick, because it's the difference between a crisp image and a noisy one at small NFE.

Now the schedule choices, which I should derive, not pull from a hat. First `α_t` and `β_t`. I'll restrict to the interpolating line, `α_t = 1 − β_t`, which puts the mean on the segment between `x_0` and `x_T`. Within that, is there any reason to prefer a fancy `β_t` over the identity `β_t = t`? Consider any invertible `β_t` giving the path `p_t^{(1)} = N((1−β_t) x_0 + β_t x_1, γ_t² I)`. Now define a second path `p_t^{(2)} = N((1−t) x_0 + t x_1, γ_{β_t^{-1}}² I)`. These two realize the *same* set of distributions — `p^{(2)}` at time `t` equals `p^{(1)}` at time `β_t^{-1}(t)` — so they have the same objective; they differ only in how `t` is distributed during training, which is a sampling-density choice, not a path choice. Since a curved `β_t` buys nothing the straight one can't, take the simplest: `α_t = 1 − t`, `β_t = t`. With these, `α̇_t = −1`, `β̇_t = 1`, and `α̇_t/α_t = −1/(1−t)` — all in closed form, no derivative approximation needed.

Next the noise envelope `γ_t`. It must vanish at both ends (`γ_0 = γ_T = 0`) and bulge in the middle. The natural one-parameter shape is `γ_t = 2 γ_max √(t^k(1−t^k))`, where `γ_max` sets the peak and `k` skews where the bulge sits. I want symmetry — noise should peak in the middle of the path, not be lopsided toward one endpoint, because a lopsided schedule (like DDBM-VP's, which piles noise near `x_T`) wastes modeling effort on one side; a symmetric one balances detail-building across the path. That's `k = 1`: `γ_t = 2 γ_max √(t(1−t))`, a clean symmetric arch. Then `γ̇_t = 2 γ_max (1 − 2t)/(2√(t(1−t))) = γ_max(1 − 2t)/√(t(1−t))`. For `γ_max`, the trade-off is visible without a sweep: too little noise (say `γ_max → 0`) and the path is nearly deterministic interpolation, so the model never learns to synthesize fine detail and the outputs look flat; too much and the middle of the path is so noisy that the bridge effectively forgets the source for a while, hurting fidelity. The sweet spot is small but nonzero — on the order of `0.125` to `0.25` — enough noise to carve detail, not so much that it drowns the conditioning.

Then the time-step placement under a tiny NFE budget. With only five denoiser calls I cannot space steps uniformly in `t`; I want them concentrated where the trajectory changes fastest and where sharpness is decided, which is near the small-`t` endpoint. EDM's `ρ`-parameterized ramp does exactly this: `t_i = (t_max^{1/ρ} + (i/N)(t_min^{1/ρ} − t_max^{1/ρ}))^ρ`, with `t_min = 0.001`, `t_max = 1 − 10^{-4}`. The exponent `ρ` controls the concentration; I want steps bunched toward `t_min`, so I take `ρ` *below* one (around `0.6`), which is the opposite of EDM's `ρ = 7` for unconditional generation — there the hard part is near the noisy end, here the hard part is the sharp endpoint, so I bias the schedule the other way.

Now the diversity problem, which I parked. Holding the source fixed, the bridge gives me almost the same image every time, because the source `x_T = x_cond` is a fully informative, fixed starting point — there's very little entropy for the noise to act on. The reflex is "inject more sampling noise," but the stochasticity-control lemma I just proved says that won't help: changing `ε_t` does not change the marginal `q(x_0 | x_T)`, so it cannot widen the conditional distribution; it only reshuffles trajectories that land in the same place. The real lever is the *base distribution* itself. Right now `π_T = π_cond` exactly. What if I smear it: `π_T = π_cond * N(0, b² I)`, convolving the source with a little Gaussian noise before starting? At first this seems strictly bad — by the data-processing inequality, blurring the input can only *lose* information about the target, never add it. But that's the point. The lost information was over-specifying the output; destroying a controlled amount of it (lossy compression of the input, exactly the regularization-by-information-bottleneck story from VAEs) lets the model fill the freed degrees of freedom with genuine variation — different colors, textures — while keeping the conditional structure. It interpolates between a pure bridge (`b = 0`, start from the exact source) and a pure diffusion model (`b → ∞`, start from noise). A small `b` is the intermediate regime that restores diversity without abandoning the conditioning. So `b` is a third knob, at the base distribution, orthogonal to the path `(α,β,γ)` and the sampler noise `ε`.

Let me also nail down the training side, because the sampler runs on a denoiser I have to train, and the reparameterization I used for the score needs preconditioning to be stable — the same EDM machinery DDBM adopted. I write the denoiser as `D_θ(x_t, x_T, t) = c_skip(t) x_t + c_out(t) F_θ(c_in(t) x_t, c_noise(t))`, choosing the scalings so the network `F_θ` always sees unit-variance inputs and is asked for a unit-variance target, and so it amplifies its own errors as little as possible. With `σ_0², σ_T², σ_{0T}` the variances and covariance of the clean target and source, the conditions "input variance 1," "effective target variance 1," and "minimize `c_out²` over `c_skip`" give `c_in = 1/√(α_t² σ_0² + β_t² σ_T² + 2 α_t β_t σ_{0T} + γ_t²)`, `c_skip = (α_t σ_0² + β_t σ_{0T}) c_in²`, `c_out = √(β_t² σ_0² σ_T² − β_t² σ_{0T}² + γ_t² σ_0²) c_in`, loss weight `λ = 1/c_out²`, and `c_noise = ¼ log t`. The training objective is then just the L2 denoising regression `∫ E[‖x̂_0(t, x_t, x_T) − x_0‖²] dt`, with `x_t` sampled from the kernel — the conditional mean it converges to is exactly the `x̂_0` my sampler calls.

Time to land this in code. The sampler walks the EDM schedule; at each step it calls the denoiser once for `x̂_0`, forms the normalized residual `ẑ`, and either takes an Euler-SDE step with `ε_t = η(γ γ̇ − (α̇/α) γ²)` or, on the last two steps, the deterministic DBIM transition. I need the path functions and their *analytic* derivatives — finite-differencing `γ̇` near the boundaries where `γ ~ O(10⁻²)` would lose a couple of digits of accuracy, so I write the derivatives out. I also have to respect the hard NFE budget: one denoiser call per step, no Heun-style double evaluation. For the linear path the route is `α = 1−t, α̇ = −1, β = t, β̇ = 1, γ = 2 γ_max √(t(1−t)), γ̇ = γ_max(1−2t)/√(t(1−t))`.

```python
import numpy as np
import torch as th


def linear_route(gamma_max):
    # α = 1-t (x_0 weight), β = t (x_T weight), γ = symmetric noise arch (k=1).
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: -th.ones_like(t)
    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: gamma_max * 2 * (t * (1 - t)) ** 0.5
    gamma_deriv = lambda t: gamma_max * 2 * (1 - 2 * t) / (2 * (t * (1 - t)) ** 0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv


def get_sigmas_karras(n, t_min, t_max, rho, device="cpu"):
    # EDM ρ-ramp; ρ<1 concentrates steps near the sharp t_min endpoint.
    ramp = th.linspace(0, 1, n, device=device)
    min_inv_rho = t_min ** (1 / rho)
    max_inv_rho = t_max ** (1 / rho)
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


def to_d_stoch(x, x0_hat, x_T, alpha, alpha_deriv, beta, beta_deriv,
               gamma, gamma_deriv, epsilon):
    # Euler-SDE drift b = α̇ x̂_0 + β̇ x_T + (γ̇ + ε/γ) ẑ, diffusion √(2ε).
    z_hat = (x - alpha * x0_hat - beta * x_T) / gamma          # normalized residual ẑ
    drift = alpha_deriv * x0_hat + beta_deriv * x_T + (gamma_deriv + epsilon / gamma) * z_hat
    diffusion = (2 * epsilon) ** 0.5
    return drift, diffusion


@th.no_grad()
def sample_stoch(
    denoiser, x, sigmas, route, progress=False, callback=None,
    churn_step_ratio=0.0, route_scaling=0, smooth=0.0
):
    x_T = x
    x = x + smooth * th.randn_like(x)        # modified base π_T = π_cond * N(0, b²I), b=smooth
    x_T_s = x
    s_in = x.new_ones([x.shape[0]])
    alpha, alpha_d, beta, beta_d, gamma, gamma_d = route
    # churn_step_ratio is η: 0 → ODE, 1 → full DDBM-strength SDE.
    epsilon = lambda t: churn_step_ratio * (
        gamma(t) * gamma_d(t) - alpha_d(t) / alpha(t) * gamma(t) ** 2)

    path, x0_est = [x.detach().cpu()], [x.detach().cpu()]
    indices = range(len(sigmas) - 1)
    for i in indices:
        x0_hat = denoiser(x, sigmas[i] * s_in, x_T)            # one budgeted denoiser call
        x0_est.append(x0_hat.detach().cpu())
        dt = sigmas[i + 1] - sigmas[i]                         # negative: sigmas decrease to 0

        if i >= len(indices) - 2:
            # Last two steps: ε=0 deterministic DBIM transition → sharp endpoint.
            x = (alpha(sigmas[i + 1]) * x0_hat + beta(sigmas[i + 1]) * x_T_s
                 + (gamma(sigmas[i + 1]) / gamma(sigmas[i]))
                 * (x - alpha(sigmas[i]) * x0_hat - beta(sigmas[i]) * x_T_s))
        else:
            # Early/middle steps: Euler-SDE; valid for any ε≥0 (no positivity constraint).
            drift, diffusion = to_d_stoch(
                x, x0_hat, x_T_s,
                alpha(sigmas[i]), alpha_d(sigmas[i]),
                beta(sigmas[i]), beta_d(sigmas[i]),
                gamma(sigmas[i]), gamma_d(sigmas[i]), epsilon(sigmas[i]))
            x = x + drift * dt + th.randn_like(x) * (dt.abs() ** 0.5) * diffusion

        path.append(x.detach().cpu())
    return x, path, x0_est
```

If instead I'm dropped into a harness that hands me the kernel through `get_abc(t) → (a, b, c)` — where `a` is the `x_T` coefficient (my `β`), `b` the `x_0` coefficient (my `α`), `c` the noise coefficient (my `γ`) — and a VP schedule with `alpha_fn = exp(−½β_min t − ¼β_d t²)`, `rho_fn = √(exp(β_min t + ½β_d t²) − 1)`, plus `f_fn = −½(β_min + β_d t)` and `g2_fn = β_min + β_d t`, the same algorithm reads off analytic derivatives from `f` and `g2` instead of from a closed-form route, and the last-two-steps deterministic branch uses `get_abc` directly. The body is identical: one denoiser call per input step, normalized residual, Euler-SDE with `ε_t = η(c ċ − (ḃ/b) c²)` early, deterministic transition `b' x̂_0 + a' x_T + (c'/c)(x − b x̂_0 − a x_T)` for the final two.

```python
@th.no_grad()
def sample_dbim(denoiser, diffusion, x, ts, eta=1.0, mask=None, seed=None, **kwargs):
    churn, rho_k = eta, 0.6
    t_max = diffusion.t_max
    ns = diffusion.noise_schedule
    rho_T2 = float(ns.rho_T) ** 2

    # Keep the framework's step count and return contract; only replace the time placement.
    t_lo, t_hi = float(diffusion.t_min), float(t_max - 5e-4)
    ramp = th.linspace(0.0, 1.0, len(ts), device=x.device, dtype=th.float64)
    ts = ((t_hi ** (1 / rho_k)) + ramp * (t_lo ** (1 / rho_k) - t_hi ** (1 / rho_k))) ** rho_k

    x_T = x
    path, pred_x0 = [x.detach().cpu()], []
    ones = x.new_ones([x.shape[0]])
    n_steps = len(ts) - 1

    def abc_and_deriv(t_scalar):
        # analytic (a,b,c) and their t-derivatives from the VP schedule (a=x_T, b=x_0, c=noise).
        t = t_scalar.clamp(min=1e-6, max=t_max - 1e-6) * ones
        alpha, alpha_bar, rho, rho_bar = [append_dims(v, x.ndim) for v in ns.get_alpha_rho(t)]
        f_t, g2_t = [append_dims(v, x.ndim) for v in ns.get_f_g2(t)]
        alpha_d = alpha * f_t                                   # α' = α·f
        rho_d = 0.5 * (rho ** 2 + 1.0) * g2_t / rho             # ρ' from g2
        alpha_bar_d = alpha_d / float(ns.alpha_T)
        rho_bar_d = -rho * rho_d / rho_bar
        a = alpha_bar * rho ** 2 / rho_T2
        b = alpha * rho_bar ** 2 / rho_T2
        c = alpha * rho_bar * rho / float(ns.rho_T)
        a_d = (alpha_bar_d * rho ** 2 + alpha_bar * 2 * rho * rho_d) / rho_T2
        b_d = (alpha_d * rho_bar ** 2 + alpha * (-2 * rho * rho_d)) / rho_T2
        c_d = (alpha_d * rho_bar * rho + alpha * rho_bar_d * rho
               + alpha * rho_bar * rho_d) / float(ns.rho_T)
        return (a, b, c), (a_d, b_d, c_d)

    nfe = 0
    generator = BatchedSeedGenerator(seed)
    first_noise = generator.randn_like(x)
    for step_idx in range(n_steps):
        s, t_next = ts[step_idx], ts[step_idx + 1]
        x0_hat = denoiser(x, s * ones)                         # one budgeted denoiser call
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)
        (a_s, b_s, c_s), (a_d, b_d, c_d) = abc_and_deriv(s)

        if step_idx >= n_steps - 2:
            a_t, b_t, c_t = [append_dims(v, x.ndim) for v in ns.get_abc(t_next * ones)]
            x = b_t * x0_hat + a_t * x_T + (c_t / c_s) * (x - b_s * x0_hat - a_s * x_T)
        else:
            eps = (churn * (c_s * c_d - (b_d / b_s) * c_s ** 2)).clamp(min=0)
            z_hat = (x - b_s * x0_hat - a_s * x_T) / c_s
            drift = b_d * x0_hat + a_d * x_T + (c_d + eps / c_s) * z_hat
            dt = t_next - s                                   # negative: sampling runs backward
            x = x + drift * dt + (2.0 * eps).sqrt() * generator.randn_like(x) * dt.abs().sqrt()

        if mask is not None:
            x = x * mask + x_T * (1 - mask)                    # keep known region fixed each step
        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1
    return x, path, nfe, pred_x0, ts, first_noise
```

Stepping back over the chain: I started with bridges that get great FID but are cramped (coupled `a_t, σ_t`) and slow (DDBM's hundred-NFE sampler), and a fast sampler (DBIM) that inherits the cramp and, through its `√(γ² − ρ²)` positivity, cannot use strong noise — plus a quiet failure of conditional diversity. Decoupling the path with the stochastic-interpolant flow map `x_t = α x_0 + β x_T + γ z` un-cramps the space; adding endpoint conditioning and a denoiser `x̂_0 = E[x_0 | x_t, x_T]` keeps it a one-model bridge. Realizing that kernel as a linear SDE gives `f, s, g` in closed form; the EDM-style score reparameterization keeps `1/γ²` out of the network. The Fokker–Planck cancellation `(∇log p)p = ∇p` proves that adding `ε_t ∇log p` drift with `√(2ε_t)` diffusion leaves every marginal fixed — so `ε_t` is a free stochasticity knob on top of `α, β, γ`, and DDBM, DBIM, I2SB, EDM all reappear as fixed choices of it, confirming the generalization. Discretizing the SDE by Euler (not DBIM's closed form) escapes the positivity constraint and lets `ε_t = η(γ γ̇ − (α̇/α) γ²)` dial the full ODE-to-DDBM range; switching the last two steps to the deterministic `ε = 0` DBIM transition sharpens the endpoint that pure Euler would smear. The schedule choices fall out of arguments rather than sweeps — the line `α = 1−t, β = t` by reparameterization-equivalence, the symmetric arch `γ = 2 γ_max √(t(1−t))` to balance detail across the path, `γ_max ~ 0.125–0.25` to add detail without drowning the conditioning, and an EDM ramp with `ρ < 1` to bunch the few steps near the sharp endpoint. And conditional diversity, which more sampling noise provably cannot fix, comes back by lossy-compressing the base distribution, `π_T = π_cond * N(0, b² I)`, trading a sliver of input information for genuine output variation. The result drops into the bridge-sampling harness as one denoiser call per step, an analytic-derivative route, an Euler-SDE early phase and a deterministic endpoint phase — fast enough for a five-call budget, flexible enough to reach the noise schedules the older samplers could not.
