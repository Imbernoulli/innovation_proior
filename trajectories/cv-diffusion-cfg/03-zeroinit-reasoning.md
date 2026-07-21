Both prior numbers are in, and together they tell a more interesting story than either alone.
Standard CFG (the guided renoise at scale 7.5) came in at FID 23.65 / 24.29 / 25.74, beating CFG++'s
23.99 / 24.89 / 25.88 on every variant. Subtract carefully: the per-variant gains are `0.34` on SD
v1.5, `0.60` on SD v2-base, and `0.14` on SDXL, a mean of about `0.36`. My standard-CFG hypothesis held —
the harder push pulled the distribution onto the sharp reference modes and lowered FID on all three.
But two features of the deltas are the real information. First, they are *small* — a tenth to
six-tenths, exactly the tenths-scale I calibrated, each on top of an absolute FID still in the
mid-twenties. Second, and more pointed, the *smallest* gain is on SDXL, `0.14` — the variant where
CFG++ was weakest and where I explicitly predicted the *largest* gain. The harder push helped least
precisely where it was needed most. One seed, so I will not over-read a single `0.14`, but a clean,
dominant sharpening effect would grow with the room to improve, not shrink.

The narrow margin is the clue and the SDXL anomaly sharpens it. That standard CFG is only marginally
ahead, and least ahead where there was most to fix, says the guided renoise is *both* helping
(sharpening toward the reference) and hurting (off-manifold distortion), with the net barely positive
— on the high-resolution model, almost exactly cancelling. This is the accounting I told myself to
watch when filling standard CFG. So the question now is not "which of these two rules is better" but "can I
keep the *helpful* part of one and drop the *harmful* part?"

Make that a small accounting model, because it turns the vague cancellation into a target. Write
standard CFG's improvement over CFG++ as `gain = S - C`, benefit `S` (sharpening onto reference
modes) minus cost `C` (off-manifold distortion). The measured gains are `S - C = 0.34, 0.60, 0.14`.
On SDXL `S ≈ C` — a wash. Now the thesis here reads in the same terms: build on CFG++, whose
informative steps carry `C = 0` (the `eps_uc` renoise is on-manifold), and add a *prefix benefit* `P`
from deleting the under-committed, unreliable high-noise steps. The predicted zero-init gain over
CFG++ is then `≈ P` with no distortion cost, and it beats standard CFG whenever `P > S - C`. That
inequality is most favorable exactly where `S - C` is smallest — SDXL — so if `P` is roughly
variant-independent, zero-init has the most headroom over standard CFG on the model that resisted the
guided renoise. A concrete, checkable consequence, not a hope.

Where does each method spend its error? The answer is in the timing. In a DDIM trajectory the early
steps are the highest-noise steps: `z_t` is almost pure noise, so both predictions are at their least
reliable, yet guidance is already allowed to push at full strength. The conditional signal `eps_c -
eps_uc` is a *difference* of two predictions, and near pure noise that difference is dominated by each
prediction's error rather than any genuine semantic gap — the network cannot localize the prompt in a
latent that is `99%` noise. So the same scale `w` that strengthens the conditional signal strengthens
the conditional *error* equally: with `eps_uc = eps_uc^* + e_uc`, `eps_c = eps_c^* + e_c`, the guided
prediction is `eps_g = eps_g^* + (e_uc + w(e_c - e_uc))`, error growing with `w`. The lever I have
not touched is *time*: not which noise renoises, but whether to apply the guided update *at all*
during the first, least-trustworthy steps.

On real images I cannot see that error, but a Gaussian path gives a measuring stick where the true
step is known in closed form. Take `x_0 ~ N(0,I)`, `x_1 ~ N(mu, I)` independent, linear path `x_t =
(1-t) x_0 + t x_1`. The optimal velocity is `E[x_1 - x_0 | x_t]`, and since `(x_t, x_1 - x_0)` is
jointly Gaussian I can write it down. With independence, `Cov(x_t, x_1 - x_0) = (1-t)(-I) + t(I) =
(2t-1) I` and `Var(x_t) = ((1-t)^2 + t^2) I`, and with `E[x_t] = t mu`, `E[x_1 - x_0] = mu`,

  `v_t^*(x) = mu + ((2t-1)/((1-t)^2 + t^2)) (x - t mu)`.

At `t = 0` the coefficient is `-1`, `v_0^*(x) = mu - x`, the mean displacement `E[x_1 - x_0 | x_0]`,
so the source end `t -> 0` — sample still near noise while guidance pushes hard — is the place to
read. That coefficient `(2t-1)/((1-t)^2 + t^2)` is the *slope* of the optimal velocity in `x`: how
much `v^*` moves per unit error in the current latent. Along the path it is `-1.00` at `t = 0`,
`-0.98` at `0.10`, `-0.80` at `0.25`, and `0` at `0.50` (where `v^* = mu`, independent of the noisy
latent). So a given error in the model's estimate of the latent — largest exactly at the source,
where the latent is nearly pure noise — is amplified into the velocity with the largest gain
precisely at the source. High noise and high sensitivity coincide at the start: the least-reliable
predictions feed the most error-amplifying step.

So the sharp question, in this controlled case: when the model is underfitted, is the *guided*
first-step move closer to the optimal one than the *zero* move? The zero move has error `||0 -
v_0^*||^2 = ||mu - x_0||^2`. Modelling the source-end guided velocity as truth plus amplified error,
`v_guided(t=0) = v_0^* + eta` with `||eta|| ~ w sigma_e` for per-prediction error `sigma_e`, the
guided error is `||eta||^2 ~ w^2 sigma_e^2` while the zero error is fixed and `w`-independent. The
guided error grows quadratically in `w`, so there is always a scale above which the guided move is
strictly worse — crossover roughly `w sigma_e > ||v_0^*||`. At `w = 7.5` and the source end where
`sigma_e` is largest, a per-prediction error a seventh the size of the optimal velocity already tips
it, and at the source the error is plausibly comparable to the velocity itself. This is a "can," not
an "always" — I cannot prove it for the frozen SD weights on every prompt — but it is robust in
shape: at the source, taking the guided step injects the largest wrong direction exactly when the
trajectory carries the least semantic information, and doing nothing is provably no worse in the
regime where prediction error dominates the semantic gap. The rule: the high-noise prefix should be
*inert*, and the fix is specifically the first steps, not a global scale reduction — once `sigma_e`
falls with the noise the crossover flips and guiding wins.

The Gaussian flow is a proxy, and the mapping to the real loop is clean: the path's source end `t
-> 0` is the pure-noise end; in the DDIM schedule that is `t` near `T`, smallest `bar_alpha_t`, which
the reverse loop visits *first*. The map from proxy `t` to step index is monotone, so "skip the
source-end steps" translates directly to "skip the first steps of the DDIM loop." What the proxy
cannot pin is the exact count — it says the sensitivity-times-noise product is maximal at the very
start and decays, not whether the error-dominated region is one step wide or five. So I take the
*shape* from the proxy (an inert prefix, short) and choose the *length* `K` myself.

Three moves address the timing, and I reject two on their own terms. Blending the two renoise
directions with a tapered `beta` is a continuous schedule with a shape and rate to tune, and it does
not implement the sharp diagnostic — which says the first few steps should carry *no* guided update,
not a gently down-weighted one. Ramping the scale `w(t)` from `0` is the same soft version of a
statement that is actually hard: the source-end predictions are not merely weak, they are
error-dominated, so the principled move is to skip them. The discrete, parameter-light move that
matches the diagnostic exactly is to hard-skip the first `K` steps — no denoise, no renoise, no
callback — leaving the latent at its initialization through the highest-noise steps, and run the
ordinary guided step on every step after. That is zero-init, a single small integer `K` rather than a
schedule.

The base I build it on is the crux, and the cancellation from the numbers decides it. Put the inert
prefix on *standard CFG* and I keep the off-manifold distortion on every informative step — the cost
that was almost eating standard CFG's sharpening. Put it on *CFG++* and once the prefix ends every
remaining step is on-manifold. And CFG++'s worst steps *are* the high-noise ones: that is where its
`eps_uc` renoise is most under-committed (the transition is dominated by the conditional-free
`sqrt(1-a)` share) and where, from the CFG++ geometry, the two denoised endpoints are farthest apart
and the estimate least trustworthy. Deleting exactly those steps removes the regime where even the
clean renoise is under-committed *and* unreliable, and running CFG++ on the rest keeps the manifold
guarantee for the steps that carry real semantics. The bet: "manifold-safe renoise, but only after
the noise has come down enough for the predictions to mean something" beats both plain CFG++ (clean
renoise applied even at the useless high-noise start) and standard CFG (hard guided renoise applied
everywhere including the unreliable start).

`K` must stay small, and quantitatively so. The argument is about the *source end* only, so it is no
license to skip arbitrarily many steps: the DDIM budget is fixed and modest, skipping `K` of `N`
forfeits a fraction `K/N` of the guidance budget, and once the field becomes informative a few steps
in, continuing to do nothing throws away useful, budget-limited steps. I set `K = 2` — skip the two
worst steps, keep the rest — small enough to sit safely inside the unreliable source end, large
enough to actually delete the error-amplifying prefix, and I do not sweep it (seed and budget pinned).
Skipping those two forfeits very little image signal, which is what makes the trade lopsided: at the
highest-noise steps the denoise contributes only the `sqrt(a_t)` signal share, and `a_t ~ 0.005` at
the first step, so `sqrt(a_t) ~ 0.07` — the first step injects on the order of `7%` of a clean-signal
component. The two skipped steps together are a small single-digit-percent share of the trajectory's
total drift toward an image, while removing the two largest error injections. The one way this
backfires is if the fixed budget `N` is so small that two steps is a large fraction of it; then the
forfeited signal is no longer a sliver — the failure mode I will watch for in the FID table.

One thing to be exact about: the full method zero-init belongs to has *two* parts and this fill uses
only one. The full version pairs the inert prefix with a per-sample *optimized scale* on the
unconditional prediction — replace `eps_uc` in the mix by `s* eps_uc` where `s* = <eps_c, eps_uc> /
||eps_uc||^2`, so guidance amplifies only the conditional *residual* the unconditional prediction
cannot already explain (a cheap dot-product-and-norm, no extra network call). This fill does **not**
include it: it keeps the plain CFG++ mix and `noise_uc` renoise and adds *only* the inert prefix,
`if step < K: continue` with `K = 2`. I set the optimized scale aside deliberately — it is the
obvious lever left on the table if the inert prefix delivers.

So the fill: start from the CFG++ step (mix with `cfg_guidance`, Tweedie-denoise with the mix,
renoise with `noise_uc`) and wrap the loop with a guard that skips the first `K = 2` iterations
entirely, so the latent stays at its initialization through the two highest-noise steps and the
manifold-safe update runs on every step after. SDXL is identical in `reverse_process()`: the same `if
step < K: continue` guard, then the CFG++ step indexing `alphas_cumprod[t]` and renoising with
`noise_uc`. The full scaffold module for both variants is in the answer.

Against the two measured results: if the high-noise early steps are really where guidance does net
harm, zeroing the first `K = 2` steps on top of the CFG++ renoise should beat *both* priors on FID —
below standard CFG's 23.65 / 24.29 / 25.74 and CFG++'s 23.99 / 24.89 / 25.88 on every variant —
recovering CFG++'s manifold cleanliness while removing its under-committed worst steps. A
mechanism-specific sub-prediction: standard CFG's gain over CFG++ was smallest on SDXL, which I read
as help and harm most nearly cancelling there; if zero-init works by *removing* harm rather than
adding help, it should move SDXL by more than that `0.14`. The failure tell is the opposite: if
skipping two steps mostly costs conditional signal at this tight budget, FID regresses toward the
no-skip CFG++ numbers. Either way, the optimized-scale projection I set aside is the explicit next
lever.
