Both prior numbers are now in, and together they tell a more interesting story than either alone.
Standard CFG (the guided renoise at scale 7.5) came in at FID 23.65 / 24.29 / 25.74 on SD v1.5 / SD
v2-base / SDXL, beating CFG++'s 23.99 / 24.89 / 25.88 on every variant. Let me subtract carefully rather
than just note the sign. The per-variant gains are `23.99 - 23.65 = 0.34` on SD v1.5, `24.89 - 24.29 =
0.60` on SD v2-base, and `25.88 - 25.74 = 0.14` on SDXL, a mean improvement of about `0.36` of a FID
point. So my rung-2 hypothesis held: the harder push, with the prompt-amplified noise driving the
renoise, pulled the generated distribution onto the sharp reference modes and lowered FID on all three.
But two features of those deltas are the real information. First, they are *small* — a tenth to
six-tenths of a point, exactly the tenths-scale I calibrated, and every one of them sits on top of an
absolute FID still in the mid-twenties. Second, and more pointed, the *smallest* gain is on SDXL, `0.14`
— the variant where CFG++ was weakest and where I explicitly predicted the largest gain. The harder push
helped *least* precisely where it was needed *most*. I only have one seed, so I will not over-read a
single `0.14`, but the pattern is not what a clean, dominant sharpening effect would produce; a clean
win would grow with the room to improve, not shrink.

The narrow margin is the clue, and the SDXL anomaly sharpens it. If the guided renoise were unambiguously
better I would expect a larger, room-tracking gap. That it is only marginally ahead, and least ahead
where there was most to fix, says the guided renoise is *both* helping (sharpening toward the reference)
and hurting (off-manifold distortion), and the net is barely positive — the two effects are nearly
cancelling, and on the high-resolution model they may be almost exactly cancelling. This is precisely the
accounting I told myself to watch at rung two: read the margin, not the sign. So the question for rung 3
is not "which of these two rules is better" but "can I keep the *helpful* part of one and drop the
*harmful* part, rather than choosing between two rules that each mix help and harm?"

Let me make the "help and harm" reading explicit as a little accounting model, because it turns the vague
cancellation into a target. Write standard CFG's FID improvement over CFG++ as a benefit minus a cost,
`gain = S - C`, where `S` is the sharpening that concentrates the distribution onto reference modes and
`C` is the off-manifold distortion the guided renoise introduces. The measured gains are `S - C = 0.34,
0.60, 0.14` on the three variants. On SDXL, `S - C = 0.14` with both `S` and `C` presumably larger in
absolute terms on a high-resolution model, so `S ≈ C` there — the sharpening and the distortion are near
a wash. Now the thesis of this rung reads cleanly in the same terms: build on CFG++, whose informative
steps carry `C = 0` (the `eps_uc` renoise is on-manifold), and add a *prefix benefit* `P` from deleting
the under-committed, unreliable high-noise steps. The predicted zero-init gain over CFG++ is then `≈ P`
with no distortion cost, and it beats standard CFG whenever `P > S - C`. That inequality is most
favorable exactly where `S - C` is smallest — SDXL, where standard CFG barely won — so if `P` is roughly
variant-independent, zero-init has the most headroom over standard CFG on the model that resisted the
guided renoise. That is a concrete, checkable consequence of the accounting, not a hope.

Let me look harder at *where* each method spends its error, because the answer is in the timing, not in
the choice of renoise. In a DDIM trajectory the early steps are the highest-noise steps: `z_t` is almost
pure noise, so both predictions `eps_uc` and `eps_c` are at their least reliable, and yet guidance is
already allowed to push at full strength. CFG++ renoises with `eps_uc` there, which is safe but
under-committed; standard CFG renoises with the guided `eps_g` there, which commits hard along a
direction the model can barely estimate at that noise level. Both prior rungs treat the first step like
every other step. But the high-noise regime is special, and I can say why mechanically: the conditional
signal `eps_c - eps_uc` is the *difference* of two predictions, and near pure noise that difference is
dominated by each prediction's error rather than by any genuine semantic gap — the network simply cannot
localize the prompt in a latent that is `99%` noise. So the same scale `w` that strengthens the
conditional signal strengthens the conditional *error* in equal measure: if `eps_uc = eps_uc^* + e_uc`
and `eps_c = eps_c^* + e_c`, the guided prediction is `eps_g = eps_g^* + (e_uc + w(e_c - e_uc))`, and the
error term grows with `w`. When the model is unreliable a large push cannot tell which component is
semantic control and which is a wrong prediction. So the lever I have not touched is *time*: not which
noise drives the renoise, but whether to apply the guided update *at all* during the first,
least-trustworthy steps.

On real images I cannot see that error, but a Gaussian path gives me a measuring stick where the true
step is known in closed form, so I can actually compare a guided step to the optimal one. Take a
source-to-target flow with `x_0 ~ N(0,I)`, `x_1 ~ N(mu, I)` independent, and the linear path `x_t =
(1-t) x_0 + t x_1`. The optimal velocity is the conditional mean `E[x_1 - x_0 | x_t]`, and because
`(x_t, x_1 - x_0)` is jointly Gaussian I can write it down. I need the two covariances. `Cov(x_t, x_1 -
x_0) = (1-t) Cov(x_0, x_1 - x_0) + t Cov(x_1, x_1 - x_0)`; with independence, `Cov(x_0, x_1 - x_0) =
-Var(x_0) = -I` and `Cov(x_1, x_1 - x_0) = Var(x_1) = I`, so `Cov(x_t, x_1 - x_0) = (1-t)(-I) + t(I) =
(2t-1) I`. And `Var(x_t) = (1-t)^2 Var(x_0) + t^2 Var(x_1) = ((1-t)^2 + t^2) I`. With `E[x_t] = t mu` and
`E[x_1 - x_0] = mu`, the Gaussian conditional mean is

  `v_t^*(x) = mu + ((2t-1)/((1-t)^2 + t^2)) (x - t mu)`.

Let me sanity the formula at the ends before I use it. At `t = 1`, the coefficient is `1/1 = 1`, so
`v_1^* = mu + (x - mu) = x`; at the target end the velocity just points along the current sample, which
is right for a fully-arrived trajectory. At `t = 0`, the coefficient is `-1/1 = -1`, so `v_0^*(x) = mu -
x`; at the source end the optimal velocity is "head from wherever you are toward `mu`," which is exactly
the mean displacement `E[x_1 - x_0 | x_0] = mu - x_0`. Both ends are sane, so the closed form is
trustworthy, and now I can read the source end — `t` near the start — which is the dangerous place: the
sample is still close to noise while guidance is already allowed to push hard.

Here is the quantitative reason the source end is dangerous. The coefficient `(2t-1)/((1-t)^2 + t^2)` is
the *slope* of the optimal velocity in `x` — how much `v^*` moves per unit error in the current latent.
Evaluate it along the path: at `t = 0` it is `-1.00`, at `t = 0.10` it is `-0.98`, at `t = 0.25` it is
`-0.80`, and at `t = 0.50` it is exactly `0`. So the optimal velocity depends on `x_t` most steeply at
the source (`|slope| = 1`) and not at all at the midpoint (`slope = 0`, where `v^* = mu`, the pure
signal, independent of the noisy latent). That means a given error in the model's estimate of the latent
— which is largest exactly at the source, where the latent is nearly pure noise — is amplified into the
velocity with the *largest* gain precisely at the source and washed out by the midpoint. High noise and
high sensitivity coincide at the start, which is the worst possible combination: the least-reliable
predictions feed the most error-amplifying step.

So I can ask the sharp question directly in this controlled case. When the model is underfitted, is the
*guided* first-step move closer to the optimal first-step move than the *zero* move is? The zero move has
error `||0 - v_0^*||^2 = ||mu - x_0||^2`. The guided move has error `||v_guided(t=0) - v_0^*||^2`. And in
the underfitted regime the diagnostic can satisfy `||v_guided(t=0) - v_0^*||^2 >= ||0 - v_0^*||^2` — the
guided step, built from unreliable predictions and amplified by the largest slope, can land *farther*
from the optimum than simply doing nothing. I want to be honest that this is a "can," not an "always": it
depends on how underfitted the model is at that noise level, and I cannot prove it holds for the frozen
SD weights on every prompt. But the structural point is robust — at the source end, taking the guided
step injects the largest wrong direction exactly when the trajectory carries the least semantic
information, and doing nothing is provably no worse in the regime where the prediction error dominates
the semantic gap. Read that as a decision rule: the high-noise prefix should be *inert*. Zero the update
for the first `K` steps and let guidance act only once the predictions become informative — a few steps
in, where the slope has fallen off toward the insensitive midpoint and the latent has enough signal that
`eps_c - eps_uc` is semantics rather than error.

I can make the "can hold" quantitative with a scaling argument, which also tells me that the *scale*
matters. Model the source-end guided velocity as the truth plus an amplified error, `v_guided(t=0) =
v_0^* + eta`, where `eta` collects the per-prediction errors amplified by the guidance scale, so its
magnitude scales like `||eta|| ~ w sigma_e` for a per-prediction error size `sigma_e`. Then the guided
error is `||v_guided - v_0^*||^2 = ||eta||^2 ~ w^2 sigma_e^2` (per coordinate), while the zero-move error
is fixed at `||v_0^*||^2 = ||mu - x_0||^2`, independent of `w`. The guided error grows quadratically in
`w` and the zero error does not, so there is always a scale above which the guided move is strictly worse:
the crossover is roughly `w sigma_e > ||v_0^*||`. At the canonical `w = 7.5`, and at the source end where
`sigma_e` is largest (the latent is nearly pure noise, so the prediction error is at its worst), this
inequality is easy to cross — a per-prediction error only a seventh the size of the optimal velocity
already tips it, and at the source the error is plausibly comparable to the velocity itself. That is the
concrete regime in which "do nothing beats guiding": high `w`, high `sigma_e`, both maximal at the start.
It also predicts the fix is specifically about the *first* steps and not a global scale reduction — once
`sigma_e` falls with the noise, the crossover flips back and guiding wins, which is exactly why I skip a
prefix rather than lowering `w` everywhere.

I should be honest that the Gaussian flow is a proxy, not the real SD schedule, so let me check the
mapping transfers rather than assert it. The linear path's "source end" `t -> 0` is the pure-noise end;
in the DDIM schedule the pure-noise end is `t` near `T`, where `bar_alpha_t` is smallest, and the reverse
loop visits it *first*. The map from the proxy's `t` to the sampler's step index is monotone — the first
reverse steps are the highest-noise, smallest-`bar_alpha` steps — so "skip the source-end steps of the
flow" translates directly to "skip the first steps of the DDIM loop." What the proxy cannot tell me is the
exact number: it says the sensitivity-times-noise product is maximal at the very start and decays, but not
whether the error-dominated region is one step wide or five for these particular frozen weights. So I take
the *shape* of the conclusion from the proxy — an inert prefix at the start, short — and the *length* `K`
from the scaffold, rather than pretending the toy model pins `K = 2` for me. That is the honest division
of labor between the diagnostic and the choice.

Now what do I actually build, and on which base? Three moves address the timing, and I want to reject two
of them on their own terms. I could blend the two renoise directions per step, `beta eps_uc + (1-beta)
eps_g`, tapering `beta` over time — but that is a continuous schedule with a shape and a rate to tune,
and it does not implement the sharp diagnostic I just derived, which says the *first few* steps should
carry no guided update at all, not a gently down-weighted one. I could ramp the guidance scale itself
from `0`, `w(t)` rising over the trajectory — also a schedule to tune, and again a soft version of a
statement that is actually hard: the source-end predictions are not merely weak, they are error-dominated,
so the principled move is to skip them, not to lightly guide them. The third move is the discrete,
parameter-light one that matches the diagnostic exactly: hard-skip the first `K` steps — no denoise, no
renoise, no callback — leaving the latent at its initialization through the highest-noise steps, and run
the ordinary guided step on every step after. That is zero-init, and it introduces a single small integer
`K` rather than a schedule.

The base I build it on is the crux, and the cancellation from the numbers decides it. If I put the inert
prefix on *standard CFG* — the guided renoise — I keep the off-manifold distortion on all the informative
steps, the very cost that (by the near-cancellation) was almost eating standard CFG's sharpening. If
instead I put the prefix on *CFG++* — the clean `eps_uc` renoise — then once the prefix ends, every
remaining step is on-manifold. And here is why that composition should beat both priors rather than just
splitting the difference: CFG++'s worst steps *are* the high-noise ones. That is where its `eps_uc`
renoise is most under-committed (the transition is dominated by the `sqrt(1-a)` noise share, which
carries no conditional signal) and simultaneously where, from rung one's geometry, the two denoised
endpoints are farthest apart and the estimate least trustworthy. Deleting exactly those steps removes the
regime where even the clean renoise is under-committed *and* unreliable, and running CFG++ on the rest
keeps the manifold guarantee for the steps that carry real semantics. So the bet is that "manifold-safe
renoise, but only after the noise has come down enough for the predictions to mean something" beats both
"manifold-safe renoise applied even at the useless high-noise start" (plain CFG++, which under-committed
worst there) and "hard guided renoise applied everywhere including the unreliable start" (standard CFG,
which distorted worst there). The inert prefix is the one move that removes the shared worst-case of both
prior rungs.

A detail on `K`, because it must stay small and I want the reason to be quantitative, not a shrug. The
inert-prefix argument is a statement about the *source end* of the trajectory, where the slope is near
its `|-1|` maximum and predictions are error-dominated; it is not a license to skip arbitrarily many
steps. The DDIM budget is fixed and modest, and every skipped step is a solver step I do not get back:
skipping `K` of `N` steps forfeits a fraction `K/N` of the guidance budget, and once the velocity field
becomes informative — a few steps in, as the slope falls toward the insensitive midpoint — continuing to
do nothing would simply throw away useful, budget-limited steps and *lose* the conditional signal I need.
`K = 2` is the scaffold's choice and a sensible one: skip the two worst steps, keep the rest. I do not
sweep it; the harness pins the seed and budget and I am comparing the rule, and two is small enough to be
safely inside the "unreliable source end" the diagnostic identifies while large enough to actually delete
the error-amplifying prefix.

There is a reassuring check that skipping those two steps forfeits very little of the actual image signal,
which is what makes the trade lopsided in my favor. At the highest-noise steps the denoise contributes
only the `sqrt(a_t)` signal share, and `a_t` there is tiny — around `0.005` at the very first step, so
`sqrt(a_t) ~ 0.07`. That is, the first step's denoise injects on the order of `7%` of a clean-signal
component into the latent; the two skipped steps together contribute a small, single-digit-percent share
of the trajectory's total drift toward an image. So leaving the latent inert for two steps forfeits only
a sliver of signal, while removing the two largest error injections (the steps where the slope-times-noise
product is maximal). Small cost, large removed harm — the asymmetry is the whole reason the prefix can be
a net win rather than a wash. The one way this backfires is if the fixed budget `N` is so small that two
steps is a large fraction of it; then the forfeited signal is no longer a sliver and the skip could cost
more than it saves. I cannot rule that out without the number, so it is the failure mode I will watch for
in the FID table.

I also want to be careful about exactly what the scaffold exposes, because the method this rung is named
for has *two* parts and the task uses only one of them. The full version of "fix guidance at the
unreliable source end" pairs the inert prefix with a per-sample *optimized scale* on the unconditional
prediction — replace `eps_uc` in the mix by `s* eps_uc` where `s* = <eps_c, eps_uc> / ||eps_uc||^2` is
the least-squares projection of the conditional prediction onto the unconditional one, so guidance
amplifies only the conditional *residual* the unconditional prediction cannot already explain. That
projection is a cheap dot-product-and-norm on the two vectors I already have, no extra network call. But
the `zeroinit` baseline I am filling here does **not** include it: its fill keeps the plain CFG++ mix and
renoise (`noise_pred = noise_uc + cfg_guidance*(noise_c - noise_uc)`, renoise with `noise_uc`) and adds
*only* the inert prefix — `if step < K: continue` with `K = 2`. So the rung I am building is the
zero-init half alone, layered on the CFG++ manifold-safe renoise, not the full optimized-scale-plus-
zero-init method. I note the omission deliberately: the optimized scale is the obvious thing left on the
table, and it is where the ladder goes next if the inert prefix delivers.

Concretely, then, the rung-3 fill is: start from the rung-1 CFG++ step (mix with `cfg_guidance`,
Tweedie-denoise with the mix, renoise with `noise_uc`), and wrap the loop with a guard that skips the
first `K = 2` iterations entirely — no denoise, no renoise, no callback — so the latent stays at its
initialization through the two highest-noise steps, and the manifold-safe CFG++ update runs on every step
after. For SDXL the fill is identical in `reverse_process()`: the same `if step < K: continue` guard with
`K = 2` at the top of the loop, then the CFG++ step indexing `alphas_cumprod[t]` and renoising with
`noise_uc`. Same two predictions per step, same fixed NFE budget, the only structural change being the
skipped prefix. The full scaffold module for both variants is in the answer.

The falsifiable expectation, against the two measured rungs. If the high-noise early steps are really
where guidance does net harm, then zeroing the first `K = 2` steps on top of the CFG++ renoise should
beat *both* priors on FID: below standard CFG's 23.65 / 24.29 / 25.74 and below CFG++'s 23.99 / 24.89 /
25.88, on every variant. The reasoning predicts the inert prefix recovers CFG++'s manifold cleanliness
while removing its under-committed worst steps, so it should clear the bar standard CFG set by sharpening
— without paying standard CFG's off-manifold cost. There is a mechanism-specific sub-prediction I can
also check: standard CFG's gain over CFG++ was smallest on SDXL (`0.14`), which I read as the guided
renoise's help and harm most nearly cancelling there; if zero-init works by *removing* the harm rather
than adding help, it should be able to move SDXL by more than that `0.14`, because it is no longer paying
the off-manifold cost on the informative steps. If instead skipping two steps mostly costs me conditional
signal — because the budget is tight and two steps is a large fraction of it — FID would regress toward
or below the no-skip CFG++ numbers, and I would conclude the prefix is too long for this step budget. The
clean test is whether all three zero-init FID numbers come in under both prior rows. And whatever the
result, the optimized-scale projection I just set aside is the explicit next lever: the full method
amplifies only the conditional residual `eps_c - s* eps_uc`, and adding that on top of this inert prefix
is the move past the strongest baseline.
