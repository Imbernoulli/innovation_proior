Both prior numbers are now in, and together they tell a more interesting story than either alone.
Standard CFG (the guided renoise at scale 7.5) came in at FID 23.65 / 24.29 / 25.74 on SD v1.5 /
SD v2-base / SDXL, beating CFG++'s 23.99 / 24.89 / 25.88 on every variant. So my rung-2 hypothesis
held: the harder push, with the prompt-amplified noise driving the renoise, pulled the generated
distribution onto the sharp reference modes and lowered FID, most on SD v2-base (0.60) and least on
SDXL (0.14). Good. But the gains are *small* — a tenth to six-tenths of a FID point — and they came
with everything CFG++ was built to avoid: over-saturation, off-manifold drift, broken inversion. The
narrow margin is the clue. If the guided renoise were unambiguously better, I would expect a larger
gap; that it is only marginally ahead says the guided renoise is *both* helping (sharpening toward
the reference) and hurting (off-manifold distortion), and the net is barely positive. The two effects
are nearly cancelling. That reframes the question for rung 3: can I keep the *helpful* part of one of
these and drop the *harmful* part, rather than choosing between two rules that each mix help and harm?

Let me look harder at where each method spends its error, because the answer is in the timing. In a
DDIM trajectory the early steps are the highest-noise steps: `z_t` is almost pure noise, the two
predictions `eps_uc` and `eps_c` are at their least reliable, and yet guidance is already allowed to
push at full strength. CFG++ renoises with `eps_uc` there, which is safe but under-committed; standard
CFG renoises with the guided `eps_g` there, which commits hard along a direction the model can barely
estimate at that noise level. Both prior rungs treat the first step like every other step. But the
high-noise regime is special: this is exactly where a guided move is most likely to be a *worse*
estimate of the right direction than no move at all, because the conditional signal `eps_c - eps_uc`
is dominated by prediction error when the latent is near-pure noise. So the lever I have not touched
is *time*: not which noise drives the renoise, but whether to apply the guided update *at all* during
the first, least-trustworthy steps.

Here is the diagnostic that makes "skip the first steps" principled rather than a hack. The trouble
with a guided mix is that the same scale that strengthens the conditional signal also strengthens the
conditional *error*: if `eps_uc = eps_uc^* + e_uc` and `eps_c = eps_c^* + e_c`, the guided
prediction carries the ideal guided part plus an error term that grows with the guidance scale, and
when the model is unreliable a large push cannot tell which component is semantic control and which is
a wrong prediction. On real images I cannot see that error, but a Gaussian path gives me a measuring
stick. Take a source-to-target flow with `x_0 ~ N(0,I)`, `x_1 ~ N(mu,I)`, and the linear path
`x_t = (1-t)x_0 + t x_1`. The optimal velocity is the conditional mean `E[x_1 - x_0 | x_t]`, and
because `(x_t, x_1 - x_0)` is jointly Gaussian with `Cov(x_t, x_1 - x_0) = (2t-1)I` and
`Var(x_t) = ((1-t)^2 + t^2)I`, it has the closed form
`v_t^*(x) = ((2t-1)/((1-t)^2 + t^2))(x - t mu) + mu`. That formula lets me compare a learned,
guided step to the true step at any `t` — and the source end, `t` near the start, is the dangerous
place: the sample is still close to noise while guidance is already allowed to push hard.

In that controlled case, when the model is underfitted, I can ask the sharp question directly: is the
*guided* first-step move closer to the optimal first-step move than the *zero* move is? And the
answer, in the underfitted regime, is that it is not — the diagnostic can satisfy
`||v_guided(t=0) - v_0^*||^2 >= ||0 - v_0^*||^2`. Read that as a decision rule. At the source end,
taking the guided step injects the largest wrong direction exactly when the trajectory carries the
least semantic information; doing nothing — leaving the latent unchanged for that step — is provably
no worse, and often better, than the guided move. So the high-noise prefix should be *inert*: zero the
update for the first `K` steps and let guidance act only once the predictions become informative.

This is the move I will fill in, and I want to be careful about exactly what the scaffold exposes,
because the method this rung is named for has *two* parts and the task uses only one of them. The
full version of "fix guidance at the unreliable source end" pairs the inert prefix with a per-sample
*optimized scale* on the unconditional prediction — replace `eps_uc` in the mix by `s* eps_uc` where
`s* = <eps_c, eps_uc> / ||eps_uc||^2` is the least-squares projection of the conditional prediction
onto the unconditional one, so guidance amplifies only the conditional *residual* the unconditional
prediction cannot already explain. That projection is a cheap dot-product-and-norm on the two vectors
I already have, no extra network call. But the task's `zeroinit` baseline does **not** include it:
its fill keeps the plain CFG++ mix and renoise (`noise_pred = noise_uc + cfg_guidance*(noise_c -
noise_uc)`, renoise with `noise_uc`) and adds *only* the inert prefix — `if step < K: continue` with
`K = 2`. So the rung I am building here is the zero-init half alone, layered on the CFG++ manifold-safe
renoise, not the full optimized-scale-plus-zero-init method. I note the omission deliberately: the
optimized scale is the obvious thing left on the table, and it is where the ladder goes next.

So the rung-3 fill is concretely: start from the CFG++ step (mix with `cfg_guidance`, Tweedie-denoise
with the mix, renoise with `noise_uc`), and wrap the loop with a guard that skips the first `K = 2`
iterations entirely — no denoise, no renoise, no callback — so the latent stays at its initialization
through the two highest-noise steps, and the manifold-safe CFG++ update runs on every step after. Why
build it on the CFG++ renoise rather than the standard-CFG renoise, given that standard CFG just beat
CFG++ on FID? Because the two diagnoses compose. Standard CFG's edge was small and bought with
off-manifold distortion; CFG++'s renoise is the clean one. If the *reason* the early steps hurt is
that guidance is unreliable at high noise, then the right base for the inert prefix is the
*clean-renoise* sampler, so that once the prefix ends and guidance resumes, every remaining step is
on-manifold. Skipping the worst two steps removes the regime where even the clean renoise is
under-committed *and* unreliable, and running CFG++ on the rest keeps the manifold guarantee for the
informative steps. The bet is that "manifold-safe renoise, but only after the noise has come down
enough for the predictions to mean something" beats both "manifold-safe renoise applied even at the
useless high-noise start" (plain CFG++) and "hard guided renoise applied everywhere including the
unreliable start" (standard CFG).

A detail on `K`. It has to stay small. The inert-prefix argument is a statement about the *source
end* of the trajectory, where predictions are unreliable; it is not a license to skip arbitrarily
many steps. Once the velocity field becomes informative — a few steps in, when the noise has dropped
— continuing to do nothing would simply throw away useful, budget-limited solver steps and *lose*
the conditional signal I need. `K = 2` is the scaffold's choice and a sensible one: skip the two
worst steps, keep the rest. I do not sweep it; the harness pins the seed and budget and I am
comparing the rule, and two is small enough to be safely inside the "unreliable source end" the
diagnostic identifies.

For SDXL the fill is identical in `reverse_process()`: the same `if step < K: continue` guard with
`K = 2` at the top of the loop, then the CFG++ step indexing `alphas_cumprod[t]` and renoising with
`noise_uc`. Same two predictions per step, same fixed NFE budget, the only structural change being the
skipped prefix. The full scaffold module for both variants is in the answer.

The falsifiable expectation, against the two measured rungs. If the high-noise early steps are really
where guidance does net harm, then zeroing the first `K = 2` steps on top of the CFG++ renoise should
beat *both* priors on FID: below standard CFG's 23.65 / 24.29 / 25.74 and below CFG++'s 23.99 / 24.89
/ 25.88, on every variant. The reasoning predicts the inert prefix recovers CFG++'s manifold
cleanliness while removing its under-committed worst steps, so it should clear the bar standard CFG
set by sharpening — without paying standard CFG's off-manifold cost. If instead skipping two steps
mostly costs me conditional signal (because the budget is tight and two steps is a large fraction of
it), FID would regress toward or below the no-skip CFG++ numbers, and I would conclude the prefix is
too long for this step budget. The clean test is whether all three zero-init FID numbers come in
under both prior rows. And whatever the result, the optimized-scale projection I just set aside is the
explicit next lever: the full method amplifies only the conditional residual `eps_c - s* eps_uc`, and
adding that on top of this inert prefix is the move past the strongest baseline.
