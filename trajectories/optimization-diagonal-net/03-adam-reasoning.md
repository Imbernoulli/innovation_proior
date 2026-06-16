Bare SGD did exactly what the mechanism predicted, and it leapt up the ladder. The numbers: `n* = 50`
on `d200_k5_s01` — the *floor of the search grid*, recovery as cheap as the harness can register — `62`
on both `d500_k10` settings, and `487` on `d10000_k50`. Scores `−5.64, −5.95, −5.95, −8.93`. Lay them
against AdaGrad's `175 / 487 / 487 / 2000` and the story is unambiguous. On the small problem SGD went
from `175` down to `50`, the grid floor: feasibility was never the issue there, only AdaGrad's bias
dilution, and removing the denominator collapsed the sample requirement to the minimum. On the giant
`d10000` problem SGD recovered at `487` where AdaGrad bottomed out at the grid ceiling of `2000` — the
single largest improvement, exactly where I bet removing the denominator would un-stall the support
escape that high dimension made fragile. The improvement being *largest where AdaGrad was worst* is the
fingerprint of the right diagnosis: AdaGrad's `sqrt(Σ g²)` denominator was fighting the multiplicative
escape, plain GD rides it, and the noise temperature at `lr = 0.1` is doing the sparse regularizing.
So SGD is the rung to beat, and it leaves one sharp question open: AdaGrad's *particular* adaptivity
hurt, but is *all* adaptivity poison on the diagonal-net, or only AdaGrad's monotone accumulation? That
is the question I want this rung to answer, because if the answer is "only AdaGrad's flavour," there is
room above SGD; if the answer is "all of it," SGD is the ceiling and I should stop.

The clean way to ask the question is to take AdaGrad's preconditioner apart and fix the *one* thing I
identified as the culprit, changing nothing else. The culprit was the *monotone accumulation*:
`state_sum += g²` only grows, so the effective rate `lr/sqrt(state_sum)` decays toward zero, and on the
support coordinates — which carry the most gradient mass — it decays fastest, stalling the escape just
when it should accelerate. The obvious surgical fix is to stop accumulating forever and start
*forgetting*: replace the growing sum with an exponential moving average of squared gradients,
`s_t = β₂ s_{t−1} + (1−β₂) g_t²`. Now the denominator tracks the *recent* gradient scale rather than the
whole history, so the effective rate no longer decays monotonically to zero — it settles to a level set
by the current gradient magnitude. This is the RMSProp move, and it directly removes the rate-decay
failure mode I watched AdaGrad suffer at `d=10000`. Add to it a smoothed *first* moment,
`m_t = β₁ m_{t−1} + (1−β₁) g_t`, so the direction is an averaged momentum rather than the raw noisy
full-batch-plus-Rademacher gradient, damping the step-to-step jitter the label noise injects. The
update is then `θ ← θ − lr · m_t / (sqrt(s_t) + eps)` — the Adam form. The reason to try it over SGD is
specific: it keeps per-coordinate adaptivity (the thing that *might* help the geometry) but throws out
the monotone-decay defect that made AdaGrad's adaptivity hurt. If adaptivity-with-forgetting recovers
as cheaply as SGD or better, then it was only AdaGrad's accumulation that was poison; if it still loses
to SGD, then any rescaling of the coordinates blurs the escape and SGD's bare step is the right object.

Here is the part I have to get exactly right, because the harness's Adam is *not* the textbook Adam.
The standard derivation of this update goes one step further: it notes that `m_t` and `s_t` start at
zero, so early in training they are biased toward zero (the EMA has not filled its window), and it
divides by `(1−β₁^t)` and `(1−β₂^t)` to de-bias them — the bias-correction terms. That correction is
what makes a near-one `β₂` safe by inflating the tiny early `s_t` up to the true scale. **This task's
Adam deliberately omits the bias correction.** The harness studies the *raw adaptive geometry* the
preconditioner imposes, so the update is literally `m_t / (sqrt(s_t) + eps)` with no `1/(1−β^t)`
factors — the moments are used exactly as the EMAs leave them, biased toward zero at the start and
relaxing to the true scale only as `t` grows. I must reason about *that* object, not the de-biased one,
because the early-training behaviour is precisely where the diagonal-net's saddle escape happens and it
is governed by the *uncorrected* moments. So what does omitting the correction do here? At `t = 1`,
`m_1 = (1−β₁) g_1` and `s_1 = (1−β₂) g_1²`, so the ratio is `(1−β₁) g_1 / ((1−β₂)^{1/2} |g_1| + eps)
≈ (1−β₁)/(1−β₂)^{1/2} · sign(g_1)` — a step whose *magnitude* is set by the ratio of the one-minus-betas,
not by `lr` cleanly. With `β₁ = 0.9, β₂ = 0.999` that prefactor is `0.1 / 0.0316 ≈ 3.16`, so the
uncorrected early step is actually *amplified* relative to the nominal `lr`, not shrunk. As `t` grows
both EMAs fill in and the ratio relaxes toward the corrected one. The upshot is that uncorrected Adam
takes a *larger, scale-normalized* step in the first dozens of iterations than the corrected version
would — and on the diagonal-net that early phase is the saddle escape from the origin. Whether that
amplified, sign-like early step helps or hurts the escape ordering is genuinely unclear a priori, which
is exactly why this rung is a measurement and not a foregone conclusion.

Before the geometry, one more consequence of the uncorrected moments worth tracing, because it governs
how the two parameter vectors interact. On the diagonal-net `u` and `v` are not independent: at init
`u = v`, and `w_i = u_i² − v_i²` is the *difference* of their squares, so the predictor only moves when
the two vectors separate. The gradients are mirror images — `grad_u_i = 2 u_i r_i`,
`grad_v_i = −2 v_i r_i` with the same residual `r_i` — so on a support coordinate where the residual is
persistently signed, one of `u_i, v_i` is driven up and the other down, and the predictor coordinate
escapes. Adam runs *separate* EMA buffers for `u` and `v` (`m_u, s_u, m_v, s_v`), so the
scale-normalization is applied independently to each side of the difference. That is benign here: both
sides see the same-magnitude residual, so their denominators track together and the normalized steps on
`u_i` and `v_i` stay balanced, preserving the cancellation that keeps off-support `w_i` near zero. It
would only matter if the two sides' gradient scales diverged, which on this symmetric init and symmetric
gradient structure they do not. So I can reason about the preconditioner one coordinate at a time, in
`w`-space, without worrying that the `u`/`v` split breaks the picture — the per-vector EMAs are just two
copies of the same coordinate-wise story.

Now the geometry, the only thing that ultimately matters for sample complexity. The Adam preconditioner
`1/(sqrt(s_t)+eps)` is, like AdaGrad's, *largest in denominator* on the coordinates with the biggest
recent squared gradient — the support coordinates the residual hammers. So it *still* damps the support
escape relative to the off-support coordinates, the same directional pressure that made AdaGrad lose.
The difference is twofold and works in Adam's favour. First, because `s_t` is an EMA, not a growing sum,
the denominator does not run away — once a support coordinate's gradient settles, its denominator
settles too, so the relative damping is bounded rather than compounding, and the rate never decays to
zero. Second, the scale-normalized step means *every* coordinate moves by roughly `lr` in
parameter-space units regardless of its raw gradient magnitude — which, combined with the
multiplicative `2u_i` factor the diagonal-net injects, can actually *accelerate* a small support
coordinate's escape compared to AdaGrad's decaying rate. So I expect Adam to recover the ground SGD won
back from AdaGrad — to be competitive with bare SGD — without quite the cleanliness of SGD's
unimpeded escape, because some support damping remains. The harness exposes only the three functions
and gives me `delta` but no noise-scale argument, so I cannot adapt anything to the noise level; `eps`
is the fixed `1e-6` floor below any healthy gradient RMS, and that is the only noise-handling the
interface affords. I keep the literal baseline configuration — `lr = 0.05`, `β₁ = 0.9`, `β₂ = 0.999`,
`eps = 1e-6`, four EMA buffers (`m_u, s_u, m_v, s_v`) zero-initialised — and the *uncorrected* update.
The lower `lr = 0.05` (against SGD's `0.1`) is the right pairing for the scale-normalized step: because
Adam's step is already normalized to roughly unit parameter-space size, the same nominal `lr` as SGD
would over-step, so the baseline halves it. The full scaffold module is in the answer.

The falsifiable expectation, against SGD's measured `50 / 62 / 62 / 487`. If forgetting cures
AdaGrad's defect, Adam should be *competitive with SGD* — recovering at or near the grid floor on
`d200_k5` and far below AdaGrad's ceiling on `d10000`. Concretely I expect Adam to match SGD's `50` on
the small problem (both at the floor, nothing more to win there) and to land *close to* SGD on the
mid-size `d500_k10` settings — perhaps a hair better, if the scale-normalized step's averaging of the
label noise tightens recovery, or a hair worse if the residual support-damping costs a few samples.
The decisive setting is `d10000_k50`: if Adam recovers *below* SGD's `487` there, then forgetting +
momentum genuinely improves on bare GD at high dimension, and the EMA preconditioner is helping the
geometry, not just un-breaking AdaGrad; if Adam lands *at or above* SGD's `487`, then the residual
support-damping is still a tax and SGD's bare step remains the cleanest. Either way the contrast with
AdaGrad's `2000` should be enormous — that is the forgetting cure, and it is the test of whether the
diagnosis ("monotone accumulation, not adaptivity per se") was right. The result will also set up the
last rung directly: if uncorrected Adam helps, the remaining free knobs are the EMA windows — and the
natural next move is to *shorten the second-moment memory* (smaller `β₂`), trading the long, smooth
denominator that lags the escape for a shorter, more responsive one that tracks each support
coordinate's escape as it happens.
