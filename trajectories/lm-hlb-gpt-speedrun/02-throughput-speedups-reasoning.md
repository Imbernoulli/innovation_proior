The baseline reaches the bar in just over six minutes, and I want to roughly halve that. I already pulled the
free correctness-neutral levers — bf16, fused attention, compile — in the baseline itself, so the time left
isn't in the *kernels*; it's in the *schedule*. The one choice I flagged there as an unjustified default is the
fixed gradient-accumulation count — the effective batch size held constant for the whole run — so that's where
I look.

State the noise story precisely enough to act on. A minibatch gradient is an unbiased estimate of the true
gradient with variance that falls as 1/B: if the per-example gradient covariance has total variance tr(Σ) and
the true gradient is g, the estimate has variance tr(Σ)/B, so SNR² = B·|g|²/tr(Σ) and SNR ∝ √B. That's the
whole story, because compute cost is *linear* in B while the SNR it buys is only *square-root* in B — the
marginal SNR per unit compute falls as 1/√B. Going B=1→4 doubles the SNR for 4× the passes; B=16→64 also
doubles it for another 4× the passes but at a quarter the marginal return. Whether averaging is worth buying
at all depends on the critical batch B_crit ≈ tr(Σ)/|g|², below which it is nearly free of benefit.

Overlay that on the run. Early, the gradient is huge and points in an obviously-right direction: |g|² is large,
B_crit is tiny, even one noisy microbatch already points downhill, so pouring several into one step buys almost
nothing but extra forward/backward passes. Late, the surface is flat, |g|² has collapsed toward the noise
floor, B_crit has grown large, and now averaging many microbatches is exactly what keeps a step from being
dominated by sampling noise. A single constant is wrong somewhere — and the arithmetic says the waste is
concentrated in the *cheap early phase*: if the constant is 8 but the early run only needs ~2, those early
steps do ~4× more forward/backward than they need, and there are a lot of them. That's plausibly where a
factor near two hides.

So I want the effective batch to *grow* over the run, small early and large late. The question is how to decide
how large. An open-loop schedule — ramp the count from 1 to N on a fixed curve — is a three-line change, but
re-tuning is exactly the tax I'm trying to stop paying: any hand-drawn curve has to be redrawn after every
other change I make to the recipe, and worse, an open-loop ramp can't respond to the run actually in front of
it, so if a change makes the gradient noisier or quieter than the curve assumed, the schedule is silently
wrong. Growing off the *loss* is tempting since loss is already computed, but loss measures how well I've
*fit*, not how *noisy* my gradient is — two runs at the same loss can have very different gradient noise. What
I actually want is a signal that reads out, per step, "the gradient is now noisy enough that you should be
averaging more." The gradient norm is that signal.

When training is healthy and the effective batch is well matched, the grad norm drifts down smoothly; if the
batch is too small, individual steps jitter it around erratically relative to the smooth trend. So the
*difference between a smoothed grad norm and the instantaneous one* reads out how much per-step noise I'm
eating: if progress is smooth the two track and the realized per-step drop is small; if the run has plateaued
or is jittering, the difference departs from the small drop I'm willing to tolerate. That's a closed-loop
signal off the quantity that directly measures averaging need, and it costs one extra reduction pass over the
gradients per step — cheap next to the forward/backward for 30M params.

The controller keeps an EMA `running_grad_norm` (decay .95, a memory of ~1/(1−.95)=20 steps, long enough to
smooth single-step spikes but short enough to follow the real downward trend), and each step forms
`per_step_diff_delta = target − (running_grad_norm − grad_norm)`. If the realized drop is *smaller* than the
target (smooth progress, no more averaging pressure needed yet) the delta is positive and I nudge the
accumulation count up; if larger (dropping hard or jittering), down. The nudge scales with the current count
and a small `accumulate_steps_lr = 5e-2`, so the batch drifts rather than jumps. The `target_per_step_decay`
of 3e-2 is the threshold for "healthy smooth progress" against grad norms of O(1): a per-step drop below ~3%
reads as a plateau wanting more averaging, above it as active descent that doesn't — too small and the
controller thinks everything is a plateau and inflates the batch, too large and it never grows. Seeding
`running_grad_norm` at 1.2 rather than 0 matters: an EMA started at zero would spend its first ~20 steps
climbing to scale, and during that transient `running_grad_norm − grad_norm` would be wildly negative and the
controller would do something erratic right when the run is most fragile — seeding near the actual initial grad
norm skips it.

The measurement is the global L2 norm √(Σᵢ‖gᵢ‖²) over every gradient, accumulated into an fp32 scalar after the
backward and before the step, detached so it never enters the graph (the module is in the answer). Combining
per-tensor norms as √(Σ‖·‖²) is the honest concatenation norm; summing the norms directly would over-count.

One subtlety or the whole thing chatters: `accumulate_steps_estimate` is a *fractional* number — say 2.3
microbatches per step — but I can only accumulate an integer count. Rounding quantizes away exactly the
fine-grained control I just built, jumping 2→3 in cliffs and oscillating at the boundary on tiny signal
changes. The fix is Bernoulli dithering: split into integer base and fractional remainder, and with
probability equal to the remainder do one extra microbatch. For 2.3 the realized count is 2 w.p. 0.7 and 3
w.p. 0.3, expectation 2·0.7 + 3·0.3 = 2.3 exactly — so the *average* effective batch tracks the fraction to
the digit while each step uses a clean integer and never pays the quantization cliff.

One correctness detail the growing batch forces: accumulating over N microbatches sums their gradients, so to
keep the *mean* gradient — the thing whose noise falls as 1/B — I divide each microbatch's loss by the current
count before its backward (`loss.div(current_accumulate_steps).backward()`). Skip it and a batch growing 1→8
would silently multiply the accumulated gradient by 8, and since AdamW's step scales with the gradient that
would inflate the effective step size exactly when the batch grows — coupling two things I want decoupled.
Dividing keeps the optimizer seeing a roughly constant-scale mean gradient whatever the batch is, so growing
the batch changes the step's *noise*, not its *size*. That separation is what lets the LR schedule own the
step size and the controller own the noise.

The ordering in the step matters. Once a full accumulation group is complete I take the optimizer step, update
the weight decay, advance the scheduler, *then* read the grad norm of the group I just stepped on (it's still
present until I zero it) and fold it into the EMA, update the fractional count and dither it to the integer
count for the *next* group, and finally zero the gradients. Reading the grad norm before `zero_grad` is the
only way to get it for free — the gradient I just consumed is the exact quantity I want — and adapting here
means this group's statistics govern the next group, which is the right causality for a controller.

Two schedule changes pair with this naturally. First, weight decay: with the effective batch now changing, I
want regularization to track the run's state, and unlike the batch size, decay *is* a question about how much
I've fit — which is exactly what the loss reads out. Tie it to the inverse squared likelihood, `(1/loss)² ×
base`. The magnitudes have to land between always-off and always-max: at the cold-start loss 10.83 the
multiplier is (1/10.83)² ≈ 0.0085, effective decay ~8.5e-6 against a base of 1e-3 — essentially off while the
model races downhill; by the 3.8 bar it's (1/3.8)² ≈ 0.069, ~6.9e-5, about 8× stronger but still only ~7% of
the base. The base would only be reached at loss 1, which this run never sees, so the schedule spends its whole
life ramping smoothly on the gentle part of the curve — no cliff. Second, the LR anneal: with the dynamic
batch front-loading cheap, slightly noisier early steps (accumulation 1 means single-microbatch gradients), a
*linear* anneal to a slightly lower final fraction gives a cleaner late descent, and I retune the run to a
longer nominal step budget (the batch now starts at 1, so more, cheaper steps) and a shorter warmup. One
tension worth naming: the linear-scaling rule says a larger batch can support a larger LR, and my batch *grows*
late — so one might expect the LR to rise late too. But that rule is about the largest *stable* step a given
noise level allows, whereas the anneal is about the step I actually *want* late in training, which is small and
careful regardless. The batch growth is spent cutting noise so the small annealed step is a *clean* small step,
not on enlarging it; the two are doing different jobs and aren't fighting.

None of this speedup can come from "turning on kernels" — bf16, fused attention, TF32, and compile were all
baked into the baseline's clock, not held in reserve — so every second I save has to come from doing *fewer or
cheaper steps*, which is exactly what running most of the run at a smaller effective batch does. If the
mechanism is right the A100 seconds should fall toward roughly half while the loss still lands at ~3.8, and the
internal accumulation-count trace should sit at 1 through the early phase and creep up only late. The risk is
the controller oscillating if the dithering or grad-norm signal is too noisy; the `max(1., …)` floor, the .95
EMA smoothing, and the small adaptation rate are the hedges, and the test is the usual one — as long as the
loss lands at ~3.8, the saved time is real.
