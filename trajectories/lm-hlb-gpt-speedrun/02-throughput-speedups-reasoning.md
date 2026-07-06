The baseline reaches the bar in just over six minutes, and I want to roughly halve that — call the aim
somewhere near three minutes. I already pulled the free, correctness-neutral levers — bf16, fused attention,
compile — in the baseline itself. So the time left on the table isn't in the *kernels*; it's in the
*schedule*. Where is the baseline doing work it doesn't need to do? Let me stare at the one choice I flagged
in the baseline as "the obvious default, not justified": the fixed gradient-accumulation count, i.e. the
fixed effective batch size held constant for the whole run.

Here's the thing about effective batch size, stated precisely enough to act on. A minibatch gradient is an
unbiased estimate of the true gradient with variance that falls as 1/B in the batch size B, so the
signal-to-noise ratio of a step grows like √B. Write it out: if the per-example gradient covariance has
total variance tr(Σ) and the true gradient is g, then the minibatch estimate has variance tr(Σ)/B, so
SNR² = |g|² / (tr(Σ)/B) = B·|g|²/tr(Σ) and SNR ∝ √B. That √B is the whole story, because compute cost is
*linear* in B while the SNR it buys is only *square-root* in B — so the marginal SNR I get per unit of
compute falls as 1/√B. Going from B=1 to B=4 doubles the SNR (1→2) for 4× the passes; going from B=16 to
B=64 also doubles it (4→8) for another 4× the passes but at a quarter the marginal return per pass. Averaging
is a diminishing-returns purchase, and whether it's worth buying at all depends on how large the true
gradient is relative to its noise — the critical batch B_crit ≈ tr(Σ)/|g|², below which averaging is nearly
free of benefit and above which it's nearly all you can do.

Now overlay that on the two ends of a run. Early in training the gradient is huge and points in an
obviously-right direction: |g|² is large, B_crit is tiny, and even a single noisy microbatch already points
downhill — pouring five microbatches into one step buys almost nothing except five forward/backward passes
per optimizer step. Late in training the loss surface is flat, the gradient is small, |g|² has collapsed
toward the noise floor, B_crit has grown large, and now the noise is a large *fraction* of the signal — here
averaging many microbatches is exactly what keeps the step from being dominated by sampling noise. The
baseline uses one constant accumulation count for both regimes. Whatever number I pick, it's wrong somewhere:
too big early (wasting compute averaging down noise that isn't hurting me), too small late (noisy steps), or
a compromise that's mediocre at both ends. A constant cannot track a need that is genuinely non-stationary,
and the arithmetic says the waste is concentrated in the cheap early phase — if the constant is, say, 8 but
the early half of the run only needs ~2, those early steps are doing ~4× more forward/backward than they
need, and there are a lot of them. That's plausibly where "just over six minutes" hides a factor near two.

So I want the effective batch size to *grow* over the run — small early, large late. The question is how to
decide *how* large. Let me walk the options honestly before committing. The simplest is an open-loop schedule:
ramp the accumulation count from 1 to some N over the run on a fixed curve. Tempting — it's a three-line
change — but it's exactly the kind of thing I'd have to re-tune every time I change the model size or any
other part of the recipe, and re-tuning is the tax I'm trying to stop paying; worse, an open-loop ramp can't
respond to the run actually in front of it, so if a change makes the gradient noisier or quieter than my
hand-drawn curve assumed, the schedule is silently wrong. And I *will* keep changing other parts of the
recipe as this ladder goes on, so any hand-tuned batch curve I draw today is a curve I'd have to redraw
after every one of those changes — an open-loop schedule quietly converts every future edit into a
batch-schedule re-tune, which is precisely the maintenance cost a closed loop pays down. A second option is to grow the batch off the *loss*:
loss is smooth and already computed. But loss measures how well I've *fit*, not how *noisy* my gradient is —
two runs sitting at the same loss can have very different gradient noise depending on the surface — so loss is
the wrong readout for a batch controller (I'll come back to loss for a different job, where "how much have I
fit" is exactly the right question). What I actually want is a signal that reads out, at each step, "the
gradient is now noisy enough that you should be averaging more." The gradient norm is that signal.

Concretely: I compute the total grad norm of the network at a step and watch how it behaves. When training
is healthy and the effective batch is well matched, the grad norm drifts down smoothly. If my effective
batch is too small, individual steps jitter the grad norm around — the per-step change is large and erratic
relative to the smooth trend. So the *difference between the smoothed grad norm and the instantaneous grad
norm* is a readout of how much per-step noise I'm eating: if the run is making smooth, un-noisy progress the
smoothed and instantaneous values track each other and the realized per-step drop is small; if the run has
plateaued or is jittering, the difference departs from the small target I'm willing to tolerate. That is a
closed-loop signal off the quantity that directly measures averaging need, and it costs one extra pass over
the gradients per step — for 30M parameters that's cheap next to the forward/backward, which is O(params ×
tokens), so I'm happy to pay it.

The measurement itself is a global L2 norm over every gradient in the network — accumulate each tensor's
squared 2-norm, sum, take the square root — computed after the backward pass and before the optimizer step,
detached so it never enters the graph:

```python
def get_grad_norm(net):
    grad_norm = torch.tensor(0., device='cuda')
    for p in net.parameters():
        if p.grad is not None:
            grad_norm += p.grad.detach().data.norm(2).square()
    return (grad_norm ** 0.5).item()
```

The `.square()` then `** 0.5` at the end is the honest way to combine per-tensor norms: √(Σᵢ ‖gᵢ‖²) is the
norm of the concatenated gradient, whereas summing the norms directly would over-count. It's one scalar per
step summarizing the whole gradient's magnitude, which is all the controller needs.

Let me make that a controller. Keep an exponential moving average of the grad norm, `running_grad_norm`,
with decay .95 so it smooths over roughly the last ~20 steps and doesn't chase a single spike. At each step
compute the actual `grad_norm`. Form `per_step_diff_delta = target − (running_grad_norm − grad_norm)`, where
`target` is the small per-step decay I'm willing to tolerate. If the realized per-step drop is *smaller* than
the target (the run is making smooth progress and doesn't need more averaging pressure yet), `per_step_diff_delta`
is positive and I nudge the accumulation count *up*; if it's larger (the run is dropping hard or jittering),
I nudge it down. I scale the nudge by the current accumulation count and a small learning rate so it adapts
smoothly:

```python
running_grad_norm = decay * running_grad_norm + (1. - decay) * grad_norm
per_step_diff_delta = target_per_step_decay - (running_grad_norm - grad_norm)
accumulate_steps_estimate += current_accumulate_steps * (accumulate_steps_lr * per_step_diff_delta)
accumulate_steps_estimate = max(1., accumulate_steps_estimate)
```

The constants aren't arbitrary and each one is doing a specific job:

```python
current_accumulate_steps = accumulate_steps_estimate = hyp['opt']['initial_accumulate_steps']  # = 1
running_grad_norm = 1.2                 # EMA of the grad norm, seeded near the initial value
running_grad_norm_decay = .95
target_per_step_decay   = 3e-2          # tolerated smooth per-step grad-norm drop
accumulate_steps_lr     = 5e-2          # smooths the batchsize adaptation rate
```

Seeding `running_grad_norm` at 1.2 rather than 0 matters: an EMA started at zero would spend its first ~20
steps climbing to the true grad-norm scale, and during that transient `running_grad_norm − grad_norm` would
be wildly negative and the controller would do something erratic right when the run is most fragile — seeding
near the actual initial grad norm skips the transient. The decay of .95 gives the EMA a memory of roughly
1/(1−.95) = 20 steps, long enough to smooth single-step spikes but short enough to follow the real downward
trend. The `target_per_step_decay` of 3e-2 is the threshold for "healthy smooth progress": against grad norms
of O(1), a per-step drop below ~3% reads as a plateau that wants more averaging, above it as active descent
that doesn't — set it too small and the controller thinks everything is a plateau and inflates the batch too
eagerly, too large and it never grows. And `accumulate_steps_lr` = 5e-2 keeps each nudge to a few percent of
the current count, so the batch size drifts rather than jumps.

Let me trace this at the two ends to make sure it does what I want, using the actual constants (target = 3e-2,
accumulate_steps_lr = 5e-2, EMA decay = .95, seeded `running_grad_norm` = 1.2, starting count 1). Early, the
grad norm is large and dropping fast: suppose it goes from ~1.2 to ~1.0 in a step. The EMA updates to
.95·1.2 + .05·1.0 = 1.19, so `running_grad_norm − grad_norm` = 1.19 − 1.0 = 0.19, and `per_step_diff_delta`
= 0.03 − 0.19 = −0.16. The count update is 1 + 1·(0.05·−0.16) = 0.992, floored back to 1. So while the
gradient is large and falling steeply, the controller *pins the count at 1* — the cheapest possible steps,
exactly when averaging buys least. Late, on a plateau, the grad norm barely moves: EMA ≈ 0.30, instantaneous
≈ 0.29, difference 0.01 which is *below* the 0.03 target, so `per_step_diff_delta` = +0.02 and a count of 2.0
updates to 2.0 + 2.0·(0.05·0.02) = 2.002 — a slow, steady creep upward. Over many plateau steps that creep
compounds and the effective batch grows. So the controller starts the run at accumulation 1 and only grows
the batch as the run actually stops making cheap progress, which is precisely the small-early/large-late
curve the noise-scale argument asked for, and the wall-clock saving comes from the long cheap early stretch
that the baseline's constant was over-averaging.

Now a subtlety I have to get right or the whole thing chatters: `accumulate_steps_estimate` is a *fractional*
number — say 2.3 microbatches per step — but I can only accumulate an integer number of microbatches. If I
round it, I quantize away exactly the fine-grained control I just built; the effective batch jumps from 2 to
3 in discrete cliffs and the controller oscillates at the boundary, flipping the whole batch size on tiny
signal changes. The fix is dithering: split the estimate into its integer base and fractional remainder, and
stochastically round — with probability equal to the remainder, do one extra microbatch. The check that this
is unbiased is a one-liner: for an estimate of 2.3, the realized count is 2 with probability 0.7 and 3 with
probability 0.3, so its expectation is 2·0.7 + 3·0.3 = 2.3 exactly. Over many steps the *average* effective
batch tracks the fractional estimate to the digit, while each individual step uses a clean integer count and
never pays the quantization cliff.

```python
base, probability = divmod(accumulate_steps_estimate, 1)
current_accumulate_steps = max(1, int(base + torch.bernoulli(torch.tensor(probability)).item()))
```

One correctness detail the growing batch forces me to be careful about: when I accumulate over N microbatches
I sum their gradients, so to keep the *mean* gradient — the thing whose noise falls as 1/B — I have to divide
each microbatch's loss by the current count before its backward:

```python
loss.div(current_accumulate_steps).backward()
```

If I skipped this, a batch that grew from 1 to, say, 8 microbatches would silently multiply the accumulated
gradient by 8, and since AdamW's step scales with the gradient that would inflate the effective step size
exactly when the batch grows — coupling two things I want decoupled. Dividing by the count keeps the optimizer
seeing a mean gradient of roughly constant scale whatever the batch is, so growing the batch changes the
*noise* of the step, not its *size*. That separation is what lets the LR schedule own the step size and the
controller own the noise.

This is the engine. It starts the run at a tiny effective batch (accumulation 1 — cheapest possible steps
while the gradient is huge and forgiving) and lets the grad-norm controller grow it only as the run actually
demands more averaging. Most of the run, especially the cheap early part, now runs at a far smaller effective
batch than the baseline's constant, which is where the wall-clock comes from.

Assembling the step makes the ordering explicit, and the ordering matters. I only act once a full
accumulation group is complete — that's the `microbatch_step % current_accumulate_steps == 0` gate — and
then, in sequence: take the optimizer step on the accumulated mean gradient; update the weight decay from the
current loss; advance the LR scheduler; *then* read the grad norm of the group I just stepped on (it's still
present until I zero it) and fold it into the EMA; update the fractional count and dither it to the integer
count for the *next* group; and finally zero the gradients:

```python
if microbatch_step % current_accumulate_steps == 0:
    opt.step()
    opt.param_groups[1]['weight_decay'] = (1. / loss.detach().item())**2. * hyp['opt']['weight_decay']
    scheduler.step()                    # linear anneal (anneal_strategy='linear')
    grad_norm = get_grad_norm(net)
    running_grad_norm = running_grad_norm_decay * running_grad_norm + (1. - running_grad_norm_decay) * grad_norm
    per_step_diff_delta = target_per_step_decay - (running_grad_norm - grad_norm)
    accumulate_steps_estimate += current_accumulate_steps * (accumulate_steps_lr * per_step_diff_delta)
    accumulate_steps_estimate = max(1., accumulate_steps_estimate)
    base, probability = divmod(accumulate_steps_estimate, 1)
    current_accumulate_steps = max(1, int(base + torch.bernoulli(torch.tensor(probability)).item()))
    opt.zero_grad(set_to_none=True)
```

Reading the grad norm *before* `zero_grad` is the only way to get it for free — the gradient I just consumed
is the exact quantity I want to measure — and adapting the count here means the decision I make on this
group's statistics governs the *next* group, which is the right causality for a controller. If I zeroed
first I'd have nothing to measure; if I measured before the step I'd be one group stale.

Two schedule changes pair with this naturally. First, weight decay. With the effective batch now changing
over the run, I want the regularization to track the run's state rather than sit at a constant — and unlike
the batch size, decay *is* a question about how much I've fit the data, which is exactly what the loss reads
out. When the loss is high (early, underfit) I want little decay so the model can move freely; as the loss
drops toward the bar (late, and on a ~100M-token corpus a 30M-param model is seeing more than once, so late
is where memorization risk lives) I want decay to strengthen. Tie it to the inverse squared likelihood —
`(1/loss)²` times the base decay — so it's gentle early and ramps up as loss falls:

```python
opt.param_groups[1]['weight_decay'] = (1. / loss.detach().item())**2. * hyp['opt']['weight_decay']
```

Let me sanity-check the magnitudes, because a decay schedule that's either always-off or always-max is
useless. At the cold-start loss ln(50304) ≈ 10.83, the multiplier is (1/10.83)² ≈ 0.0085, so the effective
decay is ~8.5e-6 against a base of 1e-3 — essentially off, which is what I want while the model is racing
downhill. By the time the loss reaches the 3.8 bar the multiplier is (1/3.8)² ≈ 0.069, giving ~6.9e-5 — about
8× stronger than at cold start, but still only ~7% of the base. The base value of 1e-3 would only be reached
at loss = 1, which this run never sees, so the schedule spends its whole life on the gentle part of the curve,
ramping smoothly by roughly an order of magnitude from start to bar and never slamming into full decay. That's
the shape I wanted: near-zero regularization while fitting, a smooth ramp as the loss falls, and no cliff.

Second, the LR anneal. The baseline cosine-anneals; with the dynamic batch front-loading cheap, slightly
noisier early steps (accumulation 1 means each early step is a single-microbatch gradient), a *linear* anneal
down to a slightly lower final fraction gives a cleaner late-run descent in my testing, and I retune the run
length and warmup to match the new dynamics — a longer nominal step budget, since the effective batch now
starts at 1 and there are more, cheaper steps, plus a shorter warmup since those early steps are cheap and
forgiving. There is a tension here worth naming rather than papering over: the linear-scaling rule says a larger batch
can support a proportionally larger LR, and my batch *grows* late — so one might expect the LR to rise late
too, not anneal down. But that rule is about the largest *stable* step a given noise level allows, whereas
the anneal is about the step I actually *want* late in training, which is small and careful regardless of
how much headroom the batch buys. The batch growth is spent on cutting noise so the small annealed step is
a *clean* small step, not on enlarging the step. So the anneal-down wins, and the growing batch and the
shrinking LR aren't fighting — they're doing different jobs, noise-reduction and step-shrinking, that happen
to both intensify late. These are tunings around the controller, not the controller itself, and I'm not
claiming they're the story; the controller is.

One attribution check before I commit, because it's easy to fool myself here. The ~6 minutes I'm trying to
halve *already* includes bf16, fused attention, TF32, and `torch.compile` — those were baked into the
baseline's clock, not held in reserve. So none of my speedup can come from "turning on kernels"; every
second I save has to come from doing *fewer or cheaper steps*, which is exactly what the controller does by
running most of the run at a smaller effective batch than the baseline's constant. That's why I keep insisting
the time left is in the schedule and not the kernels — if I found myself explaining a speedup by appeal to
precision or compilation, I'd know I'd double-counted something already spent.

The bet against the baseline's ~6 minutes: the dominant waste was a constant effective batch that
over-averaged the cheap, large-gradient early phase, and the noise-scale arithmetic says that phase is where
a factor near two can hide. Replacing it with a grad-norm-driven controller that starts at accumulation 1 and
only grows the batch when the gradient statistics actually demand it should cut the per-step cost across most
of the run. If this mechanism is right, three things should move together: the A100 seconds to the bar should
fall toward roughly half, the val loss should still land at ~3.8 and the val perplexity therefore at
e^3.8 ≈ 44.7 (the controller changes *how* I average, not *what* I'm optimizing, so the held bar has to hold
in both reported forms), and if I could watch the internal accumulation-count trace it should sit at 1 through
the early phase and creep up only late — that last one is the tell that separates "the controller worked" from
"I got lucky." The risk is the controller itself: if the dithering or the grad-norm signal is too noisy it
could oscillate the batch size and destabilize training. The `max(1., …)` floor keeps it from collapsing to
sub-unit nonsense, the EMA decay of .95 smooths the signal so single-step spikes don't swing the batch, and
the small `accumulate_steps_lr` of 5e-2 keeps each nudge gentle — those are the hedges, and the test is the
same as always: as long as the loss still lands at ~3.8, the time saved is real.
