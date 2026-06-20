The baseline reaches the bar in just over six minutes, and I want to roughly halve that. I already pulled
the free, correctness-neutral levers — bf16, fused attention, compile — in the baseline itself. So the time
left on the table isn't in the *kernels*; it's in the *schedule*. Where is the baseline doing work it
doesn't need to do? Let me stare at the one choice I flagged as "the obvious default, not justified": the
fixed gradient-accumulation count, i.e. the fixed effective batch size.

Here's the thing about effective batch size in a short run. A large effective batch averages down the
gradient noise, which lets you take a confident step. But averaging is only worth it when the noise is
actually hurting you. Early in training, the gradient is huge and points in an obviously-right direction —
even a single noisy microbatch already points downhill, and pouring five microbatches into one step buys
almost nothing except five forward/backward passes per optimizer step. Late in training, the loss surface
is flat, the gradient is small, and the noise is a large *fraction* of the signal — now averaging many
microbatches is what keeps the step from being dominated by sampling noise. The baseline uses one constant
accumulation count for both regimes. Whatever number I pick, it's wrong somewhere: too big early (wasting
compute), or too small late (noisy steps), or a compromise that's wrong at both ends. A constant can't
track a non-stationary need.

So I want the effective batch size to *grow* over the run — small early, large late. The question is how to
decide *how* large, without a hand-tuned schedule that I'd have to re-tune for every model size. I need a
signal that tells me, at each step, "the gradient is now noisy enough that you should be averaging more."
What measurable quantity carries that? The gradient norm. Concretely: I can compute the total grad norm of
the network at a step and watch how it behaves. When training is healthy and the effective batch is well
matched, the grad norm drifts down smoothly. If my effective batch is too small, individual steps jitter
the grad norm around — the per-step change is large and erratic relative to the smooth trend. So the
*difference between the smoothed grad norm and the instantaneous grad norm* is a readout of how much
per-step noise I'm eating. If that difference is bigger than some small target, I'm under-averaging and
should grow the effective batch; if it's smaller, I can shrink it and save compute.

Let me make that a controller. Keep an exponential moving average of the grad norm, `running_grad_norm`.
At each step compute the actual `grad_norm`. Form `per_step_diff_delta = target − (running_grad_norm −
grad_norm)`, where `target` is the small per-step decay I'm willing to tolerate. If the realized per-step
drop is *smaller* than the target (the run is making smooth, un-noisy progress), `per_step_diff_delta` is
positive and I nudge the accumulation count *up*; if it's larger (steps are jittery), I nudge it down. I
scale the nudge by the current accumulation count and a small learning rate so it adapts smoothly:

```python
running_grad_norm = decay * running_grad_norm + (1. - decay) * grad_norm
per_step_diff_delta = target_per_step_decay - (running_grad_norm - grad_norm)
accumulate_steps_estimate += current_accumulate_steps * (accumulate_steps_lr * per_step_diff_delta)
accumulate_steps_estimate = max(1., accumulate_steps_estimate)
```

Now a subtlety: `accumulate_steps_estimate` is a *fractional* number — say 2.3 microbatches per step — but
I can only accumulate an integer number of microbatches. If I round it, I quantize away exactly the
fine-grained control I just built; the effective batch jumps from 2 to 3 in discrete cliffs and the
controller chatters at the boundary. The fix is dithering: split the estimate into its integer base and
fractional remainder, and stochastically round — with probability equal to the remainder, do one extra
microbatch. Over many steps the *average* effective batch tracks the fractional estimate exactly, while
each individual step uses a clean integer count.

```python
base, probability = divmod(accumulate_steps_estimate, 1)
current_accumulate_steps = max(1, int(base + torch.bernoulli(torch.tensor(probability)).item()))
```

This is the engine. It starts the run at a tiny effective batch (accumulation 1 — cheapest possible steps
while the gradient is huge and forgiving) and lets the grad-norm controller grow it only as the run
actually demands more averaging. Most of the run, especially the cheap early part, now runs at a far
smaller effective batch than the baseline's constant, which is where the wall-clock comes from.

Two schedule changes pair with this naturally. First, weight decay. With the effective batch now changing
over the run, I want the regularization to track the run's state rather than sit at a constant. A clean
signal for "how much has the model fit the data" is the loss itself: when the loss is high (early,
underfit) I want little decay so the model can move freely; as the loss drops toward zero (late, at risk of
overfitting a small corpus seen more than once) I want decay to strengthen. Tie the decay to the inverse
squared likelihood — `(1/loss)²` times the base decay — so it's gentle early and ramps up as loss falls,
approaching its max only in the limit the model essentially never reaches:

```python
opt.param_groups[1]['weight_decay'] = (1. / loss.detach().item())**2. * hyp['opt']['weight_decay']
```

Second, the LR anneal. The baseline cosine-anneals; with the dynamic batch front-loading cheap, slightly
noisier early steps, a *linear* anneal down to a slightly lower final fraction gives a cleaner late-run
descent in my testing, and I retune the run length and warmup to match the new dynamics (a longer nominal
step budget but with the effective batch starting at 1, plus a shorter warmup since the early steps are
cheap and forgiving). These are tunings around the controller, not the controller itself.

The bet against the baseline's ~6 minutes: the dominant waste was a constant effective batch that
over-averaged the cheap, large-gradient early phase. Replacing it with a grad-norm-driven controller that
starts at accumulation-1 and only grows the batch when the gradient statistics actually demand it should
cut the per-step cost across most of the run, and I expect that to bring the wall-clock down toward roughly
half. The risk is the controller itself: if the dithering or the grad-norm signal is too noisy it could
oscillate the batch size and destabilize training — the `max(1., …)` floor and the smoothing LR are the
hedges, and as long as the loss still lands at ~3.8 the time saved is real.
