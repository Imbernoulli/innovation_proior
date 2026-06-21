The baseline reaches the bar in just over six minutes, and I want to roughly halve that. The free, correctness-neutral kernel levers — bf16, fused attention, compile — are already spent in the baseline itself, so the time left on the table isn't in the *kernels*; it's in the *schedule*. The one choice I flagged as "the obvious default, not justified" is the fixed gradient-accumulation count, i.e. the fixed effective batch size held constant for the whole run. A large effective batch averages down gradient noise so you can take a confident step, but averaging is only worth it when the noise is actually hurting you. Early in training the gradient is huge and points obviously downhill, so a single noisy microbatch already points the right way and pouring five microbatches into one step buys almost nothing except five forward/backward passes per optimizer step. Late in training the surface is flat, the gradient is small, and the noise is a large *fraction* of the signal — now averaging many microbatches is exactly what keeps the step from being dominated by sampling noise. A single constant can't be right at both ends: too big early wastes compute, too small late gives noisy steps, and any compromise is wrong at both.

So I propose a **grad-norm-driven dynamic effective-batch scheduler**: let the effective batch *grow* over the run — small early, large late — driven by a measurable signal instead of a hand-tuned curve I'd have to re-tune per model size. The signal is the gradient norm. When training is healthy and the effective batch is well matched, the grad norm drifts down smoothly; if the batch is too small, individual steps jitter the grad norm around so the per-step change is large and erratic relative to the smooth trend. The *difference between the smoothed grad norm and the instantaneous grad norm* is therefore a readout of how much per-step noise I'm eating. I keep an exponential moving average $\bar g$ of the grad norm (`running_grad_norm`), and at each step compute the actual norm $g$ and form $\text{per\_step\_diff\_delta} = \text{target} - (\bar g - g)$, where $\text{target}$ is the small per-step decay I'm willing to tolerate. If the realized per-step drop is *smaller* than the target (the run is making smooth, un-noisy progress) the delta is positive and I nudge the accumulation count *up*; if it's larger (steps are jittery) I nudge it down. The nudge is scaled by the current accumulation count and a small learning rate so the batch size adapts smoothly:
$$\bar g \leftarrow \text{decay}\cdot \bar g + (1-\text{decay})\cdot g, \qquad \hat n \mathrel{+}= n \cdot \big(\text{lr}\cdot(\text{target} - (\bar g - g))\big), \qquad \hat n \leftarrow \max(1, \hat n).$$

The subtlety that makes this actually work is that $\hat n$ (`accumulate_steps_estimate`) is a *fractional* number — say 2.3 microbatches per step — but I can only accumulate an integer count. If I simply round it I quantize away exactly the fine-grained control I just built: the effective batch jumps from 2 to 3 in discrete cliffs and the controller chatters at the boundary. The fix is *dithering*. I split the estimate into its integer base and fractional remainder via `divmod(·, 1)`, and stochastically round — with probability equal to the remainder, do one extra microbatch (`torch.bernoulli`). Over many steps the *average* effective batch tracks the fractional estimate exactly, while each individual step uses a clean integer count. This is the engine: it starts the run at accumulation 1 — the cheapest possible steps while the gradient is huge and forgiving — and lets the controller grow the batch only as the run actually demands more averaging, which is where the wall-clock saving comes from, since most of the run (especially the cheap early part) now runs at a far smaller effective batch than the baseline's constant.

Two schedule changes pair with the controller naturally. First, *loss-tied weight decay*. With the effective batch now changing over the run, I want regularization to track the run's state rather than sit at a constant; the loss is a clean signal for how much the model has fit the data. When the loss is high (early, underfit) I want little decay so the model can move freely, and as the loss falls toward zero (late, at risk of overfitting a small corpus seen more than once) I want decay to strengthen. So I tie it to the inverse squared likelihood — $(1/\text{loss})^2$ times the base decay — gentle early and ramping up as the loss drops, approaching its max only in the limit the model essentially never reaches. Second, the LR anneal: the baseline cosine-anneals, but with the dynamic batch front-loading cheap, slightly noisier early steps, a *linear* anneal down to a slightly lower final fraction gives a cleaner late-run descent, and I retune the run length and warmup to match (a longer nominal step budget but with the effective batch starting at 1, plus a shorter warmup since the early steps are cheap and forgiving). These are tunings around the controller, not the controller itself.

What makes the bet against the baseline's ~6 minutes pay off is that the dominant waste really was a constant effective batch over-averaging the cheap, large-gradient early phase; replacing it with a controller that starts at accumulation 1 and grows only when the gradient statistics demand it cuts the per-step cost across most of the run, bringing the wall-clock down toward roughly half while the ~3.8 loss bar is held. The real risk is the controller oscillating — if the dithering or grad-norm signal is too noisy it could destabilize training — and the $\max(1, \cdot)$ floor together with the smoothing learning rate are the hedges; as long as the loss still lands at ~3.8, the saved time is real.

```python
def get_grad_norm(net):
    grad_norm = torch.tensor(0., device='cuda')
    for p in net.parameters():
        if p.grad is not None:
            grad_norm += p.grad.detach().data.norm(2).square()
    return (grad_norm ** 0.5).item()

# init
current_accumulate_steps = accumulate_steps_estimate = hyp['opt']['initial_accumulate_steps']  # = 1
running_grad_norm = 1.2                 # EMA of the grad norm, seeded near the initial value
running_grad_norm_decay = .95
target_per_step_decay   = 3e-2          # tolerated smooth per-step grad-norm drop
accumulate_steps_lr     = 5e-2          # smooths the batchsize adaptation rate

# ... inside the loop, scaling each microbatch loss by the current accumulation count:
loss.div(current_accumulate_steps).backward()

# ... once a full accumulation group is done, take one optimizer step, then adapt:
if microbatch_step % current_accumulate_steps == 0:
    opt.step()
    opt.param_groups[1]['weight_decay'] = (1. / loss.detach().item())**2. * hyp['opt']['weight_decay']
    scheduler.step()                    # linear anneal (anneal_strategy='linear')

    grad_norm = get_grad_norm(net)
    running_grad_norm = running_grad_norm_decay * running_grad_norm + (1. - running_grad_norm_decay) * grad_norm
    per_step_diff_delta = target_per_step_decay - (running_grad_norm - grad_norm)
    accumulate_steps_estimate += current_accumulate_steps * (accumulate_steps_lr * per_step_diff_delta)
    accumulate_steps_estimate = max(1., accumulate_steps_estimate)

    # dither the fractional estimate to a clean integer; the running average tracks the estimate
    base, probability = divmod(accumulate_steps_estimate, 1)
    current_accumulate_steps = max(1, int(base + torch.bernoulli(torch.tensor(probability)).item()))

    opt.zero_grad(set_to_none=True)
```
