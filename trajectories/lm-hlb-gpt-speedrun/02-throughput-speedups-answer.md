**Problem (from rung 1).** The baseline reaches the bar in ~6 minutes but holds the effective batch size
constant for the whole run via a fixed gradient-accumulation count. A constant can't be right at both ends:
it over-averages the cheap, large-gradient early steps (wasting forward/backward passes) and risks
under-averaging the noisy, small-gradient late steps. The kernel-level levers (bf16, fused attention,
compile) are already spent, so the time left is in the *schedule*.

**Key idea.** A grad-norm-driven dynamic effective-batch controller. Start at accumulation 1 (cheapest
steps while the gradient is huge), keep an EMA of the gradient norm, and grow the accumulation count only
when the per-step grad-norm behavior says the run is under-averaging — `per_step_diff_delta =
target − (running_grad_norm − grad_norm)` nudges the fractional accumulation estimate up or down. Realize
the fractional estimate as an integer per step by Bernoulli dithering, so the *average* effective batch
tracks the estimate while each step stays integer. Pair it with loss-tied dynamic weight decay (`(1/loss)²`
× base, gentle early and ramping as loss falls) and a retuned linear LR anneal.

**Why it works.** The dominant waste was a constant effective batch over-averaging the cheap early phase.
A controller that starts tiny and grows only when the gradient statistics demand averaging cuts per-step
cost across most of the run, roughly halving the wall-clock, while the loss bar is held. Dithering keeps
the fine-grained control that integer rounding would destroy; the `max(1,·)` floor and smoothing LR keep
the controller from oscillating.

**Change / code.** The grad-norm controller, the dithered accumulation, and dynamic weight decay, inside
the step.

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
