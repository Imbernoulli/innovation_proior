I'm at 13.5 epochs and 5.1 seconds. The boosted bias learning rate helped a lot, but it also sharpens a
tension I now have to face squarely. To finish in so few epochs I am running the optimizer *fast* —
large effective steps, a short schedule, an aggressive 64× lr on the biases. Fast SGD is noisy SGD:
each minibatch gradient is a high-variance estimate of the true gradient, and with big steps the weights
bounce around their trajectory rather than gliding down it. Over a long training that noise averages out
on its own — there is time for the oscillations to cancel. But I have almost no time. In 13.5 epochs
there are only so many steps, and the weights may still be jittering around a good region at the end
rather than sitting in its center. The final weights I evaluate are a single noisy snapshot of a noisy
walk.

So the question is: can I get the *stability* of a long, well-averaged run without paying the epochs? The
classic answer to "reduce the variance of a noisy iterate" is averaging — instead of trusting the last
point, track a running average of the points the optimizer visits. Polyak/Ruppert averaging does exactly
this: maintain an exponential moving average of the weights and report the average at the end. The
average lives in the *center* of the region the noisy iterate is exploring, so it has lower variance
than any single iterate and tends to generalize better. The appeal here is that averaging is almost
free — it is a running `lerp` over the weights, no extra forward or backward passes — and it directly
attacks the thing that short, fast training suffers from: end-of-run jitter.

But a *passive* EMA that I only read at the end leaves value on the table. If the averaged weights are a
better point than the raw iterate, why let the optimizer keep wandering away from them? This is the
Lookahead idea: keep "slow" weights that are an EMA of the optimizer's "fast" weights, and periodically
*pull the fast weights back* onto the slow ones — i.e., not only track the average but every so often
reset the optimizer to it. The slow weights look ahead along the smoothed trajectory; the fast weights
do the local exploring; coupling them gives the noise-averaging benefit *during* training, not just at
the end, so the fast iterate keeps re-launching from a stable center instead of drifting. Lookahead has
been found to help training speed elsewhere — it is a natural fit for a regime that is deliberately
noisy to be fast.

Let me get the mechanics right for this setting. I keep a copy of the full model state as the slow/EMA
weights. Every k optimizer steps I update the EMA toward the current fast weights and then copy the EMA
*back into* the model — both directions, which is the Lookahead synchronization, not a passive shadow.
The EMA decay is the one design choice, and a constant decay is wrong here. Early in training the fast
weights are moving fast and far — averaging them aggressively would just smear together points from very
different parts of the loss surface and hold the model back. Late in training the fast weights are
hovering near the optimum and that is exactly when averaging pays off most. So the decay should *ramp*:
near zero early (trust the fast weights, barely average) and rising toward one late (lean heavily on the
average). A cubic ramp in the step fraction does this — it stays small for most of training and climbs
steeply near the end:

```python
alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
```

and I synchronize every 5 steps, with a final forced sync (decay 1.0) at the very end so the model I
evaluate *is* the averaged one:

```python
class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}
    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)
```

```python
if current_steps % 5 == 0:
    lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
...
if current_steps >= total_train_steps:
    lookahead_state.update(model, decay=1.0)   # final sync onto the averaged weights
```

The cost is genuinely tiny: a `lerp` over the parameters once every 5 steps, no extra gradients, and the
EMA tensors are just a second copy of the weights. So this is a near-free variance reduction, and the
question is only whether averaging buys enough stability to clear 94% in fewer than 13.5 epochs.

The prediction. The previous wins were big because they removed gross inefficiencies — un-whitened
input, scrambled deep layers, under-stepped biases. Those are mostly gone now, so what's left is the
residual *noise* of fast SGD, and Lookahead attacks exactly that. I expect a real but smaller step down:
the averaged weights generalize a touch better, the short run no longer ends on a noisy snapshot, and
the epoch count drops modestly while accuracy holds at 94%. The risk is that with the gross
inefficiencies already removed there isn't much variance left to harvest, in which case the gain is
slim — but since it costs almost nothing, even a small, reliable speedup is worth taking. The change is
the `LookaheadState` EMA with the cubic decay ramp and 5-step sync; code in the answer.
