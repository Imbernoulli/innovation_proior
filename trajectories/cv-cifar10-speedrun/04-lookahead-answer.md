**Problem (from step 3).** To finish in 13.5 epochs / 5.1 s the optimizer runs fast and noisy — large
steps, a short schedule, a 64× bias lr — so the weights jitter around their trajectory rather than gliding
down it, and the final evaluated weights are a single noisy snapshot. A long run averages that noise out
over time; a 13.5-epoch run has no time, so it may end while the iterate is still bouncing around a good
region instead of sitting in its center.

**Key idea.** Use Lookahead: maintain "slow" weights that are an exponential moving average of the
optimizer's "fast" weights, and every few steps update the EMA *and copy it back into the model*, so the
fast iterate keeps re-launching from a stable, noise-averaged center during training (not just at the end).
Ramp the EMA decay with a cubic schedule — near zero early (trust the fast, far-moving weights) rising
toward one late (lean on the average near the optimum) — and force a final sync so the evaluated model *is*
the averaged one.

**Why it works.** Averaging reduces the variance of a noisy iterate: the EMA sits in the center of the
region the fast weights explore, generalizes better, and removes the end-of-run jitter that short, fast
training suffers from. It costs almost nothing — a `lerp` over the weights once every 5 steps, no extra
forward/backward passes. The cubic decay avoids smearing together far-apart early iterates and concentrates
the averaging where it pays off, near convergence.

**Change / code.** A `LookaheadState` holding an EMA of the full state-dict, synced every 5 steps with a
cubic decay ramp and a forced decay-1.0 sync at the end.

```python
class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)

# in main():
alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
lookahead_state = LookaheadState(model)
...
    if current_steps % 5 == 0:
        lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
    if current_steps >= total_train_steps:
        lookahead_state.update(model, decay=1.0)   # final sync onto the averaged weights
        break
```
