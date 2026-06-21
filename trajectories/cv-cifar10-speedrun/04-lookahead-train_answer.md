The boosted bias learning rate brought me to 13.5 epochs and 5.1 A100-seconds, but it also sharpens a tension I now have to face. To finish in so few epochs I am running the optimizer *fast*: large effective steps, a short schedule, an aggressive 64× learning rate on the biases. Fast SGD is noisy SGD — each minibatch gradient is a high-variance estimate of the true gradient, and with big steps the weights bounce around their trajectory rather than gliding down it. Over a long training that noise averages out on its own, because there is time for the oscillations to cancel. But in 13.5 epochs there are only so many steps, and the weights may still be jittering around a good region at the end rather than sitting in its center. The final weights I evaluate are a single noisy snapshot of a noisy walk.

I propose to recover the stability of a long, well-averaged run without paying the epochs by using **Lookahead optimization**: maintain "slow" weights that are an exponential moving average of the optimizer's "fast" weights, and every few steps update the EMA *and copy it back into the model*. The classic way to reduce the variance of a noisy iterate is averaging — instead of trusting the last point, track a running average of the points the optimizer visits, the way Polyak/Ruppert averaging does. The average lives in the *center* of the region the noisy iterate explores, so it has lower variance than any single iterate and tends to generalize better. A purely passive EMA that I only read at the end leaves value on the table, though: if the averaged weights are a better point than the raw iterate, why let the optimizer keep wandering away from them? Lookahead's answer is to couple the two — the slow weights look ahead along the smoothed trajectory, the fast weights do the local exploring, and synchronizing them periodically gives the noise-averaging benefit *during* training, so the fast iterate keeps re-launching from a stable center instead of drifting off it.

The mechanics for this setting: I keep a copy of the full model state-dict as the slow/EMA weights. Every $k=5$ optimizer steps I update the EMA toward the current fast weights with a `lerp_` and then copy the EMA *back into* the model — both directions, which is the Lookahead synchronization rather than a passive shadow. The single design choice is the EMA decay, and a constant decay is wrong here. Early in training the fast weights are moving fast and far, so averaging them aggressively would just smear together points from very different parts of the loss surface and hold the model back; late in training the fast weights are hovering near the optimum, which is exactly when averaging pays off most. So the decay should *ramp*: near zero early (trust the fast weights, barely average) and rising toward one late (lean heavily on the average). A cubic ramp in the step fraction, $0.95^5\,(t/T)^3$, does this — it stays small for most of training and climbs steeply near the end. I force a final synchronization with decay 1.0 at the very last step so the model I actually evaluate *is* the averaged one. The cost is genuinely tiny — a `lerp` over the parameters once every five steps, no extra forward or backward passes, and the EMA is just a second copy of the weights — so this is a near-free variance reduction aimed at exactly what short, fast training suffers from: end-of-run jitter. With the gross inefficiencies already removed by the earlier rungs, the residual SGD noise is what is left, and Lookahead harvests it for a real if smaller step down in epochs with accuracy held.

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
