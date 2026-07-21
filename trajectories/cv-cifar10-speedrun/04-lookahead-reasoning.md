I'm at 13.5 epochs and 5.1 seconds, and the scalebias win landed as bet. The boosted bias rate took me
18 → 13.5 epochs and 6.8 → 5.1 seconds, and per-epoch cost held at 5.1/13.5 = 0.378 s — flat to three
digits, the signature of a per-group rate change that is free at run time, so its entire 4.5-epoch effect
read out in the epoch column exactly as Dirac did. That makes it the largest of the optimization-rate
tricks, more than Dirac's 3 and far less than whitening's 24, the ordering I expected. And at 5.1 s I
have finally crossed *under* hlb-CIFAR10's 6.3-second record — everything from here is new ground, and
harder-won now that the gross structural inefficiencies are gone.

The last two wins worked on the optimizer's *rate*, and they sharpen a tension I created on purpose. To
finish in so few epochs I am running the optimizer *fast* — large effective steps, a short triangular
schedule, an aggressive 64× rate on the biases — and fast SGD is noisy SGD. Each minibatch gradient is a
high-variance estimate of the true gradient, so with big steps the weights do not glide down a smooth
trajectory, they bounce around it, overshooting and correcting from one minibatch to the next. Over a
long training that noise averages out — there are enough steps for the oscillations to cancel and for
the iterate to settle into the center of a good region. But at ~49 steps per epoch, 13.5 epochs is only
about 660 steps, and after 660 noisy steps the weights may still be jittering around a good region
rather than sitting in its center. The final weights I evaluate are a *single noisy snapshot* of a noisy
walk, off-center by the amplitude of the jitter.

So can I get the *stability* of a long, well-averaged run without paying the epochs? The classical answer
to reducing the variance of a noisy iterate is averaging: instead of trusting the last point, track a
running (here exponential-moving) average of the points the optimizer visits — Polyak/Ruppert averaging.
The averaged point lives in the *center* of the explored region, so its variance is lower than any single
iterate's and it tends to generalize better. The appeal here is threefold: averaging is almost free (a
running `lerp` over the weights, no extra passes), it costs nothing in epochs, and it attacks exactly
what short-fast training suffers from — end-of-run jitter, the residual noise my last two changes
*introduced* in exchange for speed.

How *actively* to average is the design choice. The passive end is a plain shadow EMA read only at the
end (essentially SWA): the optimizer runs its normal noisy course, I keep an untouched average alongside,
and swap it in for evaluation. That captures the variance reduction but leaves value on the table — if
the averaged weights are a better point, why let the optimizer keep wandering *away* from them for the
rest of training? The active end is Lookahead: keep "slow" weights that are an EMA of the "fast" weights
and periodically *pull the fast weights back onto the slow ones*, so the noise-averaging benefit is felt
*during* training, not just at the end, because the fast iterate keeps re-launching from a stable center
instead of drifting into the jitter. In a run with so few steps, felt-during-training is worth more, so:
Lookahead, the active coupling.

The one real design choice in the mechanics is the decay. Every k steps I update the EMA toward the
current fast weights and copy it *back into* the model. In `ema_param.lerp_(net_param, 1-decay)` the new
EMA is ema·decay + net·(1−decay): the weight on the *fresh* fast weights is (1−decay), so decay near 0
essentially copies the fast weights (no averaging) and decay near 1 barely admits the new point (heavy
averaging, long memory). A *constant* decay would be wrong. Early in training the fast weights move fast
and far, so averaging them aggressively would smear together points from genuinely different regions and
drag the model backward. Late in training the fast weights hover near the optimum, and that is when
averaging pays — the points are all near each other so their mean is a cleaner estimate of the center.
So the decay should *ramp*: near zero early, rising toward one late. A cubic ramp does this, staying
small for most of training and climbing steeply only near the end:

```python
alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
```

Evaluating it confirms it turns the averaging on where I want. The prefactor 0.95⁵ = 0.7738 caps the
decay at 0.7738·(t/T)³. At the midpoint decay = 0.7738·0.125 = 0.097, weight on the fresh weights 0.903
— essentially just tracking the iterate while it still moves far. At t/T = 0.9, decay = 0.564. At the
end, decay = 0.7738, weight on net 0.226; with syncs every 5 steps that is an effective window of about
1/0.226 ≈ 4.4 syncs ≈ 22 steps ≈ 0.45 epoch. So near the finish the reported weights are the center of
roughly the last half-epoch of iterates — a real noise-reducing average — while for the first ~80% of
training the averaging is nearly inert. An EMA rather than a fixed arithmetic window is both cheaper
(O(1) memory, one extra copy updated in place, vs storing several snapshots) and better-shaped: it
weights recent points more, which is right when the iterate is still slowly drifting even near
convergence.

I sync every 5 steps and force a final sync with decay = 1.0:

```python
if current_steps % 5 == 0:
    lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
if current_steps >= total_train_steps:
    lookahead_state.update(model, decay=1.0)   # final sync onto the averaged weights
```

The final sync is what guarantees the evaluated model is the averaged one. With decay = 1.0 the `lerp`
weight on net is 0, so `ema_param.lerp_(net, 0)` leaves the EMA unchanged, and `net_param.copy_(ema_param)`
overwrites the model with the EMA — the model *is* the accumulated average, exactly. The update sweeps
the whole state dict, but a dtype guard only `lerp`s half/float tensors, so integer buffers like
`num_batches_tracked` are copied structurally, not averaged. Frozen parameters (whitening weights, BN
scales) are `lerp`ed but that is a harmless no-op — a frozen weight never changes, so its EMA and live
value are always identical — so I need not special-case them. The sweep *does* catch the BatchNorm
running mean/variance buffers, and that is a feature: those statistics are estimated from the same noisy
trajectory as the weights and jitter too, so averaging them in lockstep keeps the normalization
consistent with the averaged parameters it is paired with at evaluation.

The size of the variance reduction is the whole mechanism. If the fast iterate near convergence jitters
with per-sync variance σ² around the true center, averaging N_eff ≈ 4.4 roughly-independent syncs cuts
the reported point's variance to ~σ²/4.4 — a factor ~4.4 down, about 2.1× smaller standard deviation
(optimistic, since consecutive syncs are correlated, so the real reduction is somewhat less). So the
point I evaluate sits about twice as close to the center as a raw last-iterate snapshot. That is why this
cashes as *fewer epochs* and not just a nicer endpoint: a long run reaches a near-center endpoint by
giving oscillations time to cancel; a short run's raw endpoint is off-center by the jitter. If averaging
pulls it ~2× closer to center, my short run's reported point is about as good as a longer run's raw
point, so I can stop a couple of epochs earlier and still evaluate weights that are effectively settled.

This does not duplicate the triangular schedule's late lr decay. Decaying the lr shrinks *new* steps, so
it reduces how much *fresh* jitter is injected near the end, but it does nothing about the accumulated
displacement the iterate already has from center, and with a short schedule there is no tail for the
shrinking steps to walk that out. Lookahead *averages away* the jitter that is there rather than merely
slowing its accumulation, so the two compose — and the cubic ramp turning on late lines up naturally
with the lr ramping down late, both endgame devices aimed at the last ~20% from different angles.

The sync period k = 5 is a real choice between two failure modes. At k = 1 the fast weights barely move
between syncs, so each averages in a near-identical point — wasted work, too tight to reduce variance.
At k = 50 the fast weights drift far, so the copy-back is a large disruptive yank that throws away most
of the exploration. k = 5 is long enough that the fast weights move a meaningful, roughly-decorrelated
distance worth averaging and short enough that pulling them back is a gentle nudge. The copy-back
overwrites the weights but *not* the Nesterov momentum buffer, so after a sync the fast weights re-launch
from the averaged point while keeping their velocity — the exploration direction is preserved, only the
position recentered. The cost is genuinely tiny: a `lerp` over ~1.97M parameters once every 5 steps
against a forward+backward of billions of FLOPs is negligible, no extra gradients, and the only real cost
is a second copy of the weights (a few MB), nothing on an A100.

I expect a real but distinctly smaller step down than scalebias gave. The previous wins removed *gross*
inefficiencies — un-whitened input (24 epochs), scrambled deep layers (3), under-stepped biases (4.5) —
which are mostly gone. Lookahead attacks the *residual noise* of fast SGD, a second-order effect: it does
not remove wasted work, it just makes the endpoint cleaner. So epochs-to-94% should drop below 13.5 by
less than scalebias's 4.5, with the mean held at the bar and per-epoch cost flat. The honest downside is
that with the big inefficiencies harvested there may not be much variance left to reduce, in which case
the gain is slim — but it costs almost nothing and carries essentially no risk (averaging a jittering
iterate cannot increase its variance). If the epoch count barely moves, the fast-SGD noise was smaller
than I feared. The code is in the answer.
