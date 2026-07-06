I'm at 13.5 epochs and 5.1 seconds, and the scalebias win landed the way I bet it would, which is worth
confirming before I take the next step. The boosted bias learning rate took me 18 → 13.5 epochs and 6.8
→ 5.1 seconds: epochs down by a factor 1.33, wall-clock down by 1.33, and per-epoch cost 6.8/18 = 0.378
s to 5.1/13.5 = 0.378 s — flat to three digits. That flatness is the signature I predicted for a
per-group learning-rate change: it is free at run time, so its entire 4.5-epoch effect reads out in the
epoch column and none of it in the per-epoch column, exactly as Dirac did. And 4.5 epochs makes it the
largest of the optimization-rate tricks so far, more than Dirac's 3 and far less than whitening's 24 —
the ordering I expected. There is a milestone hidden in the seconds, too: at 5.1 s I have finally crossed
*under* hlb-CIFAR10's 6.3-second published record, which was the best public number I began beneath. So
everything from here is new ground, and the wins will be harder-won because the gross structural
inefficiencies are gone.

Both of the last two wins worked on the optimizer's *rate*, not its *starting point* — but they also
sharpen a tension I now have to face squarely, and it is a tension I created on purpose. To finish in so
few epochs I am running the optimizer *fast*: large effective steps, a short triangular schedule, an
aggressive 64× learning rate on the biases. Fast SGD is noisy SGD. Each minibatch gradient is a
high-variance estimate of the true full-batch gradient, and with big steps the weights do not glide down
a smooth trajectory — they bounce around it, overshooting and correcting from one minibatch to the next.
Over a long training that noise averages out on its own: there are enough steps for the oscillations to
cancel and for the iterate to settle into the center of a good region. But I have almost no time. At ~49
steps per epoch, 13.5 epochs is only about 660 optimizer steps, and after 660 noisy steps the weights may
still be jittering around a good region rather than sitting quietly in its center. The final weights I
actually evaluate are a *single noisy snapshot* of a noisy walk — and a single snapshot of a jittering
iterate is, by construction, off-center by the amplitude of the jitter.

So the question is whether I can get the *stability* of a long, well-averaged run without paying the
epochs. The classical answer to "reduce the variance of a noisy iterate" is averaging: instead of
trusting the last point, track a running average of the points the optimizer visits. Polyak/Ruppert
averaging does exactly this — maintain a running (here exponential moving) average of the weights and
report the average at the end. The averaged point lives in the *center* of the region the noisy iterate
is exploring, so its variance is lower than any single iterate's, and lower-variance weights tend to
generalize better. The appeal in this regime is threefold: averaging is almost free (a running `lerp`
over the weights, no extra forward or backward passes), it costs nothing in epochs, and it attacks
exactly the thing short-fast training suffers from — end-of-run jitter. This is the residual noise that
my last two rungs *introduced* in exchange for speed, so it is precisely the right target now that the
conditioning inefficiencies are already gone.

But I should decide how *actively* to average, because there is a spectrum of options. The passive end is
a plain shadow EMA that I only read at the very end (essentially SWA / stochastic weight averaging): the
optimizer runs its normal noisy course, I keep an untouched average alongside it, and I swap the average
in for evaluation. That captures the variance-reduction-at-eval benefit but leaves value on the table —
if the averaged weights are genuinely a better point than the raw iterate, why let the optimizer keep
wandering *away* from them for the rest of training? The active end is Lookahead: keep "slow" weights
that are an EMA of the optimizer's "fast" weights, and periodically *pull the fast weights back onto the
slow ones* — not only track the average but every so often reset the optimizer to it. The slow weights
look ahead along the smoothed trajectory; the fast weights do the local exploring; coupling them means
the noise-averaging benefit is felt *during* training, not just at the end, because the fast iterate
keeps re-launching from a stable center instead of drifting off into the jitter. In a run this short,
felt-during-training is worth more than felt-only-at-the-end — there are so few steps that letting the
fast weights wander uncorrected between here and the finish wastes some of them. So: Lookahead, the
active coupling, not a passive shadow.

Now the mechanics, and the one real design choice inside them is the decay. I keep a copy of the full
model state as the slow/EMA weights. Every k steps I update the EMA toward the current fast weights and
copy the EMA *back into* the model — both directions, which is the Lookahead synchronization rather than
a one-way shadow. The update is a `lerp`, and I need to be careful about which way the decay points. In
`ema_param.lerp_(net_param, 1-decay)` the new EMA is ema·decay + net·(1−decay): the weight placed on the
*fresh* fast weights is (1−decay). So decay near 0 means the EMA essentially copies the fast weights
(weight ≈ 1 on net — no averaging), and decay near 1 means the EMA barely admits the new point (weight ≈
0 on net — heavy averaging, long memory). That sign is the crux of the schedule, because a *constant*
decay would be wrong here. Early in training the fast weights are moving fast and far across the loss
surface; averaging them aggressively would smear together points from genuinely different regions and drag
the model backward toward where it has already been. Late in training the fast weights are hovering near
the optimum, and that is exactly when averaging pays — the points being averaged are all near each other,
so their mean is a cleaner estimate of the center. So the decay should *ramp*: near zero early (trust the
fast, far-moving weights; barely average) and rising toward one late (lean heavily on the average near
convergence). A cubic ramp in the step fraction does this — it stays small for most of training and climbs
steeply only near the end:

```python
alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
```

Let me actually evaluate this schedule at a few points to make sure it turns the averaging on where I
want and not before. The prefactor is 0.95**5 = 0.7738, so the decay is capped at 0.7738 · (t/T)³. At the
midpoint t/T = 0.5, (0.5)³ = 0.125, giving decay = 0.7738 · 0.125 = 0.097 — so the weight on the fresh
fast weights is 1 − 0.097 = 0.903, and the EMA is essentially just tracking the iterate, almost no
averaging, which is what I want while the weights are still moving far. At t/T = 0.9, (0.9)³ = 0.729,
decay = 0.564, weight on net 0.436 — now the EMA is genuinely averaging, holding onto the past with a bit
more than half its mass. At the very end t/T = 1, decay = 0.7738, weight on net 0.226; with syncs every 5
steps that is an effective averaging window of about 1/0.226 ≈ 4.4 syncs ≈ 22 steps ≈ 0.45 epoch. So near
the finish the reported weights are the center of roughly the last half-epoch of iterates — a real
noise-reducing average over the region the optimizer is exploring at convergence — while for the first
~80% of training the averaging is nearly inert. That is precisely the profile the cubic was chosen to
produce, and evaluating it confirms it does rather than assuming it.

Why an exponential moving average specifically, rather than a fixed arithmetic window over the last few
snapshots (the literal SWA construction)? Two reasons, both practical here. An EMA is O(1) in memory — one
extra copy of the weights, updated in place — whereas an honest arithmetic window would mean storing
several full snapshots and summing them, several times the memory for the same effect. And an EMA weights
recent points more than distant ones, which is the right bias when the iterate is still *slowly drifting*
even near convergence: I want the average centered on where the optimizer is *now*, not dragged back
toward where it was ten syncs ago. The exponential decay handles that automatically, and the cubic ramp on
the decay just modulates how long the effective memory is at each phase of training. So the EMA is both
cheaper and better-shaped than a windowed average for this job.

The synchronization cadence and the endgame:

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

I sync every 5 steps (about 132 syncs over the 660-step run) and force a final sync with decay = 1.0. Let
me trace that final sync, because it is what guarantees the evaluated model is the averaged one and not
the raw last iterate. With decay = 1.0, the `lerp` weight on net is 1 − 1 = 0, so `ema_param.lerp_(net,
0)` leaves the EMA unchanged, and then `net_param.copy_(ema_param)` overwrites the model with the EMA. So
after the final sync the model *is* the accumulated average, exactly. Without that forced sync I would
evaluate whatever noisy point the optimizer happened to stop on; with it, I evaluate the center. Note also
the dtype guard — only half/float tensors are averaged, so integer buffers like BatchNorm's
`num_batches_tracked` are copied structurally but not `lerp`ed, which is correct (averaging a counter is
meaningless). It is also worth noting the EMA sweeps the *entire* state dict, including the frozen
whitening weights and the frozen BatchNorm scales — but for those the `lerp` is a harmless no-op, because
a frozen parameter never changes, so its EMA and its live value are always identical and averaging them
returns the same tensor. So I do not need to special-case the frozen layers; the averaging simply does
nothing to them, which is exactly right — I only want to average away the *jitter*, and the frozen
parameters have none. The flip side is that the sweep *does* catch the BatchNorm running mean and variance
buffers, which are floats and so get `lerp`ed along with the weights — and that is a feature, not a bug.
Those running statistics are estimated from the same noisy trajectory as the weights, so they jitter too;
averaging them in lockstep with the weights keeps the normalization consistent with the averaged
parameters it will be paired with at evaluation. If I averaged the weights but evaluated them against the
last noisy snapshot of the running stats, the two would be slightly mismatched; sweeping the whole state
dict together avoids that. So the single `lerp` over `state_dict()` is doing exactly the coherent thing —
recentering the weights and the statistics that normalize them by the same amount, at the same time.

Let me put a number on the variance reduction, because that is the entire mechanism and I should know its
size. If the fast iterate near convergence is jittering with per-sync noise variance σ² around the true
center, then averaging N_eff roughly-independent syncs reduces the variance of the reported point to about
σ²/N_eff. At the end my effective window is N_eff ≈ 4.4 syncs, so the averaged weights have roughly σ²/4.4
variance — a factor ~4.4 down, or about 2.1× smaller *standard deviation*. So the point I evaluate sits
about twice as close to the center of the good region as a single last-iterate snapshot would. The syncs
are not perfectly independent (consecutive iterates are correlated), so 2.1× is an optimistic ceiling and
the real reduction is somewhat less — but even "meaningfully closer to center than a raw snapshot" is the
whole game, and 2.1× is the right order.

That is also precisely why this can be cashed as *fewer epochs* rather than just a nicer endpoint. A long
training reaches a clean, near-center endpoint by giving the oscillations time to cancel on their own; a
short training's raw endpoint is off-center by the jitter amplitude. If averaging pulls my short-run
endpoint ~2× closer to center, then my short run's *reported* point is about as good as a longer run's raw
point — so I can stop a couple of epochs earlier and still evaluate weights that are effectively as settled.
The variance reduction converts directly into schedule length: I am buying, in a `lerp`, some of the
settling that would otherwise cost real epochs.

There is a redundancy worth checking, because the triangular schedule already decays the learning rate to
near zero at the end, and a smaller step size *also* reduces late jitter — so does Lookahead duplicate the
lr decay? It does not, and the distinction is clean. Decaying the lr shrinks the size of *new* steps, so
it reduces how much *fresh* jitter is injected near the end; but it does nothing about the accumulated
displacement the iterate already has from the center at the moment the lr starts decaying, and with a
short schedule there is not enough tail for the shrinking steps to walk that displacement out. Lookahead
does the complementary thing: it *averages away* the jitter that is there, rather than merely slowing its
accumulation. So the two compose — the lr decay quiets the late oscillation, the EMA centers whatever
oscillation remains — and neither substitutes for the other. This is also why the cubic ramp turning on
late lines up so naturally with the lr ramping down late: both are endgame devices, and they are aimed at
the same last ~20% of training from two different angles.

The sync period k = 5 is a real choice between two failure modes. Syncing every single step (k = 1) is
maximal coupling, but the fast weights barely move in one optimizer step, so most syncs would average in a
point almost identical to the last one — wasted work and an EMA that tracks too tightly to reduce much
variance. Syncing rarely (k = 50) lets the fast weights drift far between syncs, so the copy-back becomes a
large, disruptive yank that throws away most of the exploration the fast weights just did. k = 5 sits in
between: five steps is long enough that the fast weights have moved a meaningful, roughly-decorrelated
distance worth averaging, and short enough that pulling them back onto the slow weights is a gentle nudge
rather than a jerk. One subtlety in the coupling: the copy-back overwrites the weights but *not* the
Nesterov momentum buffer, so after a sync the fast weights re-launch from the averaged point while keeping
their accumulated velocity — the exploration direction is preserved, only the position is recentered,
which is the behavior I want.

The cost is genuinely tiny, which matters because the whole pitch is "near-free variance reduction." A
`lerp` over ~1.97M parameters once every 5 steps is ~0.4M element updates per step amortized, against a
forward+backward that is on the order of billions of FLOPs — utterly negligible. There are no extra
gradients: the EMA is read-only with respect to autograd. The one real cost is memory: the EMA is a second
copy of the weights (~1.97M floats, a few MB), which on an A100 is nothing. So this buys stability at
essentially zero time cost, and the only question is whether there is enough residual noise left to
harvest to move the epoch count.

That question is exactly where my honesty about magnitude has to live. The previous wins were big because
they removed *gross* inefficiencies — un-whitened input (24 epochs), scrambled deep layers (3 epochs),
under-stepped biases (4.5 epochs). Those are mostly gone. What Lookahead attacks is the *residual noise*
of fast SGD, which is a second-order effect: it does not remove wasted work, it just makes the endpoint of
the work cleaner. So I expect a real but distinctly smaller step down than scalebias gave — the averaged
weights generalize a touch better, the short run no longer ends on a noisy snapshot, and the epoch count
drops modestly below 13.5 while accuracy holds at 94%. The honest downside is that, with the big
inefficiencies already harvested, there may not be much variance left to reduce, in which case the gain is
slim — but since it costs almost nothing in time, even a small, reliable speedup is worth taking, and it
carries essentially no risk of making things worse (averaging a jittering iterate cannot increase its
variance). Per-epoch cost should stay flat, since the `lerp` is negligible. The falsifiable prediction:
epochs-to-94% down from 13.5 with the mean held at the bar, by less than scalebias's 4.5; if instead the
epoch count barely moves, that tells me the fast-SGD noise was smaller than I feared and there was little
to harvest. The change is the `LookaheadState` EMA with the cubic decay ramp and 5-step sync plus the
forced final sync; the code is in the answer, and the epochs table will tell me how much end-of-run jitter
there really was.
