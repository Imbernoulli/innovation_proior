Let me start from what actually wastes my time when I train one of these residual networks. I run SGD with momentum, I pick a learning rate around 0.1, and I watch the same curve every single run: a burst of progress in the first handful of epochs, then a long, flat plateau where each epoch buys me almost nothing, and only when I drop the rate by a factor of ten does the accuracy lurch upward again, plateau again, drop again. Two or three of those drops and I'm done — eighty thousand iterations later. The plateaus are where the iterations die. So before I touch the optimizer, I want to understand *why* the plateau exists and whether the learning rate, the one knob I'm allowed to move per step, can be made to spend fewer iterations there.

The picture I keep coming back to is the loss-topology one from Goodfellow, Vinyals and Saxe — the trajectory makes steep, fast progress in the first few iterations, then enters a long nearly-flat valley where the slope is tiny so the per-iteration progress is tiny, and only at the very end does it have to thread a narrow trough down to the local minimum. And Dauphin and others argued the real obstacle in these high-dimensional losses isn't bad local minima, it's saddle points: broad regions where the gradient is small in many directions, so descent stalls. Now overlay the learning rate on that. In the steep early part, a big step would overshoot — I need a *small* rate to make progress in the right direction without diverging. In the long flat valley, a small rate is exactly the wrong thing: the slope is already tiny, so a small rate times a tiny slope is a microscopic step, and I sit in the plateau for tens of thousands of iterations. There I want a *large* rate, to cross the flat region fast and to punch through the saddle plateaus rather than crawling over them. And at the very end, threading the narrow trough, a large rate would bounce me out of it — I need a *small* rate again to settle in. So the rate I want, read straight off the topology, is small, then large, then small. Up, then down, once.

That already cuts against the conventional wisdom, which says keep the rate small and *only* decrease it. And it cuts against my current step-decay recipe, which is monotone non-increasing — it never goes *up*, so it never gets the fast-crossing benefit in the valley, it just grinds through with smaller and smaller steps. I've seen the idea that a temporary increase can help, though. The cyclical-learning-rate work made it concrete: let the rate oscillate linearly between a lower and an upper bound, back and forth, the whole run. Its triangular schedule, with `stepsize` the half-cycle length, is

  cycle = floor(1 + iter/(2·stepsize)),
  x     = |iter/stepsize − 2·cycle + 1|,
  lr    = base_lr + (max_lr − base_lr)·max(0, 1 − x),

so `lr` ramps from `base_lr` up to `max_lr` and back every `2·stepsize` iterations, forever. The justification there is exactly the saddle argument: increasing the rate hurts in the short term but can help overall, because a bigger step traverses a plateau that a small step stalls on. Good — that confirms going *up* is sometimes right. But look at what the triangular policy actually does over a run: many full up-and-down cycles, oscillating between the same two fixed bounds the entire time, and it never drives the rate *below* the lower bound. My topology reading says I want one rise and one fall, and then — to settle into that narrow trough at the very end — a rate driven well below where I started, not bounced back up to the top for another cycle. So I don't want to cycle repeatedly. I want a single cycle, smaller than the whole run, and then a final tail where the rate keeps dropping, several orders of magnitude below the initial rate, for the remaining iterations. That last tail is the simulated-annealing move: cool to a quiet, low-temperature search to lock in the solution. And the rise-then-fall is the curriculum move, easy-to-hard-to-settle. The schedule is curriculum plus annealing welded into one cycle.

Before I commit to that shape, I have to deal with the elephant: "a *large* rate in the valley." How large? The whole field treats the learning rate as something that must stay small, below some safe ceiling, or training diverges. If "large" means 0.15 instead of 0.1 this is a marginal trick, not an order-of-magnitude speedup. So I need to actually find out how large a rate this network tolerates, and I'd rather not grid-search it per architecture. The cyclical-LR work handed me the tool: the LR range test. Run the network for a short pre-training run, start the rate near zero, increase it linearly the whole time, and plot test accuracy against the rising rate. While the rate is small the net converges and accuracy climbs; as the rate grows accuracy holds; once the rate is too large the accuracy turns ragged and falls. The rate at that turn — the peak — is the largest value I can use as my upper bound. One short run, one curve, the bound read straight off it; the lower cyclic bound can sit a factor of three or four below the peak. No grid search.

The range-test curves for a 56-layer residual net on CIFAR-10 break my assumptions. The rate moves past the conventional ceiling, and the accuracy doesn't fall off where I expect — it stays high over a band of rates an order of magnitude larger than the 0.1 I'd normally use, all the way out near 3. Three. That's not a tweak; that's the network telling me it can tolerate rates I'd have ruled out. So the topology argument and the range test together aren't just suggesting I nudge the rate up in the valley — they're saying the upper bound can be enormous, which is exactly what could make the cross-the-valley phase fast enough to remove most of the plateau. The unusual flatness of that accuracy-versus-rate curve over a huge range is the signal that this is possible at all.

Now I have to ask *why* a rate that large doesn't wreck the final accuracy — because if the speedup costs me quality, it's not a win. The same range-test diagnostic, viewed through training loss and test loss separately, gives the clue: over the band of large rates, training loss goes *up* while test loss goes *down*. The generalization gap is the test loss minus the training loss, and it's shrinking as the rate grows. By the working definition — regularization is any modification that reduces generalization error without reducing training error — a large learning rate *is* a regularizer. It's not merely surviving the large rate; the large rate is doing the work of a regularizer, narrowing the gap. And I can see the mechanism from the noise side. Smith and Le derived the SGD gradient-noise scale as

  g ≈ eps·N / (B·(1 − m)),

with `eps` the rate, `N` the dataset size, `B` the batch size, `m` the momentum. So a larger `eps` means larger SGD noise, and the line of work on flat minima — Keskar, Jastrzębski, Hochreiter — says higher noise pushes SGD toward wider, flatter minima, which generalize better. Large rate in the middle of training injects exactly that good noise, finding a flatter region; the small rate at the end cools it into the trough. The rise-hold-fall isn't just fast, it's a regularizing noise schedule: ramp the noise up, peak it, ramp it down.

That immediately forces a consequence I'd have missed otherwise. If a large learning rate is itself a strong regularizer, and I'm *also* running my usual weight decay and dropout at their usual strengths, then I'm over-regularizing — piling the large-rate regularization on top of the conventional regularization, and the network underfits or the large rate can't be used at all. The noise-scale equation says the same thing from the other direction: `g` depends on `eps`, `B`, and `m` together, so if I crank `eps` I've changed the total noise budget, and the other knobs have to give. So the principle isn't "use a big rate," it's "the total amount of regularization has to be *balanced*; if I add a lot via the rate, I must *remove* some elsewhere." Concretely: reduce weight decay, maybe reduce dropout, possibly use a larger batch, so the big rate fits. This is why naive attempts to just turn up the rate fail — people leave all the other regularizers at their textbook values and the network can't cope. Balance is the precondition for the whole thing.

Let me get more careful about *how large* "large" should be, because I'd like a principled estimate of the optimal rate, not just "the peak of the range test." Go back to second-order optimization. Locally the loss is a quadratic,

  f(theta) ≈ f(theta_0) + (theta − theta_0)ᵀ∇f(theta_0) + ½(theta − theta_0)ᵀ H (theta − theta_0),

with `H` the Hessian. Newton's step uses `H⁻¹`, but `H` has order `N²` entries for `N`
parameters — hopeless to form. The Hessian-free idea is that I don't need the whole Hessian; the only curvature that matters is in the direction SGD is actually moving, the steepest-descent direction. And a directional second derivative is just a finite difference of gradients:

  H(theta) = lim_{delta→0} [ ∇f(theta + delta) − ∇f(theta) ] / delta,

with `delta` along the descent direction. The AdaSecant line turned that into a per-weight optimal rate by reading off the secant: the step that would have zeroed the gradient is

  eps* ≈ (theta_{i+1} − theta_i) / (∇f(theta_{i+1}) − ∇f(theta_i)),

i.e. the displacement divided by the change in gradient — small change in gradient for a given displacement means small curvature means a large optimal step. Now I substitute the actual SGD relation `theta_{i+1} = theta_i − eps·∇f(theta_i)`, which lets me trade gradients for weight differences: `∇f(theta_i) = (theta_i − theta_{i+1})/eps` and `∇f(theta_{i+1}) = (theta_{i+1} − theta_{i+2})/eps`. The denominator becomes

  ∇f(theta_{i+1}) − ∇f(theta_i) = (theta_{i+1} − theta_{i+2})/eps − (theta_i − theta_{i+1})/eps
                                = (2·theta_{i+1} − theta_i − theta_{i+2}) / eps,

and the numerator `theta_{i+1} − theta_i` divided by that gives

  eps* = eps · (theta_{i+1} − theta_i) / (2·theta_{i+1} − theta_i − theta_{i+2}),

a per-weight optimal rate written entirely in terms of three sequential weight snapshots and the rate I'm currently using. That's cheap — I'm already storing the weights. It's per-weight, though, and I want a single global rate, so I collapse it the way the no-more-pesky-rates work does, by summing numerator and denominator across weights — except their expression squares the terms to stay positive, and I'd rather not square (it changes the scale), so I sum the *absolute values* of the numerators and the absolute values of the denominators separately and divide. (Summing the squares and taking a square root gives a similar answer.) The estimate, computed during training, lands in the range of two to six for this architecture — supporting, from curvature, that the right rate really is an order of magnitude above 0.1. And the reason the estimate is so large is that the denominator, my finite-difference Hessian proxy, is *small*: small curvature. Small curvature means the minimum SGD is sitting in is wide and flat — which is precisely the flat-minimum-generalizes-better story arriving from a completely different direction. The curvature estimate and the generalization argument are pointing at the same place: large rate, flat wide minimum. I won't lean on this estimator as the method — it's noisy and I'd want to validate it properly — but it's enough to tell me the large rates the range test reveals are not a fluke, they're where the curvature says to be.

So I've fixed the shape (one cycle: up, then down, then anneal far below the start), the upper bound (the range-test peak, validated by the curvature estimate to be an order of magnitude above normal), and the precondition (rebalance the other regularizers down). Now the details of the shape. How do I move the rate from the low value to the high value and back? The cyclical-LR work used straight linear ramps — the triangular policy above — and found linear as good as parabolic or sinusoidal windows, so linear is a fine, simple default. But I have a reason to prefer a smooth curve over the sharp linear corners: a cosine ramp eases into and out of the extremes, spending a little more time near the top and the bottom and avoiding an abrupt slope change at the peak that can jolt training right when the rate is at its most dangerous. A cosine from `start` to `end` as a fraction `pct` runs from 0 to 1 is

  anneal(start, end, pct) = end + (start − end)/2 · (cos(pi·pct) + 1),

which equals `start` at `pct=0` (cos 0 = 1, so the factor is 1) and `end` at `pct=1` (cos pi = −1, factor 0), gliding monotonically between. I'll use that same function for every leg.

How long is the up-leg versus the down-leg? The topology says the steep early region is short and the flat valley plus the final descent is most of the run, so I want a short ramp up and a long ramp down. Spending roughly the first third of the run climbing and the last two-thirds descending — `pct_start ≈ 0.3` — matches that, and gives the high-rate regularizing phase enough time to do its work while leaving the bulk of the iterations for the long anneal into the trough. And where does the up-leg *start*? Not at the upper bound, obviously; well below it, so the ramp has room. I parameterize the start as the maximum divided by a division factor — `initial_lr = max_lr / div_factor` — and a factor around 25 puts the starting rate comfortably below the cyclic-minimum-of-max/3-to-4 that the range test suggests, so the warmup genuinely begins small. Then the down-leg doesn't stop at `initial_lr`; it keeps going, annealing to a `min_lr` that is `initial_lr` divided by another large factor — orders of magnitude below the start. That final annihilation is the part the plain cycle was missing: after the high-rate phase has carried me into a wide flat region, driving the rate far down lets SGD drop into a steeper, narrow local minimum *inside* that smooth region and lock it in. Two phases, then: a cosine climb from `initial_lr` to `max_lr` over the first `pct_start` of the run, and a cosine descent from `max_lr` down to `min_lr` over the rest.

Now the part I almost left at "use momentum 0.9 and forget it," which would be a mistake. Momentum and the learning rate are not independent — they push on the same lever. Write the SGD-with-momentum update plainly,

  v_{i+1}     = m·v_i − eps·∇L(theta_i),
  theta_{i+1} = theta_i + v_{i+1},

so the velocity `v` is a running average of the gradient and the actual displacement of the weights scales with *both* `eps` and `m`. A high momentum keeps a long memory of past gradients and effectively amplifies the step — Sutskever and others already noted that a high constant momentum behaves like a pseudo-increasing learning rate, speeding things up. Put that next to what I'm doing: in the up-leg I am deliberately *increasing* the learning rate toward a very large value. If I *also* hold momentum high during that ramp, I'm stacking two amplifiers — the rising `eps` and the memory-amplifying `m` — and the effective step blows past stability; the noise-scale `g ≈ eps·N/(B·(1−m))` makes it explicit, since `(1−m)` in the denominator means high `m` inflates the noise just as a high `eps` does, and I'm already maxing the `eps` contribution. So in the high-rate phase I want momentum *low*, to keep the step controlled and to put more weight on the *current* gradient rather than the accumulated history — which is exactly what I want when the large rate is supposed to be exploring new directions toward a flat region; I don't want stale momentum dragging me along the old direction. Then in the down-leg, as the rate falls and stability is no longer at risk, I want momentum *high* again to accelerate the long descent into the trough. So momentum should cycle *inversely* to the learning rate: start high, drop as the rate climbs, rise back as the rate falls.

I consider finding the momentum bounds the way I found the rate bounds — a momentum range test, sweeping momentum upward in a pre-run. But that doesn't work: when momentum rises from 0.7 toward 1, the test loss just keeps decreasing and accuracy keeps rising, with no clean peak to read off, right up until divergence. There's no informative turn. So I can't set momentum from a sweep; I have to set it from the coupling argument and a small search. The useful band is narrow: `max_momentum = 0.95` at the *start* (rate low) falling to `base_momentum = 0.85` at the *peak* (rate at `max_lr`), then back up. I cycle it with the very same cosine `anneal` function, just with the endpoints swapped relative to the rate on each leg: as the rate goes `initial_lr → max_lr`, momentum goes `0.95 → 0.85`; as the rate goes `max_lr → min_lr`, momentum goes `0.85 → 0.95`.

One regularizer I deliberately do *not* cycle: weight decay. The rate and momentum are noise/exploration knobs that should move over the run; weight decay is a fixed penalty whose job is steady regularization, so I keep it fixed and tune only its level. What weight decay *does* have to do is be *reduced* from its conventional value, by the balance principle: the large rate is now carrying a lot of the regularization load, so leaving weight decay high over-regularizes and the large rate can't be used. I pick weight decay by a small grid search on short runs — and there's a nice tell for the right value: the best weight decay shows a *small* amount of overfitting early; values that show none are too large (underfitting against the large-rate regularization), values that overfit badly are too small. So weight decay stays constant during a run, set once, lower than usual.

Let me now make sure I'm not just reinventing a neighbor. Is this SGDR — cosine annealing with warm restarts? No: SGDR cosine-decays from high to low and then *jumps back up to the top* and decays again, a sawtooth of restarts. That's the opposite of what I want at the end — I want the rate to keep going *down*, far below the start, to anneal into the trough, not to be thrown back to the maximum for another restart. Is it warmup-then-decay? That's only my up-leg followed by ordinary decay, with no annihilation tail and constant momentum — it's a special case of the rising phase, not the whole cycle. Is it the triangular cyclical policy? That oscillates between two bounds many times and never descends below the lower bound. So what I have is genuinely the single cycle — one rise, one fall, then drive far below the start — coupled with inverse momentum cycling and a rebalanced, reduced weight decay. I'll call that the one-cycle policy, "1cycle."

Let me write it as the per-step schedule that fills the empty slot in the training harness. The optimizer is unchanged plain SGD-with-momentum; all I'm designing is the function from the current step to the learning rate, plus the momentum that rides inversely along with it. I'll precompute the three derived rates — `initial_lr = max_lr/div_factor`, the same `max_lr`, and `min_lr = initial_lr/final_div_factor` — and lay out two phases with their start/end rates and start/end momenta, then at each step find the phase the step falls in, compute the within-phase fraction `pct`, and cosine-anneal both the rate and the momentum across that phase:

```python
import math


def annealing_cos(start, end, pct):
    # cosine from `start` (pct=0) to `end` (pct=1); eases into/out of the extremes
    cos_out = math.cos(math.pi * pct) + 1.0          # 2 at pct=0, 0 at pct=1
    return end + (start - end) / 2.0 * cos_out


def annealing_linear(start, end, pct):
    return (end - start) * pct + start


class OneCycle:
    """1cycle policy: one cosine up-leg then a long cosine down-leg that anneals the
    rate far below where it started, with momentum cycled inversely to the rate.
    Drives plain SGD-with-momentum; lr/momentum are set per step, the update is unchanged."""

    def __init__(self, total_steps, max_lr,
                 pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
                 max_momentum=0.95, base_momentum=0.85):
        self.total_steps = total_steps
        initial_lr = max_lr / div_factor                 # start the warmup well below the peak
        min_lr = initial_lr / final_div_factor           # final annihilation: orders below start
        up = float(pct_start * total_steps) - 1          # end step of the up-leg (~first 30%)
        # two phases: cosine climb to the large rate, then long cosine descent past the start.
        # momentum runs the opposite way on each leg (high when rate is low, low at the peak).
        self.phases = [
            dict(end_step=up,
                 lr0=initial_lr, lr1=max_lr,             # rate: initial -> max
                 m0=max_momentum, m1=base_momentum),     # momentum: 0.95 -> 0.85
            dict(end_step=total_steps - 1,
                 lr0=max_lr, lr1=min_lr,                 # rate: max -> min (far below start)
                 m0=base_momentum, m1=max_momentum),     # momentum: 0.85 -> 0.95
        ]

    def values(self, step):
        """Return (lr, momentum) for this step by cosine-annealing within the active phase."""
        start_step = 0.0
        for i, ph in enumerate(self.phases):
            end_step = ph["end_step"]
            if step <= end_step or i == len(self.phases) - 1:
                pct = (step - start_step) / (end_step - start_step)   # 0..1 within the phase
                lr = annealing_cos(ph["lr0"], ph["lr1"], pct)
                momentum = annealing_cos(ph["m0"], ph["m1"], pct)     # inverse to the rate
                return lr, momentum
            start_step = end_step


def lr_range_test(set_lr, train_one_step, lr_start, lr_end, num_steps):
    """One short pre-run: sweep the rate up linearly and record loss/accuracy vs rate.
    The largest rate before accuracy turns ragged is the max_lr to use above."""
    history = []
    for step in range(num_steps):
        pct = step / (num_steps - 1)
        lr = annealing_linear(lr_start, lr_end, pct)
        set_lr(lr)
        loss, acc = train_one_step()
        history.append((lr, loss, acc))
    return history                                       # read max_lr off the peak of acc vs lr


# the schedule drives the unchanged SGD-with-momentum loop, set per step:
def train(model, loss_fn, data_loader, optimizer, schedule, total_steps):
    step = 0
    for inputs, targets in cycle(data_loader):
        if step >= total_steps:
            break
        lr, momentum = schedule.values(step)            # 1cycle rate + inverse momentum
        optimizer.lr = lr
        optimizer.momentum = momentum
        grads = backprop(model, loss_fn, inputs, targets)
        optimizer.step(grads)                            # plain SGD-with-momentum update
        step += 1
```

The pieces now fit together. I started stuck in step-decay's plateaus, where a small monotone-decreasing rate crawls through the flat valley of the loss surface for tens of thousands of iterations. Reading the rate off the topology — small for the steep start, large to cross the flat valley and punch through saddles fast, small again to settle the trough — says the rate should rise once and fall once, not stay small and not cycle endlessly. The cyclical-LR work confirmed that a temporary rate increase helps, and handed me the LR range test, which revealed the network tolerates rates an order of magnitude above normal; the Hessian-free curvature estimate independently put the optimal rate that high and tied it to a wide, flat minimum. Plotting training loss up and test loss down across the large-rate band showed the large rate is itself a regularizer, which forces the balance principle: reduce weight decay and the other regularizers so the total stays balanced. The single cycle then needs a final tail driving the rate far below where it started — annealing into the trough — which is what separates it from a plain cycle, from warmup-then-decay, and from SGDR's restart sawtooth. Cosine legs with a short up phase and a long down phase, starting well below the peak via a division factor and annihilating orders of magnitude below it, give the shape; and because momentum and rate push on the same step-size lever, momentum cycles inversely — high when the rate is low, dropping to its floor at the peak — to keep the large-rate phase stable while letting fresh gradients steer toward the flat region. The whole thing leaves the SGD update untouched and lives entirely in a per-step function from training progress to (rate, momentum).
