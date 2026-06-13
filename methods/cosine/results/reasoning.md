Let me start from what actually slows me down when I train these wide residual nets. The model is fixed, the augmentation is fixed, the weight decay is fixed, and the optimizer is plain SGD with momentum — that part isn't up for debate, because the strongest CIFAR results right now come from exactly that, not from Adam or AdaDelta. So the one lever I have left, the thing that decides whether a run is good or wasted, is the schedule of the scalar learning rate eta_t. And the way everyone sets it is a staircase: hold eta at 0.1, then multiply it by 0.2 at epochs 60, 120, and 160, and stop at 200. ResNets do the same kind of thing with a divide-by-10 at a couple of plateaus. It works. But every time I set up a run I have to decide three drop epochs and a drop factor, and those numbers are quietly tied to a total budget of 200 epochs that I also have to commit to up front — the 60/120/160 are really 30%/60%/80% of T. Change T and I have to re-pick all of them. That's a lot of coupled hand-tuning for something that, when I squint at it, is doing one simple thing.

What is the staircase actually doing, mechanically? A big learning rate takes big steps: it moves across the landscape fast and barrels through the wide, low-gradient flats without getting stuck. A small learning rate takes tiny steps: it settles into a basin and polishes the fit. So the schedule is a coarse-to-fine transition — explore broadly with a high rate, then refine with a low one. The staircase implements that transition as two or three sudden cliffs. Fine. But why cliffs? Between drops, eta sits at a value that was chosen for the *previous* phase, so it's systematically mismatched to wherever the iterate actually is now — too high for the basin I've entered, until the next scheduled cliff finally lowers it. And the discontinuity itself is ugly: the loss visibly lurches at each drop. The deeper annoyance is anytime performance. If I stop this run at epoch 100, I'm sitting mid-plateau at a learning rate that's too high to have a clean solution; the only well-tuned point on the whole trajectory is the final epoch, after the last drop. I want a schedule where I can stop at many points and have something usable, and where I don't have to pre-commit drop epochs against a fixed budget.

So I want to do two things at once: smooth out that coarse-to-fine transition so it isn't a staircase, and get good solutions at more than just the end. Let me think about each.

There's a body of work on *restarts* that's been rattling around in my head from the optimization side, and I think it's the key. In gradient-free optimization — CMA-ES on multimodal functions — the standard move is to restart the search repeatedly, and crucially to grow the effort across restarts: start with a small population lambda, double it after each restart. The small early restarts give you a decent solution fast; the later, bigger ones do the heavy global search. The doubling is the trick that buys good anytime behavior — you're not forced to commit your whole budget to one search. I've used that pattern myself. It's the "start small, increase the budget per restart" idea, and it's exactly the anytime property I'm missing here.

And on the gradient-based side there's a sharper, more mechanistic story. O'Donoghue and Candès looked at accelerated / heavy-ball momentum methods on smooth strongly-convex functions and noticed the iterates *ripple* — the objective shows regular periodic bumps — whenever the momentum is above a critical value. They make this precise on a quadratic f(x) = (1/2) x^T A x by writing the scheme as a linear dynamical system. Diagonalize A; each eigen-mode w_i evolves by a second-order recurrence whose characteristic polynomial is r^2 - (1+beta)(1 - lambda_i/L) r + beta(1 - lambda_i/L). When beta sits above the critical beta_i* = (1 - sqrt(lambda_i/L))/(1 + sqrt(lambda_i/L)), the roots go complex and the mode oscillates — under-damped — with a solution like (beta(1 - lambda_i/L))^{k/2} cos(k psi_i - delta_i), where psi_i = arccos((1 - lambda_i/L)(1+beta)/(2 sqrt(beta(1-lambda_i/L)))). For the slowest mode, the one tied to the smallest eigenvalue mu, with lambda_i << L this collapses, using arccos(sqrt(1-theta)) ≈ sqrt(theta), to psi_mu ≈ sqrt(mu/L). So the oscillation period is proportional to 1/psi_mu ≈ sqrt(L/mu) — the square root of the condition number. The reason you can't just pick the perfect momentum is that the right value depends on this condition number, which you don't know and which varies *locally* as you move across the landscape, so in practice you live in the rippling regime. Their answer is to restart: reset the momentum to zero, take the current iterate as a fresh start, and do it periodically. If L and mu were known, a fixed restart every k* = e·sqrt(8L/mu) iterations would recover the optimal linear rate — you derive that by minimizing the per-cycle contraction (8L/(mu k^2))^j over the split jk = const, which gives k* ∝ sqrt(L/mu). Since mu is exactly the parameter I do not know, the useful practical move is to restart adaptively the moment a cheap signal says momentum is hurting: when f goes up, f(x^k) > f(x^{k-1}), or when the momentum and the negative gradient make an obtuse angle, grad f(y^{k-1})^T (x^k - x^{k-1}) > 0.

That last picture is the one I want to import. A restart is a *reset of the acceleration phase*: throw away the accumulated velocity that's now carrying you in a stale direction, and start descending fresh from where you are. Restarting is useful precisely when the curvature is unknown and drifting — which is exactly my situation, a non-convex deep net where the local geometry changes completely as the weights move. So my instinct is: don't just decay eta once over 200 epochs; periodically restart the descent.

But let me actually try to port the restart literally and watch it break, because the details matter. Option one: literally do what O'Donoghue does — zero the momentum and reset the iterate as the new origin. The iterate reset is meaningless here (there's no canonical origin in weight space; "the current iterate as a new start" just means keep x), but zeroing the momentum vector v at each restart — I could do that. Except momentum in these nets is doing real work; the velocity I've built up along consistent descent directions is hard-won, and slamming it to zero throws that away and stalls me right after every restart. I don't want a *hard* reset that discards state. Wall.

Option two: use O'Donoghue's adaptive restart *test* — restart when the loss goes up, or when the momentum-gradient angle turns obtuse. The trouble is the setting. Their analysis is on a smooth deterministic quadratic, where f(x^k) and grad f(y^k) are exact. Mine are stochastic: f_t and grad f_t are computed on a single minibatch of 128 and swing wildly from batch to batch. The loss "goes up" between consecutive batches constantly, just from sampling noise, and the gradient-angle test flips sign on noise too. To use either test honestly I'd have to denoise — average the loss and the gradient over, say, a whole epoch — before I could trust a restart trigger. That's doable but it adds machinery and a smoothing window I'd have to tune, and it makes the whole thing reactive and fiddly. Wall again, softer.

So both literal ports fail for the same underlying reason: a *hard, triggered* restart is wrong for a stochastic deep-net run. I don't want to discard state, and I can't reliably detect the right moment from noisy signals. Let me back up and ask what a restart *is*, functionally, stripped of the mechanism. It's: stop refining in the current basin, and start exploring broadly again. And I already know how to make the optimizer explore broadly versus refine — that's the learning rate. A high eta is the exploratory, large-step regime; a low eta is the refining, small-step regime. So I can *emulate* a restart without touching the momentum or the iterate at all: just push eta back up. Don't reset v; raise the learning rate, and the large steps that follow will override the stale velocity on their own — the bigger the jump in eta, the more the new gradients dominate the accumulated v, which is to say the LR jump itself is a soft knob on "how much of the old acceleration survives the restart." This is a *warm* restart: it keeps the partial progress, it needs no momentum surgery, and — the part that dissolves the second wall — it's *scheduled*, not triggered, so I never have to detect a noisy condition. I decide in advance when to raise eta. That sidesteps the stochastic-test problem entirely.

Now the real design question becomes: between two restarts, what shape should eta follow as it comes down from its high value to its low value? This is also exactly the smoothness fix I wanted for the staircase — within one cycle I'm replacing the cliffs with a continuous descent. So whatever I pick here serves double duty. Candidates: linear (a triangular ramp), exponential, polynomial, cosine. Let me not just pick one; let me figure out what properties I actually need from the within-cycle curve and let the curve fall out.

Parametrize progress through a cycle by s = T_cur / T_i in [0,1], where T_cur is how far into the current run I am and T_i is the run's length. I want a function g(s) that I'll map onto [eta_min, eta_max] by eta = eta_min + (eta_max - eta_min)·g(s). What do I need from g?

First, the endpoints: g(0) = 1 (a cycle starts at the high exploratory rate eta_max — that's the restart kick) and g(1) = 0 (a cycle ends at the low rate eta_min, where I've refined down to a clean solution). g should be monotone decreasing on [0,1].

Now the subtler conditions, which is where I think the right shape actually lives. Think about the *start* of the cycle. I just kicked eta up to eta_max to explore; I want it to *stay* high for a while, not immediately plunge — exploration needs sustained big steps, not an instant retreat. So I want the schedule to be flat at the top: g'(0) = 0, slow decay right after the restart. Now think about the *end* of the cycle. I'm approaching eta_min, where I want a gentle, fine landing into the basin — and this is also the point I'm going to restart *from*, so I want eta to ease smoothly down to its floor rather than crash into it with a steep slope and then suddenly leap back to eta_max. A discontinuity in the *derivative* at the bottom would be a kink, a mini-staircase artifact, the very thing I'm trying to remove. So I want g'(1) = 0 too: flat at the bottom.

So my requirements are g(0)=1, g(1)=0, g'(0)=0, g'(1)=0, monotone decreasing. Let me hold each candidate up against those.

Linear, the triangular ramp: g(s) = 1 - s. Endpoints are right, but g'(s) = -1 everywhere — constant slope, so g'(0) and g'(1) are both -1, not 0. It has corners at the top and the bottom: it leaves eta_max at full speed (no sustained exploration phase) and hits eta_min at full speed and then jumps back up — exactly the kink at the restart I wanted to avoid. The triangular cyclical-rate idea has this shape, and it's why it reads as "oscillate within a band" rather than "anneal to a fine finish and restart": its corners never let it settle. Reject linear on the slope conditions.

Exponential, g(s) = c^s for some c<1 (or eta decaying geometrically per step): g'(s) = c^s ln c, steepest in magnitude at s=0 and only relatively flatter at s=1 — and it never actually reaches the floor, g(1) = c ≠ 0. So it's steep at the top, which is backwards: it abandons the high exploratory rate almost immediately and spends most of the cycle already low. That's the opposite of "stay high to explore, then come down." Reject.

Polynomial, g(s) = (1-s)^p with p>1: g'(s) = -p(1-s)^{p-1}, so g'(1) = 0 (flat at the bottom, good) but g'(0) = -p ≠ 0 (steep at the top). Only one of the two flats. Reject — same failure as exponential, milder.

Now the half-cosine, g(s) = (1/2)(1 + cos(pi·s)). Check it. g(0) = (1/2)(1 + cos 0) = (1/2)(1+1) = 1. g(1) = (1/2)(1 + cos pi) = (1/2)(1 - 1) = 0. Both endpoints exact. The derivative: g'(s) = (1/2)(-pi sin(pi s)) = -(pi/2) sin(pi s). So g'(0) = -(pi/2) sin 0 = 0 and g'(1) = -(pi/2) sin pi = 0 — flat at *both* ends. And in between sin(pi s) is positive on (0,1), so g'(s) < 0 throughout: strictly monotone decreasing. The steepest descent is at the midpoint s = 1/2, where g'(1/2) = -(pi/2)·1 = -pi/2 — the curve eases out of eta_max, accelerates its drop through the middle, then eases into eta_min. That is *precisely* the profile I argued for: a sustained high-rate exploratory phase, a brisk transition, and a gentle low-rate convergent landing that also makes a clean restart point. The half-cosine is the function my requirements were describing.

I should be honest that the boundary conditions g(0)=1, g(1)=0, g'(0)=g'(1)=0 don't *uniquely* force the cosine — the cubic smoothstep 1 - (3s^2 - 2s^3) also satisfies all four (it's the lowest-degree polynomial that does, and numerically it tracks the cosine closely: at s=1/4 it gives 0.844 versus the cosine's 0.854). So the four conditions pin down the *family* of smooth, flat-ended, monotone S-curves, not one member. Why the cosine specifically? It's the natural trigonometric member of that family and the cleanest to write — one term, cos(pi s), no shape parameter to choose, no degree to pick — and a cosine is the canonical way to interpolate between two levels with zero slope at both ends. Picking the polynomial would just be choosing an equivalent S-curve with an extra arbitrary choice attached. The cosine carries no free knob, so I take one descending half-period as the cycle shape; the restart remains the intentional jump that takes the next cycle back to the high rate.

So within the i-th run the learning rate is

  eta_t = eta_min^i + (1/2)(eta_max^i - eta_min^i)(1 + cos(pi · T_cur / T_i)).

Let me sanity-check the endpoints in these original units, not just in g. At a restart, T_cur = 0, cos(0) = 1, so eta_t = eta_min + (1/2)(eta_max - eta_min)(2) = eta_min + (eta_max - eta_min) = eta_max. Good — every cycle starts at the full exploratory rate, the kick. At the end of a run, T_cur = T_i, cos(pi) = -1, so eta_t = eta_min + (1/2)(eta_max - eta_min)(0) = eta_min. Good — the cosine bottoms out at eta_min. And one implementation nicety I want to bake in: I'll update T_cur every *batch*, not every epoch, so T_cur takes fractional values like 0.1, 0.2, and the curve is genuinely smooth within an epoch rather than a per-epoch step. (On a log-axis plot the cosine's characteristic shape gets obfuscated, but on a linear axis it's the smooth S I derived.)

Now the cross-cycle structure — the restart schedule itself. Simplest: a fixed period, restart every T_0 epochs, T_i = T_0 for all i. That already gives me the smooth, anytime-friendly schedule. But I have the CMA-ES anytime trick sitting right there: start with a small period and grow it. Set T_i = T_0 · T_mult^{i} with T_mult ≥ 1 — for instance T_0 = 1, T_mult = 2 restarts after 1 epoch, then 2, then 4, then 8…, or T_0 = 10, T_mult = 2 restarts after 10, 20, 40…. Why doubling rather than fixed? Anytime performance, exactly as in the gradient-free case: the first restart comes quickly so I get a first decent annealed solution early, and each subsequent cycle is longer, giving progressively more refinement. With T_mult = 1 I recover the fixed-period schedule as the special case. I'll keep eta_max^i and eta_min^i the *same* for every i rather than decaying them per restart — I could decay them to tamp down each restart's divergence, but holding them fixed keeps the hyperparameter count minimal, which was a stated goal; the only knobs are eta_max, eta_min, and the period schedule (T_0, T_mult). For the floor, eta_min I'll set to 0 (or something tiny): I want each cycle to anneal all the way down for the sharpest possible final fit, and a floor of 0 also gives the maximum-contrast kick when the next cycle jumps back to eta_max.

One consequence I have to handle: a warm restart *raises* eta, and right after each restart the big steps temporarily *worsen* the loss before the cosine brings it back down and improves it. So if I naively report the last iterate x_t as my answer, I might catch it mid-kick at a bad point. The fix follows from the shape: the good points are the *bottoms* of the cosines, at T_cur = T_i where eta = eta_min and I've just finished refining. So my recommended (incumbent) solution after the first restart is the iterate at the end of the most recently completed run, taken at eta = eta_min — the trough of the cosine — not the literal last x_t. During the very first run, before any restart, the last x_t is fine. A pleasant by-product: because I take the recommendation from the schedule's own trough rather than by peeking at held-out performance, I don't need a separate validation set to pick the checkpoint.

Let me also notice something the periodic structure hands me for free, just by reasoning about the trajectory. Each restart kicks the iterate out of whatever basin it had settled into and sends it exploring again, so the sequence of trough iterates — the models I collect at the end of each cycle — should not be the same model nudged slightly; they should be descents into different low-loss regions of a non-convex landscape. A single run can therefore give me diverse candidate models at no extra training cost beyond the run I was doing anyway. That diversity is the kind of thing I'd want to combine, and it's a property of the restart trajectory itself, not something I have to engineer.

Now, the single most important special case for my actual problem. My CIFAR setup has a fixed 200-epoch budget and I don't necessarily want multiple restarts — I just want to fix the staircase. Set T_mult = 1 and T_0 = T = total_epochs, i.e. one cycle spanning the whole run, and eta_min = 0, eta_max = base_lr. Then T_cur is just the current epoch, T_i is total_epochs, and

  eta_t = 0 + (1/2)(base_lr - 0)(1 + cos(pi · epoch / total_epochs))
        = base_lr · (1/2)(1 + cos(pi · epoch / total_epochs)).

That's it — a single smooth cosine decay from base_lr at epoch 0 down to 0 at the final epoch, no restarts at all. It needs exactly two hyperparameters, base_lr and total_epochs, both of which I already have; there are no drop milestones and no drop factor to tune. And this is *not* a toy degenerate case — a single full-length cosine anneal already replaces the entire staircase, smoothly tracking the coarse-to-fine transition the cliffs were crudely approximating, with eta easing down so it's never grossly mismatched to the region the iterate is in. This is the schedule I'd reach for first.

Let me write the within-cycle rate as the function the training loop calls, the no-restart single-cosine version first since that's the core, filling the one empty slot in the harness:

```python
import math


def get_lr(epoch, total_epochs, base_lr, config):
    """Cosine annealing of the learning rate over the whole run.

    A single half-cosine from base_lr (at epoch 0) down to 0 (at the final epoch):
    starts flat and high (broad, exploratory steps), accelerates its descent through
    the middle, and eases gently to ~0 at the end (fine convergence). Replaces a
    hand-tuned multi-step staircase; the only knobs are base_lr and total_epochs.

        eta = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))

    epoch=0           -> cos(0)  =  1 -> eta = base_lr   (the high exploratory start)
    epoch=total_epochs-> cos(pi) = -1 -> eta = 0         (the gentle, fine-tuning floor)
    """
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
```

And the general schedule, cosine annealing *with warm restarts*, where I keep the cross-cycle bookkeeping — T_cur counting within the current run, T_i the current run length, doubled (or not) at each restart — and anneal from eta_max to eta_min along the same half-cosine inside each run. The rate calculation itself is still just the endpoint-checked formula: T_cur=0 gives eta_max, T_cur=T_i gives eta_min. The scheduler bookkeeping decides when a completed run becomes the next run, and I want that bookkeeping to match the way a real training loop calls the scheduler, including fractional epoch values when I step once per batch:

```python
import math


class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts.

    Each parameter group's initial lr is its eta_max. Within run i, anneal that
    base lr to eta_min along a half-cosine; after the run length T_i is reached,
    wrap T_cur into the next run and grow T_i by T_mult.

        eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * T_cur / T_i))
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("T_0 must be a positive integer")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("T_mult must be an integer >= 1")
        if not isinstance(eta_min, (float, int)):
            raise ValueError("eta_min must be a number")

        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.eta_min = eta_min
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.T_cur = last_epoch
        self.last_epoch = last_epoch
        self.step(0 if last_epoch < 0 else last_epoch)

    def get_lr(self):
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min)
            * (1.0 + math.cos(math.pi * self.T_cur / self.T_i))
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        # epoch may be fractional, e.g. epoch + batch_index / batches_per_epoch.
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur -= self.T_i
                self.T_i *= self.T_mult
        else:
            if epoch < 0:
                raise ValueError("epoch must be non-negative")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log(
                        epoch / self.T_0 * (self.T_mult - 1) + 1,
                        self.T_mult,
                    ))
                    self.T_cur = (
                        epoch
                        - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    )
                    self.T_i = self.T_0 * self.T_mult**n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        return lrs
```

Let me trace the causal chain back to the start. I was stuck hand-tuning a discontinuous staircase — drop epochs and a drop factor, all pinned to a fixed budget — that gave a usable solution only at the very end. The staircase was a crude two-phase explore-then-refine, so I asked for the same transition done smoothly, and for good solutions at more points than just the last. The restart literature pointed the way: in gradient-free search you restart and grow the budget each time for anytime performance, and in gradient-based optimization a momentum method ripples because the right momentum depends on an unknown, locally-varying condition number, so restarting the acceleration phase exposes a square-root-condition-number timescale and can recover fast convergence. I tried to port a restart literally — zeroing momentum throws away hard-won velocity, and the adaptive restart *test* needs deterministic f and grad I don't have under minibatch noise — so instead I emulated a restart by *raising* the learning rate: a warm restart that keeps state and is scheduled rather than triggered, dodging both walls. Then I derived the within-cycle shape from what a cycle has to do: start flat and high to keep exploring, end flat and low for a clean, fine landing that doubles as a smooth restart point — g(0)=1, g(1)=0, g'(0)=g'(1)=0, monotone — which the half-cosine (1/2)(1+cos(pi s)) satisfies exactly while linear, exponential, and polynomial each violate a slope condition; among the S-curves meeting all four conditions the cosine is the parameter-free half-period with the right flat endpoints. Mapping it to [eta_min, eta_max] gives eta_t = eta_min + (1/2)(eta_max - eta_min)(1 + cos(pi T_cur/T_i)), which hits eta_max at each restart and eta_min at each trough; growing T_i by T_mult per restart imports the anytime doubling, and reporting the trough iterate (not the kicked last iterate) gives a clean recommendation with no validation set. The whole thing collapses, when I take one cycle over the full budget with eta_min=0, to a single smooth cosine decay base_lr·(1/2)(1+cos(pi epoch/total_epochs)) from base_lr to 0 — two hyperparameters, no milestones — which is the schedule I'd drop into the training loop in place of the staircase.
