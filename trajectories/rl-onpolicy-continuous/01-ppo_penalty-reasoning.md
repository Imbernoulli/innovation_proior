I have one batch of on-policy data per iteration and the scaffold is begging me to reuse it: ten epochs
of mini-batch SGD over the same 2048 transitions before they are discarded. Let me put numbers on that
reuse before I design anything, because the numbers are the whole tension. The loop collects
`num_steps=2048` and splits each batch into `num_minibatches=32`, so a minibatch is $2048/32=64$
transitions, and it runs `update_epochs=10` passes, which is $10\times32=320$ gradient steps extracted
from a single rollout. Over `total_timesteps=1{,}000{,}000` that is about $10^6/2048\approx488$
iterations, so roughly $488\times320\approx1.56\times10^5$ Adam steps across the whole run — but only
$488$ genuinely fresh rollouts. Every one of those 320 steps per batch is trained on data drawn from the
*same* frozen $\pi_{old}$, and only the first is honestly on-policy; by step 320 the parameters have
moved and the data is stale relative to the network being updated. That reuse is the whole point of an
on-policy actor-critic — it is how I extract many gradient steps from data that cost a full rollout to
collect — but it is also exactly the thing that makes naive ascent dangerous, so the first update rule I
write has to confront that danger head-on rather than hope it does not bite.

The default fill in the scaffold is the placeholder un-clipped policy gradient $-\hat{\mathbb E}[\hat A\,r]$,
and I know without running it that across those 320 steps it will walk the policy off the cliff. The
importance ratio $r_t=\pi_\theta/\pi_{old}$ fans out as $\theta$ drifts from the data-generating
$\theta_{old}$: consider a single positive-advantage transition. The gradient of $-\hat A\,r$ with
respect to $\log\pi_\theta$ is $-\hat A\,r$, so the more the optimizer has *already* raised $r$ on that
sample, the *larger* the gradient pushing it higher still — a positive feedback loop. The optimizer
discovers that the cheapest way to raise the objective on a positive-advantage sample is to inflate
$r_t$ rather than to find a genuinely better action, and a handful of large ratios, compounded over 320
steps, yank the update around until the policy collapses. So I cannot just hand the loop the placeholder.
I need the first update rule to put a leash on how far the policy may move per batch, and I want to begin
from the *most theoretically direct* leash — the one the surrogate bound literally hands me — and see how
far it gets, because that tells me precisely what the next rung must fix.

Let me write down what I am actually optimizing so the leash has something to attach to. I have a
stochastic Gaussian policy $\pi_\theta(a|s)$ and I want to maximize expected discounted return. The
Kakade–Langford surrogate says $\eta(\pi)-\eta(\pi_{old})$ equals the new policy's expected
old-advantage, and evaluated under the old state distribution that becomes $L(\theta)=\hat{\mathbb
E}_t[r_t(\theta)\hat A_t]$ with $r_t=\pi_\theta/\pi_{old}$. This is honest only to first order at
$\theta=\theta_{old}$, where every $r_t=1$; its error grows with how far $\pi_\theta$ strays from
$\pi_{old}$, and that distance is bounded by a KL term. So improving $L$ while keeping $\pi_\theta$
KL-close to $\pi_{old}$ is guaranteed to actually improve $\eta$. That leaves the design question of
*how* to express "stay KL-close" inside the one editable slot, and I have three genuinely different
options on the table right now; let me walk each far enough to either keep it or kill it with arithmetic
rather than taste.

The first is the constrained trust region done properly — the natural-gradient move: hold $L$'s ascent
subject to a *hard* KL bound $\hat{\mathbb E}[\mathrm{KL}]\le\delta$, solved by computing the
Fisher-information matrix, solving $Fx=g$ for the natural direction by conjugate gradients, and running a
backtracking line-search to enforce the constraint exactly. This is the most principled leash there is.
But look at the edit surface it has to live in. The scaffold gives me one function that returns a
*scalar loss*, and one frozen Adam optimizer that steps on `loss.backward()`; there is no hook for
conjugate-gradient iterations, no Fisher-vector-product routine, no line-search, and the loop takes a
fixed 320 first-order steps per batch with a global gradient-norm clip at $0.5$ regardless of what I
return. A natural-gradient step is not "a scalar whose gradient Adam descends"; it is a constrained
second-order solve wrapped around the gradient, and none of that apparatus is exposed. I could not
implement it here without editing the frozen loop, which the harness forbids at runtime. So the hard
constrained solve is ruled out not because it is wrong but because it is structurally inexpressible in a
single `compute_losses` that returns a scalar — the arithmetic of the interface (one scalar, one Adam,
320 first-order steps) is what kills it.

The second option is the one that theory suggests as the first-order relaxation of that constraint, and
it *does* fit the interface: turn the hard constraint into a soft penalty and subtract it, maximize
$\hat{\mathbb E}_t[r_t\hat A_t - \beta\,\mathrm{KL}[\pi_{old},\pi_\theta]]$. This is attractive precisely
because it is *just a loss*: it differentiates cleanly, it costs nothing beyond the ratio I already
compute, and it works with the shared-optimizer K-epoch loop the scaffold gives me. That is why I start
here — it is the minimal honest leash that the interface can actually carry. The third option would be to
keep a fixed $\beta$ chosen once by hand, and I need to walk *that* a few steps too, because it is the
tempting cheap version of option two and I want a concrete reason to reject it rather than a shrug.

So let me stress a fixed $\beta$ against this specific three-environment suite. The penalty balances
$r\hat A$ against $\beta\,\mathrm{KL}$, and the balance depends on two things that move: the *scale* of
the advantages and the KL *sensitivity* of a parameter step. Even after the loop's per-minibatch
normalization the advantage signal differs across environments once the value function is inaccurate —
HalfCheetah's dense reward produces a steadier, well-estimated advantage than Swimmer's long-horizon,
high-variance one — and it also drifts *within* a run as the policy sharpens and returns grow. But there
is a sharper, purely structural reason a single $\beta$ cannot mean the same thing across these three
environments, and I can compute it. For two diagonal Gaussians differing mainly in mean, the KL is
$\mathrm{KL}\approx\tfrac12\sum_{i=1}^{d}(\Delta\mu_i/\sigma_i)^2$, a sum over the action dimensions.
HalfCheetah has $d=6$ action coordinates, Swimmer has $d=2$, InvertedDoublePendulum has $d=1$. If I want
the *total* KL of a step to sit near a target $d_{targ}$, then a roughly-even move spends $d_{targ}/d$ of
the budget per coordinate, so each coordinate may move $\Delta\mu_i/\sigma_i\approx\sqrt{2d_{targ}/d}$.
Plug in $d_{targ}=0.01$: HalfCheetah gets $\sqrt{0.02/6}\approx0.058$ per dim, Swimmer
$\sqrt{0.02/2}=0.1$, InvertedDoublePendulum $\sqrt{0.02/1}\approx0.14$. So the *same* penalty pressure
lets the single-actuator pendulum move each action coordinate about $2.4\times$ further per step than the
six-actuator cheetah. A fixed $\beta$ tuned to give reasonable per-coordinate motion on one of these
environments is systematically too tight or too loose on the others, before I even get to the
advantage-scale and drift effects. That kills the fixed-$\beta$ variant on arithmetic: there is no single
number that is simultaneously right across $d\in\{1,2,6\}$ and across 488 iterations of sharpening.

And the $\beta$ the bound *itself* prescribes is no rescue — it comes from a worst-case (max-over-states)
KL inequality, which makes it enormous, so the permitted steps are microscopic, correct but useless, no
better than not reusing the data at all. So the move that defines this baseline writes itself: stop
*guessing* $\beta$ and start *servoing* it onto a target.

Here is the servo. I cannot pre-pick the coefficient, but after I take an update I can *measure* the KL I
actually produced, and adjust $\beta$ to chase a target. Pick a target KL $d_{targ}$ — the size of the
policy move I am willing to tolerate per batch — and after the update compute the realized KL. If it
overshot, the penalty was too weak, so multiply $\beta$ up; if it undershot, the penalty was too strong,
so divide $\beta$ down. Let me check the servo has enough *reach* to matter, because a servo that cannot
cross the scale gap I just computed is theater. I double or halve, which is geometric, so to travel from
the initial $\beta=0.5$ up to $32$ (a $64\times$ increase, more than enough to cover a couple of orders
of magnitude of advantage scale) takes $\log_2 64 = 6$ doublings, i.e. six minibatches; and I have 320
minibatches per batch. So even if the initial $\beta$ is off by a factor of $\sim10^2$, the servo pulls
it to the right neighborhood inside the *first* iteration and holds it there. That is the whole reason
the adaptive version works where a fixed coefficient cannot: I am no longer fighting the advantage scale
or the run-time drift or the dimensionality by hand; I am closing a feedback loop on the quantity I
actually care about, the realized KL, and the loop has the geometric reach to catch up faster than the
policy drifts.

Now I have to land this in *this task's* edit surface, and the harness shapes several of my choices, so
let me be careful about what it gives me and what it does not. The scaffold hands `compute_losses` the
already-minibatch-normalized advantages and asks me to return `(loss, pg_loss, v_loss, entropy_loss,
approx_kl, clipfrac)`. It does **not** give me a separate, persistent place to store the adaptive $\beta$
across iterations the way a class-level training loop would — `compute_losses` is a free function called
fresh on every one of the 320 minibatches. So I attach the adaptive state to the `agent` object itself:
on the first call I lazily initialize `agent._kl_beta` and `agent._target_kl`, and thereafter I read and
mutate them in place, so the coefficient persists across minibatches, epochs, and iterations exactly as
the servo needs. I set the initial $\beta=0.5$ and target KL $=0.01$ — 0.01 is the standard small-move
regime where the surrogate bound stays tight (and, from the per-dimension arithmetic above, corresponds
to per-coordinate moves of a few hundredths of a std, which is a sane step), and 0.5 is a neutral
starting coefficient that the servo will pull to the right scale within a handful of updates regardless.

The second harness detail is *which* KL I penalize. The exact diagonal-Gaussian KL is available in closed
form, but the loop's vocabulary is the log-ratio I already have, so I use the estimator $\hat{\mathrm{KL}}
= \hat{\mathbb E}[(r-1)-\log r]$ — the same expression the scaffold computes for `approx_kl`. With
$x=\log r$ its integrand $(e^x-1)-x = \tfrac{x^2}{2}+\dots$ has the two properties I need: it is
non-negative for every sample (equality only at $x=0$), so unlike raw $-\log r$ it never becomes a
*reward* for moving; and its expectation $\tfrac12\mathbb E[(\Delta\log\pi)^2]$ is the second-order
(Fisher) expansion of the KL, a faithful trust-region proxy paid for with the log-ratio I already have.

But there is one critical difference from the diagnostic use of this same expression: the diagnostic
`approx_kl` is computed under `torch.no_grad()`, whereas my *penalty* term must carry a gradient. The
penalty is the entire mechanism by which "stay close" reaches the policy parameters; if I detached it,
the KL term would contribute nothing to the gradient and I would be back to the un-clipped placeholder
with a useless constant added — the blow-up I opened with, wearing a costume. So I compute `kl = ((ratio
- 1) - logratio).mean()` *with* gradient and use that in the policy loss, and I keep a detached copy
purely for the adaptation rule and the logging. This is the one place where getting the `detach`
placement backwards silently turns the method into the broken default, so it is worth stating plainly:
penalty KL has a gradient, adaptation KL does not.

So the policy loss is the conservative-policy-iteration surrogate (in minimize-the-negative form)
$-\hat{\mathbb E}[\hat A\,r] + \beta\cdot\hat{\mathrm{KL}}$, where I deliberately use the raw, unbounded
ratio $r$ as it comes — because the entire job of holding the policy near $\pi_{old}$ is delegated to the
KL penalty; a soft penalty on the distance is the whole of the leash here. After computing the loss I run
the adaptation: read the detached realized KL, and if it exceeds
$1.5\times$ the target, double $\beta$ (capped at 100 so a runaway minibatch cannot blow the coefficient
up); if it falls below the target over $1.5$, halve $\beta$ (floored at $10^{-4}$ so it can always
recover). The $1.5$ band gives the servo a dead-zone so it is not thrashing the coefficient on every
minibatch's noise; the doubling/halving gives it the geometric reach I checked above. The value head gets
a plain MSE loss toward the GAE returns — the critic simply regresses onto the returns, with no extra
machinery layered on top of it in this rung — folded in with the loop's `vf_coef`, and the entropy term enters through the loop's `ent_coef` (which defaults to 0 on
MuJoCo, where the Gaussian's learned log-std already supplies exploration). The full module is in the
answer.

One property of this servo matters for what follows. Modelling the realized KL as roughly inverse in
$\beta$, the $1.5\times$ dead-zone (a factor of $2.25$ wide) is wide enough that a single doubling or
halving from either edge lands back inside it rather than overshooting to the opposite edge — so the servo
does not limit-cycle; it parks $\beta$ and only nudges it as the plant drifts. But the adaptation is
strictly after-the-fact: a minibatch whose advantage estimate is an outlier can take a KL of several
$d_{targ}$ *before* the doubling reacts, and with 320 minibatches per batch such surprises are not rare.
The control is tight on average and loose in the tail — which is the mechanism behind the seed-variance
worry I close on.

The same idea has heavier realizations elsewhere, but the free-function contract fixes the shape here:
adaptation happens *inline* per minibatch, mutating `agent._kl_beta` as the 320 steps proceed rather than
once per iteration after re-running the batch — finer-grained and a little noisier, but the only clean
fit. There is no separate KL-early-stopping break (`target_kl` defaults to `None`); the penalty is the
sole brake, and the KL is the cheap log-ratio estimator above, not the closed-form Gaussian KL.

One more interaction is worth naming because it is baked into the frozen loop and my servo has to live
with it: the learning rate anneals linearly to zero over the 488 iterations. That means a fixed $\beta$
would produce ever-smaller policy moves late in training simply because the step size is shrinking, so
the realized KL per update decays toward the end even if nothing else changes. My servo reads this as
"the penalty is too strong" and halves $\beta$ to keep the KL near $d_{targ}$ — which is the *correct*
response, since the goal is a constant-size trust region measured in policy space, not a constant
coefficient. So the adaptive coefficient is not only absorbing the cross-environment and cross-run scale
differences I computed earlier; it is also silently compensating for the LR schedule, pulling $\beta$
down as the anneal bites. A hand-tuned $\beta$ would have to fight the schedule too, which is one more
reason the fixed variant was hopeless and the servo is the right shape for this substrate.

Let me close on what I expect, since this floor sets what the next update rule has to beat. The penalty
servo should keep the policy from the placeholder's outright collapse, so I expect real learning on all
three environments. But two properties I distrust are named and separable, and both trace to arithmetic
already done. The penalty is *soft*: it asks the optimizer to trade off return against distance, so on a
noisy minibatch it will sometimes pay the KL cost to chase a large advantage — it *prices* a bad move
rather than *forbidding* it. And the servo is *reactive*: it shrinks $\beta$ only *after* a minibatch has
overshot, and with 320 minibatches per batch each carrying its own noisy advantage there is ample room to
overstep before the coefficient catches up. Together they make this the *least balanced* of the update
rules I will try: it should learn everywhere but leave seed-to-seed instability on whichever environment
carries the noisiest advantages, and since the task scores by geometric mean, one unlucky environment —
or one unlucky seed within it — drags the whole score down through the aggregator that punishes the worst
factor hardest. A stronger rule on this substrate would have to hold the policy near $\pi_{old}$ without a
soft, after-the-fact leash.
