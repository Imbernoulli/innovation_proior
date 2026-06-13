I have one batch of on-policy data per iteration and the scaffold is begging me to reuse it: ten epochs
of mini-batch SGD over the same 2048 transitions before they are discarded. That reuse is the whole
point of an on-policy actor-critic — it is how I extract many gradient steps from data that cost a full
rollout to collect — but it is also exactly the thing that makes naive ascent dangerous, so the first
update rule I write has to confront that danger head-on rather than hope it does not bite. The default
fill in the scaffold is the placeholder un-clipped policy gradient $-\hat{\mathbb E}[\hat A\,r]$, and I
know without running it that across ten epochs it will walk the policy off the cliff: the importance
ratio $r_t=\pi_\theta/\pi_{old}$ fans out as $\theta$ drifts from the data-generating $\theta_{old}$,
the optimizer discovers that the cheapest way to raise the objective on a positive-advantage sample is
to inflate $r_t$ rather than to find a genuinely better action, and a handful of large ratios yank the
update around until the policy collapses. So I cannot just hand the loop the placeholder. I need the
first update rule to put a leash on how far the policy may move per batch, and I want to begin from the
*most theoretically direct* leash — the one the surrogate bound literally hands me — and see how far it
gets, because that tells me precisely what the next rung must fix.

Let me write down what I am actually optimizing so the leash has something to attach to. I have a
stochastic Gaussian policy $\pi_\theta(a|s)$ and I want to maximize expected discounted return. The
Kakade–Langford surrogate says $\eta(\pi)-\eta(\pi_{old})$ equals the new policy's expected
old-advantage, and evaluated under the old state distribution that becomes $L(\theta)=\hat{\mathbb
E}_t[r_t(\theta)\hat A_t]$ with $r_t=\pi_\theta/\pi_{old}$. This is honest only to first order at
$\theta=\theta_{old}$, where every $r_t=1$; its error grows with how far $\pi_\theta$ strays from
$\pi_{old}$, and that distance is bounded by a KL term. So improving $L$ while keeping $\pi_\theta$
KL-close to $\pi_{old}$ is guaranteed to actually improve $\eta$. The cleanest way to *express* "stay
close" as a first-order loss — no constraint solver, no second-order machinery — is to subtract a KL
penalty: maximize $\hat{\mathbb E}_t[r_t\hat A_t - \beta\,\mathrm{KL}[\pi_{old},\pi_\theta]]$. This is
the penalty form the trust-region theory suggests directly, and it is attractive precisely because it
is *just a loss*: it differentiates cleanly, it costs nothing beyond the ratio I already compute, and
it works with the shared-optimizer K-epoch loop the scaffold gives me. That is why I start here. It is
the minimal honest leash.

The catch is the coefficient. The $\beta$ the bound itself prescribes comes from a worst-case
(max-over-states) KL inequality, which makes it enormous, so the permitted steps are microscopic —
correct but useless, no better than not reusing the data at all. And if I instead pick a fixed $\beta$
by hand, it will not hold still. The right $\beta$ has to balance $r\hat A$ against $\beta\,\mathrm{KL}$,
and that balance depends on the *scale* of the advantages and on how sensitive the KL is to a parameter
step — both of which change across the three environments (HalfCheetah's dense rewards produce very
different advantage magnitudes than Swimmer's, even after the loop's per-minibatch normalization, once
the value function is inaccurate) and *over the course of a single run* as the policy sharpens and the
returns grow. A $\beta$ that gives reasonable steps at iteration 1 gives tiny steps at iteration 400,
or vice versa. So a single fixed $\beta$ is hopeless. That is the wall the penalty form runs into, and
the way past it is the move that defines this baseline: stop *guessing* $\beta$ and start *servoing* it.

Here is the servo. I cannot pre-pick the coefficient, but after I take an update I can *measure* the KL
I actually produced, and adjust $\beta$ to chase a target. Pick a target KL $d_{targ}$ — the size of
the policy move I am willing to tolerate per batch — and after the update compute the realized KL. If
it overshot, the penalty was too weak, so multiply $\beta$ up; if it undershot, the penalty was too
strong, so divide $\beta$ down. The exact thresholds are heuristic and the loop is forgiving because
$\beta$ self-corrects within a few iterations and the initial value barely matters once it is being
tuned. This is the whole reason the adaptive version works where a fixed coefficient cannot: I am no
longer fighting the advantage scale or the run-time drift by hand; I am closing a feedback loop on the
quantity I actually care about, the realized KL.

Now I have to land this in *this task's* edit surface, and the harness shapes several of my choices, so
let me be careful about what it gives me and what it does not. The scaffold hands `compute_losses` the
already-minibatch-normalized advantages and asks me to return `(loss, pg_loss, v_loss, entropy_loss,
approx_kl, clipfrac)`. It does **not** give me a separate, persistent place to store the adaptive
$\beta$ across iterations the way a class-level training loop would — `compute_losses` is a free
function called fresh on every minibatch. So I attach the adaptive state to the `agent` object itself:
on the first call I lazily initialize `agent._kl_beta` and `agent._target_kl`, and thereafter I read
and mutate them in place, so the coefficient persists across minibatches, epochs, and iterations
exactly as the servo needs. I set the initial $\beta=0.5$ and target KL $=0.01$ — 0.01 is the standard
small-move regime where the surrogate bound stays tight, and 0.5 is a neutral starting coefficient that
the servo will pull to the right scale within a handful of updates regardless.

The second harness detail is *which* KL I penalize. The exact KL between two diagonal Gaussians is
available, but the loop's vocabulary is the log-ratio: I already have `logratio = newlogprob -
mb_logprobs` for the diagnostics. The cheap, unbiased-in-expectation KL estimator that lives naturally
here is $\hat{\mathrm{KL}} = \hat{\mathbb E}[(r-1)-\log r]$, the same quantity the scaffold computes for
`approx_kl`. So I reuse it as the penalty term — but with one critical difference from the diagnostic:
the diagnostic `approx_kl` is computed under `torch.no_grad()`, whereas my *penalty* term must carry a
gradient. The penalty is the entire mechanism by which "stay close" reaches the policy parameters; if I
detached it, the KL term would contribute nothing to the gradient and I would be back to the un-clipped
placeholder with a useless constant added. So I compute `kl = ((ratio - 1) - logratio).mean()` *with*
gradient and use that in the policy loss, and I keep a detached copy purely for the adaptation rule and
the logging. This is the one place where getting the `detach` placement backwards silently turns the
method into the broken default, so it is worth stating plainly: penalty KL has a gradient, adaptation
KL does not.

So the policy loss is the conservative-policy-iteration surrogate (in minimize-the-negative form)
$-\hat{\mathbb E}[\hat A\,r] + \beta\cdot\hat{\mathrm{KL}}$, where I deliberately use the *un-clipped*
ratio $r$ — no ratio clipping anywhere — because the entire job of holding the policy near $\pi_{old}$
is delegated to the KL penalty; that is what distinguishes this rung from the clipped variants further
up the ladder. After computing the loss I run the adaptation: read the detached realized KL, and if it
exceeds $1.5\times$ the target, double $\beta$ (capped at 100 so a runaway minibatch cannot blow the
coefficient up); if it falls below the target over $1.5$, halve $\beta$ (floored at $10^{-4}$ so it can
always recover). The $1.5$ band gives the servo a dead-zone so it is not thrashing the coefficient on
every minibatch's noise; the doubling/halving gives it geometric reach so it can cross several orders of
magnitude in a few iterations if the advantage scale demands it. The value head gets a plain MSE loss
toward the GAE returns — no value clipping, since I have introduced no clipping discipline anywhere in
this rung and adding it asymmetrically would be incoherent — folded in with the loop's `vf_coef`, and
the entropy term enters through the loop's `ent_coef` (which defaults to 0 on MuJoCo, where the
Gaussian's learned log-std already supplies exploration). The full module is in the answer.

I should be honest about what this task's adaptive-KL rung is *not*, because the same idea has been
realized in heavier forms elsewhere and I want the reasoning to land exactly the harness's
implementation, not an imported one. There is no outer per-iteration adaptation loop that re-runs the
whole batch at a fixed $\beta$ and only then adjusts — the adaptation happens *inline*, per minibatch,
mutating `agent._kl_beta` as the K epochs proceed, which is finer-grained and a little noisier than a
once-per-iteration servo but is the only shape the free-function contract supports cleanly. There is no
separate KL-early-stopping break (`target_kl` in the loop defaults to `None`); the penalty is the sole
brake. And the KL is the cheap log-ratio estimator, not the closed-form Gaussian KL. These are the
harness's constraints, and the rung is the faithful realization of adaptive-KL-penalty PPO *within*
them.

Let me close on what I expect, since this is the first rung and the numbers it lands will set the floor
the next rung has to beat. The penalty servo should keep the policy from the outright collapse the
placeholder would suffer, so I expect real learning on all three environments — this is a working
on-policy method, not a broken one. But I am uneasy about two things that I think will show up as the
weakness this rung leaves on the table. First, the inline per-minibatch adaptation is reactive: it only
shrinks $\beta$ *after* a minibatch has already overshot the KL, so within a single batch the policy can
take a too-large step before the coefficient catches up, and that occasional overshoot is exactly the
kind of instability that inflates seed-to-seed variance — I would not be surprised to see one
environment swing widely across the three seeds. Second, a soft KL penalty asks the optimizer to *trade
off* return against distance, and the optimizer will sometimes pay the KL cost to chase a large
advantage on a noisy minibatch, whereas a hard band would simply forbid the move; this means on
environments where the advantage estimates are noisiest (Swimmer, with its long-horizon credit
assignment, and the unstable InvertedDoublePendulum) I expect the penalty form to be the *least*
reliable, not just the least efficient. Because the task scores by geometric mean across the three
environments, any single environment where the penalty servo gets unlucky drags the whole score down
hard. So my falsifiable expectation is concrete: this rung learns everywhere but is the *least balanced*
of the update rules I will try — it should be the one most prone to a weak environment pulling its
geometric-mean score below a method that gives up a little peak performance for a harder, non-negotiable
trust region. If that is what the numbers say, the next move writes itself: replace the soft, reactive
KL penalty with a leash that is built into the loss itself and cannot be traded away.
