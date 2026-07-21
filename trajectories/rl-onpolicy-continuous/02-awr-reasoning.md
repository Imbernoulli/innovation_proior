The adaptive-KL rung learned everywhere, exactly as I expected, but it left precisely the weakness I was
worried about, and the numbers say where. Let me read the feedback table quantitatively rather than
impressionistically, because the *shape* of the spread is the diagnosis. On HalfCheetah it landed a mean
of 1676.6 but with a huge spread — 1194.9 / 2695.0 / 1139.9 across seeds {42, 123, 456}. That best-to-
worst ratio is $2695/1140\approx2.36$, and note it is *not* a smooth spread: two seeds sit within $5\%$
of each other near 1150 while the third more than doubles them. That is a bimodal signature, not
Gaussian noise around a mean — one run got a run of lucky KL overshoots that let it escape to 2695, the
other two paid for their overshoots and stalled. Swimmer came in at 101.4 (81.6 / 113.0 / 109.5) — again
the same shape: two seeds cluster near 110 and seed-42 sits $\sim26\%$ below at 81.6, one-seed-drags-it-
down. InvertedDoublePendulum was the steadiest at 6877.1 (6764.7 / 7497.4 / 6369.2), a best-to-worst
ratio of only $7497/6369\approx1.18$. So the instability is *environment-dependent*: it is worst on the
dense high-dimensional cheetah, milder on the low-dimensional swimmer, and nearly absent on the single-
actuator pendulum. That inverse relationship with how well-behaved the advantage signal is confirms the
mechanism I predicted: the soft KL penalty does not *forbid* a bad move, it *prices* one, and on a noisy
minibatch the optimizer will sometimes pay the price and take a step it should not — so the failure mode
is seed-to-seed *variance* driven by tail overshoots, not a flat ceiling. Because the task scores by
geometric mean across the three environments, that HalfCheetah spread and the weak-seed Swimmer number
are exactly what will hold this rung down: the gmean is dominated by the *worst* environment-seed, and a
method that sacrifices some peak to never produce an 1140 or an 81.6 will beat it. The question for the
next rung is whether I can get a policy update that does not depend on the importance ratio at all —
because the ratio is the thing that fans out and the KL penalty is only a soft attempt to rein it back
in.

The diagnosis names one enemy, the importance ratio $r_t$, and there are two structurally different ways
to fight it, so let me be deliberate about which I try now. I could *keep* the ratio but hold it more
firmly than a soft, servoed penalty manages — the trust-region instinct already in the background. Or I
could *remove the ratio entirely* and frame policy improvement a different way. Both are live, but the
cleaner experiment right now is the second, and here is why I want to run it first: the penalty rung's
failure was ambiguous between "the ratio itself is unstable" and "the *soft penalty* on the ratio is
unstable." If I abandon the ratio completely and the instability changes character, I learn which half of
the diagnosis was right, and that tells me exactly what a ratio-keeping fix would need to look like. So
the decisive move now is to drop the ratio surrogate and see what a ratio-free update does.

Step back to what I actually want from a policy update: make the actions that turned out better (high
advantage) more likely, and the ones that turned out worse less likely, without moving so far that the
data I am training on stops describing the policy. The penalty form does this through $r_t\hat A_t$ minus
a KL term. But there is an older idea that frames policy improvement as plain *supervised regression*:
regress the policy directly onto the actions the agent took, with each action weighted by how good it
was. No ratio, no importance weighting, no KL — just a weighted maximum-likelihood fit. The weight is an
exponentiated advantage, $w_t = \exp(\hat A_t/\beta)$, and the policy loss is the negative weighted log-
likelihood $-\hat{\mathbb E}_t[\,w_t\,\log\pi_\theta(a_t|s_t)\,]$. This is the advantage-weighted-
regression lineage, and it is appealing here precisely because the failure I just diagnosed was *about
the ratio*: if there is no ratio in the loss, there is no ratio to fan out, and the reactive KL servo I
was unhappy with disappears entirely.

Let me convince myself the exponential weight is the right shape and not just a convenient knob, because
this is the crux of why this is a principled improvement step and not a lateral move. Frame the update as
a constrained problem: find the new policy that maximizes expected advantage subject to staying KL-close
to the data-generating policy — the *same* trust-region intuition as before, but solved in closed form
instead of penalized. Set up the Lagrangian $\max_\pi \mathbb E_\pi[A] - \beta(\mathrm{KL}[\pi\|\pi_{old}]
-\epsilon)$ and take the functional derivative with respect to $\pi(a|s)$ under the normalization
constraint; setting it to zero gives $\log\pi(a|s) = \log\pi_{old}(a|s) + A(s,a)/\beta - \text{const}$,
i.e. the exponentially-tilted policy $\pi^*(a|s)\propto\pi_{old}(a|s)\exp(A(s,a)/\beta)$, where $\beta$ is
the Lagrange multiplier for the KL constraint. I cannot represent that tilted distribution directly, but
I can *project* it back onto my Gaussian policy class by minimizing the KL from $\pi^*$ to $\pi_\theta$ —
$\min_\theta \mathrm{KL}[\pi^*\|\pi_\theta] = \min_\theta -\mathbb E_{s,a\sim\pi_{old}}[\exp(A/\beta)
\log\pi_\theta(a|s)] + \text{const}$, which is exactly weighted maximum likelihood. So the exponential
advantage weight is not a heuristic; it is the closed-form trust-region solution expressed as a
regression target. This is the trust region I wanted at rung 1, but achieved by *construction* — the
weights bake the "stay close" in — rather than by a soft penalty the optimizer can trade away. That is
the conceptual reason to expect it to be steadier on the dense-reward environment where the penalty form
swung between 1150 and 2695.

Let me sanity-check the temperature knob against its two limits before I pick a value, because a knob I
cannot reason about at the extremes is one I will mis-set in the middle. As $\beta\to\infty$, every weight
$\exp(A/\beta)\to\exp(0)=1$, so the loss becomes plain unweighted maximum likelihood on the taken actions
— pure imitation of $\pi_{old}$, zero policy improvement. As $\beta\to0$, the weight of the single highest-
advantage action dominates all others by an unbounded factor, so the regression collapses onto that one
action — greedy, winner-take-all, maximal improvement but maximal variance. So $\beta$ is literally the
knob between imitate-the-old-policy and copy-the-single-best-action, and I want to sit far enough toward
the greedy end to get real selectivity from unit-scale advantages but not so far that a single noisy
sample owns the minibatch. The $\beta=0.05$ I computed above, with the clip at $20$, lands as "select the
better-than-average half, weight them equally" — decisively past uniform, short of winner-take-all. That
is the intended operating point, and the limit check tells me exactly which failure I am courting by
choosing it: the greedy-end variance, which is the Swimmer worry.

Now I have to be very careful, because the same-named idea has a canonical realization that is *not* what
this task's harness implements, and if I import that story I will write the wrong method. The original
advantage-weighted-regression algorithm is *off-policy*: it keeps a large replay buffer, recomputes
TD($\lambda$) returns over stored paths to get advantages, fits a separate critic and a separate actor
with their own momentum optimizers and step counts, normalizes the advantage before exponentiating, uses
a temperature near $1.0$, clips the weights at $20$, and adds an action-bound penalty. The whole point
there is *reusing old data* across many iterations, which is why the regression framing matters — a ratio
surrogate degrades badly on stale data, but a weighted regression onto whatever actions are in the buffer
does not. None of that off-policy machinery exists in this scaffold. The loop here is strictly on-policy:
it collects 2048 fresh transitions, computes *GAE* advantages (not buffer TD($\lambda$)) in the frozen
reverse scan, hands me a *single* shared 2×64 actor-critic, and asks me to fill one `compute_losses` that
Adam steps on for ten epochs over that one fresh batch. There is no replay buffer to point the regression
at, no separate solvers, no second network. So the faithful realization here is *on-policy AWR*: the
advantage-weighted regression *objective* dropped into the PPO loop's plumbing, using the loop's GAE
advantages and its single optimizer. I am keeping the idea — supervised weighted regression instead of a
ratio surrogate — and discarding the off-policy apparatus the idea was born with, because the harness
does not expose it.

That harness shape forces three concrete choices, and each one is a real departure from the canonical
version that I want stated plainly rather than glossed. First, the temperature, and I want to compute it
against the input the loop actually hands me rather than assume the canonical scale. The canonical default
is $\beta\approx1.0$, but that is calibrated for *raw, separately-normalized* advantages over a replay
buffer. The loop here has already normalized `mb_advantages` to roughly zero-mean, unit-variance per
minibatch (`norm_adv=True`), so let me see what each candidate temperature does to a unit-scale advantage.
At $\beta=1.0$: a $+1\sigma$ advantage gives weight $\exp(1)\approx2.72$ and a $-1\sigma$ advantage gives
$\exp(-1)\approx0.37$ — a ratio of only $7.4\times$ between good and bad actions, so the regression is
nearly uniform and learns almost nothing selective. At $\beta=0.05$: a $+1\sigma$ advantage gives
$\exp(20)\approx4.9\times10^{8}$ and a $-1\sigma$ advantage gives $\exp(-20)\approx2\times10^{-9}\approx0$.
After the weight clip at $20$ (below), the $+1\sigma$ weight saturates and the $-1\sigma$ weight is
effectively zero, so $\beta=0.05$ turns the exponential weighting into something close to a *hard
selection*: above-average actions all get the ceiling weight, below-average actions get nothing. That is
the selectivity I need from unit-scale normalized advantages, and it is the single most important
deviation from the canonical recipe — the temperature has to be read against the scale of the input it
actually receives, not the scale the original method assumed.

Second, the weight clip and stabilization, and I want to work through what the numbers do to a minibatch
so the stabilization is a computed necessity, not a superstition. With $\beta=0.05$ the exponential
weights have an enormous dynamic range, and a single outlier advantage would produce a weight that
dominates the entire minibatch gradient — the analogue of the ratio blowing up. So I clamp the weights at
`_awr_max_weight = 20.0` (the canonical clip value, which carries over cleanly). Now trace a minibatch of
$64$ transitions with roughly symmetric normalized advantages: about half sit above the mean and get
clamped to $20$, about half sit below and collapse toward $0$. The raw clamped sum is then $\approx
32\times20 = 640$. If I fed that into $-(\mathrm{newlogprob}\cdot w).\mathrm{mean}()$ directly, the policy
loss would carry an effective scale of $640/64=10$ — ten times the entropy and value terms it shares the
optimizer with — and that scale would *swing* from minibatch to minibatch depending on how many actions
happened to land above average. So I do something the off-policy version does not need: I *self-normalize*
the clipped weights to mean one, `weights = weights / (weights.sum() + 1e-8) * weights.numel()`. On that
same minibatch this maps the $32$ surviving weights from $20$ down to $20\times64/640=2.0$ and leaves the
zeros at zero, so the loss becomes a clean weighted MLE on the better-than-average half of the batch with
a *fixed* effective scale, regardless of how many actions cleared the bar. The reason this matters is
specific to this loop: the regression loss shares Adam and the global gradient-norm clip with the value
loss, and Adam's update is sensitive to the *scale* of the gradient relative to its running second
moment, so an un-normalized minibatch whose weights happen to be mostly tiny would produce a vanishing
policy gradient and waste the step, while one with a few huge surviving weights would saturate the
gradient-norm clip at $0.5$ and starve the value head. Renormalizing to mean one keeps the effective
regression step size constant from minibatch to minibatch, which is the same robustness the loop's per-
minibatch advantage normalization buys on the input side — this is on-policy AWR's way of being stable
inside a shared-optimizer K-epoch loop, and it is machinery the canonical buffer-based version simply does
not have.

Third, what computes the weights and what carries the gradient. The weights are a *target*, not a path:
the entire weight computation — exp, clamp, renormalization — sits under `torch.no_grad()`, or the
optimizer could cheat by reshaping the advantage estimate; only `newlogprob` carries gradient into
$-\hat{\mathbb E}[w\,\log\pi_\theta]$. For the Gaussian that per-sample gradient is $-w\cdot(a-\mu)/
\sigma^2$, pushing $\mu$ *toward* each selected action with magnitude bounded by the renormalized weight
(at most $\sim2$) — there is no $r$ to compound across epochs, which is the fan-out immunity I came for.
But the same structure carries the danger I predict below: the weights are frozen from *this rollout's*
GAE advantages, so the regression target does not move as the policy moves through the ten epochs. A fixed
target is stable when the advantage estimate is trustworthy and a trap when it is not — the policy
regresses toward whichever action drew the top advantage for all ten epochs with nothing to correct it.
The value head keeps rung 1's plain MSE toward the GAE returns, deliberately unchanged: I am swapping
exactly one thing — the policy objective — so whatever the numbers do is attributable to that swap and not
to a critic I quietly improved. The full module is in the answer.

Now to falsifiable expectations against the rung-1 numbers, and why Swimmer specifically is where the
fixed-target trap bites hardest. The frozen GAE scan uses $\gamma=0.99$, $\lambda=0.95$, so the advantage
is a geometric
sum of TD residuals with decay $\gamma\lambda=0.99\times0.95=0.9405$, giving an effective mixing horizon
of $1/(1-0.9405)\approx16.8$ steps. On a dense-reward environment like HalfCheetah, where reward accrues
every step, a $\sim17$-step advantage window aggregates plenty of signal and the estimate is relatively
tight. On Swimmer the reward is the small forward progress of a low-dimensional swimmer over a long
episode, and credit for a good stroke shows up many steps later — the signal-to-noise inside any $17$-step
window is far lower, so the per-minibatch advantage estimate carries much more variance. AWR then takes
that noisy estimate, computes a near-binary weight from it, and *freezes* it as the regression target for
all ten epochs. So the noisiest advantages get the most decisive, least-correctable commitment — the
worst possible pairing. That is the mechanism, quantified, behind the asymmetric prediction below.

My central bet is *reliability through construction*: with the trust region baked into the weights rather
than servoed by a soft penalty, I expect AWR to win where the advantage signal is clean and the
better-than-average actions are well separated — HalfCheetah, clearing the penalty rung's 1676.6, and
InvertedDoublePendulum, where the large achievable return rewards decisive exploitation and the sharp
$\beta=0.05$ concentration should beat 6877.1. But the same sharpness is a liability on the low-signal
environment. On Swimmer the near-binary selection commits for all ten epochs to whichever half of a
minibatch beat its noisy mean — frequently noise — with no ratio and no KL to pull it back. So my
prediction is asymmetric: AWR beats the penalty rung on HalfCheetah and InvertedDoublePendulum but comes
in *worse* on Swimmer than 101.4, with the collapse concentrated on whichever seed draws the noisiest
advantages. Because the score is a geometric mean, that one Swimmer collapse would cap AWR overall even if
it wins the other two — decisiveness on the clean environments bought at the cost of reliability on
exactly the noisy one the aggregator punishes hardest.
