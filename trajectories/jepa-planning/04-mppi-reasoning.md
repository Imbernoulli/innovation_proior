iCEM's numbers came back, and they confirmed the bet *and* exposed the risk I flagged when I built it.
The bet paid off at the short horizon exactly as predicted: CEM's weakest benchmark, horizon 30, went
from 0.70 to 0.80 and the residual `mean_dist` dropped from 9.5 to 5.7 — the colored, far-ranging
excursions plus the across-step warm start did reach the door inside a single `plan()` call, which is
precisely what white CEM could not do cold. At horizon 60 iCEM held respectably, 0.90 with residual
4.1. But look at horizon 90: iCEM scored 0.85 — *below* CEM's 0.95 there — with residual 4.8 against
CEM's 3.4. That is the regression I named in advance: at the long horizon, where CEM's white sampling
plus the re-planning loop had plenty of room to converge on a clean route, iCEM's aggressive elite
reuse and smooth low-frequency colored draws *over-commit*. When the kept elites and the persistent
shifted mean all agree on a route, and that route is the wrong side of the door for a given start/goal,
the search has even less spread to escape than CEM did — the hard top-k elite cut plus the colored
over-smoothing leaves no candidate that disagrees enough to pull the distribution onto the other route.
So iCEM traded long-horizon robustness for short-horizon reach. Averaged across the three benchmarks it
is roughly a wash with CEM (mean success ~0.85 vs ~0.83), which is not the clean dominance I wanted.

The root is the *hard* elite selection, and I should be precise about why. Both CEM and iCEM refit the
distribution by taking the top-k lowest-cost sequences and computing their plain mean and variance.
That throws away two things. First, it throws away *how much* better the best elite was than the worst:
a sequence that is dramatically better and one that barely cleared the elite cut count exactly the same
in the mean — each elite gets weight 1/k. Second, the hard cut is brittle near the decision boundary
between the two routes through the door: a handful of mediocre elites on the wrong route can drag the
mean and, worse, keep the variance from shrinking onto the right route — or, in iCEM's over-committed
case, the elites all land on one route and the variance collapses there with no soft pressure from the
near-elite samples on the other route. I want an update that is *soft*: every sampled sequence
contributes, weighted by how good it is, so a markedly-better rollout pulls the mean and tightens the
variance far more than a marginal one, and no information is discarded at a hard threshold. And I want
it to be the *principled* soft update, not an ad-hoc reweighting — the same demand I made when I moved
from random shooting to CEM.

So go back to the object and ask what the right weighting is. I have sampled action sequences and their
costs, and I want the new control distribution that best concentrates on low cost. Frame it as the
free-energy / partition-function object: assign each trajectory τ a Boltzmann weight that is
exponential in its negative cost, w(τ) ∝ exp(−S(τ)/λ), where S(τ) is the cost of rolling that sequence
through the model and λ is a temperature. Low-cost trajectories get large weight, high-cost ones are
suppressed, and λ controls how sharply: λ → 0 puts all the weight on the single best sequence (a hard
argmax), λ → ∞ weights everything equally (back to plain averaging). The optimal control is then the
weighted average of the sampled controls under these weights — the soft, information-preserving analogue
of CEM's hard elite mean. This is exactly the path-integral reading of optimal control: the new mean is
the cost-weighted average of the perturbations, u_i ← Σ_k w_k · a_{i,k} with w_k = exp(−S(τ_k)/λ) / Σ_j
exp(−S(τ_j)/λ). Each sequence votes with a weight that decays smoothly in its cost, so the
dramatically-better wrong-route-avoiding sequence dominates without me having to draw a hard line, and a
cluster of mediocre wrong-route sequences gets exponentially suppressed rather than averaged in at full
weight — which is exactly the long-horizon failure iCEM showed.

Now I have to be honest about the specific form the task's harness gives me, because the full
path-integral derivation assumes a control-affine system with a particular noise-cost coupling and
yields a Girsanov likelihood-ratio correction for shifting both the mean and the covariance of the
sampling distribution. I do not have a control-affine analytic model here — I have a learned JEPA world
model I can only roll forward and a black-box cost — so I am not going to import the covariance-scaling
Girsanov machinery. What I land instead is the discrete, sampling-based core of the method as it is
actually used on top of learned models: keep the CEM skeleton — sample a batch around the current mean,
roll through the model, take the lowest-cost elites — but replace the *update* with the exponential
cost-weighting. Concretely, after scoring I still select the top elites (here num_elites = 20, top 10%),
then compute weights over those elites, score_k = exp(temperature · (min_cost − cost_k)), normalized to
sum to 1, and set the new mean to the weighted average of the elite sequences and the new std to the
weighted standard deviation. Subtracting min_cost inside the exponential is the standard numerical
stabilization — it shifts the largest exponent to zero so nothing overflows, and because the weights are
normalized the shift cancels and leaves the Boltzmann weighting unchanged. The temperature here is
small, 0.005, which keeps the weighting gentle relative to the cost scale — sharp enough that a clearly
better rollout dominates, soft enough that the near-best sequences still contribute and the variance
does not collapse onto one route the way iCEM's hard reuse did.

I keep the sampling *white* here — plain `torch.randn` perturbations, not colored noise. That is a
deliberate choice given what iCEM's numbers showed: the colored over-smoothing is part of what made
iCEM over-commit at the long horizon, and the soft weighting is supposed to do the reach-and-route work
through the *update*, not through a far-ranging noise prior. White sampling with a soft cost-weighted
mean lets the distribution be pulled toward whichever route the rollouts actually score well, without a
low-frequency prior baking in a committed direction before the cost is read. I also widen the initial
spread to max_std = 2.0 (above CEM's 1.5), because the soft update is more forgiving of a wide start —
it will not be dragged around by a few wide outliers the way a hard elite mean can be, since those
outliers carry exponentially small weight — so I can afford to explore both routes more aggressively at
the first iteration and let the Boltzmann weighting concentrate from there.

There is one more place the soft framing changes the *output*, and it matters for the maze. CEM and
iCEM return the distribution's center (the mean, or iCEM's best-so-far fallback). But the mean of a
soft-weighted distribution, in a bimodal cost surface with two routes, can land *between* the two modes
— in the wall — which is worse than either route. So instead of executing the mean, I select the
executed sequence by *sampling* from the final elite weights: draw one elite with probability
proportional to its score, and return that actual evaluated sequence. This is the path-integral
controller's stochastic action selection, and it is the right thing in a multi-modal surface: it commits
to one real, rolled-out route rather than averaging two incompatible routes into an infeasible
compromise. The chosen sequence is one I actually scored, so it is feasible and known-good, not an
untested centroid. Nothing persists across env steps in this rung — each `plan()` re-optimizes from the
zero mean and wide std — so `t0` is irrelevant again, and the whole loop runs under `no_grad` because,
like every rung before it, MPPI reads only function values of the cost and never touches the gradient
the differentiable model could hand it. That gradient is still sitting unused; it is the lever a method
beyond this ladder would pull.

So the falsifiable expectations against iCEM's numbers. The soft cost-weighting is aimed squarely at
iCEM's long-horizon regression, so the cleanest prediction is at horizon 90: where iCEM slipped to 0.85
by over-committing to one route, the Boltzmann weighting should recover the long-horizon performance
CEM had — I expect horizon 90 back up to ~0.95 with the residual `mean_dist` back down toward CEM's 3.4,
because the soft update plus stochastic-elite action selection stops a few wrong-route elites from
collapsing the search and stops the planner from executing an in-the-wall mean. At horizon 60 I expect
a clear lift over iCEM's 0.90 — the soft update is most valuable in the middle regime where both routes
are live and a hard cut is brittle. At horizon 30 the open question is whether dropping iCEM's colored
noise costs me the short-horizon reach: I expect to roughly *hold* iCEM's 0.80 rather than beat it,
because white sampling reaches the door less reliably than colored excursions inside one short call —
and if horizon 30 instead *drops* below 0.75, that is the signal that the short horizon genuinely needs
the colored prior and the right method would be MPPI's soft update *over* colored noise, combining this
rung's robustness with iCEM's reach. But the headline bet is that the soft, information-preserving update
gives the first planner that is strong at *all three* horizons at once — short reach near iCEM's, and
long-horizon robustness back at CEM's level — which is the clean dominance neither hard-cut baseline
achieved. The full scaffold module — the literal `CustomPlanner` fill for MPPI — is in the answer.
