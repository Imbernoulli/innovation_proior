The distributional object was the strongest single change to the floor — 164% median, the best any *one*
modification bought. I now have six improvements, each measured in isolation against the DQN floor:
decoupling the target (79$\to$117), learned parametric exploration (117$\to$118), prioritized replay
(118$\to$140), the dueling head (140$\to$151), and the distributional value object (151$\to$164). The obvious
question is whether these compound. If the others each add a real and *independent* improvement on top of the
best single component, an agent with all six should sit well above 164%. But "independent" is load-bearing,
and I cannot assume it: these are not orthogonal knobs in a config file, they are changes to *overlapping*
parts of the same agent, and the only way to know whether they collide or compose is to ask, for each pair
that touches the same machinery, whether they attack the *same* weakness or *different* ones.

Bound the two extremes first. Each component's gain over the floor is its reported value minus $79$:
decoupled $38$, noisy $39$, prioritized $61$, dueling $72$, distributional $85$. If those were perfectly
independent and simply *added*, the combined agent would reach $79+38+39+61+72+85=374\%$ — obviously absurd, which is
useful precisely because it proves the components *cannot* be independent-and-additive: they overlap in what
they fix (three of them broadly improve the same "value estimate on the typical game," so their benefits are
not disjoint pieces I can sum). The other extreme is perfect *redundancy*: if all six attacked the identical
weakness, the combined agent would do no better than the single best component, $\approx164\%$. The truth
lies between — above $164$ because they are not identical, below $374$ because they are not disjoint — and the
pairwise analysis of what each touches is meant to locate *where*.

Run down the list by what each one touches. The decoupled target changes *how the bootstrap action is
selected and evaluated*. Prioritized replay changes *which transitions I sample*. The dueling head changes
*the architecture that produces the values*. Learned noise changes *how I explore*. The distributional object
changes *what the values are*. Five genuinely different surfaces — target construction, sampling,
architecture, exploration, representation — and there is no reason fixing the sampling should undo fixing the
target's bias, or a better-shaped head conflict with a learned exploration scheme. The improvements are
*largely* independent because they address *largely* independent weaknesses, exactly the precondition for
compounding rather than cancelling. So the hypothesis is plausible — but the places where two of them touch
the *same* object need actual design, not assumption, and that is the work here.

Where they touch the same object: the value is now a *distribution*, and three components were specified for
a *scalar*. I have to re-express each over the categorical distribution rather than bolt it on.

*Double Q-learning over a distribution.* The decoupled target needs a greedy next action, but greedy with
respect to *what*, now that there is no scalar $Q$? The policy stays mean-greedy, so select the bootstrap
action by the *online* net's mean, $a^\star=\arg\max_a z^\top p_\theta(S',a)$, then evaluate by taking the
*target* net's whole *distribution* for that action, $p_{\bar\theta}(S',a^\star)$. The value entering the
target is a different network's estimate than the one that selected $a^\star$ — exactly the independence the
scalar double target bought, now at the level of distributions. The tempting shortcut, selecting $a^\star$
from the *target* net's mean too (it is already being evaluated), collapses selection and evaluation back
onto one network and reintroduces the overestimation coupling the decoupled target removed. So keeping the
selection on the online net is what carries the anti-overestimation property across, and I have to be
deliberate about it or I would silently lose it.

*Dueling over a distribution.* Lift the value/advantage split to the *logits*: each action's $N$
atom-logits are a per-atom value stream plus a per-atom advantage stream with the mean-over-actions
subtracted, $\text{logits}_i(x,a)=V_i(x)+A_i(x,a)-\frac1{|\mathcal A|}\sum_{a'}A_i(x,a')$, and *then* softmax
over the atom dimension to get $p_i(x,a)$. So the identifiability-fixing aggregator lives on the logits, per
atom, and for each atom index the same argument as the scalar dueling head pins the offset. Crucially the
softmax comes *after* the subtraction, so the aggregator shapes the logits and the categorical normalization
stays exact; the other order would break the probability interpretation. The reshape and the categorical
representation compose cleanly because they act on different axes — dueling on the action axis, the softmax on
the atom axis.

*Prioritized replay over a distribution.* The priority was $|\delta|$, a *scalar* TD error, which no longer
exists — the loss is the per-sample cross-entropy / KL between the projected target distribution and the
prediction. So make the *KL loss itself* the priority source, $p_i\propto L_t^\omega$. This is the *more*
principled quantity — it measures how surprising the whole return distribution is, not just its mean — and it
is already computed for the gradient. See why it is strictly more informative: a transition whose predicted
*mean* is already correct but whose predicted *shape* is wrong (the network thinks the return is a tight bump
at $1$ when it is really a $50/50$ split between $0$ and $2$) has near-zero scalar TD error, so
prioritization-over-mean would rate it boring — yet it is exactly a transition the distributional agent has a
lot left to learn from, and the KL flags it correctly. So switching the priority source is not a mechanical
port but an upgrade the distributional representation makes possible.

*Learned noise over everything.* The noisy linear layers replace the fully-connected layers of the (now
dueling, now distributional) head, and acting is greedy under the sampled net with $\epsilon=0$ —
$\epsilon$-greedy is gone entirely. There are now *four* FC layers to make noisy, because the head has two
streams each with a hidden and an output layer, and the online and target nets each get independent noise
draws as before. No conflict in principle: the noise is on the head's weights, orthogonal to *what* the head
computes. The one genuine interaction is with prioritization: a transition that looks surprising only because
a particular noise draw mispredicted it could be over-prioritized, so noise-driven and learning-driven
surprise partially blur — a real integration wrinkle I flag for the risk accounting rather than wave away.

Assembling the six surfaces one more lever I could not justify as its own isolated change — it is not in the
per-component ablation — yet it composes naturally and the distributional machinery makes it nearly free:
**multi-step returns**. Every change so far bootstrapped after *one* step, $r+\gamma(\cdot)$. A one-step
target is maximally biased toward the current (wrong) value estimate and propagates reward information back
only one state per update. An $n$-step target $R_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}$ followed by a
bootstrap $\gamma^n(\cdot)$ uses $n$ steps of *real* reward before trusting the estimate. Quantify both
sides. Credit assignment: a one-step update moves reward information back one state per replay, so a reward
$L$ states back needs the chain replayed $\sim L$ times; an $n$-step update moves it $n$ states at once, an
$n$-fold speedup — at $n=3$, three times faster. Reliance on the estimate: the bootstrap enters weighted by
$\gamma^n=0.99^3\approx0.970$ instead of $0.99$, so a slightly larger share of the target is real reward
rather than the network's early wrong guess. The costs grow with $n$: the $n$-step return sums $n$ stochastic
rewards so its variance rises, and — the sharper cost — the intermediate actions were taken by a *stale*
behavior policy, so a long $n$-step return is increasingly off-policy and biased for the greedy return I
want. $n=3$ is the balance where the propagation speedup is real and the few-step staleness is still small.
Over the distribution this is the cleanest possible edit, and it is nearly free because the distributional
Bellman update was *already* a shift-and-scale of atoms: C51's target $\hat{\mathcal T}z_j=r+\gamma z_j$
becomes $\hat{\mathcal T}z_j=R_t^{(n)}+\gamma^n z_j$ — shift by the accumulated $n$-step return instead of the
one-step reward, contract by $\gamma^n$ instead of $\gamma$ — then the *same* projection and cross-entropy.
The only code that changes is what number I shift by and what power of $\gamma$ I scale by, whereas bolting an
$n$-step return onto a scalar target would have needed its own separate handling.

So the integrated target: take the online net's mean to pick $a^\star$, take the target net's distribution
there, shift by the $n$-step return and contract by $\gamma^n$, project back onto the fixed $[-10,10]$
$51$-atom grid, and minimize the cross-entropy of the projected target against the prediction — the
per-sample loss feeding both the gradient (importance-weighted, $\beta$-annealed) and the replay priority, on
a dueling-over-logits, noisy-layer head. The hyperparameters that make a *single* set work across all 57
games are the careful part, and the learning rate is where the interactions concentrate: the floor ran Adam
near $10^{-4}$; prioritized replay alone wanted that cut about fourfold for the raised gradient scale, the
multi-step target pushes effective gradient magnitude up again, and the distributional cross-entropy carries
its own scale — three pressures toward a smaller step. Landing at Adam $6.25\times10^{-5}$, roughly the
floor's rate over $1.6$, reconciles them: low enough that no one stacked change drives the update past the
stable band. The rest follow their components — Adam $\epsilon=1.5\times10^{-4}$, target sync every 32K
frames, priority exponent $\omega=0.5$ with $\beta:0.4\to1.0$, $n=3$, $51$ atoms on $[-10,10]$, noisy-layer
$\sigma_0=0.5$, $\epsilon=0$ for acting. These are *one* set frozen across all 57 games — each chosen for the
interaction, not tuned per game.

The one real risk in the "compound" hypothesis is that the components are largely but not perfectly
independent, and the pairs that interact — multi-step and prioritized replay both shifting the gradient scale
(hence the LR drop), the distributional loss redefining "priority," noisy exploration blurring into
prioritization — are exactly the integration points where a naive concatenation of six papers would collide.
The re-expressions above (distributional priority, logit-level dueling, mean-greedy distributional double
selection) are what keep them composing instead.

Now the bar, stated so I can be proven wrong. The floor was 79%, the best single component 164%. If the six
were perfectly redundant — all attacking the same weakness — the combined agent would do no better than
$\approx164\%$, the null this tests against. If instead they are largely independent, it should clear 164%
*decisively* and land far above — I expect it to roughly double the floor into the low 200s, the level no
single-component agent came close to. That lands inside the bracket: above the redundancy floor of $164$, far
below the impossible additive ceiling of $374$, and closer to the bottom of the bracket than its top, which
is what "largely but not perfectly independent" has to look like numerically — I recover only a fraction of
the naive sum, but combining is not pointless. It is falsifiable in both directions: a result stuck near
$164$ would mean redundancy, one near $374$ would mean I mis-measured the singles. The finer prediction is
*which* components carry the result: an ablation removing each from the full agent should hurt most for
prioritized replay and the multi-step/distributional pieces (which touch every game's learning broadly) and
least for the learned noise (whose isolated gain was a tail-only sliver). If removing the distributional
object or prioritized replay *doesn't* hurt, those were redundant and the compounding story is wrong. But if
the integration is right — six largely independent fixes to six largely independent weaknesses, re-expressed
so they share one distributional, dueling, noisy, prioritized, multi-step agent — then this single agent
should be the best value-based learner on the suite, clearing the 164% bar by a wide margin.
