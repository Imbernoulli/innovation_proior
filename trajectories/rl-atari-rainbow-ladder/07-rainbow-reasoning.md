The distributional object was the strongest single change to the floor — 164% median, the best any *one*
modification bought. I now have six improvements, each measured in isolation against the DQN floor:
decoupling the target (79$\to$117), learned parametric exploration (117$\to$118), prioritized replay
(118$\to$140), the dueling head (140$\to$151), and the distributional value object (151$\to$164). The
obvious question — the one that has been implicit since rung 2 — is whether these compound. If I take the
best single component, 164%, and the others are each adding a real and *independent* improvement, then an
agent with all six should sit well above 164%. But "independent" is the load-bearing word, and I cannot
assume it: these are not orthogonal knobs in a config file, they are changes to *overlapping* parts of the
same agent, and the only way to know whether they collide or compose is to ask, for each pair that touches
the same machinery, whether they were attacking the *same* weakness or *different* ones.

Bound the two extremes first, to know what I am even hoping for. Each component was measured as the floor
*plus that one change*, so its gain over the floor is its reported value minus $79$: decoupled target
$117-79=38$, noisy $118-79=39$, prioritized $140-79=61$, dueling $151-79=72$, distributional $164-79=85$.
If those five gains were perfectly independent and simply *added* on top of the floor, the combined agent
would reach $79+38+39+61+72+85=374\%$ — an obviously absurd ceiling, which is useful precisely because it is
absurd: it
proves the components *cannot* be independent-and-additive, because they overlap in what they fix (three of
them broadly improve the same "value estimate on the typical game," so their benefits are not disjoint
pieces I can just sum). The other extreme is perfect *redundancy*: if all six attacked the identical
underlying weakness, the combined agent would do no better than the single best component, $\approx164\%$,
because once that weakness is fixed the rest add nothing. The truth has to lie between these — above $164$
because the components are not identical, below $374$ because they are not disjoint — and the interesting
question is *where*, which the pairwise analysis of what each one touches is meant to answer.

Run down the list by what each one touches. The decoupled target changes *how the bootstrap action is
selected and evaluated*. Prioritized replay changes *which transitions I sample*. The dueling head changes
*the architecture that produces the values*. Learned noise changes *how I explore*. The distributional
object changes *what the values are*. Five of these touch genuinely different surfaces of the agent —
target construction, sampling, architecture, exploration, representation — and there is no reason fixing the
sampling distribution should undo the benefit of fixing the target's bias, or that a better-shaped head
should conflict with a learned exploration scheme. The improvements are *largely* independent because they
address *largely* independent weaknesses, which is exactly the precondition for combining them to compound
rather than cancel. So the hypothesis is plausible — but the places where two of them touch the *same*
object need actual design, not assumption, and that is where the work of this rung is.

Where they touch the same object: the value is now a *distribution*, and three of the components were
specified for a *scalar*. I have to re-express each of those over the categorical distribution rather than
bolt it on. Take them one at a time.

*Double Q-learning over a distribution.* The decoupled target needs a greedy next action, but greedy *with
respect to what*, now that there is no scalar $Q$? The policy I commit to is still mean-greedy — I act on
the expected return — so select the bootstrap action by the *online* network's mean,
$a^\star=\arg\max_a z^\top p_\theta(S',a)$, and then evaluate by taking the *target* network's whole
*distribution* for that action, $p_{\bar\theta}(S',a^\star)$. Online net selects (via its mean), target net
supplies the distribution to bootstrap — the decoupling is preserved, now at the level of distributions.
Check that this really is the decoupling and not a lookalike: the whole point of the double target was that
the network which *selects* the bootstrap action must not be the same one that *supplies its value*, so the
selection cannot inflate the value it reads back. Here the online net picks $a^\star$ from its own mean, and
the value that enters the target is the *target* net's distribution at $a^\star$ — a different network's
estimate, exactly the independence the scalar double target bought. The tempting shortcut would be to select
$a^\star$ from the *target* net's mean too (it is already being evaluated), but that collapses selection and
evaluation back onto one network and reintroduces the very overestimation coupling rung 2 removed. So the
distributional double target is not a free reshuffle; keeping the selection on the online net is what
carries the anti-overestimation property across into the distributional setting, and I have to be deliberate
about it or I would silently lose it.

*Dueling over a distribution.* The value/advantage split was defined on scalars. Lift it to the *logits*:
each action's $N$ atom-logits are formed as a per-atom value stream plus a per-atom advantage stream with
the mean-over-actions subtracted, $\text{logits}_i(x,a)=V_i(x)+A_i(x,a)-\frac1{|\mathcal A|}\sum_{a'}
A_i(x,a')$, and *then* softmax over the atom dimension to get each action's $p_i(x,a)$. So the
identifiability-fixing aggregator lives on the logits, per atom, and the categorical structure is recovered
by the softmax — the dueling architecture and the distributional head are the same head. Check the shapes
close: the value stream emits $(B,1,N)$ atom-logits, the advantage stream emits $(B,|\mathcal A|,N)$, I
broadcast-add them and subtract the mean over the action axis to get $(B,|\mathcal A|,N)$ logits, then
softmax over the atom axis $N$ to get a proper per-action distribution $(B,|\mathcal A|,N)$ summing to one
along $N$. The mean-subtraction is per atom $i$: $\text{logit}_i(x,a)=V_i(x)+A_i(x,a)-\frac1{|\mathcal A|}
\sum_{a'}A_i(x,a')$, so for each atom index the same identifiability argument as the scalar dueling head
applies — a per-(state,atom) constant added to $V_i$ and removed from every $A_i$ would otherwise leave the
logits unpinned, and the mean reference pins it. And crucially the softmax comes *after* the subtraction, so
the aggregator shapes the logits and the categorical normalization is exact; doing it in the other order
would break the probability interpretation. The dueling reshape and the categorical representation compose
cleanly because they act on different axes — dueling on the action axis, the softmax on the atom axis.

*Prioritized replay over a distribution.* The priority was $|\delta|$, the magnitude of a *scalar* TD
error. There is no scalar TD error anymore — the loss is the per-sample cross-entropy / KL between the
projected target distribution and the prediction. So make the *KL loss itself* the priority source:
prioritize transitions by their distributional loss $L_t$, $p_i\propto L_t^\omega$. This is actually the
*more* principled quantity — it measures how surprising the whole return distribution is, not just its mean
— and it is already computed for the gradient, so the priority is free. See why it is strictly more
informative: consider a transition whose predicted *mean* return is already correct but whose predicted
*shape* is wrong — say the network thinks the return is a tight bump at $1$ when it is really a
$50/50$ split between $0$ and $2$ (same mean, different distribution). The old scalar TD error is near zero,
so uniform-over-mean prioritization would rate this transition as boring and rarely replay it — yet it is
exactly a transition the distributional agent has a lot left to learn from. The KL between the projected
target distribution and the prediction is large there, so the distributional priority correctly flags it as
surprising and replays it more. The scalar priority was blind to shape error by construction; the KL
priority is not, so switching the priority source to the distributional loss is not a mechanical port, it is
an upgrade that the distributional representation makes possible.

*Learned noise over everything.* The noisy linear layers simply replace the fully-connected layers of the
(now dueling, now distributional) head, and acting is greedy under the sampled net with $\epsilon=0$ —
$\epsilon$-greedy is gone entirely, exploration comes only from the parameter noise. There are now *four* FC
layers to make noisy, because the head has two streams each with a hidden layer and an output layer — the
value stream's $3136\to512$ and $512\to N$, and the advantage stream's $3136\to512$ and
$512\to|\mathcal A|N$ — so all four become noisy linear layers, and the online and target nets each get
their own independent noise draws as before. No conflict in principle: the noise is on the head's weights,
orthogonal to *what* the head computes (the mean-subtracted logits) — perturbing the weights changes which
value function is sampled, not the aggregation rule or the softmax. The one genuine interaction I have to
keep in mind is with prioritization: a transition that looks surprising only because a particular noise draw
mispredicted it could be over-prioritized, so the noise-driven surprise and the learning-driven surprise
partially blur. That is a real integration wrinkle, not a clean orthogonality, and I flag it for the risk
accounting rather than waving it away.

That accounts for combining the six. But assembling them surfaces one more lever I could not justify adding
*as its own rung* — it is not in the per-component ablation — yet it composes naturally here and the
distributional machinery makes it nearly free: **multi-step returns**. Every rung so far bootstrapped after
*one* step, $r+\gamma(\cdot)$. A one-step target is maximally biased toward the current (wrong) value
estimate and propagates reward information backward only one state per update — slow credit assignment. An
$n$-step target $R_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}$ followed by a bootstrap
$\gamma^n(\cdot)$ uses $n$ steps of *real* reward before trusting the estimate, which trades a little
variance for much faster reward propagation and less reliance on the early, badly-wrong value function.
Quantify both sides. Credit assignment: a one-step update moves reward information back exactly one state
per replay of a transition, so propagating a reward $L$ states back needs the chain replayed $\sim L$ times;
an $n$-step update moves it back $n$ states at once, an $n$-fold speedup in how fast a discovered reward
reaches the states that lead to it — at $n=3$, three times faster. Reliance on the estimate: the bootstrap
now enters weighted by $\gamma^n=0.99^3\approx0.970$ instead of $\gamma=0.99$, so a slightly larger share of
the target is *real* reward rather than the network's own (early, wrong) guess, which matters most early in
training when that guess is worst. The costs grow with $n$ too: the $n$-step return sums $n$ stochastic
rewards, so its variance rises with $n$, and — the sharper cost here — the intermediate actions
$r_{t+1},\dots,r_{t+n}$ were taken by a *stale* behavior policy, so a long $n$-step return is increasingly
off-policy and biased for the greedy return I actually want. Small $n$ keeps that off-policy bias tolerable;
large $n$ speeds credit assignment. $n=3$ is the balance point where the propagation speedup is real and the
few-step staleness is still small.
Over the distribution this is the cleanest possible edit, and that it is nearly free is not luck — it is
because the distributional Bellman update was *already* a shift-and-scale of atoms. C51's target was
$\hat{\mathcal T}z_j=r+\gamma z_j$: shift every atom by the reward, contract toward zero by $\gamma$, project
back to the grid. The $n$-step target is the identical operation with two substitutions,
$\hat{\mathcal T}z_j=R_t^{(n)}+\gamma^n z_j$ — shift by the accumulated $n$-step return $R_t^{(n)}$ instead
of the one-step reward, contract by $\gamma^n$ instead of $\gamma$ — then the *same* projection $\Phi$ and
the *same* cross-entropy. There is no new machinery: the only code that changes is what number I shift by
and what power of $\gamma$ I scale by. This is exactly the kind of composition that justifies calling
multi-step "free" here — it slots into the atom shift-and-scale that distributional learning already
required, whereas bolting an $n$-step return onto a scalar target would have needed its own separate
handling. $n=3$ is the balance: long
enough to speed credit assignment, short enough that the off-policy-ness of an $n$-step return over a few
stale actions is tolerable. I add it because it composes for free with the distributional target and
attacks a weakness (slow one-step credit assignment) none of the six touched.

So the integrated target is: take the online net's mean to pick $a^\star$, take the target net's
distribution there, shift it by the $n$-step return and contract by $\gamma^n$, project back onto the fixed
$[-10,10]$, $51$-atom grid, and minimize the cross-entropy of the projected target against the prediction —
with the per-sample loss feeding both the gradient (importance-weighted, $\beta$-annealed) and the replay
priority, on a dueling-over-logits, noisy-layer head. One agent, every axis improved at once. The
hyperparameters that make a *single* set of them work across all 57 games are the careful part, and the
learning rate is where the interactions concentrate. The floor ran Adam near $10^{-4}$; prioritized replay
alone already wanted that cut by about four for the raised gradient scale, the multi-step target pushes
effective gradient magnitude up again, and the distributional cross-entropy carries its own scale — three
pressures all pointing toward a smaller step. Landing at Adam $6.25\times10^{-5}$, roughly the floor's rate
divided by $1.6$, is the reconciliation of those pressures: low enough that no one of the stacked changes
drives the update size past the stable band the encoder needs. The rest follow the components they come
from — Adam $\epsilon=1.5\times10^{-4}$ for numerical stability at this small step, a target sync every 32K
frames, priority exponent $\omega=0.5$ with importance-sampling $\beta:0.4\to1.0$ (the same anneal logic as
the prioritized rung), $n=3$, $51$ atoms on $[-10,10]$, noisy-layer $\sigma_0=0.5$, and $\epsilon=0$ for
acting since exploration now lives entirely in the noisy weights. The discipline is that these are *one* set
frozen across all 57 games — the benchmark's whole constraint — so each had to be chosen for the
interaction, not tuned per game.

I should be honest about the one real risk in the "compound" hypothesis: the components are largely but not
perfectly independent, and a couple of pairs *do* interact in ways that could either help or hurt. Multi-step
returns and prioritized replay both change the effective gradient scale, which is why the learning rate
drops; the distributional loss changes what "priority" even means, which I had to redefine; and the noisy
exploration interacts with prioritization (surprising-but-noise-driven transitions could be over-replayed).
These are exactly the integration points where a naive concatenation of six papers would collide — and the
re-expressions above (distributional priority, logit-level dueling, mean-greedy distributional double
selection) are what keep them composing instead.

Now the bar, stated so I can be proven wrong. The floor was 79%. The best *single* component was the
distributional object at 164%. If the six improvements were perfectly redundant — all attacking the same
underlying weakness — the combined agent would do no better than that best single component, $\approx164\%$,
and the whole "compound" premise would be false; that is the null result this rung is testing against. If
instead they are largely independent, the combined agent should clear 164% *decisively* and land far above
it — I am expecting it to roughly double the floor and reach the low 200s in median human-normalized score,
the level no single-component agent came close to. That lands the prediction inside the bracket I set at the
start: above the redundancy floor of $164$ but far below the impossible additive ceiling of $374$, and
closer to the floor of the bracket than its top, which is what "largely but not perfectly independent" has
to look like numerically — the components share enough of their benefit that I recover only a fraction of the
naive sum, but not so much that combining is pointless. A result in the low 200s is the quantitative
signature of partial overlap; it is the specific number my "compound, don't collide" story predicts, and it
is falsifiable in both directions — a result stuck near $164$ would mean redundancy, and an impossible
result near $374$ would mean I mis-measured the singles. And the finer, falsifiable prediction is *which*
components carry the combined result: an ablation that removes each one from the full agent should hurt most
when it removes prioritized replay and the multi-step/distributional pieces (the ones that touch every
game's learning broadly), and least when it removes the component whose isolated gain was already a
tail-only sliver (the learned noise, which barely moved the median on its own). If removing the
distributional object or prioritized replay from the full agent *doesn't* hurt, then those components were
redundant with the rest and the compounding story is wrong. But if the integration is right — six largely
independent fixes to six largely independent weaknesses of the DQN floor, re-expressed so they share one
distributional, dueling, noisy, prioritized, multi-step agent — then this single agent should be the best
value-based learner on the suite, clearing the 164% best-single-component bar by a wide margin.
