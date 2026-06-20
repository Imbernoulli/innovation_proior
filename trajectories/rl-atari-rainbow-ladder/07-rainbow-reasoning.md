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

*Dueling over a distribution.* The value/advantage split was defined on scalars. Lift it to the *logits*:
each action's $N$ atom-logits are formed as a per-atom value stream plus a per-atom advantage stream with
the mean-over-actions subtracted, $\text{logits}_i(x,a)=V_i(x)+A_i(x,a)-\frac1{|\mathcal A|}\sum_{a'}
A_i(x,a')$, and *then* softmax over the atom dimension to get each action's $p_i(x,a)$. So the
identifiability-fixing aggregator lives on the logits, per atom, and the categorical structure is recovered
by the softmax — the dueling architecture and the distributional head are the same head.

*Prioritized replay over a distribution.* The priority was $|\delta|$, the magnitude of a *scalar* TD
error. There is no scalar TD error anymore — the loss is the per-sample cross-entropy / KL between the
projected target distribution and the prediction. So make the *KL loss itself* the priority source:
prioritize transitions by their distributional loss $L_t$, $p_i\propto L_t^\omega$. This is actually the
*more* principled quantity — it measures how surprising the whole return distribution is, not just its mean
— and it is already computed for the gradient, so the priority is free.

*Learned noise over everything.* The noisy linear layers simply replace the fully-connected layers of the
(now dueling, now distributional) head, and acting is greedy under the sampled net with $\epsilon=0$ —
$\epsilon$-greedy is gone entirely, exploration comes only from the parameter noise. No conflict: the noise
is on the head's weights, orthogonal to what the head computes.

That accounts for combining the six. But assembling them surfaces one more lever I could not justify adding
*as its own rung* — it is not in the per-component ablation — yet it composes naturally here and the
distributional machinery makes it nearly free: **multi-step returns**. Every rung so far bootstrapped after
*one* step, $r+\gamma(\cdot)$. A one-step target is maximally biased toward the current (wrong) value
estimate and propagates reward information backward only one state per update — slow credit assignment. An
$n$-step target $R_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}$ followed by a bootstrap
$\gamma^n(\cdot)$ uses $n$ steps of *real* reward before trusting the estimate, which trades a little
variance for much faster reward propagation and less reliance on the early, badly-wrong value function.
Over the distribution this is the cleanest possible edit: shift the target atoms by the accumulated return
$R_t^{(n)}$ and contract them by $\gamma^{(n)}_t=\gamma^n$ before projecting,
$\hat{\mathcal T}z_j=R_t^{(n)}+\gamma^n z_j$ — exactly the C51 shift-and-scale with $r\to R_t^{(n)}$ and
$\gamma\to\gamma^n$, then the same projection $\Phi$ and the same cross-entropy. $n=3$ is the balance: long
enough to speed credit assignment, short enough that the off-policy-ness of an $n$-step return over a few
stale actions is tolerable. I add it because it composes for free with the distributional target and
attacks a weakness (slow one-step credit assignment) none of the six touched.

So the integrated target is: take the online net's mean to pick $a^\star$, take the target net's
distribution there, shift it by the $n$-step return and contract by $\gamma^n$, project back onto the fixed
$[-10,10]$, $51$-atom grid, and minimize the cross-entropy of the projected target against the prediction —
with the per-sample loss feeding both the gradient (importance-weighted, $\beta$-annealed) and the replay
priority, on a dueling-over-logits, noisy-layer head. One agent, every axis improved at once. The
hyperparameters that make a *single* set of them work across all 57 games are the careful part: Adam at
$6.25\times10^{-5}$ (lower, because prioritization and the multi-step target both raise effective gradient
magnitude), Adam $\epsilon=1.5\times10^{-4}$, a target sync every 32K frames, priority exponent
$\omega=0.5$ with importance-sampling $\beta:0.4\to1.0$, $n=3$, $51$ atoms on $[-10,10]$, noisy-layer
$\sigma_0=0.5$, and $\epsilon=0$ for acting.

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
the level no single-component agent came close to. And the finer, falsifiable prediction is *which*
components carry the combined result: an ablation that removes each one from the full agent should hurt most
when it removes prioritized replay and the multi-step/distributional pieces (the ones that touch every
game's learning broadly), and least when it removes the component whose isolated gain was already a
tail-only sliver (the learned noise, which barely moved the median on its own). If removing the
distributional object or prioritized replay from the full agent *doesn't* hurt, then those components were
redundant with the rest and the compounding story is wrong. But if the integration is right — six largely
independent fixes to six largely independent weaknesses of the DQN floor, re-expressed so they share one
distributional, dueling, noisy, prioritized, multi-step agent — then this single agent should be the best
value-based learner on the suite, clearing the 164% best-single-component bar by a wide margin.
