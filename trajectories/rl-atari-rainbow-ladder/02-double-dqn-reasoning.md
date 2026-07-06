The DQN floor lands at 79% median human-normalized score — a working agent, but on the typical game it
sits below the human reference, and I have a list of broken pieces to choose from. Read that $79\%$
literally: on the median game the agent closes $79\%$ of the gap between random play and the human
reference, so it is genuinely learning — this is not a broken floor, it is a working one — but it is still
short of human on the typical title, sitting $21$ points under the $100\%$ line. The floor did its job of
being stable everywhere; the job now is to lift the middle of the distribution. I want the *one* change
that buys the most on the median game, and the discipline of the median tells me where to look: it is not
moved by squeezing more out of games the agent already wins, it is moved by fixing whatever quietly
degrades the agent across many games at once. So I want a defect that is not specific to one title but is
baked into the update I run on all 57. Run the candidate axes past that criterion before committing. Exploration — the $\epsilon$-greedy dithering
— is crude, but its failures are concentrated on the minority of games that need long deliberate action
sequences; on the typical game the agent already sees enough reward signal, so fixing exploration would
lift a few tails and leave the median roughly where it is. The value object being a scalar is a deep issue,
but it is a *representational* overhaul rather than a recipe tweak, and I want to spend the cheap
high-leverage fixes before the expensive ones. The uniform replay sampling I adopted only for
decorrelation is a candidate too, but it is not *provably* wrong the way the target is — it is at worst an
inefficiency, whereas the target has a specific, demonstrable defect I can write down. The most suspect piece of the floor is therefore the
bootstrap target itself, because it is computed the same way on all $57$ games — so any bias in it is a
tax on the whole suite, exactly what a median responds to — and I already flagged it as theoretically
broken. Let me look hard at it.

DQN regresses $Q(S_t,A_t;\theta)$ onto $Y_t=R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta^-)$. The reward and
$\gamma$ are fixed; the only thing in that target that is a function of my noisy estimates is
$\max_a Q(S_{t+1},a;\theta^-)$. So if the target is biased, the bias lives in that max. Does the max bias
it? Suppose my per-action estimates are individually *unbiased* — $Q(s',a)=Q_*(s',a)+\epsilon_a$ with each
$\epsilon_a$ zero-mean. Is $\max_a Q(s',a)$ then an unbiased estimate of $\max_a Q_*(s',a)$? No, because
$\max$ is convex, and Jensen runs the wrong way:
$\mathbb{E}[\max_a(Q_*(s',a)+\epsilon_a)]\ge\max_a(Q_*(s',a)+\mathbb{E}[\epsilon_a])=\max_a Q_*(s',a)$.
The expected max of noisy values is at least the max of the true values. The intuition is sharper than the
inequality: the $\max$ hunts across all the actions and selects the largest estimate, which preferentially
picks whichever action's noise landed on the high side — positive noise gets selected, negative noise gets
discarded. Then I bootstrap that inflated value into $Q(S_t,A_t)$, which becomes part of the *next* state's
target, and the inflation propagates backward through the chain of states. And here is why a bias that
looks small per step is not small in aggregate: if every state's max is inflated by roughly a constant $b$,
then the value at a state accumulates the discounted inflation of all the states downstream of it,
$b+\gamma b+\gamma^2 b+\cdots=b/(1-\gamma)$, which at $\gamma=0.99$ is $100\,b$. A per-step overestimate of a
tenth of a point becomes a ten-point distortion in the learned values. Over 200M frames on a function
approximator whose errors are large early and never zero, this is not a rare event that averages out; the
$\max$ re-inflates every update, the discounting sums it up the chain, and it is the steady state.

There is a way to sanity-check that this is really happening and not a theoretical worry, and it falls out
of the floor's own reward clipping. With rewards clipped to $\{-1,0,+1\}$ and $\gamma=0.99$, the largest
return the agent can *actually* receive is bounded — a run of all-$+1$ rewards sums to at most
$1/(1-\gamma)=100$, and real games are nowhere near that dense, so achievable $Q$ values live in a modest
range. If the learned $Q$ values instead drift well above what any policy can achieve, that gap is not
signal — it is precisely the accumulated overestimation, made visible because clipping fixed the scale of
what is achievable. So the disease is not only derivable, it is observable as value estimates that exceed
the clipped-return ceiling, and fixing the max should pull those estimates back toward the achievable range
at the same time as it improves the policy.

I want this quantitative, not just "biased up," and I want to know *which* games it hurts. Take the
cleanest stress state: all true action values tied, $Q_*(s,a)=V_*(s)$ for every $a$ — the worst case,
because there is genuinely nothing to choose and any apparent winner is pure noise. Let the estimation
errors be balanced, $\sum_a\epsilon_a=0$, with mean-squared spread $\frac1m\sum_a\epsilon_a^2=C>0$ over
$m\ge2$ actions. Then $\max_a Q(s,a)\ge V_*(s)+\sqrt{C/(m-1)}$, and it is tight. Let me verify the tightness construction
rather than assert it: put $\epsilon_a=\sqrt{C/(m-1)}$ on $m-1$ of the actions and
$\epsilon_a=-\sqrt{(m-1)C}$ on the last. The balance constraint holds — the sum is
$(m-1)\sqrt{C/(m-1)}-\sqrt{(m-1)C}=\sqrt{(m-1)C}-\sqrt{(m-1)C}=0$ — and so does the spread constraint,
because $\frac1m\big[(m-1)\cdot\frac{C}{m-1}+(m-1)C\big]=\frac1m[C+(m-1)C]=\frac{mC}{m}=C$. And the max over
these errors is exactly $\sqrt{C/(m-1)}$, since that positive value beats the single negative one. So the
bound is achieved, and *any* balanced error pattern of spread $C$ forces the single-estimator target to
overshoot by at least $\sqrt{C/(m-1)}$, with no independence or distribution assumption — a worst-case
guarantee, not an average-case guess. And the companion typical case — errors i.i.d. uniform on
$[-\epsilon,\epsilon]$ — gives $\mathbb{E}[\max_a\epsilon_a]=\epsilon\frac{m-1}{m+1}$, which *increases*
with the action count $m$. Put numbers on it: at $m=2$ actions the expected overestimate is $\epsilon/3$;
at $m=6$ it is $\frac{5}{7}\epsilon\approx0.71\epsilon$; at the full $m=18$-action games it is
$\frac{17}{19}\epsilon\approx0.89\epsilon$, nearly the entire noise amplitude. The inflation is small on the
few games with a handful of actions and large on the many games with the full button set — and since a
large share of the $57$ have big action sets, this is a defect that touches most of the suite, hardest
where the action count is biggest. That
is the cross-suite lever I was looking for: the inflation is worst on games with the largest action sets,
and it is present on every game with more than one action — exactly the kind of broad, systematic damage
that moves a median.

The worry is not just that values are wrong. If the overestimation were *uniform* — every action lifted by
the same constant — the $\arg\max$ would not change and the policy would be fine. But there is no reason it
is uniform: it depends on the action count, on the shape of the per-state errors, on which states got more
data. A non-uniform additive distortion scrambles the *relative* ordering of actions, and the greedy
policy reads off nothing but that ordering. Make it concrete: two next-states $s'_A$ and $s'_B$ with true
values $V_*(s'_A)=1.0$ and $V_*(s'_B)=1.1$, so a correct agent prefers $s'_B$. But $s'_A$ has $18$ actions
and $s'_B$ has $4$; by the $\epsilon\frac{m-1}{m+1}$ arithmetic the max at $s'_A$ is inflated by roughly
$0.89\epsilon$ and at $s'_B$ by only $0.71\epsilon$, so with, say, $\epsilon=0.3$ the estimated values
become $1.0+0.27=1.27$ for $s'_A$ and $1.1+0.21=1.31$ for $s'_B$ — order preserved here, but nudge the
true gap smaller or the action-count gap larger and the inflation on the many-action state overtakes the
genuinely-better few-action state, and the agent walks toward the wrong one. The bias is not a harmless
constant offset I could ignore; because it is *action-count-dependent*, it distorts comparisons *between*
states unequally, and those comparisons are the policy. So I am propagating wrong relative judgments about
which states are worth more — that can absolutely produce a worse policy across many games at once.

Now where does the fix live? Rewrite the target to expose its structure:
$\max_a Q(S_{t+1},a;\theta^-)=Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta^-);\theta^-\big)$. Written this
way the disease is obvious — one set of numbers does two jobs: it *selects* the greedy next action (the
inner $\arg\max$) and *evaluates* it (the outer $Q$), both on $\theta^-$. The action I pick is by
construction the one whose $\theta^-$-estimate is largest — the one whose noise is most positive — and then
I read off that same inflated estimate as its value. Selection and evaluation are perfectly correlated in
their error, and that correlation is what turns "noisy" into "biased high." If I select an action *because*
its estimate is highest, I should not trust that same estimate to tell me its worth; I have conditioned on
it being large.

So evaluate with a *different* set of estimates. If I had a second value function $\theta'$ whose errors
were independent of $\theta$'s, I could select with $\theta$ and evaluate with $\theta'$:
$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta'\big)$. The action chosen by
$\theta$ is some particular $a^\star$; $\theta'$'s error on $a^\star$ is independent of *why* $a^\star$ was
selected, so the evaluation is not conditioned-on-being-large — it is unbiased for $Q_*(S_{t+1},a^\star)$.
In the all-tied stress state, $a^\star$ is whichever action $\theta$ inflated most, but $\theta'$ neither
knows nor cares, so its expected value is $V_*(S_{t+1})$ — no upward bias. Where the single max was forced
to overshoot by $\sqrt{C/(m-1)}$, the decoupled estimator's floor is zero. Let me trace one concrete state
to see the mechanism bite. Four actions, all truly worth $0$, and $\theta$'s errors happen to be
$(+0.9,-0.3,-0.3,-0.3)$ — balanced (they sum to zero). The single-estimator target reads
$\max=+0.9$: it selects action $1$ *and* reports $0.9$ as its value, so the target overshoots the true $0$
by nine tenths of a point. Now decouple: $\theta$ still selects action $1$ (its estimate is highest), but I
read that action's value off $\theta'$, whose error on action $1$ is some *independent* draw with mean $0$
— on average it reports $0$, the true value, because $\theta'$ has no idea action $1$ was chosen *for*
being large. The overshoot collapses from $0.9$ to $0$ in expectation. The selection kept the inflated
choice; the independent evaluation refused to pay the inflated price. That is the whole idea in one state.
(This is the two-estimator idea of van Hasselt, 2010: two tables, one to pick, one to score.)

Before I reach for the target network I should weigh the honest alternatives for getting a second
estimator, because the choice is not forced. I can think of three. First, the literal two-estimator recipe:
train and store a whole second network $\theta'$ with its own independent initialization, and on each
update randomly assign the transition to update one of the two while using the other to evaluate, swapping
roles symmetrically. This gives genuinely decorrelated errors — the ideal $\theta'$ — but it doubles the
parameter count from about $1.7$M to $3.4$M and, more to the point, it doubles the learning signal split
across two networks so each sees half the updates, and it muddies the question I actually want answered,
which is whether *just* fixing the max helps, not whether "DQN plus a second network trained differently"
helps. Second, I could try to *estimate and subtract* the bias as a correction constant — but the
overestimate $\sqrt{C/(m-1)}$ depends on the per-state error spread $C$, which I do not know and which
varies by state and over training, so any constant I subtract is wrong almost everywhere and could easily
overcorrect into a downward bias. Third, I could average an ensemble of heads to shrink the noise variance
$C$ before the max — but that attacks the *size* of the noise, not the *selection bias*, and even with
small $C$ the max still preferentially picks positive noise; halving $C$ only shrinks the overshoot by
$\sqrt2$, it never removes it. The averaging alternative treats a symptom; the two-estimator idea treats
the disease, which is the *coupling* between selection and evaluation, not the magnitude of the noise. So
the decoupling is the right mechanism — the only question is where to get the second estimator cheaply.

The literal two-estimator recipe's machinery muddies the isolation I want. But the floor already hands me a
second set of weights: the target network $\theta^-$, a frozen copy of $\theta$, sitting there so the
regression target holds still. So let $\theta$ do the selection and $\theta^-$ the evaluation:
$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big)$. The online net picks the
greedy next action; the target net scores it. Compared to plain DQN — both jobs on $\theta^-$ — the *only*
change is whose $\arg\max$ I use: $\theta$ instead of $\theta^-$. No new network, no new parameters, no
extra forward pass beyond one I am essentially already doing, replay/$\epsilon$-greedy/target-sync all
untouched. This is the smallest edit that introduces the decoupling, which is exactly what makes it the
right next rung: it isolates the effect of the decoupled target from everything else, so the change in
median HNS is attributable to fixing the max and to nothing else. Everything the floor built stays exactly
as it was: the replay buffer, the uniform sampling, the reward clipping to $\{-1,0,+1\}$, the Huber/clipped
TD error with its unit threshold, the $\epsilon$-greedy schedule, the target-sync period — all untouched.
The only line that changes is which network supplies the inner $\arg\max$ in the target: $\theta^-\to\theta$
for selection, $\theta^-$ retained for evaluation. The extra cost is a single forward pass of the online
net on the next observations to get its $\arg\max$ — the online net is already being evaluated on the
*current* observations for the prediction, so this adds one more evaluation on the *next* observations, a
conv forward pass that is negligible against the backward pass it shares a step with. That the change is a
single line is not a cosmetic virtue; it is what guarantees the experiment answers the exact question I
posed — does decoupling the max move the median — with no confound from any other altered piece.

One more property I want to check before I trust this as a clean single-axis change: does it alter the
*policy class* or only the *learning*? The agent still acts $\epsilon$-greedily on the online net's scalar
$Q$, and it still bootstraps toward a scalar target — the object being learned is the same mean-return
scalar, the acting rule is the same $\arg\max$, and the only thing that moved is how the *target* for that
scalar is constructed. So this is not a new kind of agent; it is the same agent estimating the same quantity
with a less biased target. That is precisely the property that makes any shift in median HNS attributable to
the decoupling and not to a changed objective — the same discipline I will want to hold to on every rung, so
that the ladder measures one mechanism at a time. If I had also changed how the agent acts, I could not say
whether a gain came from the better target or the different behavior.

I should be honest about how much it buys. $\theta$ and $\theta^-$ are not independent the way the
idealized $\theta'$ was — $\theta^-$ is a *stale copy* of $\theta$, so their errors are correlated, and
right after a sync $\theta^-=\theta$ exactly and the target reverts to a plain max for that interval. So
this removes much, not all, of the bias. But the staleness gives partial decoupling: between syncs
$\theta$ has moved on while $\theta^-$ has not, so the action current-$\theta$ thinks is best is no longer
guaranteed to be the one $\theta^-$ inflated, and that is enough to take a real bite out of the
overestimation. How much decoupling the staleness actually provides scales with the sync period: the longer $\theta$ has
drifted since the last copy, the more the action it now favors can differ from the one $\theta^-$ inflated,
so more of the bias is removed — but right after each sync the two coincide and the bias is momentarily
back at full strength, then decays over the interval until the next copy resets it. So the effect is a
sawtooth: strongest just before a sync, zero just after, and what I buy on average is the area under that
sawtooth. It is partial by construction, which is exactly why I should expect a real but bounded gain
rather than a total elimination of overestimation.

The reason I am confident this moves the *median* and not just a few tails is the cross-suite arithmetic I
did earlier: the overshoot $\epsilon\frac{m-1}{m+1}$ is present on every game with more than one action and
grows toward the full noise amplitude as the action count rises, so it is a broad tax the floor was paying
on most of the $57$, hardest on the large-action games. Fixing it therefore lifts the broad middle of the
distribution, not a handful of outliers — the signature of a change that moves a median. The fix is broad,
it helps most where the action count is largest, and it is free. The cost is that I have only changed the
*target*; the value object is still a single scalar per action, exploration is still $\epsilon$-greedy,
replay is still uniform. Those are the remaining levers. But fixing the systematic upward bias in the one
update I run on all 57 games is the highest-leverage single change to the floor, so I expect a clear, broad
lift in the median — the first real climb off 79%, and enough of one that the typical game should cross the
$100\%$ human line rather than merely inch up. If instead the median barely moves, that would tell me the
overestimation was not actually the binding constraint on the typical game and I would have to look
elsewhere; a broad jump is the falsifiable prediction this rung stakes.
