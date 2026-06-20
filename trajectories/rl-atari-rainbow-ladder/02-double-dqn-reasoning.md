The DQN floor lands at 79% median human-normalized score — a working agent, but on the typical game it
sits below the human reference, and I have a list of broken pieces to choose from. I want the *one* change
that buys the most on the median game, and the discipline of the median tells me where to look: it is not
moved by squeezing more out of games the agent already wins, it is moved by fixing whatever quietly
degrades the agent across many games at once. So I want a defect that is not specific to one title but is
baked into the update I run on all 57. The most suspect piece of the floor is the bootstrap target itself,
because it is computed the same way everywhere and I already flagged it as theoretically broken. Let me
look hard at it.

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
target, and the inflation propagates backward through the chain of states. Over 200M frames on a function
approximator whose errors are large early and never zero, this is not a rare event; it is the steady state.

I want this quantitative, not just "biased up," and I want to know *which* games it hurts. Take the
cleanest stress state: all true action values tied, $Q_*(s,a)=V_*(s)$ for every $a$ — the worst case,
because there is genuinely nothing to choose and any apparent winner is pure noise. Let the estimation
errors be balanced, $\sum_a\epsilon_a=0$, with mean-squared spread $\frac1m\sum_a\epsilon_a^2=C>0$ over
$m\ge2$ actions. Then $\max_a Q(s,a)\ge V_*(s)+\sqrt{C/(m-1)}$, and it is tight (put
$\epsilon_a=\sqrt{C/(m-1)}$ on $m-1$ actions and $-\sqrt{(m-1)C}$ on the last; both constraints hold with
the max exactly at the bound). So *any* balanced error pattern of spread $C$ forces the single-estimator
target to overshoot by at least $\sqrt{C/(m-1)}$, with no independence or distribution assumption. And the
companion typical case — errors i.i.d. uniform on $[-\epsilon,\epsilon]$ — gives
$\mathbb{E}[\max_a\epsilon_a]=\epsilon\frac{m-1}{m+1}$, which *increases* with the action count $m$. That
is the cross-suite lever I was looking for: the inflation is worst on games with the largest action sets,
and it is present on every game with more than one action — exactly the kind of broad, systematic damage
that moves a median.

The worry is not just that values are wrong. If the overestimation were *uniform* — every action lifted by
the same constant — the $\arg\max$ would not change and the policy would be fine. But there is no reason it
is uniform: it depends on the action count, on the shape of the per-state errors, on which states got more
data. A non-uniform additive distortion scrambles the *relative* ordering of actions, and the greedy
policy reads off nothing but that ordering. So I am propagating wrong relative judgments about which states
are worth more — that can absolutely produce a worse policy across many games at once.

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
to overshoot by $\sqrt{C/(m-1)}$, the decoupled estimator's floor is zero. (This is the two-estimator idea
of van Hasselt, 2010: two tables, one to pick, one to score.)

The literal recipe would train and store a whole second network and arrange symmetric random-assignment
updates — a lot of machinery, and it muddies the question I actually want answered, which is whether *just*
fixing the max helps, not whether "DQN plus a second network" helps. But the floor already hands me a
second set of weights: the target network $\theta^-$, a frozen copy of $\theta$, sitting there so the
regression target holds still. So let $\theta$ do the selection and $\theta^-$ the evaluation:
$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big)$. The online net picks the
greedy next action; the target net scores it. Compared to plain DQN — both jobs on $\theta^-$ — the *only*
change is whose $\arg\max$ I use: $\theta$ instead of $\theta^-$. No new network, no new parameters, no
extra forward pass beyond one I am essentially already doing, replay/$\epsilon$-greedy/target-sync all
untouched. This is the smallest edit that introduces the decoupling, which is exactly what makes it the
right next rung: it isolates the effect of the decoupled target from everything else, so the change in
median HNS is attributable to fixing the max and to nothing else.

I should be honest about how much it buys. $\theta$ and $\theta^-$ are not independent the way the
idealized $\theta'$ was — $\theta^-$ is a *stale copy* of $\theta$, so their errors are correlated, and
right after a sync $\theta^-=\theta$ exactly and the target reverts to a plain max for that interval. So
this removes much, not all, of the bias. But the staleness gives partial decoupling: between syncs
$\theta$ has moved on while $\theta^-$ has not, so the action current-$\theta$ thinks is best is no longer
guaranteed to be the one $\theta^-$ inflated, and that is enough to take a real bite out of the
overestimation. The fix is broad — it touches every game with more than one action and helps most where
the action count is largest — and it is free. The cost is that I have only changed the *target*; the value
object is still a single scalar per action, exploration is still $\epsilon$-greedy, replay is still
uniform. Those are the remaining levers. But fixing the systematic upward bias in the one update I run on
all 57 games is the highest-leverage single change to the floor, so I expect a clear, broad lift in the
median — the first real climb off 79%.
