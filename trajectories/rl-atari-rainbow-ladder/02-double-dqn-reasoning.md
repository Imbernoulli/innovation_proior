The DQN floor lands at 79% median human-normalized score. Read literally: on the median game the agent
closes $79\%$ of the gap between random play and the human reference, so it is genuinely learning — a working
floor, not a broken one — but still $21$ points under the $100\%$ line on the typical title. The floor did
its job of being stable everywhere; the job now is to lift the middle of the distribution. The median is not
moved by squeezing more out of games the agent already wins; it is moved by fixing whatever quietly degrades
the agent across many games at once. So I want a defect baked into the update I run on all 57, not one
specific to a few titles. Exploration — the $\epsilon$-greedy dithering — is crude, but its failures are
concentrated on the minority of games needing long deliberate action sequences, so fixing it would lift a
few tails and leave the median where it is. The scalar value object is a deep issue but a representational
overhaul, and I want the cheap high-leverage fixes first. Uniform replay is a candidate, but it is at worst
an inefficiency, not *provably* wrong. The most suspect piece is the bootstrap target itself: it is computed
the same way on all $57$ games, so any bias in it is a tax on the whole suite, and I already flagged it as
theoretically broken. Look hard at it.

DQN regresses $Q(S_t,A_t;\theta)$ onto $Y_t=R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta^-)$. The reward and
$\gamma$ are fixed; the only thing in that target that is a function of my noisy estimates is
$\max_a Q(S_{t+1},a;\theta^-)$. Does the max bias it? Suppose my per-action estimates are individually
*unbiased* — $Q(s',a)=Q_*(s',a)+\epsilon_a$ with each $\epsilon_a$ zero-mean. Is $\max_a Q(s',a)$ then an
unbiased estimate of $\max_a Q_*(s',a)$? No, because $\max$ is convex, and Jensen runs the wrong way:
$\mathbb{E}[\max_a(Q_*(s',a)+\epsilon_a)]\ge\max_a Q_*(s',a)$. The $\max$ hunts across all the actions and
selects the largest estimate, which preferentially picks whichever action's noise landed on the high side —
positive noise gets selected, negative discarded. Then I bootstrap that inflated value into $Q(S_t,A_t)$,
which becomes part of the *next* state's target, and the inflation propagates backward. Here is why a bias
that looks small per step is not small in aggregate: if every state's max is inflated by roughly a constant
$b$, the value at a state accumulates the discounted inflation of all downstream states,
$b+\gamma b+\gamma^2 b+\cdots=b/(1-\gamma)$, which at $\gamma=0.99$ is $100\,b$. A per-step overestimate of a
tenth of a point becomes a ten-point distortion. On a function approximator whose errors are large early and
never zero, the $\max$ re-inflates every update and the discounting sums it up the chain; this is the steady
state, not a rare event that averages out. And it is observable, thanks to the floor's own reward clipping:
with rewards clipped to $\{-1,0,+1\}$ and $\gamma=0.99$ the largest achievable return is $1/(1-\gamma)=100$,
so learned $Q$ values that drift well above that ceiling are precisely the accumulated overestimation made
visible.

I want this quantitative, and I want to know *which* games it hurts. Take the cleanest stress state: all
true action values tied, $Q_*(s,a)=V_*(s)$ for every $a$ — the worst case, because there is genuinely
nothing to choose and any apparent winner is pure noise. With balanced errors, $\sum_a\epsilon_a=0$ and
mean-squared spread $\frac1m\sum_a\epsilon_a^2=C>0$ over $m\ge2$ actions, one has
$\max_a Q(s,a)\ge V_*(s)+\sqrt{C/(m-1)}$, tight (achieved by putting $\sqrt{C/(m-1)}$ on $m-1$ actions and
$-\sqrt{(m-1)C}$ on the last). So *any* balanced error pattern of spread $C$ forces the single-estimator
target to overshoot by at least $\sqrt{C/(m-1)}$ — a worst-case guarantee, no independence assumption. The
companion typical case — errors i.i.d. uniform on $[-\epsilon,\epsilon]$ — gives
$\mathbb{E}[\max_a\epsilon_a]=\epsilon\frac{m-1}{m+1}$, which *increases* with the action count $m$: at
$m=2$ it is $\epsilon/3$; at $m=6$, $\frac57\epsilon\approx0.71\epsilon$; at the full $m=18$-action games,
$\frac{17}{19}\epsilon\approx0.89\epsilon$, nearly the entire noise amplitude. The inflation is small on the
few games with a handful of actions and large on the many with the full button set. That is the cross-suite
lever: a defect present on every game with more than one action, hardest where the action count is biggest —
exactly the kind of broad, systematic damage that moves a median.

The worry is not just that values are wrong. If the overestimation were *uniform* — every action lifted by
the same constant — the $\arg\max$ would not change and the policy would be fine. But it is not uniform; it
depends on the action count and the per-state error shape. A non-uniform additive distortion scrambles the
*relative* ordering of actions, and the greedy policy reads off nothing but that ordering. Concretely: two
next-states with true values $V_*(s'_A)=1.0$ and $V_*(s'_B)=1.1$, so a correct agent prefers $s'_B$; but
$s'_A$ has $18$ actions and $s'_B$ has $4$, so by the $\epsilon\frac{m-1}{m+1}$ arithmetic the max at $s'_A$
is inflated by $\approx0.89\epsilon$ and at $s'_B$ by $\approx0.71\epsilon$. With $\epsilon=0.3$ the
estimates become $1.27$ and $1.31$ — order preserved here, but shrink the true gap or widen the action-count
gap and the inflation on the many-action state overtakes the genuinely-better few-action state, and the
agent walks toward the wrong one. Because the bias is *action-count-dependent* it distorts comparisons
*between* states unequally, and those comparisons are the policy.

Now where does the fix live? Rewrite the target to expose its structure:
$\max_a Q(S_{t+1},a;\theta^-)=Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta^-);\theta^-\big)$. Written this
way the disease is obvious — one set of numbers does two jobs: it *selects* the greedy next action (the
inner $\arg\max$) and *evaluates* it (the outer $Q$), both on $\theta^-$. The action I pick is by
construction the one whose $\theta^-$-estimate is largest — whose noise is most positive — and then I read
off that same inflated estimate as its value. Selection and evaluation are perfectly correlated in their
error, and that correlation is what turns "noisy" into "biased high." If I select an action *because* its
estimate is highest, I should not trust that same estimate to tell me its worth.

So evaluate with a *different* set of estimates. If I had a second value function $\theta'$ whose errors
were independent of $\theta$'s, I could select with $\theta$ and evaluate with $\theta'$:
$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta'\big)$. The action chosen by
$\theta$ is some $a^\star$; $\theta'$'s error on $a^\star$ is independent of *why* $a^\star$ was selected, so
the evaluation is not conditioned-on-being-large — it is unbiased for $Q_*(S_{t+1},a^\star)$. Where the
single max was forced to overshoot by $\sqrt{C/(m-1)}$, the decoupled estimator's floor is zero. One state
makes it concrete: four actions all truly worth $0$, $\theta$'s errors $(+0.9,-0.3,-0.3,-0.3)$ — balanced.
The single-estimator target selects action $1$ *and* reports its inflated $0.9$ as the value, overshooting
by nine tenths. Decouple: $\theta$ still selects action $1$, but I read its value off $\theta'$, whose
error there is an *independent* mean-zero draw — on average $0$, the true value, because $\theta'$ has no
idea action $1$ was chosen *for* being large. The selection keeps the inflated choice; the independent
evaluation refuses to pay the inflated price. (This is the two-estimator idea of van Hasselt, 2010 — two
tables, one to pick, one to score.)

The alternatives for getting a second estimator all fall short of the target network. A literal second
network with independent initialization gives genuinely decorrelated errors but doubles the parameters and
splits the learning signal, and it muddies the question I actually want answered — whether *just* fixing the
max helps. Estimating and subtracting the bias as a constant fails because $\sqrt{C/(m-1)}$ depends on the
unknown, state-varying spread $C$. Averaging an ensemble to shrink $C$ attacks the *magnitude* of the noise,
not the *selection* bias — even with small $C$ the max still preferentially picks positive noise. The
disease is the *coupling* between selection and evaluation, not the size of the noise, and the floor already
hands me a second set of weights for free: the target network $\theta^-$, a frozen copy of $\theta$.

So let $\theta$ select and $\theta^-$ evaluate:
$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big)$. Compared to plain DQN —
both jobs on $\theta^-$ — the *only* change is whose $\arg\max$ I use: $\theta$ instead of $\theta^-$. No new
network, no new parameters, replay/$\epsilon$-greedy/target-sync all untouched; the extra cost is a single
online-net forward pass on the next observations, negligible against the backward pass it shares a step with.
That the change is one line is what guarantees the experiment answers the exact question — does decoupling
the max move the median — with no confound. And it changes only the *learning*, not the policy class: the
agent still acts $\epsilon$-greedily on the online net's scalar $Q$ and still bootstraps toward a scalar
target, so any shift in median HNS is attributable to the less-biased target and nothing else.

I should be honest about how much it buys. $\theta$ and $\theta^-$ are not independent the way the idealized
$\theta'$ was — $\theta^-$ is a *stale copy* of $\theta$, so their errors are correlated, and right after a
sync $\theta^-=\theta$ exactly and the target reverts to a plain max. So this removes much, not all, of the
bias. The staleness gives *partial* decoupling that grows over the sync interval: the longer $\theta$ has
drifted since the last copy, the more the action it now favors can differ from the one $\theta^-$ inflated,
so the effect is a sawtooth — strongest just before a sync, zero just after — and what I buy on average is
the area under it. It is partial by construction, so I expect a real but bounded gain rather than total
elimination.

The reason I am confident this moves the *median* and not just a few tails is the cross-suite arithmetic:
the overshoot $\epsilon\frac{m-1}{m+1}$ is present on every game with more than one action and grows toward
the full noise amplitude as the action count rises, so it is a broad tax the floor was paying on most of the
$57$, hardest on the large-action games. Fixing it lifts the broad middle, not a handful of outliers — the
signature of a median-moving change, and it is free. The cost is that I have only changed the *target*; the
value object is still a scalar, exploration still $\epsilon$-greedy, replay still uniform, those the remaining
levers. So I expect a clear, broad lift — the first real climb off 79%. If instead the median barely moves,
that would tell me the overestimation was not the binding constraint on the typical game and I would look
elsewhere; a broad jump is what this change stakes.
