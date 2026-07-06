The scaffold hands me a frozen Nature-DQN encoder feeding a single linear head, a uniform replay
buffer, $\epsilon$-greedy acting, and a periodic target copy, and asks me to fill in the value head and
its TD update. The floor here — the thing the whole ladder reacts to — is ordinary deep Q-learning, so
that is where I have to start, but I want to start by being honest about its one structural disease,
because diagnosing it is what tells me the minimal first edit. Let me instrument the obvious thing.
During training I freeze the greedy policy and roll it out, and I record two numbers: the value the
network *predicts* for that policy, averaged over the states it visits, and the *actual* discounted
return that same policy collects. If the value function were any good these two would sit on top of
each other — the predicted value of a policy is, by definition, the return you expect from running it.
Instead the predicted value keeps climbing well above the realized return, and on the harder games it
doesn't just sit high, it can run away while the score stagnates or falls. So this is not harmless
calibration error. The estimates are biased upward, the bias is large, and it coincides with the policy
failing to improve.

Where could a systematic *upward* bias come from? Not from the optimizer — least-squares regression
toward a target is unbiased for the target. So the bias has to be in the target itself. Write down what
DQN regresses onto. For a transition $(S_t,A_t,R_{t+1},S_{t+1})$ the Q-learning target is
$Y_t = R_{t+1} + \gamma \max_a Q(S_{t+1},a;\theta^-)$, where $\theta^-$ is the frozen target network the
scaffold already maintains. The reward is what it is; $\gamma$ is a constant. The only thing in that
expression that is a function of my noisy estimates is $\max_a Q(S_{t+1},a;\theta^-)$. So the bias, if
there is one, lives in that max.

What does the max do to estimation error? Say the true optimal value at $s'=S_{t+1}$ is $Q_*(s',a)$ and
my network gives noisy estimates $Q(s',a)=Q_*(s',a)+\epsilon_a$. Suppose for the moment the errors are
zero-mean for each action, so my per-action estimates are individually unbiased. Is
$\max_a Q(s',a)$ then an unbiased estimate of $\max_a Q_*(s',a)$? No — because $\max$ is convex. For any
convex $f$ and random vector $X$, Jensen gives $\mathbb{E}[f(X)]\ge f(\mathbb{E}[X])$, and the pointwise
maximum is convex in its arguments, so
$\mathbb{E}[\max_a(Q_*(s',a)+\epsilon_a)] \ge \max_a(Q_*(s',a)+\mathbb{E}[\epsilon_a]) = \max_a Q_*(s',a)$.
The expected max of my noisy values is at least the max of the true values. Even with perfectly unbiased
per-action estimates the bootstrap target is biased upward. Intuitively the max goes hunting through all
$m$ actions and selects the largest estimate, which preferentially picks the action whose noise landed
on the high side; positive noise gets selected, negative noise gets discarded. Then I bootstrap: this
inflated target is regressed into $Q(S_t,A_t)$, which becomes part of the next state's target, and the
inflation propagates backward through the chain of states. That is the runaway I see in the curves.

And "runaway" is not hyperbole — the discount tells me how badly a per-step bias compounds. Suppose the
max injects a roughly constant upward bias $b$ into every state's target. Bootstrapping means the bias
at one state feeds the target of its predecessor, discounted by $\gamma$: the accumulated inflation of a
value satisfies, to first order, $\Delta \approx b + \gamma\,\Delta$, whose fixed point is
$\Delta \approx b/(1-\gamma)$. With the scaffold's $\gamma=0.99$ that is a multiplier of
$1/(1-\gamma)=100$. So a per-step overestimate of even a fraction of a point, iterated through the
bootstrap, can settle into a predicted value that sits *tens of points* above the realizable return —
which is exactly the shape of the divergence I instrumented, the prediction climbing far past the score
rather than tracking it. The discount that makes long-horizon credit assignment possible is the same
factor that amplifies a small target bias into a large value bias, and the longer-horizon games have the
most steps over which it can accumulate.

I want a feel for how the size of this scales with the action count, because the three games differ
exactly there — Breakout has four actions, Pong six, Seaquest eighteen. Take the tractable typical case:
per-action errors i.i.d. uniform on $[-\epsilon,\epsilon]$, so each action carries the *same* noise
scale regardless of how many actions there are. The expected max of $m$ such draws works out to
$\mathbb{E}[\max_a\epsilon_a]=\epsilon\frac{m-1}{m+1}$. Put the numbers in: $m=2$ gives $0.333\,\epsilon$,
$m=4$ (Breakout) gives $0.600\,\epsilon$, $m=6$ (Pong) gives $0.714\,\epsilon$, and $m=18$ (Seaquest)
gives $0.895\,\epsilon$. So holding per-action noise fixed, going from Breakout's four actions to
Seaquest's eighteen lifts the expected inflation by a factor $0.895/0.600\approx1.5$ — half again as
much overestimation baked into every Seaquest bootstrap. The max is taken over more independent noisy
draws, so its expectation creeps toward the top of the noise range. This is the first quantitative
reason to expect Seaquest to be the game where this disease bites hardest.

Before I call this fatal there is a counterargument I half-believe. If the overestimation were
*uniform* — every action in every state lifted by the same constant — then $\arg\max_a Q(s,a)$ wouldn't
change at all, since adding a constant preserves the ordering; the values would be wrong but the policy
fine. And optimism about uncertain things is a respectable exploration strategy. So why worry? Two
reasons, and they are exactly what separates this from benign optimism. First, this bias appears *after*
I have updated, baked into values I now treat as settled — overoptimism in the face of apparent
certainty, not optimism before acting. Second, and decisively, there is no reason the bias is uniform:
the $\epsilon\frac{m-1}{m+1}$ arithmetic just showed it depends on how many actions there are, and it
depends further on the shape of the per-state errors and on which states got more data. A state with
eighteen noisy actions gets a bigger upward kick from the max than a state with four; a poorly-fit
region gets a bigger kick than a well-fit one. So the overestimate is state- and action-dependent, and a
non-uniform additive distortion scrambles the *relative* ordering, which is the only thing the greedy
policy reads off. Combined with bootstrapping I am propagating wrong relative judgments about which
states are worth more. That can absolutely produce a worse policy.

I want this quantitative in the other direction too — not just "typically inflated" but "provably
inflated even in the least favorable arrangement." Take the cleanest stress state: all true action
values equal, $Q_*(s,a)=V_*(s)$ for every $a$ — the worst case for "the max manufactures value out of
noise," because there is genuinely nothing to choose between. Let the estimates be balanced around the
truth, $\sum_a(Q_t(s,a)-V_*(s))=0$, with a fixed mean-squared error
$\frac1m\sum_a(Q_t(s,a)-V_*(s))^2=C>0$, $m\ge2$. Define $\epsilon_a=Q_t(s,a)-V_*(s)$, so
$\sum_a\epsilon_a=0$ and $\sum_a\epsilon_a^2=mC$. I claim $\max_a\epsilon_a\ge\sqrt{C/(m-1)}$. Argue by
contradiction: suppose $\max_a\epsilon_a<\sqrt{C/(m-1)}$. At least one error is non-positive (if all
were positive, the zero-sum forces all zero, contradicting nonzero spread), so the number of strictly
positive errors is $n\le m-1$. Each positive error is below the max, so the positive mass
$\sum\epsilon^+_i< n\sqrt{C/(m-1)}$; the zero-sum makes the negative mass equal to it, so each negative
magnitude is also bounded by $n\sqrt{C/(m-1)}$. Converting magnitudes to squares via the $\ell_1/\ell_\infty$
bound, the positive squared mass is $<n\cdot C/(m-1)$ and the negative is $<n^2C/(m-1)$, summing to
$<\frac{n(n+1)}{m-1}C\le\frac{(m-1)m}{m-1}C=mC$ since $n\le m-1$. That contradicts
$\sum_a\epsilon_a^2=mC$. So $\max_a Q_t(s,a)\ge V_*(s)+\sqrt{C/(m-1)}$, and it is tight: put
$\epsilon_a=\sqrt{C/(m-1)}$ on $m-1$ actions and $-\sqrt{(m-1)C}$ on the last, and both constraints hold
with the max exactly at the bound. So *any* balanced error pattern with spread $C$ forces the
single-estimator target to overshoot by at least $\sqrt{C/(m-1)}$, with no independence or distribution
assumption — and early in training the estimates are always wrong, so this is the norm, not a corner
case.

Now there is a subtlety here I should not paper over, because the two calculations point opposite ways in
$m$ and I want to be sure I am reading them right. The typical-case inflation $\epsilon\frac{m-1}{m+1}$
*grows* with the action count, while the worst-case floor $\sqrt{C/(m-1)}$ *shrinks* with it: at fixed
per-action spread the floor is $0.577\sqrt C$ for four actions, $0.447\sqrt C$ for six, $0.243\sqrt C$
for eighteen. These are not in conflict — they answer different questions. The floor is the *minimum*
overshoot achievable over all balanced patterns of a given total spread; with more actions the mass can
be dispersed into many small positive errors, so the guaranteed-unavoidable overshoot can be made
smaller. The typical case is the *average* overshoot when the errors are independent draws of a fixed
scale, which grows because a max of more independent things creeps higher. Read together they say
exactly the two things I need: overestimation is *unavoidable* (bounded below by a strictly positive
floor even in the most favorable arrangement), and in the realistic regime of roughly-independent
per-action noise it gets *typically worse as the action count grows*. Seaquest's eighteen actions are
where both the "it happens at all" and the "it happens more" land hardest.

Now the fix has to live where the disease lives. Rewrite the target as
$\max_a Q(S_{t+1},a;\theta^-) = Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta^-);\theta^-\big)$. Written
this way the disease is obvious: the max does two jobs with one set of numbers — it *selects* the
greedy action (the inner $\arg\max$) and *evaluates* it (the outer $Q$), both on $\theta^-$. The action
I pick is by construction the one whose $\theta^-$-estimate is largest — the one whose noise is most
positive — and then I read off that same inflated estimate as its value. Selection and evaluation are
perfectly correlated in their error, and that correlation is what turns "noisy" into "biased high." If
I select an action *because* its estimate is highest, I should not trust that same estimate to tell me
its value; I have conditioned on it being large.

What if I evaluated with a *different* set of estimates? Suppose I had a second value function $\theta'$
whose errors are independent of $\theta$'s. Select with $\theta$, evaluate with $\theta'$:
$Y_t=R_{t+1}+\gamma\,Q(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta')$. The action chosen by $\theta$
is some particular $a^\star$, and $\theta'$'s error on $a^\star$ is independent of *why* $a^\star$ was
selected, so the evaluation is not conditioned-on-being-large — it is an unbiased estimate of
$Q_*(S_{t+1},a^\star)$. In the all-equal stress state, $a^\star$ is whichever action $\theta$ inflated
most, but $\theta'$ doesn't know or care, so its expected value is $V_*(S_{t+1})$ — no upward bias. In
the tight example above I can set the second estimator on the selected action to exactly $V_*$, so the
floor on the decoupled estimate's error is *zero* where the single max is forced to overshoot by
$\sqrt{C/(m-1)}$. Decoupling selection from evaluation is the cure. (This is the idea behind the tabular
two-estimator scheme of van Hasselt, 2010: two tables, each experience updates one at random, one picks
and the other scores.)

Let me make that concrete on one transition, because the abstract argument is easy to nod along with and
I want to see the numbers move. Take a next state $s'$ where the truth is all-equal, $Q_*(s',a)=0$ for
every action, and $m=4$. Say the online net's noisy estimates come out
$Q(s',\cdot;\theta)=[+0.8,\,-0.3,\,+0.1,\,-0.6]$, and the stale target net, being a different snapshot,
gives $Q(s',\cdot;\theta^-)=[-0.2,\,+0.5,\,0.0,\,+0.3]$. Plain DQN takes $\max_a Q(s',a;\theta^-)=+0.5$
(action 1) — an overshoot of $+0.5$ above the true $0$. The decoupled target instead lets $\theta$
select: $\arg\max_a Q(s',a;\theta)=$ action $0$ (its $+0.8$), and evaluates *that* action with
$\theta^-$, reading $-0.2$. So the decoupled target is $-0.2$, an *undershoot* of $0.2$ — a magnitude of
$0.2$ against the plain max's $0.5$. The mechanism is exactly what the algebra promised: $\theta^-$'s
error on action $0$ ($-0.2$) has nothing to do with why $\theta$ liked action $0$ (its $+0.8$), so the
evaluation is not conditioned on being large, and averaged over the noise it sits near the true $0$
rather than reaching for the top of the pile. One draw proves nothing on its own, but it shows the two
targets are computed from genuinely different quantities and that the decoupled one is not systematically
grabbing the inflated estimate.

The literal two-estimator recipe would double the value functions — train and store a whole second
network and arrange symmetric random-assignment updates. That is a lot of machinery to bolt onto a
delicate system, and it muddies the comparison: I want to know whether *just* fixing the max helps, not
whether "DQN plus a second network" helps. Before I settle on the cheaper route let me be honest about
what else is on the table at this first rung, because the background hands me two other named heads and I
should say why neither is the first move. A dueling head splits the output into a state-value stream
$V(s)$ and an advantage stream $A(s,a)$ and recombines them — a real improvement, but it targets a
*different* pathology (letting the net learn a state is valuable without committing to every per-action
value), and it does nothing about the selection-evaluation correlation inside the max; it also restructures
the head and adds streams, which spends my capacity budget on something orthogonal to the bias I just
diagnosed. A fixed-atom distributional head would model the whole return law, which is a genuinely larger
change I can see coming, but it drags in a support range and a projection and it is far from the *minimal*
edit — I would be changing what the critic represents before I have even removed the point-estimate bias
that corrupts the mean. Both are premature. The disease I measured is the max, and the smallest thing
that touches the max is what I want first.

And the scaffold already hands me a second set of weights for free: the target network $\theta^-$, a
frozen copy of $\theta$, sitting there so the regression target doesn't move every step. So let $\theta$
do the selection and $\theta^-$ the evaluation:
$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big)$. The online net picks
the greedy action; the target net scores it. Compared to plain DQN — selection and evaluation both on
$\theta^-$ — the *only* change is whose argmax I use, $\theta$ instead of $\theta^-$. No new network, no
new parameters, and the target-sync rule stays exactly as the scaffold sets it. The one thing it does
cost is a forward pass: plain DQN needs $\theta^-$ on the next states (for the max) and $\theta$ on the
current states (for the prediction); the decoupled target additionally needs $\theta$ on the next states
to compute its argmax, so I pay one extra online forward per update — three encoder passes where DQN
does two, on a shared frozen encoder, which is negligible. This is the smallest possible edit that
introduces the decoupling, which is exactly what makes it the right first rung: it isolates the effect of
the decoupled target from everything else.

I should be honest about how much this buys. $\theta$ and $\theta^-$ are not independent the way the
idealized $\theta'$ was — $\theta^-$ is a stale copy of $\theta$, so their errors are correlated, and
right after a target refresh $\theta^-=\theta$ exactly and the whole thing collapses back to plain
Q-learning for that interval. Let me quantify the staleness the scaffold gives me, because it sets how
much decoupling I actually get. The target syncs every `target_network_frequency = 1000` *environment*
steps, and the loop trains once every four environment steps, so between two hard copies the online net
takes $1000/4 = 250$ gradient steps while $\theta^-$ sits frozen. Only the single update immediately
after a copy is genuinely $\theta^-=\theta$ (pure Q-learning); the other $\approx249$ of every $250$
updates run against a $\theta^-$ that lags by up to $250$ steps of drift. So the decoupling is *active*
for the overwhelming majority of updates, and its strength grows across each window as $\theta$ walks
away from $\theta^-$ — the action the current $\theta$ thinks is best is decreasingly likely to be the
one $\theta^-$ inflated. That is enough to take a real bite out of the overestimation, though not to kill
it, since the correlation is only partial and resets every window. The lever to push it further is
obvious from the analysis: widen the gap between target copies so $\theta$ and $\theta^-$ drift apart and
spend even less time freshly-synced. The scaffold fixes that gap with a hard copy (`tau = 1.0`, so the
soft-update formula degenerates to an outright copy every 1000 steps), and I keep it — I am not tuning
the decoupling knob here, only installing the decoupling itself.

Now the concrete fill, because the edit surface is specific and I must match it, not a generic paper
harness. The head stays the scaffold default: the fixed `NatureDQNEncoder` → 512 features → a single
`nn.Linear(512, n_actions)`, no change to the architecture, no capacity added. It is worth doing the
budget arithmetic once so I know I am not sneaking capacity in: that head is $512\cdot|\mathcal A|+|\mathcal A|
=513\,|\mathcal A|$ parameters — $2052$ for Breakout's four actions, $3078$ for Pong's six, $9234$ for
Seaquest's eighteen — and since I keep the scaffold's default head unchanged, this is *identical* to the
baseline head the budget check measures against, so I am trivially inside the $1.05\times$ bound on all
three games. The contribution cannot be capacity; it is entirely the target construction. The
`select_action` is greedy on those Q-values (the loop handles $\epsilon$ externally). The whole edit is
in `update`: under `no_grad`, the **online** network's argmax over $Q(S_{t+1},\cdot)$ selects the next
action, the **target** network gathers that action's value, and
$y=R_{t+1}+\gamma\,Q(\cdot;\theta^-)(1-\text{done})$. Then I regress $Q(S_t,A_t;\theta)$ onto $y$ with
the mean-squared error the loop expects, step Adam at the scaffold's `learning_rate = 1e-4`, and on
every `target_network_frequency`-th step do the hard target copy via the soft-update formula at
$\tau=1$. One detail I respect from the harness rather than the textbook: the loss is plain `mse_loss`
on the TD error, not a Huber — the scaffold's protocol and budget check are built around the squared
loss, and the only thing I am changing relative to DQN is the selection net inside the target. (The
full scaffold module is in the answer.)

Now what should this floor do, because that is the point of running it. Decoupling selection from
evaluation should pull the predicted values back down toward the realized returns and, because the bias
it removes is the *non-uniform* part that corrupts relative ordering, should give a steadier, slightly
better greedy policy than plain DQN across all three games — most visibly where the action count is
largest and the inflation worst (Seaquest, eighteen actions, the $0.895\,\epsilon$ end of that
typical-case arithmetic), least where the game is near-trivial to saturate (Pong, which any working
value learner should drive close to its ceiling). What it cannot do is enrich the value object: $Q$ is
still a single scalar per action, the mean of a return distribution it never sees, and the target is
still a point estimate with no notion of its own spread. So I expect this to be a *correct but modest*
rung — a real reduction in overestimation, a real but not dramatic score, with Pong essentially solved,
Breakout in the low hundreds, and Seaquest the most variable because its long horizon and large action
set leave the most residual bias and the most room for the relative-ordering errors to matter. That
ceiling — a scalar-mean critic with a decoupled but still correlated target — is exactly what the next
rung will have to break by learning the whole return distribution rather than its mean.
