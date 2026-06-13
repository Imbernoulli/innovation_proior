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

Before I call this fatal there is a counterargument I half-believe. If the overestimation were
*uniform* — every action in every state lifted by the same constant — then $\arg\max_a Q(s,a)$ wouldn't
change at all, since adding a constant preserves the ordering; the values would be wrong but the policy
fine. And optimism about uncertain things is a respectable exploration strategy. So why worry? Two
reasons, and they are exactly what separates this from benign optimism. First, this bias appears *after*
I have updated, baked into values I now treat as settled — overoptimism in the face of apparent
certainty, not optimism before acting. Second, and decisively, there is no reason the bias is uniform:
it depends on how many actions there are, on the shape of the per-state errors, on which states got
more data. A state with eighteen noisy actions (Seaquest) gets a bigger upward kick from the max than a
state with four (Breakout); a poorly-fit region gets a bigger kick than a well-fit one. So the
overestimate is state- and action-dependent, and a non-uniform additive distortion scrambles the
*relative* ordering, which is the only thing the greedy policy reads off. Combined with bootstrapping I
am propagating wrong relative judgments about which states are worth more. That can absolutely produce a
worse policy.

I want this quantitative, not just "biased upward." Take the cleanest stress state: all true action
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
case. (The companion typical-case figure, errors i.i.d. uniform on $[-\epsilon,\epsilon]$, gives
$\mathbb{E}[\max_a\epsilon_a]=\epsilon\frac{m-1}{m+1}$, which *increases* with the number of actions —
so Seaquest's eighteen actions are exactly where I should expect the worst inflation.)

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

The literal two-estimator recipe would double the value functions — train and store a whole second
network and arrange symmetric random-assignment updates. That is a lot of machinery to bolt onto a
delicate system, and it muddies the comparison: I want to know whether *just* fixing the max helps, not
whether "DQN plus a second network" helps. But the scaffold already hands me a second set of weights:
the target network $\theta^-$, a frozen copy of $\theta$, sitting there so the regression target
doesn't move every step. So let $\theta$ do the selection and $\theta^-$ the evaluation:
$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big)$. The online net picks
the greedy action; the target net scores it. Compared to plain DQN — selection and evaluation both on
$\theta^-$ — the *only* change is whose argmax I use, $\theta$ instead of $\theta^-$. No new network, no
new parameters, no extra forward pass beyond one I am essentially already doing, and the target-sync
rule stays exactly as the scaffold sets it. This is the smallest possible edit that introduces the
decoupling, which is exactly what makes it the right first rung: it isolates the effect of the
decoupled target from everything else.

I should be honest about how much this buys. $\theta$ and $\theta^-$ are not independent the way the
idealized $\theta'$ was — $\theta^-$ is a stale copy of $\theta$, so their errors are correlated, and
right after a target refresh $\theta^-=\theta$ exactly and the whole thing collapses back to plain
Q-learning for that interval. So this will not fully kill the bias. But the staleness gives partial
decoupling: between refreshes $\theta$ has moved on while $\theta^-$ has not, so the action the current
$\theta$ thinks is best is no longer guaranteed to be the one $\theta^-$ inflated, and that is enough to
take a real bite out of the overestimation. The lever to push it further is obvious from the analysis:
widen the gap between target copies so $\theta$ and $\theta^-$ drift apart and spend less time
freshly-synced. The scaffold fixes that gap at `target_network_frequency = 1000` with a hard copy
(`tau = 1.0`, so the soft-update formula degenerates to an outright copy every 1000 steps), and I keep
it — I am not tuning the decoupling knob here, only installing the decoupling itself.

Now the concrete fill, because the edit surface is specific and I must match it, not a generic paper
harness. The head stays the scaffold default: the fixed `NatureDQNEncoder` → 512 features → a single
`nn.Linear(512, n_actions)`, no change to the architecture, no capacity added (the budget check is moot
since I touch nothing). The `select_action` is greedy on those Q-values (the loop handles $\epsilon$
externally). The whole edit is in `update`: under `no_grad`, the **online** network's argmax over
$Q(S_{t+1},\cdot)$ selects the next action, the **target** network gathers that action's value, and
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
largest and the inflation worst (Seaquest, eighteen actions), least where the game is near-trivial to
saturate (Pong, which any working value learner should drive close to its ceiling). What it cannot do
is enrich the value object: $Q$ is still a single scalar per action, the mean of a return distribution
it never sees, and the target is still a point estimate with no notion of its own spread. So I expect
this to be a *correct but modest* rung — a real reduction in overestimation, a real but not dramatic
score, with Pong essentially solved, Breakout in the low hundreds, and Seaquest the most variable
because its long horizon and large action set leave the most residual bias and the most room for the
relative-ordering errors to matter. That ceiling — a scalar-mean critic with a decoupled but still
correlated target — is exactly what the next rung will have to break by learning the whole return
distribution rather than its mean.
