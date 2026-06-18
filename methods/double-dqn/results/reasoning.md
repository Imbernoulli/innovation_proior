I start from the discrepancy I can actually observe. I have a DQN-style agent
whose network predicts action values, and during evaluation I can compare the
network's value for its own greedy policy with the discounted return the same
policy obtains when I roll it out. If the value estimates are calibrated, those
two quantities should meet. Instead the predicted values can sit well above the
returns, and in the worst games they climb while score falls. So I am looking
for a mechanism that creates upward value error and that can change the policy,
not just a harmless offset.

The Q-learning target points directly at the suspect:
$$
Y_t^{\mathrm Q}=R_{t+1}+\gamma \max_a Q(S_{t+1},a;\theta_t).
$$
DQN stabilizes this by using a target network,
$$
Y_t^{\mathrm{DQN}}=R_{t+1}+\gamma \max_a Q(S_{t+1},a;\theta_t^-),
$$
but the statistical operation is still a maximum over estimated values. I can
rewrite that maximum as a selection followed by an evaluation:
$$
\max_a Q(s',a;\theta^-)
=Q\left(s',\arg\max_a Q(s',a;\theta^-);\theta^-\right).
$$
The same numbers choose the action and score it.

Suppose, just to isolate the effect, that each estimated next-state action value
is a true value plus zero-mean error:
$$
\widehat Q(s',a)=Q_*(s',a)+\epsilon_a,\qquad
\mathbb E[\epsilon_a]=0.
$$
The max is convex, so Jensen's inequality gives
$$
\mathbb E\left[\max_a \widehat Q(s',a)\right]\ge
\max_a \mathbb E[\widehat Q(s',a)]=\max_a Q_*(s',a).
$$
The max is biased upward because it tends to select whichever action happens to
have positive error. The target then bootstraps from that selected overestimate,
so the inflated value can be copied backward into predecessor states.

I need to be careful about why this matters. A uniform positive shift of every
action value in a state does not change an argmax. Optimism can also be useful
when it is deliberately tied to uncertainty. But this overestimation is not a
deliberate exploration bonus, and it is not uniform. It depends on local action
count, approximation error, sample coverage, and previous bootstrap errors.
Those differences change relative action and state values, and the greedy policy
depends exactly on those relative values.

Now I want the lower-bound statement to be exact. Consider a state where every
action is truly tied: $Q_*(s,a)=V_*(s)$ for $m\ge 2$ actions. Let
$$
\epsilon_a=Q_t(s,a)-V_*(s),
$$
with balanced error
$$
\sum_a\epsilon_a=0
$$
and mean squared error
$$
\frac1m\sum_a\epsilon_a^2=C>0.
$$
I want to prove
$$
\max_a Q_t(s,a)\ge V_*(s)+\sqrt{\frac{C}{m-1}}.
$$

Let $M=\max_a\epsilon_a$. Because the errors sum to zero and are not all zero,
there is at least one positive error and at least one negative error. Let the
$n$ positive errors be $\epsilon_i^+$, so $1\le n\le m-1$, and put
$B=\sqrt{C/(m-1)}$. If I assume $M<B$, then the positive squared mass is less
than $nB^2$. The positive total mass $p=\sum_i\epsilon_i^+$ is less than $nB$,
and the negative errors have total absolute mass $p$ because the full sum is
zero. For any fixed total negative mass $p$, the negative squared mass is at
most $p^2$, so it is less than $n^2B^2$. Therefore
$$
\sum_a\epsilon_a^2<nB^2+n^2B^2
=\frac{n(n+1)}{m-1}C
\le mC.
$$
That contradicts the condition $\sum_a\epsilon_a^2=mC$. Hence
$M\ge B$, which is the desired lower bound. The bound is tight: set
$$
\epsilon_a=\sqrt{\frac{C}{m-1}}\quad\text{for }a=1,\ldots,m-1,
\qquad
\epsilon_m=-\sqrt{(m-1)C}.
$$
The errors sum to zero, their squared sum is
$C+(m-1)C=mC$, and the maximum error is exactly
$\sqrt{C/(m-1)}$.

This proof also shows why the bound decreasing with $m$ is not a contradiction.
It is a floor over all balanced error patterns. With more actions, I can spread
the positive side across more entries and make the smallest possible maximum
smaller. That says nothing about the typical case.

For the typical uniform-error calculation, stay in the all-tied state and take
$\epsilon_a$ i.i.d. uniform on $[-1,1]$. For $x\in(-1,1)$,
$$
P(\epsilon_a\le x)=\frac{1+x}{2},
$$
so independence gives
$$
P(\max_a\epsilon_a\le x)=\left(\frac{1+x}{2}\right)^m.
$$
The density of the maximum is
$$
f_{\max}(x)=\frac m2\left(\frac{1+x}{2}\right)^{m-1}.
$$
Then
$$
\mathbb E[\max_a\epsilon_a]
=\int_{-1}^1 x\,\frac m2\left(\frac{1+x}{2}\right)^{m-1}\,dx
=\left[\left(\frac{x+1}{2}\right)^m\frac{mx-1}{m+1}\right]_{-1}^{1}
=\frac{m-1}{m+1}.
$$
Scaling the interval to $[-\epsilon,\epsilon]$ multiplies this by $\epsilon$,
and a one-step discounted bootstrap multiplies it by $\gamma$. This is the
source of the $\gamma\epsilon(m-1)/(m+1)$ overestimation term in the older
uniform-error analysis.

The bound tells me the single-estimator max has no escape once balanced error is
present. The action selected by the max is selected partly because its estimate
is high, and then the same high estimate is used as the value. The fix must
break that correlation. If one estimator chooses the greedy action and another
estimator scores that chosen action, the evaluator's error is not the reason the
action was selected. In the all-tied case, an independent unbiased evaluator has
expected value $V_*(s)$ on the selected action. The lower bound on the absolute
error of such a double estimate is zero: choose a legal selecting estimate that
picks $a_1$, for example
$$
\epsilon_1=\sqrt{C(m-1)},\qquad
\epsilon_i=-\sqrt{\frac{C}{m-1}}\quad(i>1),
$$
which has zero sum and average squared error $C$, and let the second estimator
have $Q'_t(s,a_1)=V_*(s)$. The selected action's evaluated error is then exactly
zero. The point is not that the double estimate is always perfect; it can even
underestimate. The point is that the forced positive floor created by evaluating
with the selected high estimate is gone.

The literal tabular construction keeps two value functions and randomly assigns
updates to one or the other. In the deep Atari agent, adding a second fully
trained network would change more than the target. But DQN already has two sets
of weights: the online weights $\theta_t$ and the periodically copied target
weights $\theta_t^-$. The target network is not independent, but between copies
it is stale, so it is at least partly decoupled from the online network.

That suggests the smallest useful change. Keep the DQN architecture, replay
buffer, behavior policy, optimizer, target-copy rule, and loss machinery. Change
only the next-state target from
$$
R_{t+1}+\gamma Q\left(S_{t+1},
\arg\max_a Q(S_{t+1},a;\theta_t^-);\theta_t^-\right)
$$
to
$$
Y_t^{\mathrm{DoubleDQN}}
=R_{t+1}+\gamma Q\left(S_{t+1},
\arg\max_a Q(S_{t+1},a;\theta_t);\theta_t^-\right).
$$
The online network selects the greedy next action. The target network evaluates
that selected action.

This does not fully recreate independent Double Q-learning. Immediately after a
target copy, $\theta_t^-=\theta_t$, so the first target computed before the
online weights move again is the same target DQN would compute. As soon as the
online network takes further gradient steps, the two networks diverge. That also
explains the tuning knob: copying less often, such as increasing the copy period
from $10{,}000$ to $30{,}000$ frames in the tuned condition, keeps the selector
and evaluator apart for longer. But the core method does not require that
tuning; with the untuned DQN hyperparameters, the only algorithmic difference is
the target formula.

The code should reflect exactly that. In a reference implementation, DQN
computes a TD error from online values at time $t-1$, target-network values at
time $t$, and the Q-learning max. The double version additionally computes
online values at time $t$ for selection, target-network values at time $t$ for
evaluation, and forms the temporal-difference error
$$
\delta_t=
R_{t+1}+\gamma_t
Q(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta_t);\theta_t^-)
-Q(S_t,A_t;\theta_t).
$$
Here $\gamma_t$ is zero on terminal transitions and otherwise equals the
discount. The loss and optimizer remain the DQN choices, including the clipped
TD-error gradient used in the reference agent. That is the faithful artifact:
decouple selection from evaluation in the bootstrap target, and leave the rest
of DQN alone.
