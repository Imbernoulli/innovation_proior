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
The same numbers choose the action and score it. That coupling is what I want to
interrogate.

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

I need to be careful about why this matters, because not every upward shift is
harmful. A uniform positive shift of every action value in a state does not
change an argmax. Optimism can also be useful when it is deliberately tied to
uncertainty. But this overestimation is not a deliberate exploration bonus, and
I do not think it is uniform: it depends on local action count, approximation
error, sample coverage, and previous bootstrap errors. Those differences change
relative action and state values, and the greedy policy depends exactly on those
relative values.

Jensen tells me the bias is nonnegative on average, but I would like to know how
hard the floor is in a single state, because an average over realizations of the
error could be small even if every realization is positive. Let me try to push
the inequality into a deterministic, per-state lower bound. Consider a state
where every action is truly tied: $Q_*(s,a)=V_*(s)$ for $m\ge 2$ actions. Let
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
I want to know the smallest the maximum can be over all error vectors meeting
those two constraints. I will guess the answer is
$$
\max_a Q_t(s,a)\ge V_*(s)+\sqrt{\frac{C}{m-1}}
$$
and try to prove it; if the guess is wrong, the proof attempt should break and
tell me where.

Let $M=\max_a\epsilon_a$. Because the errors sum to zero and are not all zero,
there is at least one positive error and at least one negative error. Let the
$n$ positive errors be $\epsilon_i^+$, so $1\le n\le m-1$, and put
$B=\sqrt{C/(m-1)}$. Suppose for contradiction that $M<B$. Then the positive
squared mass is less than $nB^2$. The positive total mass $p=\sum_i\epsilon_i^+$
is less than $nB$, and the negative errors have total absolute mass $p$ because
the full sum is zero. For any fixed total negative mass $p$, the negative squared
mass is at most $p^2$ (worst case, all of it on one entry), so it is less than
$n^2B^2$. Therefore
$$
\sum_a\epsilon_a^2<nB^2+n^2B^2
=\frac{n(n+1)}{m-1}C
\le mC,
$$
where the last step uses $n\le m-1$ so $n(n+1)\le(m-1)m$. That contradicts the
condition $\sum_a\epsilon_a^2=mC$. Hence $M\ge B$.

Before I trust this I want to check that the bound is actually attainable, not
just an inequality I derived loosely. The contradiction chain used two
inequalities ($p<nB$ and negative mass $\le p^2$); if both can be made tight
simultaneously the bound should be reached. Try
$$
\epsilon_a=\sqrt{\frac{C}{m-1}}\quad\text{for }a=1,\ldots,m-1,
\qquad
\epsilon_m=-\sqrt{(m-1)C}.
$$
Sum: $(m-1)\sqrt{C/(m-1)}-\sqrt{(m-1)C}=\sqrt{(m-1)C}-\sqrt{(m-1)C}=0$, good.
Squared sum: $(m-1)\cdot\frac{C}{m-1}+(m-1)C=C+(m-1)C=mC$, so the mean square is
$C$, good. And the maximum entry is exactly $\sqrt{C/(m-1)}=B$. I checked this
numerically for a few $m$ with $C=1.7$: $m=2$ gives max $1.3038$ against bound
$1.3038$, $m=3$ gives $0.9220$ against $0.9220$, $m=5$ gives $0.6519$ against
$0.6519$, every time the sum is zero and the mean square is $1.7$ to machine
precision. So the floor is tight, and the all-positive-tied side of the
construction is what forces the max up.

One thing nags at me: the bound $\sqrt{C/(m-1)}$ shrinks as $m$ grows, which
sounds backwards if I expect more actions to give more overestimation. But this
is a floor over the worst-balanced error pattern, not the typical case. With
more actions I can spread the positive side across more entries (as the tight
construction does) and make the smallest possible maximum smaller. It says
nothing about what happens with random errors, which is the case I actually care
about for an explanation of the DQN curves.

So let me do the typical-case calculation separately. Stay in the all-tied state
and take $\epsilon_a$ i.i.d. uniform on $[-1,1]$. For $x\in(-1,1)$,
$$
P(\epsilon_a\le x)=\frac{1+x}{2},
$$
so independence gives
$$
P(\max_a\epsilon_a\le x)=\left(\frac{1+x}{2}\right)^m,
$$
and the density of the maximum is the derivative,
$$
f_{\max}(x)=\frac m2\left(\frac{1+x}{2}\right)^{m-1}.
$$
Then
$$
\mathbb E[\max_a\epsilon_a]
=\int_{-1}^1 x\,\frac m2\left(\frac{1+x}{2}\right)^{m-1}\,dx.
$$
I will guess the antiderivative
$F(x)=\left(\frac{x+1}{2}\right)^m\frac{mx-1}{m+1}$ and check it by
differentiating rather than trusting the form. Writing $u=(x+1)/2$ so
$u'=1/2$, $F=u^m(mx-1)/(m+1)$ and
$$
F'=\frac{m u^{m-1}\cdot\tfrac12\,(mx-1)+u^m\cdot m}{m+1}
=\frac{m u^{m-1}}{m+1}\left(\frac{mx-1}{2}+u\right).
$$
The bracket is $\frac{mx-1}{2}+\frac{x+1}{2}=\frac{mx+x}{2}=\frac{(m+1)x}{2}$,
so $F'=\frac{m u^{m-1}}{m+1}\cdot\frac{(m+1)x}{2}=\frac m2 u^{m-1}x$, which is
exactly the integrand. (I also confirmed this symbolically for $m=2,3,4$: the
derivative minus the integrand simplifies to $0$.) Evaluating,
$$
\mathbb E[\max_a\epsilon_a]=F(1)-F(-1).
$$
At $x=1$, $u=1$ and $F(1)=\frac{m-1}{m+1}$; at $x=-1$, $u=0$ and $F(-1)=0$. So
$$
\mathbb E[\max_a\epsilon_a]=\frac{m-1}{m+1}.
$$
Quick sanity check at $m=2$: this predicts $1/3$, and the expected maximum of two
independent uniforms on $[-1,1]$ is indeed $1/3$ (the larger of two uniforms on
$[0,1]$ has mean $2/3$, rescaled to $[-1,1]$ gives $1/3$). A Monte Carlo of four
million draws gives $0.3330$ at $m=2$, $0.4999$ at $m=3$, $0.5998$ at $m=4$, and
$0.8181$ at $m=10$, against $\frac{m-1}{m+1}=0.3333,0.5,0.6,0.8182$. Scaling the
interval to $[-\epsilon,\epsilon]$ multiplies by $\epsilon$, and a one-step
discounted bootstrap multiplies by $\gamma$, giving an overestimation term of
$\gamma\epsilon(m-1)/(m+1)$. This grows with $m$, which is the dependence I
expected for the random case, and it matches the older uniform-error analysis.

Both calculations point the same way: once balanced error is present, the
single-estimator max has no escape on average, and in the worst balanced pattern
it cannot even be driven to zero. The action selected by the max is selected
partly because its estimate is high, and then the same high estimate is used as
the value. So the fix has to break that correlation. If one estimator chooses the
greedy action and another estimator scores that chosen action, the evaluator's
error is not the reason the action was selected.

I should check that this actually removes the floor and is not just intuition. In
the all-tied case, an independent unbiased evaluator has expected value $V_*(s)$
on whichever action is selected, so the expected bias is zero rather than
positive. And the hard per-state floor disappears too: I can pick a legal
selecting error vector that chooses $a_1$, for example
$$
\epsilon_1=\sqrt{C(m-1)},\qquad
\epsilon_i=-\sqrt{\frac{C}{m-1}}\quad(i>1).
$$
Its sum is $\sqrt{C(m-1)}-(m-1)\sqrt{C/(m-1)}=\sqrt{C(m-1)}-\sqrt{C(m-1)}=0$ and
its mean square is $\frac1m\big(C(m-1)+(m-1)\frac{C}{m-1}\big)=\frac1m(C(m-1)+C)=C$,
so it satisfies the same constraints as before and its argmax is $a_1$ (I
confirmed sum, mean square, and argmax numerically for $m=2,3,5$). Now let the
second estimator have $Q'_t(s,a_1)=V_*(s)$ exactly. Then the selected action's
evaluated error is $0$, not the forced $\sqrt{C/(m-1)}>0$ I was stuck with
before. The double estimate is not always perfect — it can even underestimate —
but the positive floor created by evaluating with the selected high estimate is
gone.

The literal tabular construction of this idea keeps two value functions and
randomly assigns updates to one or the other. In the deep Atari agent, adding a
second fully trained network would change a lot more than the target — memory,
compute, another set of optimizer state. But DQN already carries two sets of
weights: the online weights $\theta_t$ and the periodically copied target weights
$\theta_t^-$. The target network is not independent, but between copies it is
stale, so it is at least partly decoupled from the online network. That is enough
of a second estimator to try.

So the smallest change I can make is to keep the DQN architecture, replay buffer,
behavior policy, optimizer, target-copy rule, and loss machinery, and change only
the next-state target from
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
The online network selects the greedy next action; the target network evaluates
that selected action.

Let me trace this on a tiny case to be sure it does what I intend and is not
accidentally identical to DQN. Take one next-state with two actions where the two
networks disagree about the best action: online values $[1.0,\,0.3]$ so the
online argmax is action $0$, and target values $[0.2,\,0.9]$ so the target's own
max is action $1$. With $\gamma=1$ and reward $0$, the Double DQN target is the
target's value of the online-chosen action $0$, which is $0.2$. The plain DQN
target is the target's own max, $0.9$. So Double DQN here reports $0.2$ and DQN
reports $0.9$: the new target does evaluate the online pick instead of taking the
target network's optimistic maximum, and it comes out lower exactly when the two
networks' high errors land on different actions. That is the decoupling I wanted,
on a concrete input.

This does not fully recreate independent Double Q-learning, and I should be
honest about when it degenerates. Immediately after a target copy,
$\theta_t^-=\theta_t$, so the online and target values are identical; running the
same trace with both networks at $[1.0,0.3]$ gives Double target $1.0$ and DQN
target $1.0$, the same number. So the first target computed before the online
weights move again is exactly the DQN target. As soon as the online network takes
further gradient steps, the two networks diverge and the decoupling reappears.
That also suggests a tuning knob rather than a core change: copying less often
(for example raising the copy period from $10{,}000$ to $30{,}000$ frames) keeps
the selector and evaluator apart for longer. But the core method does not depend
on that tuning; with the untuned DQN hyperparameters, the only algorithmic
difference is the target formula.

The code should reflect exactly that. In a reference implementation, DQN computes
a TD error from online values at the current state, target-network values at the
next state, and the Q-learning max. The double version instead computes online
values at the next state for selection, target-network values at the next state
for evaluation, and forms the temporal-difference error
$$
\delta_t=
R_{t+1}+\gamma_t
Q(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta_t);\theta_t^-)
-Q(S_t,A_t;\theta_t).
$$
Here $\gamma_t$ is zero on terminal transitions and otherwise equals the
discount. The loss and optimizer remain the DQN choices, including the clipped
TD-error gradient used in the reference agent. The change is local: decouple
selection from evaluation in the bootstrap target, and leave the rest of DQN
alone.
