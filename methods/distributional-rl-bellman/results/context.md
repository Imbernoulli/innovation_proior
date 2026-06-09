# Context: a Bellman equation for the *distribution* of returns

## Research question

Value-based reinforcement learning rests on a single scalar summary of an agent's future: the
expected return. For a Markov decision process $(\mathcal{X},\mathcal{A},R,P,\gamma)$ and a fixed
policy $\pi$, the quantity of interest is

$$Q^\pi(x,a) = \mathbb{E}\Big[\textstyle\sum_{t\ge 0}\gamma^t R(x_t,a_t)\Big],\qquad x_0=x,\ a_0=a,$$

and almost everything we know how to do — temporal-difference learning, SARSA, Q-learning, fitted
value iteration — is built on the fact that this expectation satisfies Bellman's equation and that the
associated operator contracts.

But the object inside that expectation, the *random return* $Z^\pi(x,a)=\sum_{t\ge0}\gamma^t
R(x_t,a_t)$, is a full random variable. Stochastic rewards, stochastic transitions, and a stochastic
policy compound at every step, so the return is in general spread out, often multimodal, sometimes
heavy-tailed. Taking the expectation collapses all of that to a point. The question this context sets
up is whether one can put a Bellman-style recursion directly on the *distribution* of $Z^\pi$ rather
than on its mean — and whether such a recursion is mathematically well-posed in the same strong sense
the scalar one is: a unique fixed point, reached by iteration at a geometric rate. A second, sharper
question concerns *control*: the scalar optimality operator is a contraction and value iteration
converges to $Q^*$; does the distributional analogue inherit that good behaviour, or does maximizing
break something that fixing a policy did not?

A solution would have to (i) define the return distribution and a distributional operator precisely as
random-variable equations; (ii) supply a metric on distributions in which the operator provably
contracts, so iteration converges and a unique return distribution exists; (iii) settle the control
case honestly, including any instability; and (iv) explain how to *represent and fit* such
distributions with a finite-parameter model in a way that survives learning from sampled transitions.

## Background

**Bellman's equation and the contraction that makes RL work.** For a fixed policy, the value function
satisfies $Q^\pi(x,a)=\mathbb{E}R(x,a)+\gamma\,\mathbb{E}_{P,\pi}Q^\pi(x',a')$; the optimal value
function satisfies $Q^*(x,a)=\mathbb{E}R(x,a)+\gamma\,\mathbb{E}_P\max_{a'}Q^*(x',a')$
(Bellman, 1957). Viewing value functions as vectors in $\mathbb{R}^{\mathcal{X}\times\mathcal{A}}$, the
Bellman operator $\mathcal{T}^\pi$ and the optimality operator $\mathcal{T}$ are both
$\gamma$-contractions in the supremum norm; by Banach's fixed-point theorem each has a unique fixed
point ($Q^\pi$, $Q^*$) and repeated application from any $Q_0$ converges geometrically
(Bertsekas & Tsitsiklis, 1996). This contraction is the load-bearing fact of approximate dynamic
programming, and it is what any distributional generalization must reproduce — but now for whole
distributions, where "supremum norm" is no longer the obvious yardstick.

**Distributional recursions and the contraction method.** The idea that a recursively defined random
variable is pinned down as the fixed point of a contraction in a metric *between distributions* is well
established outside RL. Rösler's (1991) analysis of Quicksort is the cleanest example: the normalized
comparison count $(X_n-\mathbb{E}X_n)/n$ converges to a limit $Y$ whose law is the unique fixed point
of a map $S$ on distributions, and $S$ is shown to be a contraction in the **Wasserstein (Mallows)
metric** $d_2$. The fixed-point relation then yields recursive formulas for all the higher moments,
and convergence holds in every $\ell_p$-metric. Rösler (1992) generalizes this into the contraction
method for stochastic recursive equations of sum and max type. This body of work lives in the
probabilistic analysis of algorithms; whether its machinery bears on the Bellman recursion has not
been examined.

**The Wasserstein metric.** For two cumulative distribution functions $F,G$ on $\mathbb{R}$, the
$p$-Wasserstein distance (called the Mallows metric by Bickel & Freedman, 1981) is

$$d_p(F,G):=\inf_{U,V}\lVert U-V\rVert_p,$$

the infimum over all couplings — pairs of random variables with marginals $F,G$. The infimum is
attained by the *inverse-cdf (quantile) coupling*, giving the closed form
$d_p(F,G)=\lVert F^{-1}(\mathcal{U})-G^{-1}(\mathcal{U})\rVert_p$ with $\mathcal{U}\sim\mathrm{Unif}[0,1]$,
and for $p<\infty$, $d_p(F,G)=\big(\int_0^1|F^{-1}(u)-G^{-1}(u)|^p\,du\big)^{1/p}$. Crucially, $d_p$
measures *horizontal* transport along the value axis: shifting a distribution by a constant or scaling
it scales the distance correspondingly. It remains defined even for distributions whose supports do not
overlap.

**Earlier distributional and risk-sensitive work.** Studying more than the mean of the return is almost
as old as Bellman's equation. Jaquette (1973) introduced a *moment optimality* criterion that imposes a
total ordering on return distributions and showed it is achievable by a stationary policy. Sobel (1982)
derived Bellman-style recursions for the *higher moments* of the return (not the full distribution): in
particular, once the mean-dependent transition term is fixed, the successor-variance component is
multiplied by $\gamma^2$, so the variance recursion has a $\gamma^2$ contraction in that component.
Chung & Sobel (1987) studied convergence of the
distributional recursion and showed, importantly, that it is **not** a contraction in total-variation
distance. White (1988) studied nonstandard MDP criteria. In the learning literature the distribution of
returns had been used for specific ends — Gaussian-process value functions (Engel et al., 2005), a
Normal-Gamma model of parametric uncertainty (Dearden et al., 1998), Kalman-filter value estimates
(Geist & Pietquin, 2010), and parametric/nonparametric cdf models for risk-sensitive control
(Morimura et al., 2010a,b). These either approximate only low-order moments, restrict to Gaussian or
cdf representations, target risk rather than the full learning problem, or analyze policy evaluation
only. None puts a general Bellman operator on the full distribution, proves its contraction, and
confronts the control case.

## Baselines

- **Expected-value dynamic programming (Bellman, 1957; Bertsekas & Tsitsiklis, 1996).** Iterate
  $Q_{k+1}=\mathcal{T}^\pi Q_k$ or $Q_{k+1}=\mathcal{T}Q_k$. Sup-norm $\gamma$-contraction ⇒ unique
  fixed point, geometric convergence. *Gap:* models only the mean; discards multimodality and the
  intrinsic randomness of the return. In the approximate setting the single scalar target averages
  qualitatively different futures (e.g. "survive" vs. "die") into one number that may not be realizable
  by any state.

- **DQN (Mnih et al., 2015).** Deep Q-learning: a convolutional network outputs $Q(x,a)$, trained by
  regressing the scalar TD target $r+\gamma\max_{a'}Q_{\tilde\theta}(x',a')$ under a squared loss, with a
  target network $\tilde\theta$ and experience replay. *Gap:* a single real-valued output per action; no
  representation of the return's shape, and the squared regression target collapses any next-state
  distribution to its mean.

- **Moment- and risk-based return models (Sobel, 1982; Tamar et al., 2016; Prashanth & Ghavamzadeh,
  2013; Mannor & Tsitsiklis, 2011).** Bellman recursions or linear function approximation for the
  variance (and higher moments) of the return, used mainly for risk-sensitive objectives. *Gap:* track
  a few moments, not the distribution; control results are largely negative (variance-constrained
  optimal control is hard), and there is no full-distribution operator with a convergence guarantee.

- **Distributional cdf models (Morimura et al., 2010a,b).** Parametric and nonparametric models of the
  return's cumulative distribution function for risk-sensitive RL, with a policy-evaluation consistency
  result in the nonparametric case. *Gap:* tailored to risk; the control setting and the contraction
  behaviour of a general distributional operator are not analyzed.

- **Compress and Control (Veness et al., 2015).** Predicts the return distribution with density models
  and obtains fast learning, but via Monte-Carlo–style return prediction rather than a Bellman operator.
  *Gap:* leaves open whether a *practical, bootstrapped* distributional algorithm can be built on the
  Bellman equation itself.

## Evaluation settings

The natural yardstick for a value-based control method is the Arcade Learning Environment
(Bellemare et al., 2013): Atari 2600 games from raw pixels, the standardized DQN preprocessing and
network, per-game scores reported relative to a random agent and a human baseline, with mean/median
human-normalized score across the suite and counts of games exceeding human level. For a *policy
evaluation* check of the theory, a small tabular domain with a known return distribution is the
appropriate testbed — e.g. CliffWalk (Sutton & Barto, 1998), where a ground-truth return distribution
can be estimated by many Monte-Carlo rollouts and an approximation compared to it under the
$1$-Wasserstein distance, swept over whatever knobs the chosen representation and loss expose. The
setting is specified by the domains, the metrics, and the protocol.

## Code framework

A value-based deep-RL scaffold has the usual replay, target-network, greedy-action, and loss pieces,
with one open representation slot for the random return and its sampled backup.

```python
import numpy as np

# --- a value-based agent over an MDP, DQN-style ---

class ReplayBuffer:
    """Stores (x, a, r, x', terminal) transitions; samples minibatches."""
    def add(self, x, a, r, x_next, terminal): ...
    def sample(self, batch_size): ...

class QNetwork:
    """Convolutional torso (as in DQN), mapping a state to per-action outputs.
    Today those outputs are scalar action-values Q(x, a). The shape of the
    head is the open design question below."""
    def __init__(self, num_actions, head):
        ...
        self.head = head            # TODO: what does the network emit per action?

    def forward(self, x):
        ...                         # returns per-action outputs of head's shape


# --- the open slot: how do we represent a return, and back it up? -------------

class ReturnRepresentation:
    """A finite-parameter stand-in for the return random variable Z(x, a).
    TODO: decide what is stored per (x, a). The mean alone (a scalar) is the
    status quo; whether more of the return's shape should be carried, and in
    what coordinates, is the open design question."""
    pass


def bellman_target(batch, net_target, gamma):
    """Form the learning target from a sampled transition.
    Today: the scalar TD target  r + gamma * max_a' Q_target(x', a').
    TODO: the target as a function of the chosen ReturnRepresentation, and the
    operator applied to it before it is used as a target."""
    # scalar status quo:
    # q_next = net_target.forward(batch.x_next)          # (B, num_actions)
    # a_star = q_next.argmax(axis=1)
    # return batch.r + gamma * (1 - batch.terminal) * q_next[range(len(a_star)), a_star]
    raise NotImplementedError  # TODO


def loss_fn(pred, target):
    """Discrepancy between prediction and target, minimized by SGD.
    Today: squared error on scalars. TODO: the loss appropriate to the
    ReturnRepresentation — it must be minimizable from *sampled* transitions."""
    raise NotImplementedError  # TODO


def train_step(batch, net, net_target, gamma, opt):
    target = bellman_target(batch, net_target, gamma)
    pred = net.forward(batch.x)                          # select chosen actions
    loss = loss_fn(pred, target)
    opt.minimize(loss)
    return loss
```

The scaffold separates the network output shape (`head` / `ReturnRepresentation`), the sampled
Bellman target (`bellman_target`), and the trainable discrepancy (`loss_fn`).
