# Context

## Research question

The task is to make a parameterized stochastic policy good at a sequential
control problem by maximizing the expected total reward
$\mathbb{E}\!\left[\sum_{t=0}^{\infty} r_t\right]$, directly by gradient ascent,
with a neural network mapping observations to actions. Policy gradient methods
are attractive here precisely because they optimize the reward objective
directly and place no special structure on the function approximator. The
obstacle is the *credit assignment problem*: an action's consequence for reward
can arrive many timesteps later, so the learner must figure out which of the
thousands of actions in a trajectory deserve credit (or blame) for the outcome.

This causes two concrete failures that a usable method must solve.

First, **variance**. The standard likelihood-ratio (score-function) gradient
estimator multiplies each $\nabla_\theta \log \pi(a_t\mid s_t)$ by some measure
of the return that follows. Because the return after $a_t$ also contains the
effects of every *other* action (and of environment stochasticity), the signal
attributed to $a_t$ is buried in noise from everything else. The variance of the
estimator grows with the horizon. High variance forces tiny, conservative steps
and enormous sample counts; for high-dimensional continuous control (tens of
state dimensions, ~$10^4$ network parameters) this is the binding constraint.

Second, **stability under nonstationarity**. The data distribution is generated
by the current policy, so it shifts every time the policy changes. A naive
gradient step that is too large can collapse the policy and never recover, and
because each batch reflects only the current policy, an update that overfits the
latest batch is dangerous.

A solution has to drive down the variance of the gradient estimate without
introducing so much bias that the method converges to a poor policy, and it has
to take steps that improve reliably despite the moving data distribution. Bias
is the more dangerous of the two: more variance only costs more samples, but
bias can make the procedure converge to something that is not even a local
optimum of the true objective, no matter how many samples are collected.

## Background

**The policy gradient and its forms.** For a parameterized stochastic policy,
the gradient of expected return can be written
$$
g = \mathbb{E}\!\left[\sum_{t=0}^{\infty} \Psi_t\, \nabla_\theta \log \pi_\theta(a_t\mid s_t)\right],
$$
and $\Psi_t$ admits several interchangeable choices that all give the same
expectation: the total trajectory reward; the reward-to-go
$\sum_{t'\ge t} r_{t'}$; the reward-to-go minus a state-dependent baseline
$b(s_t)$; the state–action value $Q^\pi(s_t,a_t)$; the advantage
$A^\pi(s_t,a_t)=Q^\pi-V^\pi$; or the one-step temporal-difference residual
$r_t+V^\pi(s_{t+1})-V^\pi(s_t)$. These are standard (Williams 1992;
Sutton et al. 1999; Baxter & Bartlett 2000). They are equal in expectation but
have very different variances.

**Baselines and the advantage.** Subtracting a function of the state only,
$b(s_t)$, from the return does not bias the gradient, because
$\mathbb{E}_{a\sim\pi}[\,\nabla_\theta\log\pi(a\mid s)\,b(s)]=0$. Choosing
$b=V^\pi$ yields the advantage $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$, which measures
whether an action is better or worse than the policy's average behavior from
that state. Using the advantage gives close to the lowest achievable variance
among these choices, since it removes the part of the return that is explained
by the state alone (Greensmith, Bartlett & Baxter 2004 analyze this rigorously).
The catch is that $A^\pi$ is not known and must be estimated.

**Value functions and temporal-difference learning.** The state-value function
$V^\pi(s_t)=\mathbb{E}[\sum_{l\ge 0} r_{t+l}]$ and action-value
$Q^\pi(s_t,a_t)$ are the central objects for credit assignment: they let the
learner estimate the goodness of an action before the delayed reward arrives.
Temporal-difference methods (Sutton & Barto 1998) estimate $V$ from the
bootstrapped relation $V(s_t)\approx r_t+\gamma V(s_{t+1})$, whose error
$\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)$ is the TD residual. TD($\lambda$) forms
a geometrically-weighted average of $n$-step returns — the $\lambda$-return —
trading off the bias of short bootstrapped backups against the variance of long
Monte-Carlo backups with a single parameter $\lambda\in[0,1]$. This averaging
construction, applied to *value* estimation, is the template that the present
problem will reuse for *advantage* estimation.

**Discounting as variance reduction.** Even when the problem is posed without a
discount (maximize $\sum_t r_t$), introducing a discount $\gamma<1$ inside the
estimator downweights rewards in the distant future. This drops the
highest-variance, longest-delay credit terms at the cost of bias relative to the
undiscounted objective. Treating $\gamma$ as a variance-reduction knob rather
than part of the problem statement was analyzed by Marbach & Tsitsiklis (2003),
Kakade (2001, *Optimizing average reward using discounted rewards*) and Thomas
(2014, *Bias in natural actor-critic algorithms*), who characterize the bias it
introduces.

**Reward shaping.** Ng, Harada & Russell (1999) showed that transforming the
reward by a potential function $\Phi$,
$\tilde r(s,a,s') = r(s,a,s') + \gamma\Phi(s') - \Phi(s)$,
leaves the optimal policy and (for the discounted objective) the policy gradient
unchanged. The discounted sum of shaped rewards telescopes to the discounted sum
of original rewards minus $\Phi(s_t)$. This invariance is a load-bearing tool: it
means a value-function-like potential can be folded into the reward without
changing what is being optimized.

**The actor-critic precedent and its known limitation.** Actor-critic methods
(Konda & Tsitsiklis 2003; Hafner & Riedmiller 2011) replace the empirical return
in the gradient by a learned value function, lowering variance but introducing
bias. The diagnostic fact that motivates everything below is the
**bias/variance asymmetry along the spectrum of estimators**: a one-step
bootstrapped estimate $\delta_t$ has low variance but is heavily biased whenever
the value function is imperfect (which it always is for a neural net being
trained); the full Monte-Carlo return has no such bias but high variance. Neither
endpoint is good enough on its own, and prior actor-critic work using a
parameterized $Q$-function is effectively pinned to the high-bias, one-step end
of this spectrum. A closely related geometrically-weighted estimator had appeared
in *online* actor-critic work with eligibility traces (Kimura & Kobayashi 1998;
Wawrzyński 2009), but without an analysis that carries over to the batch setting
or that connects it to discounting and shaping.

**Natural gradients.** Plain gradient ascent in parameter space is sensitive to
the arbitrary parameterization of the policy. The natural policy gradient
(Kakade 2001, *A natural policy gradient*; Peters & Schaal 2008, natural
actor-critic) preconditions the gradient by the inverse Fisher information
matrix, giving a step that is invariant to reparameterization and that follows
the steepest-ascent direction in distribution space. Computing it scales poorly
if the Fisher matrix is formed explicitly, but it can be approximated with
conjugate gradient using only Fisher–vector products.

**Compatible features.** Konda & Tsitsiklis (2003) observed that, because the
policy has limited representation power, the policy gradient only depends on the
projection of the advantage function onto the subspace spanned by the compatible
features $\nabla_{\theta}\log\pi(a\mid s)$. Projecting an advantage estimate onto
this subspace by least squares recovers the natural policy gradient. This theory
says nothing about how to exploit the *temporal* structure of credit assignment
to estimate the advantage better.

## Baselines

**REINFORCE / vanilla policy gradient (Williams 1992; Sutton et al. 1999).**
Estimate $g$ with $\Psi_t = \sum_{t'\ge t} r_{t'} - b(s_t)$, an unbiased
estimator using the empirical return-to-go and a baseline. *Core math:* score
function identity $\nabla_\theta \mathbb{E}[R] = \mathbb{E}[R\,\nabla_\theta\log
\pi]$. *Gap:* variance grows with horizon because the empirical return after
$a_t$ contains all the noise from later actions and dynamics; convergence is slow
and sample-hungry.

**Actor-critic with a learned value/Q function (Konda & Tsitsiklis 2003;
Hafner & Riedmiller 2011).** Replace the empirical return by a bootstrapped
estimate using a critic — e.g. $\Psi_t = r_t+\gamma V(s_{t+1})-V(s_t)$ or
$\Psi_t = Q(s_t,a_t)-V(s_t)$. *Core math:* if $V=V^\pi$, the one-step residual is
an unbiased advantage; in general it bootstraps. *Gap:* a one-step bootstrap is
strongly biased whenever the critic is imperfect, and a parameterized $Q$-critic
offers no smooth way to dial that bias down toward the Monte-Carlo end. The bias
from a one-step estimate is, empirically, prohibitively large for hard control.

**TD($\lambda$) for value estimation (Sutton & Barto 1998).** The
$\lambda$-return geometrically averages $n$-step returns to fit $V$, trading bias
against variance with one knob. *Gap (for the present purpose):* it estimates the
*value* function, not the *advantage* used in the policy gradient; the analogous
construction for the advantage, in the discounted-as-variance-reduction batch
setting, and its relationship to reward shaping, is not established.

**Online actor-critic with eligibility traces (Kimura & Kobayashi 1998;
Wawrzyński 2009).** Use eligibility-trace-style geometric weighting of TD
residuals in an online update. *Gap:* presented as an online algorithmic device;
lacks the general bias/variance analysis that would let the same estimator be
dropped into a batch, trust-region policy optimizer, and lacks the
discounting/shaping interpretation.

**Natural policy gradient / natural actor-critic (Kakade 2001; Peters & Schaal
2008).** Precondition the gradient by $F^{-1}$, $F$ the Fisher information.
*Core math:* steepest ascent under the KL metric on the policy distribution.
*Gap:* addresses the *direction/metric* of the update, not the *estimation of the
advantage* that feeds it; still needs a low-variance advantage estimate and a
principled step-size rule.

## Evaluation settings

The natural yardstick is episodic continuous-control simulation. A classic
small-scale task is cart-pole balancing with the standard physical parameters
(Barto, Sutton & Anderson 1983), collecting a batch of trajectories per
iteration with a maximum episode length. The challenging regime is 3D robotic
locomotion in a physics simulator (MuJoCo; Todorov, Erez & Tassa 2012): a biped
(~33 state dimensions, 10 actuated joints) and a quadruped (~29 state
dimensions, 8 joints) controlled at the torque level, plus a getting-up-from-the
-ground task. Policies map raw kinematic state directly to joint torques. The
protocol is batch policy optimization: simulate the current policy for a fixed
number of timesteps per batch (tens of thousands up to hundreds of thousands),
update, repeat for many iterations, average over several random seeds. The
performance signal is the episodic total reward (equivalently its negative,
cost) as a function of optimization iterations or simulated experience; reward
functions reward forward velocity / posture and lightly penalize torque and
impact. A natural ablation axis is to sweep the estimator's bias/variance
parameters and to compare against a state-independent (time-dependent) baseline.

## Code framework

The primitives that already exist: a simulator with a step function, an
automatic-differentiation library for the policy and value networks, a
trust-region / natural-gradient optimizer driven by conjugate gradient with
matrix–vector products, and a batch rollout loop. The contribution will be the
estimator that turns rewards and value predictions into the per-timestep
multiplier $\Psi_t$ in the policy gradient — that slot is left empty here.

```python
import numpy as np

# --- existing primitives ---------------------------------------------------

class Policy:
    """Stochastic policy network a ~ pi(.|s); gives log-prob and its gradient."""
    def act(self, s): ...
    def logp(self, s, a): ...

class ValueFunction:
    """Scalar state-value network V(s)."""
    def predict(self, s): ...
    def fit(self, states, targets): ...   # regress V onto target values

def collect_batch(env, policy, n_timesteps):
    """Roll out the current policy, recording s, a, r, and V(s) per timestep,
    split into episodes (with terminal flags)."""
    ...
    return batch  # states, actions, rewards, values, episode boundaries

def trust_region_step(policy, states, actions, advantages, kl_limit):
    """Natural-gradient / KL-constrained policy update (CG + Fisher-vector
    products, line search to satisfy the KL constraint)."""
    ...

# --- the slot the method will fill -----------------------------------------

def estimate_targets(rewards, values, episode_boundaries, **knobs):
    """Turn a batch of rewards and value predictions into:
       - the per-timestep multiplier Psi_t fed to the policy gradient, and
       - regression targets for the value function.
    This per-timestep credit signal is the whole contribution; everything
    else above already exists.
    """
    # TODO: design Psi_t to trade off bias against variance of the gradient.
    pass

# --- training loop ---------------------------------------------------------

def train(env, policy, value_fn, n_iters, n_timesteps, kl_limit, **knobs):
    for _ in range(n_iters):
        batch = collect_batch(env, policy, n_timesteps)
        psi, v_targets = estimate_targets(
            batch.rewards, batch.values, batch.boundaries, **knobs)
        trust_region_step(policy, batch.states, batch.actions, psi, kl_limit)
        value_fn.fit(batch.states, v_targets)
```
