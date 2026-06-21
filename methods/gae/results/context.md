# Context

## Research question

The task is to make a parameterized stochastic policy good at a sequential
control problem by maximizing the expected total reward
$\mathbb{E}\!\left[\sum_{t=0}^{\infty} r_t\right]$, directly by gradient ascent,
with a neural network mapping observations to actions. Policy gradient methods
optimize the reward objective directly and place no special structure on the
function approximator. The central difficulty is *credit assignment*: an
action's consequence for reward can arrive many timesteps later, so the learner
must figure out which of the thousands of actions in a trajectory deserve credit
(or blame) for the outcome.

Two quantities govern how well this works. The first is the **variance** of the
gradient estimate. The standard likelihood-ratio (score-function) estimator
multiplies each $\nabla_\theta \log \pi(a_t\mid s_t)$ by some measure of the
return that follows. The return after $a_t$ also contains the effects of every
*other* action and of environment stochasticity, and the variance of the
estimator grows with the horizon. The second is **bias**: replacing the
empirical return by a learned estimate can lower variance but can also shift the
estimator away from the true gradient. For high-dimensional continuous control
(tens of state dimensions, ~$10^4$ network parameters) both quantities matter,
and the policy's data distribution shifts every time the policy changes, so each
update must improve reliably under a moving distribution.

The question is how to construct the per-timestep multiplier $\Psi_t$ in the
policy gradient — and the targets used to fit any learned value function —
so as to trade off the bias and variance of the gradient estimate.

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
among these choices, since it removes the part of the return explained by the
state alone (Greensmith, Bartlett & Baxter 2004 analyze this rigorously).
$A^\pi$ is not known and must be estimated.

**Value functions and temporal-difference learning.** The state-value function
$V^\pi(s_t)=\mathbb{E}[\sum_{l\ge 0} r_{t+l}]$ and action-value
$Q^\pi(s_t,a_t)$ are the central objects for credit assignment: they let the
learner estimate the goodness of an action before the delayed reward arrives.
Temporal-difference methods (Sutton & Barto 1998) estimate $V$ from the
bootstrapped relation $V(s_t)\approx r_t+\gamma V(s_{t+1})$, whose error
$\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)$ is the TD residual. TD($\lambda$) forms
a geometrically-weighted average of $n$-step returns — the $\lambda$-return —
trading the bias of short bootstrapped backups against the variance of long
Monte-Carlo backups with a single parameter $\lambda\in[0,1]$. This averaging
construction is defined and analyzed for *value* estimation.

**Discounting as variance reduction.** Even when the problem is posed without a
discount (maximize $\sum_t r_t$), introducing a discount $\gamma<1$ inside the
estimator downweights rewards in the distant future, dropping the
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
of original rewards minus $\Phi(s_t)$. This is a known invariance of the
discounted objective under potential-based reward transformations.

**Actor-critic methods.** Actor-critic methods (Konda & Tsitsiklis 2003;
Hafner & Riedmiller 2011) replace the empirical return in the gradient by a
learned value function. A one-step bootstrapped estimate $\delta_t$ has low
variance; the full Monte-Carlo return is unbiased relative to the discounted
objective but high variance. Geometrically-weighted combinations of TD residuals
have appeared in *online* actor-critic work with eligibility traces
(Kimura & Kobayashi 1998; Wawrzyński 2009).

**Natural gradients.** Plain gradient ascent in parameter space is sensitive to
the arbitrary parameterization of the policy. The natural policy gradient
(Kakade 2001, *A natural policy gradient*; Peters & Schaal 2008, natural
actor-critic) preconditions the gradient by the inverse Fisher information
matrix, giving a step that is invariant to reparameterization and that follows
the steepest-ascent direction in distribution space. It can be approximated with
conjugate gradient using only Fisher–vector products.

**Compatible features.** Konda & Tsitsiklis (2003) observed that, because the
policy has limited representation power, the policy gradient only depends on the
projection of the advantage function onto the subspace spanned by the compatible
features $\nabla_{\theta}\log\pi(a\mid s)$. Projecting an advantage estimate onto
this subspace by least squares recovers the natural policy gradient.

## Baselines

**REINFORCE / vanilla policy gradient (Williams 1992; Sutton et al. 1999).**
Estimate $g$ with $\Psi_t = \sum_{t'\ge t} r_{t'} - b(s_t)$, an unbiased
estimator using the empirical return-to-go and a baseline. *Core math:* score
function identity $\nabla_\theta \mathbb{E}[R] = \mathbb{E}[R\,\nabla_\theta\log
\pi]$.

**Actor-critic with a learned value/Q function (Konda & Tsitsiklis 2003;
Hafner & Riedmiller 2011).** Replace the empirical return by a bootstrapped
estimate using a critic — e.g. $\Psi_t = r_t+\gamma V(s_{t+1})-V(s_t)$ or
$\Psi_t = Q(s_t,a_t)-V(s_t)$. *Core math:* if $V=V^\pi$, the one-step residual is
an unbiased advantage; in general it bootstraps.

**TD($\lambda$) for value estimation (Sutton & Barto 1998).** The
$\lambda$-return geometrically averages $n$-step returns to fit $V$, trading bias
against variance with one knob.

**Online actor-critic with eligibility traces (Kimura & Kobayashi 1998;
Wawrzyński 2009).** Use eligibility-trace-style geometric weighting of TD
residuals in an online update.

**Natural policy gradient / natural actor-critic (Kakade 2001; Peters & Schaal
2008).** Precondition the gradient by $F^{-1}$, $F$ the Fisher information.
*Core math:* steepest ascent under the KL metric on the policy distribution.

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
matrix–vector products, and a batch rollout loop. The piece to design is the
estimator that turns rewards and value predictions into the per-timestep
multiplier $\Psi_t$ in the policy gradient.

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
    Everything else above already exists; this is the unresolved design slot.
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
