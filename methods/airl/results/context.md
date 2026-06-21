## Research question

Deep reinforcement learning has removed much of the feature engineering once needed for
policies and value functions, but a reward function still has to be specified by hand. Some
objectives are hard to write down at all (what is the reward for "socially acceptable"
behavior?), and deep RL is sensitive to reward sparsity and magnitude. Inverse reinforcement
learning (IRL) — inferring an expert's reward function from demonstrations — *acquires* a
reward automatically instead.

The setting here is to recover a reward function from a fixed set of expert demonstrations on
high-dimensional continuous control with unknown dynamics, where the demonstrations are
trajectories from MuJoCo-style locomotion tasks, and then to ask what happens when the learned
reward is taken to a new environment whose *dynamics* differ from the one the demonstrations
came from and a policy is re-optimized under it.

## Background

By this time the dominant probabilistic framing of IRL is **maximum entropy IRL** (Ziebart et
al. 2008) and its causal refinement, **maximum causal entropy IRL** (Ziebart 2010). The setup
is an entropy-regularized MDP `(S, A, T, r, gamma, rho_0)` whose forward objective is

```
pi* = arg max_pi  E_{tau ~ pi} [ sum_t gamma^t ( r(s_t,a_t) + H(pi(.|s_t)) ) ].
```

Two facts from this framework are load-bearing. First, the optimal policy is a Boltzmann
distribution in the soft Q-function, `pi*(a|s) ∝ exp(Q*_soft(s,a))`, with the soft value
`V*(s) = log sum_a exp(Q*(s,a))` supplying the normalizer. Equivalently
`pi*(a|s) = exp(Q*(s,a) - V*(s)) = exp(A*(s,a))`, so the soft advantage is the log of the
optimal policy. Second, IRL becomes a maximum-likelihood
problem over an energy-based model of trajectories: with the dynamics and initial-state
distribution held fixed at the MDP's own and only the reward parametrized,

```
p_theta(tau) ∝ p(s_0) prod_t p(s_{t+1}|s_t,a_t) exp{ sum_t gamma^t r_theta(s_t,a_t) },
```

which under deterministic dynamics reduces to `p_theta(tau) ∝ exp( sum_t gamma^t r_theta )`.
Maximizing `E_{tau~D}[log p_theta(tau)]` is the MaxEnt IRL objective. Its gradient is
`E_D[ sum gamma^t d/dtheta r_theta ] - E_{p_theta}[ sum gamma^t d/dtheta r_theta ]` — a positive phase over
demonstrations minus a negative phase over the model — and the negative phase requires samples
from `p_theta`, equivalently from the partition function `Z = ∫ exp(...)`, which is intractable
in large or continuous spaces.

A separate result frames what reward learning can pin down. **Reward shaping**
(Ng, Harada & Russell 1999) shows that the transformation

```
r_hat(s,a,s') = r(s,a,s') + gamma * Phi(s') - Phi(s)
```

leaves the optimal policy unchanged for *any* potential function `Phi: S -> R`, and — without
prior knowledge of the dynamics — this potential-based class is the *only* class of reward
transformations that preserves the optimal policy. A consequence: an IRL algorithm that only
observes demonstrations from an optimal agent has no way to tell the true reward `r` apart from
any shaped `r_hat` in this class, unless the class of learnable rewards is restricted. With
deterministic dynamics `T`, the shaped reward
`r_hat(s,a) = r(s,a) + gamma*Phi(T(s,a)) - Phi(s)` depends on `T` through the successor state;
change `T` to `T'` and `r_hat` no longer sits in the policy-invariant class for the new MDP, so
the optimal policy under `r_hat` changes.

Two more facts close the background. **Generative adversarial networks** (Goodfellow et al.
2014) train a generator `G` against a discriminator `D`; for a fixed generator the optimal
discriminator is `D*(x) = p(x)/(p(x)+q(x))` where `p` is the data density and `q` the generator
density. And the **occupancy measure** view: a policy `pi` induces an occupancy measure
`rho_pi(s,a) = pi(a|s) sum_t gamma^t P(s_t=s|pi)`, and there is a one-to-one correspondence
between policies and occupancy measures, so "imitate the expert" can be posed as matching
`rho_pi` to the expert's `rho_E`.

## Baselines

These are the methods a new reward-learning algorithm would be measured against and would react
to.

**Maximum entropy / guided cost learning (Ziebart 2008; Finn, Levine & Abbeel 2016a, "Guided
Cost Learning").** MaxEnt IRL fits the energy-based trajectory model above by maximum
likelihood, with the partition function `Z` to estimate. Guided cost learning (GCL)
estimates `Z` with importance sampling, learning an *adaptive sampler* — a policy — that is
improved alongside the cost so that samples concentrate where the model puts mass. Concretely
it alternates: take a gradient step on the cost using demonstrations (positive phase) versus
importance-weighted samples from the sampler (negative phase), and improve the sampler toward
the current cost. A mixture `mu = (1/2) pi + (1/2) p_hat` of the policy and a rough density
estimate `p_hat` on the demonstrations is used as the importance distribution to reduce
variance when the early policy covers the demonstrations poorly. GCL operates over
whole trajectories and uses domain-specific regularization.

**GAN-GCL — adversarial cost learning over trajectories (Finn, Christiano, Abbeel & Levine
2016b, "A Connection between GANs, IRL, and Energy-Based Models").** This work shows that GCL is
*equivalent* to training a GAN with a particular, non-generic discriminator. Instead of letting
`D` directly output `p/(p+q)`, it plugs in the known generator (policy) trajectory density `q(tau)`
and lets the discriminator estimate only the data density via a Boltzmann form:

```
D_theta(tau) = exp{ f_theta(tau) } / ( exp{ f_theta(tau) } + pi(tau) ),
```

i.e. the sigmoid's input is `f_theta(tau) - log pi(tau)` — the learned term minus the filled-in
log generator density. This makes the optimal discriminator independent of the
generator (it is optimal exactly when `exp{f_theta} ∝ p`), and a reward can be read
back out of `f_theta`: at optimality `f*(tau) = R*(tau) + const`. The formulation is
trajectory-centric, so the discriminator's logit is a sum over an entire episode; Finn et al.'s
treatment is theoretical and reports no working implementation.

**GAIL — generative adversarial imitation learning (Ho & Ermon 2016).** Poses imitation as
matching the expert's occupancy measure, and shows that with a particular regularizer the
problem becomes minimizing the Jensen-Shannon divergence between `rho_pi` and `rho_E`. The
practical algorithm is a GAN over single state-action pairs with a *generic* discriminator
`D_w(s,a) -> (0,1)`: alternate an Adam step on `D_w` to increase
`E_pi[log D_w(s,a)] + E_{pi_E}[log(1 - D_w(s,a))]` and a TRPO policy step that minimizes the
same objective, using cost `log D_w(s,a)` (so the policy is rewarded for confusing `D`).
Single-transition discrimination is low-variance and scalable, and it matches expert
performance on continuous control. Because `D_w` is an unstructured classifier, at the
GAN optimum it outputs `0.5` for every `(s,a)`; GAIL recovers a *policy*.

## Evaluation settings

The natural yardsticks already in use for continuous-control imitation and IRL:

- **MuJoCo / Gym locomotion tasks** — HalfCheetah, Hopper, Walker2d (and Ant, Swimmer),
  high-dimensional continuous state and action spaces with unknown dynamics. Expert
  demonstrations are pre-generated trajectories (on the order of tens of demonstrations / a few
  thousand state-action pairs), produced by running a policy optimizer on the true reward.
- **A simple tabular / point-mass MDP** with a known ground-truth reward, used to inspect the
  recovered reward directly (tabular function approximators), where the recovered reward can be
  compared to the true one.
- **Transfer protocol** — learn the reward in one environment, then instantiate a second
  environment that shares the reward but has *modified dynamics* (e.g. a re-articulated agent,
  altered morphology, or changed transition structure), re-optimize a fresh policy under the
  learned reward, and measure how good the resulting policy is under the *true* environment
  reward.
- **Policy optimization** is by a standard on-policy method (TRPO-style trust-region updates
  or a clipped-surrogate variant in a modern harness), with entropy regularization, large
  on-policy batches per update, and the learned reward used as the training signal.
- **Metric:** mean episodic return under the *true* environment reward over evaluation
  episodes (higher is better), in both the matched-dynamics and transfer settings.

## Code framework

The reward-learning algorithm plugs into a standard on-policy adversarial-imitation harness:
an outer loop collects policy rollouts, trains a discriminator to separate expert transitions
from policy transitions, turns the discriminator into a per-step reward, and runs a policy
optimizer (PPO/TRPO) on that reward. What already exists is the GAN-over-transitions scaffold:
a `RewardNetwork` whose internal structure is still open, and a training loop that knows how
to do binary logistic regression of expert vs. policy and how to feed a reward to the policy
optimizer. The open slots are the discriminator score and the reward handed to the policy.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RewardNetwork(nn.Module):
    """Network whose output drives the discriminator and the policy's reward.
    Its internal structure and input dependencies are what we have to design."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        # TODO: the network we will design and the scalar score it produces.

    def forward(self, state, action, next_state, done=None):
        # TODO: the scalar the discriminator and policy reward are built from.
        raise NotImplementedError


class IRLAlgorithm:
    """Adversarial reward learning: alternate (a) train the reward network as the
    learnable part of a binary classifier separating expert from policy
    transitions, and (b) hand a per-step reward to the policy optimizer."""

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos          # dict: obs, acts, next_obs
        self.device = device
        self.args = args
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)

    def compute_reward(self, obs, acts, next_obs, dones, log_policy_act_prob):
        # TODO: turn the reward network's output into the per-step reward fed to PPO.
        raise NotImplementedError

    def update(self, expert_batch, policy_batch):
        # Binary logistic regression of expert vs. policy transitions, where the
        # reward network is the learnable part of the classifier's score.
        # Batches contain obs, acts, next_obs, dones, and policy log-probabilities.
        # TODO: form the per-sample classifier scores, then minimize binary
        #       cross-entropy (expert=1, policy=0).
        raise NotImplementedError
```

The discriminator's exact score and the map from network output to the policy's reward are
the slots the method fills in.
