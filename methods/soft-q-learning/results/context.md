# Context

## Research question

How can an agent learn a genuinely stochastic, *multimodal* policy over continuous states and actions — one that captures the entire range of good ways to perform a task, not just one — and do so with a policy representation expressive enough to be arbitrarily multimodal?

The standard reinforcement-learning objective maximizes expected return, and under full observability its optimum is a deterministic policy. Stochasticity, when present, is added by hand: injecting exploration noise, or initializing a parametric stochastic policy with high entropy that then collapses. But there are tasks where a stochastic policy is genuinely preferable as the *solution*, not just as an exploration trick:

- **Multimodal reward landscapes.** When several distinct behaviors are (near-)equally good, a unimodal policy must pick one mode early and risks committing to a suboptimal one. A policy that keeps all good modes alive explores better.
- **Compositionality / transfer.** A policy that has learned *all* the ways to, say, move forward is a far better initialization for finetuning to a specific skill than a near-deterministic expert that only knows one way.
- **Robustness** to perturbations and uncertain dynamics — multiple ways to accomplish a task give more options to recover.

Achieving this requires two things at once: (1) an *objective* that rewards stochasticity rather than penalizing it, and (2) a *representation* for the policy distribution rich enough to be multimodal in a high-dimensional continuous action space — together with a *tractable* way to train it and to draw actions from it online.

## Background

**Control as inference / maximum-entropy RL.** Casting optimal control as probabilistic inference (Todorov 2008; Toussaint 2009) yields stochastic policies as the *optimal* answer: instead of the single lowest-cost behavior, one infers a distribution over actions that captures the whole range of low-cost behaviors, explicitly maximizing the policy's entropy. The resulting objective augments reward with an entropy bonus at every visited state. This is distinct from greedy Boltzmann exploration, which maximizes entropy only at the current step; the maximum-entropy objective maximizes the entropy of the entire *trajectory* distribution, so it plans to reach future states where it will *also* have high entropy (Ziebart 2008, 2010; Levine 2014).

**Soft / entropy-regularized value functions.** When the objective carries an entropy term, the Bellman recursion picks up a "soft" form: the hard maximum over next actions is replaced by a log-sum-exp (a "soft max") — `V(s) = α log ∫ exp(Q(s,a)/α) da`. As the temperature `α → 0` this collapses back to the ordinary greedy maximum. This soft backup appears in linearly-solvable MDPs / Z-learning (Todorov 2006), path-integral control (Kappen 2005), G-learning (Fox et al. 2015), and Ψ-learning (Rawlik 2012).

**Energy-based models (EBMs).** A distribution written as `p(x) ∝ exp(−E(x))` can represent *any* density given an expressive enough energy `E`; with a neural-network energy this is a universal family. The price is that the normalizer (partition function) is intractable, so both density evaluation and sampling are hard. A standard EBM toolkit exists for the sampling problem: MCMC (slow, sequential), or training a separate feed-forward *sampling network* whose outputs approximate draws from the EBM (Zhao 2016; Kim 2016).

**Stein variational gradient descent (SVGD).** Liu & Wang (2016) give a deterministic particle method that transports a set of particles toward a target `p(x) ∝ exp(−E(x))` known only up to normalization. The optimal perturbation of the particles — the steepest descent of `KL(q‖p)` within the unit ball of a reproducing-kernel Hilbert space with kernel `κ` — is `φ*(·) = E_{x∼q}[ κ(x,·) ∇_x log p(x) + ∇_x κ(x,·) ]`. The first term pulls particles toward high-probability regions; the second is a repulsive term that keeps them from collapsing together, so the particle cloud covers the modes. Feng, Wang & Liu's *amortized* SVGD (2017) turns this into the training signal for a stochastic neural sampler: nudge the network's output along `φ*` and backpropagate that displacement into the network's weights, yielding a fast feed-forward sampler for the target.

**Off-policy value learning with deep networks.** Deep Q-learning (Mnih 2013, 2015) established the practical machinery this problem can borrow: a replay buffer to decorrelate samples, and a delayed *target network* to stabilize bootstrapped value regression. Optimization is by ADAM (Kingma & Ba 2014).

## Baselines

- **DDPG** (Lillicrap et al. 2015; deterministic policy gradient, Silver 2014; NFQCA, Hafner 2011). Maintains a Q-function critic updated with *hard* Bellman targets, and an actor trained by backpropagating the critic's action-gradient `∇_a Q` into the policy network. The actor thus chases the single `argmax_a Q(s,a)` — a deterministic, unimodal MAP action. Strong sample efficiency on continuous control, but cannot represent or explore multiple modes; it commits to one.
- **Normalized advantage functions (NAF)** (Gu et al. 2016). Makes continuous Q-learning tractable by forcing the Q-function to be quadratic in the action (advantage = a negative-definite quadratic), so the max and the greedy action are closed-form. Tractable but the implied action distribution is a single Gaussian — unimodal by construction.
- **PGQ / entropy-regularized policy gradient** (O'Donoghue et al. 2016). Connects Boltzmann exploration and policy gradients by adding a per-step entropy bonus. But it maximizes entropy only at the *current* timestep, not the trajectory's entropy, so it does not plan toward future high-entropy states, and it is demonstrated with simple (e.g. discrete multinomial) policy classes.
- **Stochastic policies with simple parametric families** (e.g. conditional Gaussian, as in Rawlik 2012; multinomial in PGQ). Even with a neural network producing the *parameters*, the distribution itself stays unimodal — the representational power is capped by the family, so it cannot capture genuinely multimodal behavior.
- **Tabular / analytically-normalizable maximum-entropy methods** (Z-learning, MaxEnt IRL, message-passing inference, G-learning). Solve the maximum-entropy problem exactly but only in discrete/tabular settings or with normalizable distributions; they do not scale to high-dimensional continuous spaces with expressive energies.

The common gap: prior continuous methods are either deterministic/unimodal, or capped by a simple distribution family; prior expressive maximum-entropy methods stall at discrete/tabular settings. No existing approach delivers a genuinely multimodal policy in high-dimensional continuous spaces while staying fast enough to act online.

## Evaluation settings

- **Didactic 2D multi-goal environment.** A point mass at the origin must reach one of four symmetric goals; reward is a mixture of Gaussians at the goals. Used to inspect whether the learned policy is genuinely multimodal — the optimal maximum-entropy behavior picks all four goals at random and the action distribution should track the Q-landscape (unimodal/convex/bimodal at different states).
- **Exploration tasks with multimodal structure.** (1) A simulated swimming snake rewarded by speed along an axis (either direction), with a larger bonus past a "finish line" — the good strategy is to explore both directions until the bonus is found. (2) A quadrupedal 3D robot (adapted from Schulman 2015) navigating a maze with two initially-identical passages (one blocked), reward a Gaussian at the target — again requires keeping both options open.
- **Pretraining / finetuning.** Pretrain a quadruped to locomote in arbitrary directions (reward = center-of-mass speed), then finetune in hallway / narrow-hallway / U-maze environments that demand a specific direction.
- **Protocol.** Infinite-horizon discounted MDP with continuous state/action spaces; simulated robotics (MuJoCo-style). Comparison method: DDPG, itself shown stronger than REINFORCE, TRPO, and A3C on these continuous-control tasks. Metrics: traveled distance / minimum distance-to-goal over training (exploration), and average discounted return over training iterations across random seeds (finetuning).

## Code framework

The pieces that already exist: a continuous-control environment with a replay buffer, a feed-forward network builder, ADAM, and the deep-Q-learning scaffolding of target networks and minibatch bootstrapping. We have a Q-function approximator `Q_θ(s,a)` and we must design two things — how the value-regression target is formed, and how a stochastic policy is represented and trained. The slots below are empty.

```python
import numpy as np, tensorflow as tf

EPS = 1e-6

def feedforward_net(inputs, layer_sizes, activation_fn, output_nonlinearity=None):
    # standard MLP: concatenate inputs, stack Dense+activation layers
    ...

def kernel_fn(xs, ys):
    """A kernel over action particles and its gradient w.r.t. the first arg.
    Returns {'output': K(xs,ys), 'gradient': d/dxs K(xs,ys)}."""
    # TODO: choose a kernel for the particle sampler we will design
    pass

class Policy:
    """A state-conditioned policy. How it represents the action distribution
    is exactly the thing to be designed."""
    def __init__(self, env_spec, hidden_layer_sizes):
        self._action_dim = ...
        self._obs_dim = ...

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        # TODO: produce action(s) for given observations.
        pass

class ValueBasedAlgorithm:
    def __init__(self, env, pool, qf, policy, qf_lr=1e-3, policy_lr=1e-4,
                 discount=0.99, reward_scale=1, td_target_update_interval=1):
        self.env, self.pool, self.qf, self.policy = env, pool, qf, policy
        self._create_placeholders()
        self._create_td_update()      # value-regression loss
        self._create_policy_update()  # how the policy is trained
        self._create_target_ops()

    def _create_td_update(self):
        # TODO: form the bootstrapped target for the next state and regress Q onto it.
        pass

    def _create_policy_update(self):
        # TODO: train the policy against the current Q-function.
        pass

    def _create_target_ops(self):
        # delayed hard copy of online params into target params
        ...

    def _do_training(self, iteration, batch):
        self._sess.run(self._training_ops, self._get_feed_dict(batch))
        if iteration % self._qf_target_update_interval == 0:
            self._sess.run(self._target_ops)
```
