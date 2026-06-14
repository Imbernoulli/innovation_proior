# Context: task inference and online exploration in meta-reinforcement learning (circa 2018-2019)

## Research question

An agent is dropped into an unknown Markov decision process (MDP) drawn from a known family
`p(M)`: the states, actions, discount, and horizon are shared across the family, but the reward
function `R` and the transition function `T` differ from task to task and are not revealed. The
agent must act well *while it is still figuring out which task it is in*. The yardstick is not
the return after the agent has converged on a new task; it is the expected return accumulated
*during learning*, from the very first steps — the regime that matters in settings where every
interaction has a real cost (a robot, a treatment policy, a tutoring system). Doing well at this
forces the agent to trade off exploration (acting to reduce its uncertainty about which MDP it
faces) against exploitation (acting to collect reward given what it currently believes), and to
do so *online*, within a single deployment.

There is a precise notion of the optimal trade-off. If the agent maintains a posterior belief
over which MDP it is in and acts to maximize expected return *with respect to that belief over
the remaining horizon*, it is Bayes-optimal: it takes an information-seeking action exactly when,
and only when, the expected future return from resolving that uncertainty outweighs the cost of
the detour. The trouble is that computing this policy exactly is intractable for anything beyond
toy problems, and even *maintaining* the belief — the posterior over reward and transition
functions given the experience so far — is itself intractable in general. The goal is a method
that recovers approximately Bayes-optimal online behaviour at the scale of deep RL, given only
that we can sample related tasks during a meta-training phase. The open question is how to turn a
stream of interaction `(s_0, a_0, r_1, s_1, a_1, r_2, …, s_t)` into a usable representation of the
agent's current belief about the task, cheaply and online, and how to train that representation
so the belief is calibrated, sharpens as evidence accumulates, and drives good exploration.

## Background

**The Bayes-adaptive MDP (BAMDP).** The principled formalization of "act optimally under task
uncertainty" is the Bayes-adaptive MDP (Bellman 1956; Duff 2002; reviewed in Ghavamzadeh et al.
2015). Put a prior `b_0 = p(R, T)` over the unknown reward and transition functions. After
observing the history `τ_{:t} = {s_0, a_0, r_1, s_1, …, s_t}`, the agent holds the posterior
`b_t(R, T) = p(R, T | τ_{:t})`, its *belief*. Augmenting the environment state with this belief
gives a hyper-state `s⁺_t = (s_t, b_t) ∈ S × B`. The hyper-state transitions deterministically in
the belief component (Bayes' rule updates `b_t → b_{t+1}` given the new transition) and
stochastically in the environment component (the next state is drawn from the posterior-averaged
transition), so

```
T⁺(s⁺_{t+1} | s⁺_t, a_t, r_t) = E_{b_t}[ T(s_{t+1} | s_t, a_t) ] · δ( b_{t+1} = p(R,T | τ_{:t+1}) ),
R⁺(s⁺_t, a_t, s⁺_{t+1})       = E_{b_{t+1}}[ R(s_t, a_t, s_{t+1}) ].
```

This is a special case of a belief MDP (a POMDP recast with beliefs as states), with the crucial
simplification that in a BAMDP the hidden quantity — the task `(R, T)` — is *constant within a
task*, unlike a general POMDP whose hidden state drifts each step. The Bayes-optimal policy
maximizes `J⁺(π) = E[ Σ_t γ^t R⁺(·) ]` over the BAMDP horizon `H⁺` (which may span several MDP
episodes, `H⁺ = N × H`, so the agent acts Bayes-optimally across `N` episodes). The framework is
beautiful and exactly captures the explore/exploit optimum, but it has three intractabilities,
each load-bearing for what follows: (i) we usually do not know the parametric form of the true
`R` and `T`; (ii) the belief update `p(R, T | τ_{:t})` is intractable; and (iii) even given the
belief, planning in belief space is intractable.

**Posterior sampling as the tractable shortcut, and where it falls short.** The standard way to
sidestep belief-space planning is posterior sampling (Strens 2000; Osband et al. 2013), which
lifts Thompson sampling (Thompson 1933) from bandits to MDPs: periodically — typically at the
start of an episode — sample a single hypothesis MDP from the current posterior, follow the
policy that is optimal *for that sampled MDP* until the next resample, then update the posterior.
Planning is now on an ordinary MDP, not a BAMDP, which is far cheaper. But committing to one
sampled hypothesis for a whole episode explores inefficiently relative to the Bayes-optimal
strategy: the agent will, for instance, revisit states it has already ruled out, because the
sampled hypothesis does not *systematically* reduce uncertainty the way a belief-conditioned plan
would. The didactic picture is a gridworld with an unknown goal: the Bayes-optimal agent sweeps
the still-possible goal cells in a route that shrinks the candidate set fastest, whereas posterior
sampling darts to one sampled goal, finds nothing, resamples another, and so on. Posterior
sampling is the tractability bar to clear; Bayes-optimality is the performance bar to reach.

**Amortized variational inference.** The machinery for turning an intractable posterior into a
fast, learned one comes from the variational auto-encoder (Kingma & Welling 2013; Rezende et al.
2014). For a latent-variable model `p_θ(x, z) = p_θ(x|z) p(z)` with intractable posterior
`p_θ(z|x)`, introduce a *recognition network* (an amortized encoder) `q_φ(z|x)` and bound the log
evidence below:

```
log p_θ(x) = KL( q_φ(z|x) || p_θ(z|x) ) + L(θ, φ; x)  ≥  L(θ, φ; x),
L(θ, φ; x) = E_{q_φ(z|x)}[ log p_θ(x|z) ] − KL( q_φ(z|x) || p(z) ).
```

The first term is a reconstruction term ("decode `z` back to `x`"); the second pulls the
posterior toward the prior. When `q_φ` and `p` are Gaussian the KL is analytic, and the
reconstruction expectation is made differentiable in `φ` by the reparameterization trick
`z = μ_φ(x) + σ_φ(x) ⊙ ε`, `ε ~ N(0, I)`, so a single Monte-Carlo sample gives a low-variance,
backprop-able estimate. Amortization is the key economic property: instead of solving a fresh
optimization for the posterior of every new input, one forward pass through `q_φ` produces it.

**Recurrent networks as online history compressors.** Sequence models — in particular gated
recurrent units (GRU; Cho et al. 2014) and LSTMs (Hochreiter & Schmidhuber 1997) — maintain a
fixed-size hidden state `h_t` that is updated in `O(1)` per step from the previous hidden state
and the current input, `h_t = f(h_{t-1}, x_t)`. They are the standard tool for compressing a
growing, ordered history into a constant-size running summary, and the gating mitigates the
vanishing/exploding-gradient problem that plagues vanilla recurrence over long horizons. This is
the natural substrate whenever a quantity that depends on an entire variable-length history has to
be maintained incrementally.

**The meta-training setup.** All of the above is put to work in the meta-learning setting:
batches of tasks `M_i ~ p(M)` are sampled during meta-training, with the reward/transition pair
drawn from `p(R, T)`; across tasks the dynamics and rewards vary but share structure (a goal
position, a target velocity, a system-parameter vector). A hidden-parameter (HiP-) MDP view
(Doshi-Velez & Konidaris 2016) and the contextual-MDP view (Hallak et al. 2015) both formalize
the assumption that a low-dimensional latent factor indexes the family. At meta-test time the
agent is scored by its average return *during learning* on held-out tasks from `p`. Doing this
well needs two things at once: transferring knowledge acquired on related tasks, and reasoning
about task uncertainty when selecting actions.

## Baselines

**Recurrent black-box meta-RL — RL² (Duan et al. 2016; Wang et al. 2016).** Cast "learning to
learn" as a single RL problem over a recurrent policy. At each step the network receives, in
addition to the state, the *previous action and reward* (and a done flag): the input is
`φ(s_t, a_{t-1}, r_{t-1}, d_t)`, fed to a GRU/LSTM whose output goes through a fully-connected
layer and a softmax over actions. Crucially, the hidden state is *carried across the episodes of a
task* (in Duan et al.) rather than reset, so that adaptation happens entirely in the recurrent
dynamics: after meta-training, the weights are frozen and "learning" within a new task is just the
evolution of the activations as experience streams in. The objective maximizes total reward over a
trial (multiple episodes of a fixed sampled MDP), which forces the network to integrate everything
it has seen and adapt its behaviour continually — an end-to-end-learned RL algorithm living in the
RNN. **Limitation that it leaves open.** The hidden state is a single deterministic vector that the
training signal shapes *only* through the policy-gradient/return objective; nothing in the method
makes the hidden state an explicit representation of *which task* the agent is in or *how
uncertain* it is. There is no distribution to read an uncertainty off of, and the hidden state is
trained purely to be useful for the next action rather than to be a calibrated summary of the task.
In the gridworld and locomotion settings, a high-dimensional deterministic hidden state (e.g.
128-dim) is observed to drift and behave unstably when the task is reset across multiple rollouts,
and there is no separate signal grounding the hidden state in the task structure.

**Probabilistic per-transition context — PEARL (Rakelly et al. 2019).** Learn a latent context
variable `z` summarizing the task, infer its posterior with an amortized encoder, and condition an
off-policy (SAC) actor-critic on samples of `z`; exploration is by posterior sampling over `z`.
The encoder is built to be **permutation-invariant** over the collected transitions `c_{1:N}`: each
transition `c_n = (s_n, a_n, s'_n, r_n)` is passed independently through a network `φ` that emits a
Gaussian factor `Ψ_φ(z | c_n) = N(μ_φ(c_n), σ_φ(c_n))`, and the factors are combined as a product
of Gaussians,

```
q_φ(z | c_{1:N}) ∝ Π_{n=1}^N Ψ_φ(z | c_n),
```

which has a closed form for the combined mean and variance. The posterior is regularized toward a
unit Gaussian, `KL(q_φ(z|c) || N(0, I))`, read as an information bottleneck that keeps `z` to the
minimal task-relevant content. The product form is what lets the encoder operate inside an
off-policy loop: order-invariance means context batches resampled from a replay buffer in any
order give the same posterior, which is convenient when the context-sampling and the RL batches
are decoupled. **Limitation that it leaves open.** The order-invariant product factorizes the
posterior as if the transitions were *conditionally independent given `z`* and carry no sequential
information; the encoder is, by construction, blind to the order in which experience arrived. It
also yields posterior-sampling-style exploration (act according to a sampled `z` for a stretch),
which inherits posterior sampling's inefficiency relative to a belief-conditioned, systematically
uncertainty-reducing strategy, and it is not trained toward an online-Bayes-optimal objective.

**Posterior sampling / Thompson sampling for MDPs (Strens 2000; Osband et al. 2013).** As above:
maintain a posterior over MDPs, sample one, act optimally for it for an episode, repeat. It is the
tractable Bayesian baseline and the conceptual reference point for "use the posterior to explore,"
but its exploration is provably less efficient than Bayes-optimal behaviour and so accrues lower
return during learning.

**Gradient-based meta-RL — MAML and successors (Finn et al. 2017; E-MAML, Stadie et al. 2018;
ProMP, Rothfuss et al. 2018).** Meta-learn an initialization from which a few policy-gradient
steps on a new task yield a good policy; E-MAML and ProMP add terms to account for the exploratory
value of the pre-adaptation trajectories. **Limitation that it leaves open.** These methods
typically separate an exploration phase (collect trajectories) from an exploitation phase (after
the gradient update), spending whole rollouts on exploration before adapting, rather than trading
off the two online within the first episode; they need substantially more interaction in each new
task before they act well.

## Evaluation settings

The natural yardsticks at this time, all measuring return *during learning* on held-out tasks from
`p(M)`:

- **A didactic gridworld** (e.g. `5×5`) with an unobserved goal placed uniformly at random in an
  allowed region, deterministic actions {up, down, left, right, stay}, a sparse reward (small
  negative per step, `+1` at the goal), MDP horizon `H` (e.g. 15) and a BAMDP horizon `H⁺ = N×H`
  spanning several episodes (so the agent must explore until it finds the goal and then exploit on
  resets). Small enough that a hard-coded Bayes-optimal and a hard-coded posterior-sampling policy
  can be computed as reference behaviours, and the agent's belief can be visualized cell-by-cell.
- **MuJoCo continuous-control meta-RL families** widely used in the literature and taken from the
  PEARL reference environments: a half-cheetah that must run at an unknown target velocity
  (`cheetah-vel`), forward/backward direction tasks (cheetah-dir, ant-dir), goal-reaching for an
  ant (ant-goal), and walker tasks with randomized system parameters. Horizon `H = 200`; for online
  evaluation the BAMDP horizon spans one or a few rollouts. Metric: average return on held-out test
  tasks, reported per rollout (the first-rollout return is the quantity of interest for online
  adaptation).
- **Point-robot navigation** families: a 2-D point mass that must reach goals drawn from a region
  (dense negative-distance reward) or from a half-circle with sparse `+1`-within-radius reward,
  testing task inference from dense versus sparse signals.
- **Reference comparators** computed where tractable: a hard-coded optimal policy *with privileged
  access to the true task* (an upper bound), the hard-coded Bayes-optimal policy, and hard-coded
  posterior sampling.

## Code framework

The encoder plugs into a fixed meta-RL harness that already exists: an outer loop that samples
tasks from `p(M)`, rolls out a policy to collect ordered context, calls an experience encoder to
produce a latent task representation, conditions the policy on it, and trains everything by RL plus
(optionally) a model/auxiliary objective. The latent task representation is consumed downstream as
Gaussian parameters (a mean and a variance per latent dimension), which the agent turns into the
task posterior. What is *not* settled — the one empty slot — is the encoder itself: the module that
maps the agent's experience so far into those latent-task parameters, including whatever internal
state it keeps. Everything around it (the data pipeline, the policy/value backbone, the optimizer,
the way the latent is aggregated into the posterior and fed to the policy) is given.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    """Maps the agent's interaction experience into latent-task parameters.

    Input:  per-transition features built from (s, a, r [, s']).
    Output: (*, output_size) numbers that the agent reads as Gaussian task
            parameters (mean and a (log-)variance per latent dimension).
    The agent aggregates the encoder's output into the task posterior and
    conditions the policy on it. The internal structure of this module --
    whether/how it keeps state across a trajectory -- is the open design.
    """

    def __init__(self, hidden_sizes, input_size, output_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # TODO: the encoder architecture we will design — the mapping from
        #       experience to latent-task parameters, plus any state it keeps.
        pass

    def forward(self, input, return_preactivations=False):
        # input: per-transition features for the collected experience.
        # return: (*, output_size) latent-task parameters.
        # TODO: produce the latent-task parameters from the experience.
        pass

    def reset(self, num_tasks=1):
        # Called at the start of a new task, before any experience is seen.
        # TODO: reset any internal state the encoder keeps (if it keeps any).
        pass


# ---- fixed surrounding machinery (already exists) ----------------------------

def product_of_gaussians(mus, sigmas_squared):
    """Combine per-factor Gaussian parameters into one Gaussian posterior."""
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1.0 / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class Agent(nn.Module):
    """Holds the encoder and a policy; infers the task posterior and acts on it."""

    def __init__(self, latent_dim, context_encoder, policy):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_encoder = context_encoder
        self.policy = policy

    def infer_posterior(self, context):
        # encode collected context -> latent-task parameters -> task posterior
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        # aggregate into the posterior over the latent task, sample z, condition policy
        ...

    def reset_belief(self, num_tasks=1):
        # reset the belief to the prior at the start of a new task
        self.context_encoder.reset(num_tasks)
        ...


def meta_train(task_distribution, agent, rl_optimizer):
    for batch_of_tasks in task_distribution:        # sample tasks ~ p(M)
        for task in batch_of_tasks:
            agent.reset_belief()                    # start from the prior
            context = collect_context(agent, task)  # roll out, gather ordered experience
            agent.infer_posterior(context)          # experience -> task posterior
            update_policy(agent, task, rl_optimizer)
            # (optional) train the encoder with an auxiliary model objective
```

The final encoder code fills exactly the `ContextEncoder` stubs above (`__init__`, `forward`,
`reset`); the surrounding loop, the posterior aggregation, and the policy are unchanged.
