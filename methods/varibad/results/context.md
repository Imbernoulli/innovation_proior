# Context: fast adaptation under task uncertainty in deep meta-RL (circa 2018-2019)

## Research question

An agent is dropped into an environment it has never seen, drawn from a known family of related
environments. It can interact, but every interaction in an unknown environment is a gamble between
*learning about the environment* (exploration) and *acting on what it already believes*
(exploitation). The quantity we care about is the **return accumulated during learning** — the agent
should be good *from the first steps*, not only after a long separate adaptation phase. This matters
acutely wherever the early interactions are themselves costly: treatment decisions, tutoring, a robot
that must not waste its first hundred actions flailing.

The optimal way to balance that trade-off is known in principle. A policy that conditions its action
not only on the current state but on its own *uncertainty about which environment it is in* will
explore exactly as much as pays off within the time it has left, and no more. Computing such a policy
requires knowing how to parameterise the unknown reward and transition functions, maintaining a
posterior belief over them as data arrives, and planning in the space of beliefs.

So the precise problem is: given a distribution over related tasks to train on, produce an agent that,
at test time on a *held-out* task, adapts within a handful of interactions and maximises online return.

## Background

By this time, meta-reinforcement learning is an established setting. We have a distribution `p(M)`
over Markov decision processes (MDPs), each `M_i = (S, A, R_i, T_i, T_{i,0}, gamma, H)` sharing the
state/action spaces, discount, and horizon but with task-specific reward `R_i` and transition `T_i`
functions that vary while sharing structure (e.g. a goal position, a target velocity). We meta-train
by repeatedly sampling tasks; at meta-test we draw fresh tasks from `p(M)` and measure the average
return the agent achieves *during learning*. Doing this well demands two things at once: carrying over
prior knowledge from related tasks, and reasoning about task uncertainty when choosing actions.

The principled formalism for optimal behaviour under task uncertainty is the **Bayes-Adaptive MDP
(BAMDP)** (Bellman 1956; Duff & Barto 2002; surveyed by Ghavamzadeh et al. 2015). Put a prior
`b_0 = p(R, T)` on the unknown reward and transition functions. As the agent collects experience
`tau_{:t} = (s_0, a_0, r_1, s_1, ..., s_t)`, it maintains a belief `b_t = p(R, T | tau_{:t})`, the
posterior over the MDP. Augment the environment state with this belief to form a **hyper-state**
`s^+_t = (s_t, b_t)` in `S^+ = S x B`. Hyper-states transition by

```
T^+(s^+_{t+1} | s^+_t, a_t, r_t) = E_{b_t}[ T(s_{t+1} | s_t, a_t) ] * delta( b_{t+1} = p(R,T | tau_{:t+1}) ),
```

i.e. the environment state advances under the current posterior over dynamics, and the belief updates
deterministically by Bayes' rule. The reward on hyper-states is `R^+ = E_{b_{t+1}}[R(s_t, a_t,
s_{t+1})]`. The agent maximises `J^+(pi) = E[ sum_t gamma^t R^+ ]` over the BAMDP horizon `H^+`. The
optimiser of `J^+` is the **Bayes-optimal policy**: it takes information-seeking actions only insofar
as reducing its uncertainty raises expected return within the horizon. Note `H^+` need not equal the
single-episode horizon `H` — one often wants the agent to be Bayes-optimal across the first `N`
episodes, `H^+ = N x H`, since how much exploration is worth it depends on how much time remains. A
BAMDP is a special case of a belief MDP / partially observable MDP, with the crucial structural
property that the hidden quantity — the task's `(R, T)` — is **constant** over the task rather than
changing each step.

A separate line of work supplies a tool for cheap approximate posterior inference. In **amortised
variational inference** (Kingma & Welling 2014), an intractable posterior `p(z|x)` is replaced by a
learned recognition network `q_phi(z|x)` trained jointly with a generative decoder `p_theta(x|z)` by
maximising the evidence lower bound

```
log p(x) >= E_{q_phi(z|x)}[ log p_theta(x|z) ] - KL( q_phi(z|x) || p(z) ),
```

optimised by gradient descent through the reparameterisation trick `z = mu + sigma . eps`,
`eps ~ N(0,I)`. The payoff is *fast* inference: once trained, a single forward pass of the recognition
net produces approximate posterior parameters. With a diagonal-Gaussian `q`, the entire posterior is
summarised by two small vectors, a mean and a (log-)variance.

A complementary observation about recurrent agents closes the conceptual picture: when a recurrent
network is trained to act well across a family of tasks, receiving its past actions and rewards as
input, its hidden state comes to track the statistics needed to act — it implicitly performs
something like Bayesian filtering of the data stream, amortising the posterior update into the
recurrence (Ortega et al. 2019). Recurrence is thus a candidate substrate for online inference about
the task as experience accumulates.

## Baselines

These are the prior methods a new meta-RL agent would be measured against and would react to.

**Posterior sampling / Thompson sampling for RL** (Thompson 1933; Strens 2000; Osband et al. 2013).
Maintain a posterior over MDPs; periodically (e.g. at the start of each episode) sample one hypothesis
MDP from it, compute the policy optimal *for that single sampled MDP*, and follow it until the next
sample. Core idea: randomise over your uncertainty to explore.

**Classical Bayesian RL / BAMDP planners** (Asmuth & Littman 2011; Guez et al. 2012, 2013; Brunskill
2012; Poupart et al. 2006). These attack the BAMDP head-on with sample-based tree search or analytic
solutions over a belief that is updated by an explicit, hand-specified prior and Bayes rule. Core
idea: approximate Bayes-optimal planning directly.

**RL² / "learning to reinforcement learn"** (Duan et al. 2016; Wang et al. 2016). Structure the agent
as a recurrent network whose input at each step is the observation concatenated with the *previous
action, previous reward, and a done flag*. Preserve the hidden state across episodes within a "trial"
(a sequence of episodes on one fixed MDP) and train, by ordinary RL, to maximise return over the whole
trial. The within-task learning then lives entirely in the recurrent dynamics: the net becomes a
fast learning algorithm, distilled by slow outer-loop RL. Core idea: let a black-box RNN discover the
adaptation procedure.

**Gradient-based meta-RL: MAML and descendants** (Finn et al. 2017; ProMP, Rothfuss et al. 2019;
E-MAML, Stadie et al. 2018). Meta-learn an initialisation such that a few policy-gradient steps on a
new task yield a good policy. Core idea: adaptation = a short inner-loop optimisation from a learned
start. Lightweight (typically a feedforward policy).

**Probabilistic-context off-policy meta-RL: PEARL** (Rakelly et al. 2019). Encode the collected
context transitions into a probabilistic latent with a *permutation-invariant* encoder — a product of
per-transition Gaussians — and train it alongside an off-policy SAC backbone; explore by sampling a
latent and acting greedily for it. Core idea: a probabilistic task latent + posterior-sampling-style
exploration, made sample-efficient by off-policy training.

**Supervised task-inference** (Humplik et al. 2019). Condition the policy on a posterior over the MDP,
but meta-train the inference network with *privileged* supervision — true task descriptions or IDs.
Core idea: directly learn the belief, supervised by ground-truth task labels.

## Evaluation settings

The natural yardsticks already in use for meta-RL at this time:

- **A didactic gridworld** for which the Bayes-optimal and posterior-sampling exploration strategies
  can be hard-coded for reference. The agent starts in a corner and must reach an *unknown* goal in a
  designated region; the posterior over goal cells is uniform over not-yet-excluded cells. Trained and
  evaluated over a BAMDP horizon of several short episodes (e.g. `H^+ = N x H`, with the agent reset
  to the start each episode). The metric is average return per episode across all possible goals; the
  reference curves are the privileged-goal optimal policy, the hard-coded Bayes-optimal strategy, and
  hard-coded posterior sampling.
- **MuJoCo continuous-control meta-RL families** widely used in the meta-RL literature: half-cheetah
  with a per-task *target velocity* (dense reward for matching the velocity), half-cheetah with a
  per-task *direction* (run left vs right), and locomotion families where the dynamics vary across
  tasks. Continuous state and action spaces; an episode horizon of a couple hundred steps; the BAMDP
  horizon spanning a small number of episodes per task. Metric: online return on held-out test tasks,
  reported at the first rollout and at later rollouts to show how fast the agent adapts.
- Protocol: meta-train on the training task distribution; at meta-test, roll out the policy in
  randomly sampled held-out tasks via forward passes only (no privileged task information). Compare
  against the meta-RL baselines above and, where available, the hard-coded reference strategies.

## Code framework

The agent plugs into a standard meta-RL training harness that already exists. There is an outer loop
over meta-training iterations; within each, the agent is rolled out for a batch of steps across
sampled tasks, the transitions are stored, and a learner performs gradient updates. A standard
on-policy RL algorithm (an actor-critic with a policy network and value head) is available to optimise
return, with its own optimiser and rollout buffer. Diagonal-Gaussian / categorical policy heads, MLP
feature extractors, a GRU primitive, and the reparameterisation sampler are all standard building
blocks on hand.

What is *not* settled is how the agent should represent and infer the task from its stream of
experience, and how the policy should be conditioned so as to adapt online — that representation and
its training signal are exactly what is to be designed. So the substrate is only the generic harness,
with one big empty slot where the contribution will go:

```python
import torch
import torch.nn as nn


def build_mlp(in_dim, out_dim, hidden, n_layers):
    """Generic MLP feature extractor (already available)."""
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def build_policy(state_dim, task_repr_dim, action_dim, hidden):
    """Standard actor-critic policy head conditioned on state plus a task
    representation of size task_repr_dim. Returns (action, value, log_prob)."""
    ...  # existing building block


class MetaRLAgent(nn.Module):
    """The agent we are designing. It sees a stream of transitions on an
    unknown task and must (a) form some representation of the task from the
    experience so far, and (b) condition the policy on it so as to adapt
    online. How the task is represented, how it is inferred from experience,
    and what trains that inference are exactly the open questions."""

    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        # TODO: the task-representation module(s) we will design, and
        #       whatever is needed to train them. (open)
        self.policy = None  # build_policy(state_dim, <task_repr_dim?>, action_dim, hidden)

    def task_representation(self, context):
        """Map the experience collected so far on the current task to whatever
        representation the policy will condition on."""
        # TODO: the inference we will design. (open)
        raise NotImplementedError

    def act(self, state, context, deterministic=False):
        # TODO: condition the policy on the state and the task representation. (open)
        raise NotImplementedError


def auxiliary_objective(agent, trajectories):
    """Any extra training signal the task-representation module needs, beyond
    the RL return. What this should be — if anything — is open."""
    # TODO: the representation-learning objective we will design. (open)
    raise NotImplementedError


# existing meta-RL training loop the agent plugs into
def meta_train(agent, task_distribution, rl_algorithm, num_iters):
    for _ in range(num_iters):
        tasks = task_distribution.sample_batch()
        trajectories = rollout(agent, tasks)          # collect experience
        rl_algorithm.update(agent.policy, trajectories)   # optimise return
        aux = auxiliary_objective(agent, trajectories)    # train the representation
        # (separate optimisers / buffers as the design requires)
```

The outer loop, the RL optimiser, and the standard network primitives are given; the agent's task
representation, its inference from experience, and its training objective are the empty slots the
method will fill.
