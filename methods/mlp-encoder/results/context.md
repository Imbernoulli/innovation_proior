## Research question

An agent faces a whole *distribution* of related tasks `p(T)` — each a Markov decision process
that shares structure with the others (same robot, different target velocity; same arena,
different goal) but differs in its reward function or dynamics, both unknown until the agent
acts. We want an agent that, dropped into an unseen task, adapts in a handful of trajectories
rather than learning from scratch. The agent only gets to *observe* the task through the
transitions it collects: a stream of tuples `(s, a, r, s')`. The core question is how to use
a small, growing history of those observations to condition behavior on *this* task without
assuming a fixed, hand-coded task identifier.

## Background

By this time deep RL has shown that powerful function approximators can solve individual
continuous-control tasks, but each task is learned from scratch at a cost of millions of
interactions. The field state on the relevant axes:

- **Meta-learning's matching principle.** The governing assumption in meta-learning is that the
  conditions at meta-training must match those at meta-test: a few-shot image classifier tested
  on 5-example episodes is meta-trained on 5-example episodes (Vinyals et al. 2016). Carried into
  RL, this says: since at test time the agent adapts from data it collects *on-policy*, it must
  be meta-trained on on-policy data too. This principle is the reason essentially all meta-RL of
  the period — RL² (Duan et al. 2016), MAML (Finn et al. 2017), ProMP (Rothfuss et al. 2018),
  MAESN (Gupta et al. 2018) — is on-policy.

- **Off-policy actor-critic methods.** Off-policy actor-critic methods (DDPG, TD3, SAC)
  are empirically one to two orders of magnitude more sample-efficient than on-policy policy
  gradients on the same continuous-control benchmarks, because they reuse a replay buffer
  instead of discarding each batch of experience after one gradient step.

- **Adaptation as inference / a POMDP.** Adapting to an unknown task is formally RL in a
  partially observed MDP where the unobserved part of the state is the task identity (Kaelbling
  et al. 1998). Maintaining a *belief* over the hidden task and acting on it is the principled
  response; Igl et al. (2018) estimate such a belief variationally for general POMDPs. The
  meta-learning setting adds exploitable structure: the task is fixed within an episode, while
  the collected Markov transitions are observations of that same hidden task.

- **Amortized variational inference.** The standard machinery for an intractable posterior
  `p(z | x)` is to train an *inference network* `q_φ(z | x)` that approximates it, optimizing an
  evidence lower bound and backpropagating through sampled `z` via the reparameterization trick
  (Kingma & Welling 2014; Rezende et al. 2014). In the variational autoencoder this maps a
  single datapoint `x` to a latent `z`, with a `KL(q(z|x) || p(z))` term pulling the posterior
  toward a unit-Gaussian prior.

- **The information-bottleneck reading of that KL.** Alemi et al. (2016) show that the same
  `β·KL(q(z|x) || p(z))` term is a variational upper bound on the mutual information `I(z; x)`,
  so penalizing it *compresses* `z` to retain only the bits of `x` that the downstream objective
  actually needs — a regularizer against memorizing the input. This gives the KL a second
  interpretation beyond "match the prior": it is a knob on how much information the latent is
  allowed to carry.

- **Permutation-invariant aggregation of a set.** When the input is an unordered set rather than
  a fixed vector, the function consuming it should not depend on order. Prototypical networks
  (Snell et al. 2017) represent a class by the *mean* of its embedded support examples,
  `v_c = (1/|S_c|) Σ_i h(x_i)` — an order-invariant aggregation of per-element embeddings. Deep
  Sets (Zaheer et al. 2017) make this general: a function on a set is permutation-invariant iff
  it can be written `ρ(Σ_i φ(x_i))`, a per-element map summed and then transformed. These give a
  recipe for consuming variable-sized unordered collections.

- **Posterior sampling for exploration.** In classical RL, posterior sampling (Strens 2000;
  Osband et al. 2013) keeps a distribution over possible MDPs, samples one, and acts optimally
  for it for an entire episode; because the same hypothesis drives a whole episode, the agent
  gets *temporally extended* ("deep") exploration that can test hypotheses whose payoff is
  delayed. Osband et al. (2016) realize a version of this in deep RL via an approximate
  posterior over value functions maintained with bootstraps. The prerequisite is a representation
  of *uncertainty* to sample from.

## Baselines

The prior methods a new adaptation mechanism would be measured against and reacts to.

**RL² / recurrent meta-RL (Duan et al. 2016; Wang et al. 2016).** Run a recurrent network across
the stream of transitions within and across episodes; its hidden state aggregates experience,
and the policy is conditioned on that hidden state. Meta-training is on-policy policy-gradient.
Core idea: let the RNN learn its own adaptation rule end-to-end.

**MAML (Finn et al. 2017).** Meta-learn an initialization such that one (or a few) policy-
gradient steps on a new task's on-policy data yields a good task policy. Core idea: adaptation =
gradient descent from a learned starting point.

**MAESN (Gupta et al. 2018).** Introduces per-task latent variables and is probabilistic,
and adapts the latent by gradient descent on on-policy data and explores by sampling the latent
from its *prior*. Core idea: structured exploration through a learned latent noise distribution.

**Soft actor-critic (Haarnoja et al. 2018).** The off-policy backbone of the period. Maximizes
entropy-augmented return; learns a soft critic with a clipped double-Q target and a
reparameterized squashed-Gaussian actor:
```
y = r + γ ( min_{j=1,2} Q̄_j(s', ã') − α log π(ã' | s') ),   ã' ~ π(· | s')
L_critic = E[(Q(s,a) − y)²],   L_actor = E[ α log π(ã|s) − min_i Q_i(s, ã) ],  ã = tanh(μ+σ⊙ξ)
```
Sample-efficient and stable, with a probabilistic policy.

**Variational POMDP belief estimation (Igl et al. 2018).** Estimate a belief over hidden state
in a general POMDP via amortized variational inference, conditioning the policy on the belief.
Core idea: principled uncertainty over the unobserved state.

## Evaluation settings

The natural yardsticks already in use, all simulated in MuJoCo (Todorov et al. 2012):

- **Locomotion task families varying the reward** — Half-Cheetah-Vel (achieve a target forward
  velocity), Half-Cheetah-Fwd-Back and Ant-Fwd-Back (move forward vs. backward),
  Humanoid-Direc-2D and Ant-Goal-2D (target direction / goal on a 2-D grid). Tasks differ only
  in the reward function. Introduced as meta-RL benchmarks by Finn et al. (2017) and Rothfuss
  et al. (2018).
- **Locomotion varying the dynamics** — Walker-2D-Params (random system parameters), where tasks
  differ in transition dynamics rather than reward.
- **Sparse-reward 2-D navigation** — a point robot must reach an unseen goal on a semicircle,
  with reward given only inside a small radius around the goal (radii such as 0.2 and 0.8). Dense
  reward may be assumed at meta-train time as a simplification.
- Protocol: a fixed set of training tasks and a disjoint set of held-out test tasks sampled from
  `p(T)`; horizon 200; adaptation performed at the trajectory level (collect a trajectory,
  update the task representation, repeat); the figure of merit is test-task return as a function
  of environment samples consumed during meta-training, averaged over several seeds.

## Code framework

The adaptation mechanism plugs into an off-policy actor-critic harness that already exists: a
replay buffer per training task; actor and critic modules that can be given an additional
task-conditioning input; a sampler that draws transition batches; and a training loop that
updates actor and critic from off-policy batches. What does not exist yet is the module that
constructs that task-conditioning input from the history collected in the current task. That
module is left as one empty design slot.

```python
class TaskAdaptationSlot:
    """Turns task history into the conditioning object used by actor and critic."""

    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.task_state = None

    def clear(self, num_tasks=1):
        self.task_state = None

    def infer(self, context):
        # context: (num_tasks, num_observations, feature_dim)
        # TODO: design the task-conditioning object.
        raise NotImplementedError

def train_step(task_slot, policy, qf1, qf2, vf, batch, context):
    task_state = task_slot.infer(context)
    obs, actions, rewards, next_obs, terms = batch
    q1 = qf1(obs, actions, task_state)
    q2 = qf2(obs, actions, task_state)
    # ... soft Bellman target, actor/critic/value updates (existing SAC machinery) ...
    # the empty task-adaptation slot is updated by backprop from this loop
```

The task-adaptation body is the empty slot; everything else — buffer, actor, double critic, soft
Bellman update — is the pre-existing off-policy machinery.
