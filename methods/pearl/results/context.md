# Context: meta-reinforcement learning for fast adaptation (circa 2017-2019)

## Research question

We want an agent that, after training across a distribution of tasks `p(T)`, can solve a *new*
task drawn from that distribution from only a small amount of interaction. Each task is a Markov
decision process `T = {p(s_0), p(s'|s,a), r(s,a)}` with the transition and reward unknown but
samplable; tasks differ in their reward function (e.g. a target velocity or goal location) or in
their dynamics (e.g. different physical parameters). The agent never observes which task it is in;
it must infer that from the transitions `(s,a,r,s')` it collects, and turn that inference into good
behavior fast.

Two distinct kinds of sample-efficiency are at stake, and the pain is that existing methods buy one
by paying heavily in the other:

1. **Adaptation efficiency** — at test time, reach good performance on the new task after as few
   interactions as possible. This is the headline promise of meta-learning, and it is hardest
   precisely when reward is *sparse*: if the agent only gets a signal once it stumbles onto the
   goal, then identifying the task requires *exploring* deliberately, and naive per-timestep
   action noise wanders without ever testing a coherent hypothesis. Fast adaptation in sparse
   reward demands reasoning about *which tasks are still possible* and acting to disambiguate them.

2. **Meta-training efficiency** — the number of environment samples consumed *while learning to
   adapt*. This is the cost that is usually swept under the rug. Meta-RL trains across many tasks
   and many trials per task, so if the meta-training algorithm cannot reuse past data, the total
   sample count explodes — millions of interactions per task, times a large task set.

A solution would have to deliver both at once: drive meta-training with reusable (off-policy) data
so the meta-learner is cheap to train, *and* equip the test-time agent with a structured way to
explore an unfamiliar task and home in on it from a handful of trajectories. The tension between
these two — explained below — is the crux.

## Background

**The meta-RL setup.** Meta-RL assumes a task distribution `p(T)` and a set of training tasks
sampled from it. Meta-training learns some adaptation procedure; meta-testing measures how well a
*held-out* task is solved after limited interaction. Two broad families had emerged. **Context-based**
methods (Duan et al. 2016, "RL^2"; Wang et al. 2016, "Learning to reinforcement learn") aggregate the
stream of recent experience into a hidden representation — typically the hidden state of a recurrent
network — and condition the policy on it; the network *is* the adaptation, learned end to end. The
hidden state ingests `(s,a,r,s')` tuples as they arrive and the policy reads off it. **Gradient-based**
methods (Finn et al. 2017, "MAML"; Rothfuss et al. 2018, "ProMP") learn an initialization (or a loss,
or hyperparameters) such that a few policy-gradient steps on the new task's data produce a good policy;
adaptation is literally fine-tuning.

**A structural fact about tasks: the context is a set.** Identifying an MDP from experience does not
depend on the order of the transitions. A collection `{(s_i,a_i,s'_i,r_i)}` carries the same
information about the reward and dynamics however it is permuted: the Markov model is specified by
local transition and reward laws, not by the order in which those local observations happened to
arrive. So whatever summarizes the context to identify the task should treat that context as an
unordered set, not a sequence.

**Amortized variational inference.** A latent-variable model `z ~ p(z)`, `x ~ p_θ(x|z)` can be
trained even when the posterior `p(z|x)` is intractable by introducing a *recognition network*
`q_φ(z|x)` that predicts the variational parameters with one shared network (Kingma & Welling 2014;
Rezende et al. 2014). The training objective is the evidence lower bound, `log p(x) ≥
E_{q_φ}[log p_θ(x|z)] − KL(q_φ(z|x) ‖ p(z))`, and the **reparameterization trick** — write
`z = μ_φ(x) + σ_φ(x) ⊙ ε`, `ε ~ N(0,I)`, so the sampling measure no longer depends on `φ` — lets
gradients flow through the sampled `z` with low variance. The same KL-to-a-prior term reappears in
the **deep variational information bottleneck** (Alemi et al. 2016): `KL(q_φ(z|x) ‖ p(z))` is a
variational upper bound on the mutual information `I(Z;X)`, so penalizing it squeezes the
representation toward keeping only the information about `x` that the downstream objective actually
needs.

**Permutation-invariant set embeddings.** In few-shot supervised learning, prototypical networks
(Snell et al. 2017) embed each support example with a shared network and aggregate by *averaging* the
embeddings into a class prototype; the average makes the representation invariant to the order and
count of the support examples. This is the canonical recipe for turning a variable-size set into a
fixed-size summary.

**Posterior sampling for exploration.** In classical RL, posterior sampling (Strens 2000; Osband
et al. 2013) maintains a posterior over MDPs, draws one MDP from it, and acts *optimally for that
sampled MDP* for the duration of an episode. Because the agent commits to a single hypothesis for a
whole episode rather than re-randomizing each step, it performs *temporally extended* ("deep")
exploration — it can take a coherent sequence of actions to test a hypothesis even when no single
action is immediately informative. As experience accumulates the posterior concentrates and behavior
becomes more and more optimal. Osband et al. (2016) realized a version of this in deep RL by
maintaining an approximate posterior over value functions via bootstrap. Relatedly, adapting to an
unknown task can be framed as RL in a partially observed MDP (Kaelbling et al. 1998) with the task
identity as the hidden state, where one maintains a *belief* over that hidden state (Igl et al. 2018).

**Maximum-entropy off-policy actor-critic.** Soft actor-critic (Haarnoja et al. 2018) is an
off-policy actor-critic for continuous control that optimizes the maximum-entropy objective
`Σ_t E[r(s_t,a_t) + α H(π(·|s_t))]`. It is built from soft policy iteration: a soft value
`V(s) = E_{a~π}[Q(s,a) − α log π(a|s)]`, a soft Bellman backup for the critic, and a policy improvement
step that is an information projection of a Gaussian policy onto `exp(Q/α)`. In practice it trains two
Q-networks (taking their minimum to curb overestimation), a separate value network with a slowly
tracked target, and a squashed-Gaussian actor whose actions are reparameterized so the policy
gradient flows pathwise through `−∇_a Q`. Because it learns from a replay buffer, it is far more
sample-efficient than on-policy methods, and its stochastic actor and probabilistic
(energy-based) interpretation make it natural to extend.

**The diagnostic that frames the problem.** Modern meta-learning is built on the principle that the
data distribution used to adapt should *match* between meta-training and meta-testing — a five-shot
image classifier is meta-trained on five-example episodes because that is what it will face at test
(Vinyals et al. 2016). At test time, an RL agent adapts from the *on-policy* experience it gathers by
exploring the new task. Taken literally, the matching principle then says meta-training must also use
on-policy data — which is exactly why every efficient context- and gradient-based meta-RL method of
the time is on-policy, and why their meta-training is so expensive. Replaying old, off-policy
transitions to train the meta-learner clashes with this principle: those transitions are
systematically unlike the on-policy data the adapted agent will encounter. It had further been
observed that value-based off-policy methods, which only minimize temporal-difference error, have no
direct handle on the *distribution of states the agent visits*, so they cannot by themselves be made
to learn an exploration strategy — whereas policy-gradient methods, which act directly on the policy,
can, but are on-policy and inefficient. Attempts to graft recurrent context encoders onto off-policy
value learning (recurrent DDPG, Heess et al. 2015) had been demonstrated only on much simpler or
discrete tasks (Hausknecht & Stone 2015), and were observed not to train well on these continuous
control meta-RL benchmarks.

## Baselines

**RL^2 / recurrent context meta-RL (Duan et al. 2016; Wang et al. 2016).** A recurrent network
consumes the running stream of `(s,a,r,s')` and the policy conditions on its hidden state; the whole
thing is meta-trained end to end with a policy-gradient algorithm so that, after a few episodes on a
new task, the hidden state encodes enough to act well. Core idea: let one network learn *both* to
infer the task and to act, and let backprop-through-time discover the adaptation rule.
*Limitations:* it is on-policy, so meta-training is sample-inefficient; the single recurrent network
is asked to simultaneously do task inference and control, with no separable belief over the task; it
processes the context as an ordered sequence even though the underlying information is order-free; and
training a recurrent representation with off-policy value learning over long horizons is unstable in
this regime.

**MAML / ProMP (Finn et al. 2017; Rothfuss et al. 2018).** Learn an initialization `θ` such that one
(or a few) inner policy-gradient steps on a new task's trajectories yields a good policy:
`θ'_i = θ − α∇_θ L^{T_i}(θ)`, with `θ` meta-optimized so the post-adaptation return is high. Core
idea: bake fast adaptability into the starting point, so test-time adaptation is plain fine-tuning.
*Limitations:* the inner and outer updates both use on-policy policy gradients, which are
high-variance and sample-hungry; exploration at test time is whatever the pre-adaptation policy
happens to do (sampling its own action noise), with no mechanism to deliberately disambiguate the
task; and the asymptotic returns reached were observed to lag behind what a strong off-policy learner
attains on the same continuous-control families.

**MAESN (Gupta et al. 2018).** The closest prior method that reasons about tasks probabilistically:
it introduces per-task latent variables with a meta-learned prior, adapts by gradient descent *on the
latent variables* for a new task, and explores by sampling the latent from the *prior*. Core idea:
structured exploration via a learned latent space plus gradient-based adaptation of that latent.
*Limitations:* it is on-policy and gradient-based, so meta-training again costs on the order of
`1e8` timesteps; exploration draws the latent from the prior and is not refined *within* test-time
interaction by conditioning on what has been seen so far; and adaptation by latent-space gradient
descent is slower per task than a single forward pass would be.

**Recurrent off-policy value learning (Heess et al. 2015; Hausknecht & Stone 2015).** Memory-based
control with a recurrent network trained by an off-policy value method (recurrent DDPG / DRQN). Core
idea: get off-policy efficiency *and* memory by recurrently encoding history inside a Q-learner.
*Limitations:* it had been shown to work on much simpler or discrete-action POMDPs; it does not
explicitly form a belief over the task, leaving both inference and optimal behavior to the RNN; and
it requires training on trajectories, which correlates the samples and was observed to destabilize
the value updates on these continuous-control task families.

**Soft actor-critic (Haarnoja et al. 2018).** The strongest single-task off-policy continuous-control
learner available (see Background). It is not itself a meta-learner — it solves one fixed MDP — but it
is the natural efficient, stable, replay-driven engine a meta-RL method would want to be built on.
*Limitation as a baseline:* on its own it has no notion of a task distribution or of conditioning on
inferred task context; run per-task from scratch it pays the full single-task sample cost every time.

## Evaluation settings

The natural yardsticks were continuous-control locomotion meta-RL benchmarks simulated in MuJoCo
(Todorov et al. 2012), previously introduced for MAML and ProMP:

- **Reward-varying locomotion families** — Half-Cheetah running forward/backward and at a target
  velocity, Ant forward/backward and to a 2-D goal, Humanoid in a target 2-D direction. Tasks share
  dynamics but differ in the reward function (direction / target speed / goal). Dense reward in most;
  the metric is average return on held-out test tasks after adaptation.
- **Dynamics-varying family** — Walker-2D with randomized system parameters; tasks differ in
  dynamics and the agent must still move forward.
- **Sparse-reward 2-D navigation** — a point robot must reach an unseen goal on a half-circle, with
  reward given only inside a small radius around the goal (radii such as 0.2 and 0.8 tested). This is
  the setting that stresses *structured exploration*: with no signal until the goal is touched,
  per-step noise is hopeless, so it is the discriminating test for whether an agent can explore by
  testing hypotheses.

Protocol: a fixed split of training vs. test tasks per family; a fixed horizon per family (200 steps
for the MuJoCo-style locomotion families, 20 for the point-robot family);
adaptation performed by collecting exploration trajectories on a held-out task and then evaluating;
the yardstick is *test-task return versus number of meta-training samples* — i.e. both final
performance and how many samples meta-training consumed to get there. Returns are averaged over
several random seeds.

## Code framework

The pieces that already exist are an off-policy actor-critic engine and a multi-task data pipeline.
We have a replay buffer per task, an environment that can be reset to any task, an actor-critic
learner (a stochastic policy, Q-functions, a value function, target networks, soft updates), and a
trajectory sampler. What does *not* yet exist is how the agent should summarize the experience it has
collected on a task into something the policy can condition on, and how to meta-train that summary
together with the actor-critic across the task distribution. Those are the empty slots.

```python
import torch
import torch.nn as nn


def build_mlp(input_dim, output_dim, hidden_dim, n_layers):
    """Generic MLP block (exists)."""
    ...


def build_policy(obs_dim, action_dim, latent_dim, net_size):
    """Squashed-Gaussian stochastic actor, conditioned on (obs, task_summary)."""
    ...


def build_qf(obs_dim, action_dim, latent_dim, net_size): ...   # Q(s, a, task_summary)
def build_vf(obs_dim, latent_dim, net_size): ...               # V(s, task_summary)


class MetaRLAgent(nn.Module):
    """Conditions an off-policy actor-critic on a summary of task experience.
    The summary mechanism is exactly what we have to design."""

    def __init__(self, obs_dim, action_dim, latent_dim, net_size):
        super().__init__()
        self.latent_dim = latent_dim
        # TODO: the module that turns collected context (s,a,r,s') into the
        #       task summary the policy/critic condition on -- to be designed.
        self.context_module = None          # <- the contribution goes here
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf  = build_vf(obs_dim, latent_dim, net_size)

    def update_context(self, transition):
        """Append one online transition (s,a,r,s') to the running context."""
        ...                                  # bookkeeping exists

    def adapt(self):
        """Turn the collected context into the task summary the policy uses."""
        # TODO: how to summarize context into the conditioning variable -- to be designed.
        pass

    def get_action(self, obs, deterministic=False):
        # TODO: condition the policy on obs and the current task summary
        pass


class MetaRLAlgorithm:
    """Meta-trains the agent across tasks with an off-policy actor-critic.
    The data the actor-critic sees and the data the context_module sees
    are sampled by the loop below; how to couple them is part of the design."""

    def __init__(self, agent, env, train_tasks, eval_tasks,
                 replay_buffer, enc_replay_buffer, config):
        self.agent = agent
        self.env = env
        self.train_tasks = train_tasks
        self.replay_buffer = replay_buffer            # exists
        self.enc_replay_buffer = enc_replay_buffer    # exists
        # off-policy actor-critic optimizers exist (policy, qf1, qf2, vf)
        # TODO: an optimizer for the context_module once it is designed
        ...

    def train_iteration(self, iteration_idx):
        # 1) collect data on sampled tasks and store in the buffers
        # 2) for each gradient step:
        #    - sample an RL batch (for the actor-critic)
        #    - sample a context batch (for the context_module)
        #    - take a meta-gradient step
        # TODO: the gradient update that ties task summarization to the
        #       actor-critic objective -- to be designed.
        return {}
```

The actor-critic objectives, the replay buffers, the samplers, and the network builders are given.
The single open problem is the `context_module` — what it computes, what it is trained against, and
how its data is drawn relative to the RL data — and the meta-gradient step that ties it to the
actor-critic.
