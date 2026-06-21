## Research question

The setting is reinforcement learning that must be both *offline* and *meta*. Offline means the
agent never interacts with the environment: it learns a policy from a fixed, pre-collected dataset
of transitions, logged by some unknown behavior policy. Meta means the agent faces a whole
distribution of tasks `p(T)` — each task an MDP with shared state and action space but its own
transition function `P` and reward function `R` — and at test time is handed a small batch of logged
transitions from a *new, unseen* task and must adapt to it, still without any interaction. This is
the regime that would make RL usable where exploration is dangerous or expensive (healthcare,
autonomous driving, controlled-environment agriculture): learn once from static multi-task logs,
then drop into a new task and act well from a handful of recorded transitions.

The **offline** side. A value-based RL learner bootstraps: it regresses `Q(s,a)` toward
`r + γ Q(s',a')` where `a'` is chosen by the *learned* policy. When the learner can collect new data,
an over-optimistic `Q` value at an out-of-distribution `(s',a')` gets corrected the next time that
action is tried. Offline, there is no "next time": the policy-improvement step queries `Q` at OOD
actions, and the values are backed up from the fixed dataset alone.

The **task-inference** side. To behave well on a new task the agent must compress the context (the
few available transitions) into a compact task representation `z`, condition its policy and value
functions on `z`, and learn that compression jointly with control — from few samples, and without
exploring to disambiguate which task it is in.

The question: an end-to-end, model-free algorithm that (1) infers task identity from a few logged
transitions with no test-time interaction; (2) conditions an actor-critic on that inferred task
variable; and (3) trains the whole thing on static multi-task data, across continuous-control
meta-RL benchmarks.

## Background

**Markov decision processes and bootstrapping.** A task is an MDP
`M = (S, A, P, R, ρ_0, γ)`. The action-value function satisfies
`Q_π(s,a) = R(s,a) + γ E_{s'∼P}[V_π(s')]`, and value-based methods iterate a Bellman backup
`(BQ)(s,a) := R(s,a) + γ E_{P}[max_{a'} Q(s',a')]`. With neural-network function approximators the
backup is realized by regression of `Q_ψ` toward a target built from a slowly-updated target network.
A phenomenon studied in the offline setting is **bootstrapping error** (Kumar et al. 2019): deep
networks generalize poorly to actions outside the behavior policy's support, the policy-improvement
step queries `Q` at exactly such out-of-distribution actions, and with no new data those values are
propagated through the backups.

**Behavior-constrained offline RL.** One family of methods keeps the learner close to the behavior
policy. BCQ (Fujimoto et al. 2019) parameterizes the policy as small learned perturbations of
behavior actions, so the policy selects actions near the data; BEAR (Kumar et al. 2019) penalizes the
policy for diverging from the behavior support using a sample-based kernel MMD and takes a
conservative minimum over a target-Q ensemble, with the constraint strength tuned as a Lagrange
multiplier against a divergence threshold. The common theme — regularize the learner toward the
logged transitions — was unified by Wu, Tucker & Nachum (2019) into the **behavior-regularized
actor-critic (BRAC)** framework: insert a divergence `D(π_θ(·|s), π_b(·|s))` between the learner and
behavior policies into the actor-critic objective, either as a *value penalty* added inside the
target Q,
```
V^D_π(s) = Σ_t γ^t E[ R_π(s_t) − α D(π_θ(·|s_t), π_b(·|s_t)) ],
```
or as a *policy regularizer* applied only in the actor update (α set to zero in the Q update). The
divergence `D` can be a kernel MMD, a Wasserstein distance, or any f-divergence. For an f-divergence,
a trick avoids ever estimating a cloned behavior density: every f-divergence has a **dual
(Fenchel) form** (Nowozin et al. 2016),
```
D_f(p,q) = max_g  E_{x∼p}[ g(x) ] − E_{x∼q}[ f*(g(x)) ],
```
with `f*` the convex conjugate of `f`; for KL, `f(x) = −log x` and `f*(t) = −log(−t) − 1`. One then
learns a discriminator `g` by minimax instead of fitting a generative model of the behavior policy.
BCQ and BEAR both reappear as special cases of BRAC. This whole family, inheriting from SAC (Haarnoja
et al. 2018), uses a reward scale and a regularization strength `α` set per environment.

**Context-based meta-RL.** Meta-RL trains across `p(T)` so the agent can adapt quickly to a new task.
The context-based view treats the unknown task as an unobserved part of the state — a latent `z`
inferred from past experience — turning meta-RL into RL on an augmented MDP whose state is `(s, z)`.
An inference network `q_φ(z|c)` maps context `c` (a set of transitions) to `z`, and a universal
policy `π_θ(a|s,z)` and value function condition on it (Schaul et al. 2015). The off-policy instance
PEARL (Rakelly et al. 2019) makes three choices. (i) The encoder is **probabilistic and
permutation-invariant**: each transition `c_n` produces a Gaussian factor `Ψ_φ(z|c_n) = N(μ_n, σ_n)`,
and the factors are combined by a **product of Gaussians**,
```
q_φ(z|c_{1:N}) ∝ Π_n Ψ_φ(z|c_n),    σ² = 1 / Σ_n σ_n^{−2},   μ = σ² Σ_n μ_n σ_n^{−2},
```
so the order of transitions does not matter. (ii) A variational **information bottleneck**: the
encoder is trained with `+ β·KL( q_φ(z|c) ‖ N(0,I) )`, which both constrains `z` to the minimal
task-relevant information and supplies a prior to sample from. (iii) That probabilistic posterior
enables **posterior-sampling exploration** at test time — sample `z` from the prior, act for an
episode, update the posterior from what was seen, and act again as the belief narrows — a
temporally-extended way to explore an unknown task. PEARL's encoder is trained by the **Bellman
gradients** of the critic (Rakelly et al. found recovering the state-action value function works
better than reconstructing rewards/dynamics), and it decouples context sampling from RL-batch
sampling so that off-policy learning is stable.

**Metric-based few-shot learning.** A separate lineage learns task/class structure directly in an
embedding space. Prototypical networks (Snell et al. 2017) embed each support example with a
deterministic network `f_φ`, form a class prototype as the **mean** of its support embeddings
`c_k = (1/|S_k|) Σ f_φ(x_i)`, and classify a query by a softmax over (squared Euclidean) distances to
the prototypes. The choice of squared Euclidean distance is principled: it is the Bregman divergence
whose cluster representative is the mean, so prototype-as-mean and distance-as-squared-Euclidean are
the matched pair (the model is equivalent to mixture-density estimation with a spherical-Gaussian
class density). The embedding is permutation-invariant (a mean) and fully deterministic. This is the
metric-learning view: instead of inferring a posterior, shape the geometry of an embedding so that
same-class points cluster and different-class points separate.

**Distance metric learning.** The workhorse objective of that lineage is the contrastive loss (Chopra
et al. 2005; Hadsell et al. 2006). For a pair of inputs with embeddings `q_i, q_j` and a same/different
label,
```
L_cont(x_i, x_j) = 1{y_i = y_j} ‖q_i − q_j‖₂²  +  1{y_i ≠ y_j} max(0, m − ‖q_i − q_j‖₂)²,
```
a "spring model": same-label pairs are pulled together by an attractive quadratic, different-label
pairs are pushed apart by a margin-`m` hinge. Hadsell et al. note that the attractive term over
similar pairs needs an explicit repulsive term between dissimilar pairs, and that the repulsive margin
term acts like a spring *only within radius `m`*: it exerts zero force once a pair is already farther
than `m` apart, and its short-range force is bounded rather than singular when a dissimilar pair starts
very close together.

## Baselines

**PEARL ported to offline (Rakelly et al. 2019).** Take PEARL — probabilistic permutation-invariant
encoder, KL information bottleneck, posterior-sampling exploration, encoder trained by Bellman
gradients, SAC backbone — and run it on logged data, without exploration at train or test. Core idea
and math as above. The encoder learns through the value-learning (Bellman) gradient.

**Contextual / task-augmented BCQ (Fujimoto et al. 2019).** Put the inferred latent `z` into the
state and run offline BCQ on the augmented MDP, with the task-inference module trained through the
value-learning (Bellman) gradients. Core idea and math as above. The behavior constraint is BCQ's
action-perturbation device.

**Two-stage model-based multi-task batch RL (Li et al. 2019).** Train, per task, an offline BCQ
policy and explicit reward/dynamics models, then distill across tasks with a metric-learning term.
Core idea: bring metric learning into multi-task batch RL via per-task models; the task representation
is read out of the separate model fits.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Continuous-control meta-environments** introduced by Finn et al. (2017) and Rakelly et al.
  (2019): tasks that vary by *reward* (Half-Cheetah-Vel — match a target running velocity;
  Half-Cheetah-Fwd-Back and Ant-Fwd-Back — run forward vs backward; Sparse-Point-Robot — reach a
  goal on a circle with a sparse +reward only near the goal) and tasks that vary by *transition
  dynamics* (Point-Robot-Wind; Walker-2D-Params — randomized mass/inertia/friction). Shared state and
  action spaces within each family; horizon 200 for the MuJoCo families, 20 for the point robot.
- **Static-dataset generation.** For each task, train a per-task SAC agent, save policy checkpoints
  from random through expert quality, and roll them out to log trajectories. The offline dataset for
  each task is a chosen selection of those logs, so the performance level (expert / medium / random)
  and state-action coverage (narrow expert vs diverse mixed) can be controlled — making distribution
  shift between training and testing datasets a tunable axis.
- **Protocol.** Meta-train on the training tasks' static datasets; at test time, on a held-out task,
  draw a batch of logged transitions as context, encode it to `z`, and roll out the policy
  *deterministically* conditioned on `z` with no exploration. The metric is average episodic return
  on the held-out tasks, averaged over random seeds.
- **Embedding diagnostics.** t-SNE or PCA visualizations of per-task embedding vectors, simple
  cluster-separation statistics, and the dimensionality of the latent space.

## Code framework

The substrate is a context-based meta-RL actor-critic harness: a deterministic-MLP toolkit, a
TanhGaussian policy and Q/V networks that take the latent task variable as an extra input, per-task
replay buffers over the static datasets, a context sampler, and an outer meta-training loop. The
open slot is how the agent turns context into a task variable and what objective trains that mapping.
The agent below collects context and exposes an `adapt()` that must produce the task variable `z`; the
algorithm samples per-task RL batches and context batches and takes a gradient step. The bodies that
(a) aggregate per-transition embeddings into `z` and (b) supply the objective that trains the encoder
are left as stubs.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# --- existing building blocks (already available) ---
def build_mlp(input_dim, output_dim, hidden_dim, n_layers): ...   # plain MLP
def build_policy(obs_dim, action_dim, latent_dim, net_size): ...  # TanhGaussian policy π(a|s,z)
def build_qf(obs_dim, action_dim, latent_dim, net_size): ...      # Q(s,z,a)
def build_vf(obs_dim, latent_dim, net_size): ...                  # V(s,z)
def sample_context_from_buffer(enc_buf, indices, batch_size, **kw): ...  # context transitions
def sample_sac_batch(buf, indices, batch_size): ...               # (obs, act, rew, next_obs, term)


class ContextMetaRLAgent(nn.Module):
    """Encodes logged context into a task variable z and conditions an actor-critic on it."""

    def __init__(self, obs_dim, action_dim, latent_dim, net_size,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        context_input_dim = obs_dim + action_dim + reward_dim
        if use_next_obs_in_context:
            context_input_dim += obs_dim
        # encoder maps one transition (s,a,r[,s']) to an embedding
        self.context_encoder = build_mlp(context_input_dim, latent_dim,
                                         hidden_dim=200, n_layers=3)
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self._context = None

    def update_context(self, transition_tuple):
        # accumulate one logged transition into self._context
        ...

    def infer_posterior(self, context):
        # encode the context transitions and produce the task variable z
        embeddings = self.context_encoder(context)
        # TODO: how the per-transition embeddings become the task variable z
        pass

    def adapt(self):
        if self._context is not None:
            self.infer_posterior(self._context)

    def get_action(self, obs, deterministic=False):
        in_ = torch.cat([torch.as_tensor(obs)[None], self.z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)


class ContextMetaRLAlgorithm:
    """One meta-training step: encode context, train the encoder, train the actor-critic."""

    def __init__(self, agent, env, train_tasks, replay_buffer, enc_replay_buffer, config):
        self.agent = agent
        self.train_tasks = train_tasks
        self.replay_buffer, self.enc_replay_buffer = replay_buffer, enc_replay_buffer
        lr = 3e-4
        self.context_optimizer = optim.Adam(agent.context_encoder.parameters(), lr=lr)
        self.policy_optimizer  = optim.Adam(agent.policy.parameters(),          lr=lr)
        self.qf1_optimizer     = optim.Adam(agent.qf1.parameters(),             lr=lr)
        self.qf2_optimizer     = optim.Adam(agent.qf2.parameters(),             lr=lr)
        self.vf_optimizer      = optim.Adam(agent.vf.parameters(),              lr=lr)

    def _encoder_objective(self, indices, context):
        # TODO: the objective that trains the context encoder
        pass

    def _update(self, indices, context):
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)
        # encode context -> z, condition the actor-critic, train both
        # TODO: train the encoder via self._encoder_objective(...)
        # then run the offline actor-critic update conditioned on z
        pass
```

The final method fills `infer_posterior` (how the per-transition embeddings combine into `z`) and
`_encoder_objective` / the encoder side of `_update` (what trains the encoder, and how its gradients
relate to the value-learning gradients), together with the offline behavior-regularized actor-critic
update that the value side of `_update` performs.
