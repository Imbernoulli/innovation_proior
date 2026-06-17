## Research question

We want a reinforcement-learning algorithm that trains humanoid control policies — whole-body
locomotion and dexterous manipulation, continuous actions clipped to `[-1, 1]`, tens to a few
hundred action dimensions — both *fast in wall-clock time* and *capable*, on a single GPU. The
pain that makes this urgent is iteration speed: in the HumanoidBench suite even strong RL
algorithms fail to solve many tasks after 48 hours of training, and reward design in robotics
is inherently iterative (shape reward → retrain → inspect gait → reshape), so each retrain that
takes a day or more throttles the whole research loop. A usable algorithm therefore has to
clear two bars at once. It must be *sample-efficient enough* to reach good policies without an
astronomical number of environment steps, and *time-efficient enough* that those steps and the
gradient updates over them finish in hours, not days — and it must stay *stable* across a range
of tasks while doing so. The harder, unstated bar is simplicity: a method that only an expert
can implement and tune will not actually accelerate the field even if it is fast in a benchmark
table.

The concrete target setting (the one we are measured on): three HumanoidBench locomotion tasks
— `h1hand-stand-v0`, `h1hand-walk-v0`, `h1hand-run-v0` — a fixed budget of 100,000 gradient
steps with 128 parallel environments, continuous actions in `[-1, 1]`, deterministic actions at
evaluation, and a single GPU. The question is which off-policy algorithm, under exactly that
budget, gets the highest mean episode return.

## Background

**The two camps of deep RL for control, and why neither alone fits.** By this time the
practitioner default for training deployable simulation policies is Proximal Policy
Optimization (PPO; Schulman et al. 2017). PPO is *on-policy*: it learns from rollouts of the
current policy and throws them away after each update. With massively parallel simulation —
thousands of environments stepping on a GPU — PPO collects enormous batches of fresh on-policy
data per iteration and learns behaviors very fast in wall-clock terms (Heess et al. 2017;
Hwangbo et al. 2019). But being on-policy, it is *not sample-efficient*: it cannot reuse past
experience, which makes it awkward to fine-tune from real-world interaction or to initialize
from demonstrations (Hester et al. 2018).

The other camp is *off-policy* RL, which keeps a replay buffer and reuses every transition many
times. A recent sample-efficiency line (SR-SAC, D'Oro et al. 2023; BBF, Schwarzer et al. 2023;
BRO, Nauman et al. 2024; TD-MPC2, Hansen et al. 2023; Simba, Lee et al. 2024) pushes the
update-to-data ratio high and reaches strong returns from few environment steps — but at the
cost of algorithmic complexity and long wall-clock training, because high-UTD off-policy
learning is unstable and has to be propped up with architectural stabilizers (LayerNorm,
residual blocks, hyperspherical normalization) and extra machinery.

**The deadly triad.** The instability these stabilizers fight has a name: the combination of
*bootstrapping* (targets built from the network's own predictions), *function approximation*
(a neural net Q), and *off-policy* learning (training on data from an older policy) can make
value learning diverge (Sutton & Barto 2018). It is the central obstacle to making off-policy
RL both aggressive and stable.

**Distributional value learning.** Bellemare, Dabney & Munos (2017) argued for learning the
full distribution of the return `Z(s,a)`, not only its mean `Q = E[Z]`. The distributional
Bellman operator `T^π Z(s,a) =^D R(s,a) + γ Z(S', A')` is a γ-contraction in a maximal
Wasserstein metric, so the recursion is sound. To make it tractable they fix a categorical
support `{z_i = v_min + i·Δz : i = 0,…,N-1}` with `Δz = (v_max - v_min)/(N-1)`, have the network
emit logits whose softmax gives atom probabilities `p_i(s,a)`, and fit them by cross-entropy to
a *projected* Bellman target. The empirical finding that carried over into control work: a
distributional critic is, in many settings, a more stable and more performant critic than a
scalar one.

**Massive parallel simulation changes the economics of off-policy RL.** Li et al. (2023, PQL)
made a concrete, load-bearing observation: with thousands of GPU-simulated environments feeding
a replay buffer, off-policy Q-learning can be made *both* fast and sample-efficient — they
found parallel simulation, large batch sizes, and a distributional critic to be the important
ingredients. Two diagnostic facts from that work are directly relevant here. First, the number
of parallel environments matters: the value-based method scaled robustly with `N` and
benefited from more environments as long as the algorithm could exploit the extra data.
Second, a per-environment *mixed* exploration noise — giving each parallel environment its own
Gaussian noise scale drawn from a range `[σ_min, σ_max]`, with a large `σ_max` — removed the
need to tune one exploration scale per task, since the parallel fleet covers a spread of noise
levels for free.

**Deterministic continuous-control RL.** The off-policy backbone for continuous actions is the
deterministic-policy line. The deterministic policy gradient (Silver et al. 2014) gives, for a
deterministic actor `μ_φ` and critic `Q_θ`, `∇_φ J = E_{s∼ρ}[ ∇_a Q_θ(s,a)|_{a=μ_φ(s)}
∇_φ μ_φ(s) ]` — push the actor uphill on the critic by the chain rule. DDPG (Lillicrap et al.
2016) made this work with neural nets via a replay buffer, slow-moving (soft-updated) target
networks `θ' ← τθ + (1-τ)θ'`, additive exploration noise on the deterministic action, and
input normalization to cope with observation coordinates that live on very different physical
scales. A documented failure mode of this scalar-critic, single-critic setup is *value
overestimation*: bootstrapping through a `max`/`argmax`-like target systematically inflates
`Q`, and the inflated value is then chased by the policy.

## Baselines

**DDPG (Lillicrap et al. 2016).** Actor-critic for continuous control. Critic trained by
Bellman regression `L(θ) = E[(Q_θ(s,a) - y)^2]`, `y = r + γ Q_{θ'}(s', μ_{φ'}(s'))`; actor by
the deterministic policy gradient above; both with soft-updated targets and a replay buffer;
Gaussian (or Ornstein–Uhlenbeck) exploration noise. *Limitation:* a single scalar critic
bootstrapped off its own target overestimates value, and the actor amplifies that error by
maximizing the inflated `Q`; training is brittle and sensitive to hyperparameters.

**TD3 (Fujimoto, van Hoof & Meger 2018).** Three modifications to DDPG that directly target the
overestimation and the actor–critic coupling. (1) *Clipped Double Q-learning*: maintain two
critics `Q_{θ1}, Q_{θ2}` and form the shared target with the minimum,
`y = r + γ min_{i=1,2} Q_{θ'_i}(s', π_{φ'}(s') + ε)`, which upper-bounds the more biased
estimate by the less biased one. Since a noisy high-variance estimate is more likely to be
penalized by the minimum, this target also biases learning toward more reliable value estimates.
(2) *Target policy smoothing*: add clipped noise to
the target action, `ε ∼ clip(N(0, σ̃), -c, c)`, so the target averages the critic over a small
neighborhood of the chosen action — a regularizer that stops the policy from exploiting sharp,
spurious peaks in the approximate critic. (3) *Delayed policy updates*: update the actor and the
target networks only every `d` critic updates, a two-timescale scheme that lets the critic
settle before each policy step. *Limitation:* TD3's deterministic actor explores poorly on its
own — a single additive-noise behavior policy gives thin, low-diversity data — and in its
original single-environment, modest-batch, scalar-critic form it is sample- and wall-clock-hungry
on high-dimensional humanoid tasks.

**SAC (Haarnoja et al. 2018).** Off-policy actor-critic with a stochastic (maximum-entropy)
policy; the entropy bonus drives exploration and tends to be robust. *Limitation:* maximizing
action entropy is hard in very high-dimensional action spaces (whole-body humanoids), where it
can destabilize training; and in its standard form it inherits the same wall-clock cost as
single-environment off-policy methods.

**PPO (Schulman et al. 2017).** On-policy clipped policy-gradient; the fast, robust default
for parallel-sim policy training. *Limitation:* on-policy, so no experience reuse — not
sample-efficient, and ill-suited to fine-tuning from logged or real-world interaction or to
demo initialization.

**PQL (Li et al. 2023).** Showed off-policy RL can be fast *and* sample-efficient by scaling
through massively parallel simulation, large batches, and a distributional critic, with per-env
mixed exploration and n-step returns. *Limitation:* its core mechanism — three *asynchronous*
parallel processes (a data-collecting actor, a policy learner, a value learner) with explicit
inter-process update-frequency ratios to balance them — carries heavy implementation
complexity, which has held back adoption even though the underlying scaling insight is sound.

**High-UTD off-policy methods (BRO, Simba, SR-SAC, TD-MPC2).** Reach strong sample efficiency
by doing many gradient updates per environment step. *Limitation:* aggressive UTD makes the
deadly triad bite, so they require architectural stabilizers and added machinery, and they tend
to be slow in wall-clock time and complex to reproduce.

## Evaluation settings

- **HumanoidBench** (Sferrazza et al. 2024): simulated humanoid whole-body locomotion and
  manipulation, including dexterous-hand tasks; the natural capability yardstick, and the suite
  on which strong RL had been failing within 48 hours. The target tasks here are the three
  locomotion environments `h1hand-stand-v0`, `h1hand-walk-v0`, `h1hand-run-v0`.
- **IsaacLab** (Mittal et al. 2023) and **MuJoCo Playground** (Zakka et al. 2025): GPU-native
  massively-parallel suites with humanoid and dexterous control, including rough terrain and
  domain randomization; the natural wall-clock-efficiency yardsticks against PPO.
- **Protocol.** Continuous actions clipped to `[-1, 1]`; observation normalization (running
  mean/variance); deterministic policy at evaluation (`actor(obs)` with no exploration noise);
  performance = mean episode return over evaluation rollouts. For the target setting: 100,000
  gradient steps, 128 parallel environments, single GPU, return averaged over 3 evaluation
  rollouts at end of training. Reference baselines from the suite's own runs (DreamerV3, SAC,
  TD-MPC2) are trained for fixed long wall-clock budgets, with curves plotted against
  interpolated wall-clock time. Mixed-precision (AMP, bfloat16) and `torch.compile` are
  available training accelerators on the GPU.

## Code framework

The substrate is a standard off-policy actor-critic training harness for continuous control,
the part that is fixed and already exists: a vectorized environment stepping `num_envs`
instances in parallel, a replay buffer holding transitions on the GPU, observation
normalization, AdamW optimizers with a cosine learning-rate schedule, soft critic-target
updates, an evaluation loop, and mixed-precision (`autocast`) plus `torch.compile`. What is
*not* settled is the agent itself — the actor network and its exploration, the critic network
and its value parametrization, and the critic/actor update objectives. Those are the empty
slots to be designed and filled into the harness below.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Deterministic continuous-control policy: obs -> action in [-1, 1].
    The architecture, the exploration strategy, and any per-environment noise
    state are to be designed."""

    def __init__(self, n_obs, n_act, num_envs, init_scale, hidden_dim,
                 std_min, std_max, device=None):
        super().__init__()
        self.n_act = n_act
        # TODO: define the actor network (obs -> action) and any exploration state.
        raise NotImplementedError

    def forward(self, obs):
        # deterministic action used at evaluation
        raise NotImplementedError

    def explore(self, obs, dones=None):
        # action with exploration noise used during data collection
        raise NotImplementedError


class Critic(nn.Module):
    """Action-value network(s) for a continuous-control off-policy agent.
    How many critics, what the value head emits, and how a scalar value is read
    out of it are to be designed."""

    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
        super().__init__()
        # TODO: define the critic network(s), value parametrization, and support.
        raise NotImplementedError

    def forward(self, obs, actions):
        raise NotImplementedError


def build_algorithm(n_obs, n_act, num_envs, device, **cfg):
    """Construct actor, critic, target critic, optimizers, schedulers, and any
    auxiliary modules. The fixed substrate (parallel envs, GPU replay buffer,
    obs normalization, AdamW, cosine LR, AMP, compile) lives in the outer loop."""
    actor = Actor(n_obs, n_act, num_envs, cfg["init_scale"], cfg["actor_hidden_dim"],
                  cfg["std_min"], cfg["std_max"], device=device)
    qnet = Critic(n_obs, n_act, cfg["num_atoms"], cfg["v_min"], cfg["v_max"],
                  cfg["critic_hidden_dim"], device=device)
    qnet_target = Critic(n_obs, n_act, cfg["num_atoms"], cfg["v_min"], cfg["v_max"],
                         cfg["critic_hidden_dim"], device=device)
    qnet_target.load_state_dict(qnet.state_dict())
    # ... AdamW optimizers + cosine schedulers for actor and qnet ...
    return actor, qnet, qnet_target


def update_main(data, logs_dict):
    """Compute the critic target from the next state/action and the reward,
    then a critic loss against the current critic prediction, and step it.
    The target construction and the loss are to be designed."""
    # TODO: build the bootstrapped target distribution and critic objective.
    raise NotImplementedError


def update_pol(data, logs_dict):
    """Compute the policy objective from the critic and step the actor.
    The objective is to be designed."""
    # TODO: define the policy objective.
    raise NotImplementedError


@torch.no_grad()
def soft_update(src, tgt, tau):
    src_ps = [p.data for p in src.parameters()]
    tgt_ps = [p.data for p in tgt.parameters()]
    torch._foreach_mul_(tgt_ps, 1.0 - tau)
    torch._foreach_add_(tgt_ps, src_ps, alpha=tau)


# existing off-policy training loop the agent plugs into
def train(envs, actor, qnet, qnet_target, rb, normalize_obs, cfg):
    obs = envs.reset()
    dones = None
    for global_step in range(cfg["total_timesteps"]):
        actions = actor.explore(normalize_obs(obs), dones=dones)   # collect data
        next_obs, rewards, dones, infos = envs.step(actions)
        rb.extend(...)                                             # store transition
        obs = next_obs
        if global_step > cfg["learning_starts"]:
            logs_dict = {}
            for i in range(cfg["num_updates"]):                   # update-to-data ratio
                data = rb.sample(cfg["batch_size"] // cfg["num_envs"])
                logs_dict = update_main(data, logs_dict)
                if i % cfg["policy_frequency"] == 1:              # delayed policy update
                    logs_dict = update_pol(data, logs_dict)
                soft_update(qnet, qnet_target, cfg["tau"])
```

The outer loop, buffer, normalization, optimizers, and target-update mechanics are given; the
agent — actor, critic, and the two update objectives — is the slot to fill.
