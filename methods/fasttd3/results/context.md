## Research question

We want a reinforcement-learning algorithm that trains humanoid control policies — whole-body
locomotion and dexterous manipulation, continuous actions clipped to `[-1, 1]`, tens to a few
hundred action dimensions — on a single GPU, within a wall-clock budget measured in hours. The
broad question is how to train such policies on a fixed budget of environment steps and gradient
updates: what actor, what critic, and what update objectives reach high return under that budget.

The concrete target setting (the one we are measured on): three HumanoidBench locomotion tasks
— `h1hand-stand-v0`, `h1hand-walk-v0`, `h1hand-run-v0` — a fixed budget of 100,000 gradient
steps with 128 parallel environments, continuous actions in `[-1, 1]`, deterministic actions at
evaluation, and a single GPU. The question is which off-policy algorithm, under exactly that
budget, gets the highest mean episode return.

## Background

**The two camps of deep RL for control.** By this time the practitioner default for training
deployable simulation policies is Proximal Policy Optimization (PPO; Schulman et al. 2017). PPO
is *on-policy*: it learns from rollouts of the current policy and discards them after each
update. With massively parallel simulation — thousands of environments stepping on a GPU — PPO
collects large batches of fresh on-policy data per iteration and learns behaviors fast in
wall-clock terms (Heess et al. 2017; Hwangbo et al. 2019).

The other camp is *off-policy* RL, which keeps a replay buffer and reuses every transition many
times. A recent sample-efficiency line (SR-SAC, D'Oro et al. 2023; BBF, Schwarzer et al. 2023;
BRO, Nauman et al. 2024; TD-MPC2, Hansen et al. 2023; Simba, Lee et al. 2024) pushes the
update-to-data ratio high and reaches strong returns from few environment steps, using
architectural stabilizers (LayerNorm, residual blocks, hyperspherical normalization) and
related machinery.

**The deadly triad.** The combination of *bootstrapping* (targets built from the network's own
predictions), *function approximation* (a neural net Q), and *off-policy* learning (training on
data from an older policy) is known to be able to make value learning diverge (Sutton & Barto
2018).

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
scales.

## Baselines

**DDPG (Lillicrap et al. 2016).** Actor-critic for continuous control. Critic trained by
Bellman regression `L(θ) = E[(Q_θ(s,a) - y)^2]`, `y = r + γ Q_{θ'}(s', μ_{φ'}(s'))`; actor by
the deterministic policy gradient above; both with soft-updated targets and a replay buffer;
Gaussian (or Ornstein–Uhlenbeck) exploration noise.

**TD3 (Fujimoto, van Hoof & Meger 2018).** Three modifications to DDPG. (1) *Clipped Double
Q-learning*: maintain two critics `Q_{θ1}, Q_{θ2}` and form the shared target with the minimum,
`y = r + γ min_{i=1,2} Q_{θ'_i}(s', π_{φ'}(s') + ε)`, which upper-bounds the more biased
estimate by the less biased one. Since a noisy high-variance estimate is more likely to be
penalized by the minimum, this target biases learning toward more reliable value estimates.
(2) *Target policy smoothing*: add clipped noise to the target action,
`ε ∼ clip(N(0, σ̃), -c, c)`, so the target averages the critic over a small neighborhood of the
chosen action — a regularizer over sharp peaks in the approximate critic. (3) *Delayed policy
updates*: update the actor and the target networks only every `d` critic updates, a
two-timescale scheme that lets the critic settle before each policy step. TD3's actor is
deterministic, with exploration supplied by additive noise on the chosen action.

**SAC (Haarnoja et al. 2018).** Off-policy actor-critic with a stochastic (maximum-entropy)
policy; the entropy bonus drives exploration and tends to be robust.

**PPO (Schulman et al. 2017).** On-policy clipped policy-gradient; the fast, robust default
for parallel-sim policy training. Being on-policy, it learns from current-policy rollouts and
discards them after each update.

**PQL (Li et al. 2023).** Showed off-policy RL can be fast *and* sample-efficient by scaling
through massively parallel simulation, large batches, and a distributional critic, with per-env
mixed exploration and n-step returns. Its mechanism runs three *asynchronous* parallel processes
(a data-collecting actor, a policy learner, a value learner) with explicit inter-process
update-frequency ratios to balance them.

**High-UTD off-policy methods (BRO, Simba, SR-SAC, TD-MPC2).** Reach strong sample efficiency
by doing many gradient updates per environment step, using architectural stabilizers and added
machinery to support the high update-to-data ratio.

## Evaluation settings

- **HumanoidBench** (Sferrazza et al. 2024): simulated humanoid whole-body locomotion and
  manipulation, including dexterous-hand tasks; the capability yardstick. The target tasks here
  are the three locomotion environments `h1hand-stand-v0`, `h1hand-walk-v0`, `h1hand-run-v0`.
- **IsaacLab** (Mittal et al. 2023) and **MuJoCo Playground** (Zakka et al. 2025): GPU-native
  massively-parallel suites with humanoid and dexterous control, including rough terrain and
  domain randomization; the wall-clock-efficiency yardsticks against PPO.
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
