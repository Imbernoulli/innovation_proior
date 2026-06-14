## Research question

We have `n` agents acting in a shared, partially-observed environment and earning a single shared
reward. The standard model is a cooperative Markov game `⟨N, O, A, R, P, γ⟩`: each agent `i` sees a
local observation `o^i`, takes an action `a^i` under its policy `π^i`, the team collects a joint
reward `R(o, a)`, and the objective is the discounted joint return `J(π) = E[Σ_t γ^t R(o_t, a_t)]`.
Two difficulties are coupled and both must be solved at once. First, for *each* agent we must find a
direction that improves its contribution. Second — and this is the hard one — the per-agent updates
must *combine* so that the **joint** return goes up. These are not the same problem: it is entirely
possible for every agent to move in a locally improving direction and for the joint return to drop.

On top of that, the search space is brutal. If we reason about the joint action directly, its size is
the product `∏_i |A^i|`, which explodes with the number of agents. We would like a method whose
per-iteration search cost grows additively, like `Σ_i |A^i|`, rather than multiplicatively. We want
it to work for **heterogeneous** agents (different roles, different action meanings) as well as
**homogeneous** ones (interchangeable units), to scale to many agents, and ideally to come with a
guarantee that each training step does not make the team worse. A practical solution should also be
parallelizable: anything that forces the `n` agents to be updated strictly one after another becomes
a wall-clock bottleneck when `n` is large. The gap to close is a single training paradigm that gives
all of (a) joint — not just individual — improvement, (b) additive search cost, (c) a monotonic
joint-improvement guarantee, (d) heterogeneity and variable agent counts, and (e) parallel updates.

## Background

The field has converged on **centralized training with decentralized execution (CTDE)**: agents act
on local observations at execution time, but during training a centralized value function may see
extra global information to reduce the variance of the policy-gradient estimate. The single-agent
engine underneath almost everything here is the **trust-region / proximal** family. TRPO (Schulman et
al. 2015a) is built on a monotonic-improvement bound: for a current policy `π` and a candidate `π̄`,
with the surrogate `L_π(π̄) = J(π) + E_{s∼ρ_π, a∼π̄}[A_π(s,a)]`,

```
J(π̄) ≥ L_π(π̄) − C · D_KL^max(π, π̄),   C = 4γ max_{s,a}|A_π(s,a)| / (1−γ)^2.
```

Maximizing the surrogate inside a small KL ball therefore cannot decrease `J`. PPO (Schulman et al.
2017) replaces the KL constraint with a clipped ratio objective
`E[min(r·A, clip(r, 1±ε)·A)]`, `r = π_θ(a|s)/π_{θ_k}(a|s)`, keeping the trust-region effect with only
first-order gradients.

The **credit-assignment** problem is the defining pain of the cooperative setting: because the reward
is shared, an individual agent cannot read off its own contribution from the team's success or
failure (Chang et al. 2003). Naively plugging the joint value function into a multi-agent policy
gradient (MAPG) makes the estimate's variance grow with the number of agents (Kuba et al. 2021).
This motivated value-functional notions for agents — local value functions and counterfactual
baselines — and a general object, the **multi-agent observation-value function**: for ordered disjoint
subsets `i_{1:m}` and `j_{1:h}`,

```
Q_π(o, a^{i_{1:m}}) = E[ R^γ | o_0 = o, a^{i_{1:m}}_0 = a^{i_{1:m}} ],
A^{i_{1:m}}_π(o, a^{j_{1:h}}, a^{i_{1:m}}) = Q^{j_{1:h}, i_{1:m}}_π − Q^{j_{1:h}}_π,
```

recovering the ordinary `Q` when `m = n` and the ordinary `V` when `m = 0`. The multi-agent advantage
asks how much better than average it is for agents `i_{1:m}` to take `a^{i_{1:m}}` once `j_{1:h}` have
already committed to `a^{j_{1:h}}`.

A separate observed phenomenon shaped expectations about what *kind* of agent population a method
faces. On benchmarks with **homogeneous** agents (e.g. the marine-only StarCraft maps, where every
unit is interchangeable) sharing one set of parameters across agents is highly sample-efficient —
each agent's experience trains the shared network, so with 25 identical marines you get ~25× the data
per parameter update. On **heterogeneous** benchmarks (e.g. a multi-agent cheetah whose "thigh" and
"foot" joints do different jobs) the opposite holds: training one body part with another's experience
hurts, because the agents represent genuinely different functions. Any general method has to handle
both regimes.

The other relevant background is the rise of **sequence models** outside RL. The Transformer (Vaswani
et al. 2017) maps an input sequence to an output sequence with an encoder-decoder structure built
almost entirely from attention. Its core operation is scaled dot-product attention,

```
Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V,
```

the `1/sqrt(d_k)` scaling there because, if query and key coordinates are independent with mean 0
and variance 1, then `var(q·k) = d_k`; without rescaling the logits grow with width and push softmax
into a tiny-gradient regime. Heads
are run in parallel (`h` projections to `d_k`, concatenated and projected). The encoder uses full
self-attention; the decoder uses *masked* self-attention so that position `i` can attend only to
positions `< i`, plus cross-attention into the encoder output, with residual connections and
`LayerNorm(x + Sublayer(x))` around every sublayer and a position-wise feed-forward block (inner
width `4·d_model`). These same techniques were already being imported into single-agent RL — e.g.
casting offline trajectories as return-conditioned sequences and predicting actions auto-regressively
(Chen et al. 2021). That line of work showed that RL objects could be represented as sequences, but
it did not solve the cooperative MARL problem of guaranteeing a jointly improving update across
multiple policies.

## Baselines

**MAPPO (Yu et al. 2021).** The most direct way to bring PPO to MARL: give all agents one shared set
of parameters, aggregate their trajectories, and optimize the shared clip objective
`Σ_i E[min(r^i A_π, clip(r^i, 1±ε) A_π)]`, `r^i = π_θ(a^i|o)/π_{θ_k}(a^i|o)`, with a **centralized**
value function `V_φ(s)` used only during training for variance reduction. A careful study of its
value-function input found that the global-state representation matters a lot: concatenating all local
observations grows in dimension with `n` and can still omit truly global features; the
environment-provided global state omits agent-specific features; an agent-specific state that
concatenates the global state with `o^i` helps but adds redundancy. **Gap:** parameter sharing imposes
the constraint `θ^i = θ^j` for all `i, j`; on tasks where the optimal joint policy is genuinely
heterogeneous this constraint caps the achievable return, and the cap can be exponentially below
optimal as `n` grows. Concretely, in a one-state game with joint action space `{0,1}^n` and reward 1
only on the two patterns `(0^{n/2}, 1^{n/2})` and `(1^{n/2}, 0^{n/2})`, the best shared-parameter
return over the optimum is `2 / 2^n` (Kuba et al. 2021, Prop. 1). And the centralized critic
itself is a plain MLP over a hand-chosen state vector: it conditions on global information but does
not model how the agents relate to one another, and the policy side has no built-in awareness that the
agents are supposed to cooperate.

**COMA / MADDPG (Foerster et al. 2018; Lowe et al. 2017).** Actor-critic methods with a centralized
critic that conditions on the joint action and global state during training while keeping
decentralized actors. COMA adds a counterfactual baseline: it marginalizes out a single agent's
action (keeping the others fixed) to produce an agent-specific advantage, attacking credit assignment
in one forward pass. **Gap:** the critic is again a monolithic feed-forward network over the joint
input; it does not represent the ordered "who acts in light of whom" structure of a team's decision,
and the actors come with no monotonic joint-improvement guarantee.

**Sequential trust-region MARL: the multi-agent advantage decomposition and HAPPO (Kuba et al.
2021).** This line establishes a decomposition of the joint advantage into per-agent terms and uses
it to run trust-region updates one agent at a time. The decomposition (their Lemma 1) is a pure
telescoping identity: for *any* agent permutation, any state, any joint action, with no assumption
that the value is decomposable,

```
A^{i_{1:n}}_π(o, a^{i_{1:n}}) = Σ_{m=1}^{n} A^{i_m}_π(o, a^{i_{1:m-1}}, a^{i_m}),
```

where each term `A^{i_m}_π(o, a^{i_{1:m-1}}, a^{i_m})` is agent `i_m`'s advantage *given the actions
its predecessors `i_{1:m-1}` have taken*. Building on this, HAPPO draws a random permutation each
iteration and updates agents in that order, each agent `i_m` maximizing a clip objective
`E[min(r^{i_m} A^{i_{1:m}}, clip(r^{i_m}, 1±ε) A^{i_{1:m}})]` whose expectation is taken under the
*newly updated* policies of its predecessors `i_{1:m-1}`. Each no-op update leaves its corresponding
trust-region surrogate contribution at zero, so selecting nonnegative conditioned updates makes the
telescoped lower bound on `J` nondecreasing; randomizing the order yields Nash limit points at
convergence. **Gap:** the update is
*strictly sequential* — agent `i_m` must wait for `i_{1:m-1}` to finish updating (and uses their
updated distributions for the importance-sampling term) before it can be optimized, so the `n` updates
cannot run in parallel and the procedure becomes a wall-clock bottleneck as `n` grows. The agents also
still optimize a handcrafted maximization objective rather than being trained as one cooperating
whole.

## Evaluation settings

The natural yardsticks at the time, all pre-existing cooperative benchmarks:

- **StarCraft Multi-Agent Challenge (SMAC)** unit micromanagement — maps spanning Easy / Hard /
  Hard+ difficulty (e.g. `3m`, `8m`, `MMM`, `2c vs 64zg`, `5m vs 6m`, `27m vs 30m`, `MMM2`, `6h vs
  8z`, `3s5z vs 3s6z`), with marine-only maps as the homogeneous regime; metric is **test win rate**.
  A pure-Python reimplementation that drops the StarCraft II binary dependency makes these maps
  cheap to run; training budgets per map are on the order of `5e5`–`2e7` environment steps.
- **Multi-Agent MuJoCo** continuous control — a single robot's joints split across agents (e.g.
  HalfCheetah), the canonical *heterogeneous* benchmark; metric is episode return. A faulty-joint
  variant (disable one leg segment) gives a family of related transfer tasks.
- **Bimanual Dexterous Hands Manipulation (Bi-DexHands)** and **Google Research Football** —
  additional cooperative continuous-control and team-sport scenarios with 2–4 agents; metric is
  return / scenario success.
- Protocol: baselines run at their original best-performing hyperparameters; common settings include
  GAE for advantage estimation with advantage normalization, value clipping, Adam, gradient-norm
  clipping, a discount `γ = 0.99` (`0.96` on Bi-DexHands), hidden width 64, and PPO clip / epochs
  tuned per map. Zero-shot / few-shot transfer is measured by pre-training on a set of source tasks
  and fine-tuning on harder held-out tasks with 0%, 1%, 5%, 10% of new data.

## Code framework

A standard CTDE on-policy training harness already exists: a rollout phase that collects each agent's
observations and actions and stores transitions in a buffer, and a training phase that samples a
minibatch, fits the value function, computes advantages with GAE, and applies a PPO-style clipped
update. What is *not* settled is the model that turns a team's observations into the team's actions and
values — that joint model is exactly what is to be designed. So the substrate below is only the generic
machinery; the one big empty slot is the joint policy/value model.

```python
import torch
import torch.nn as nn


class JointModel(nn.Module):
    """Maps a team's per-agent observations to per-agent actions and values.
    The internal architecture is exactly what we have to design."""

    def __init__(self, state_dim, obs_dim, action_dim, n_agent, hidden_dim):
        super().__init__()
        self.n_agent = n_agent
        self.action_dim = action_dim
        # TODO: the joint observation->action/value model we will design.

    def get_values(self, state, obs):
        # obs: (batch, n_agent, obs_dim)
        # returns per-agent value estimates (batch, n_agent, 1)
        raise NotImplementedError  # TODO

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # produce one action per agent for acting in the environment
        raise NotImplementedError  # TODO

    def evaluate(self, state, obs, actions, available_actions=None):
        # for a training minibatch: log-probs of the stored actions, values, entropy
        raise NotImplementedError  # TODO


# existing on-policy CTDE training loop the model plugs into
def train(model, buffer, optimizer, clip, ppo_epochs, value_coef, entropy_coef):
    advantages = buffer.compute_gae(model)              # GAE on the model's values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    for _ in range(ppo_epochs):
        for sample in buffer.minibatches(advantages):
            state, obs, actions, old_log_probs, returns, adv, avail = sample
            log_probs, values, entropy = model.evaluate(state, obs, actions, avail)
            ratio = torch.exp(log_probs - old_log_probs)          # importance weight
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()          # PPO clip
            value_loss = ((returns - values) ** 2).mean()          # value regression
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
```

The training loop, GAE, PPO clip, value regression, and optimizer are fixed; the joint
observation→action/value model is the slot to fill.
