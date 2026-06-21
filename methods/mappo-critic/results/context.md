# Context: centralized value functions for cooperative multi-agent policy gradients (circa 2018-2021)

## Research question

A team of `n` agents acts in a partially observable cooperative environment and shares a single
reward. Formally this is a decentralized partially observable Markov decision process (DEC-POMDP;
Oliehoek & Amato 2016) `⟨S, A, O, R, P, n, γ⟩`: there is a global state `s ∈ S`, but at execution time
agent `i` sees only a local observation `o_i = O(s; i)`, every agent draws an action from a policy
`π_θ(a_i | o_i)` that may only condition on `o_i`, and all agents receive the *same* shared reward
`R(s, A)` for the joint action `A = (a_1,…,a_n)`. The team objective is the usual discounted return
`J(θ) = E[Σ_t γ^t R(s^t, A^t)]`.

Policy-gradient training of such a team runs into the variance wall that has dogged REINFORCE since the
beginning: an unbiased gradient estimate `E[Σ_t ∇log π(a^t|o^t) G_t]` has enormous variance, and the
standard cure is to subtract a state-dependent baseline and replace the Monte-Carlo return with an
advantage estimate. In single-agent RL that baseline is a learned value function `V(s)`. The question
is what the value function should be in the *multi-agent* case, where a subtlety appears that has no
single-agent analogue: the actor may only look at `o_i`, but training happens in a simulator or lab
where the full global state — and every agent's observation — is available. A baseline used only during
training is free to look at more than the actor can. So the precise problem is: **what should the
team's value function condition on, and how should it be built?** This is a question about an architecture — the inputs
and the network of the value function (the "critic").

## Background

The setting is *centralized training with decentralized execution* (CTDE). Two extremes bracket it.
Fully *centralized* learning (Claus & Boutilier 1998) trains a single controller that emits the joint
action of all agents; it does not respect the per-agent partial-observability constraint and its
action space is the product `|A|^n`. Fully *decentralized* learning (Littman 1994) has each agent
optimize its own reward independently. CTDE sits between them: the agents execute decentralized
policies on `o_i`, but during training a centralized component may use global information that is
unavailable at execution. The whole CTDE family is built on the observation that the
partial-observability constraint binds only at *execution* time, so training is free to exploit the
simulator's privileged access.

The on-policy machinery that a policy-gradient method in this setting would reuse is already mature in
single-agent RL:

- **The advantage / baseline identity.** For a policy `π`, `∇_θ J = E[∇_θ log π(a|s) · A^π(s,a)]` for
  *any* baseline subtracted inside the advantage, because `E[∇_θ log π(a|s) · b(s)] = 0`. A good
  baseline (the state value `V(s)`) leaves the gradient unbiased while sharply cutting its variance.
  Crucially the baseline can depend on anything that is independent of the action `a` given the state —
  it is a free design choice constrained only by variance, not correctness.

- **Generalized Advantage Estimation, GAE (Schulman et al. 2016).** With the TD residual
  `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`, the estimator is the exponentially weighted sum
  `Â_t^{GAE(γ,λ)} = Σ_{l≥0} (γλ)^l δ_{t+l}`. The knob `λ` interpolates a bias-variance trade-off:
  `λ=0` gives the one-step `δ_t` (low variance, biased by `V`'s error), `λ=1` gives
  `Σ_l γ^l r_{t+l} − V(s_t)` (unbiased Monte-Carlo, high variance). Intermediate `λ ≈ 0.95` is the
  usual sweet spot. Every term of this sum is an error of the value function `V`, so the *quality of
  `V`* directly controls the variance of the advantage and hence of the gradient.

- **Proximal Policy Optimization, PPO (Schulman et al. 2017).** Reuse a batch of on-policy data over
  several epochs by optimizing a clipped surrogate
  `L^{CLIP}(θ) = E[min(r_t(θ) Â_t, clip(r_t(θ), 1−ε, 1+ε) Â_t)]`, with importance ratio
  `r_t(θ) = π_θ(a_t|s_t)/π_{θ_old}(a_t|s_t)`. The clip prevents any single update from moving the
  policy too far from the data-collecting policy, which is what makes the multi-epoch reuse safe.
  PPO carries a body of single-agent implementation lore — orthogonal initialization, advantage
  normalization, value clipping, GAE — that is known to be load-bearing for its empirical performance
  (Engstrom et al. 2020; Andrychowicz et al. 2021).

- **Value-target normalization (van Hasselt et al. 2016, "PopArt").** Regression targets in RL can
  span orders of magnitude and drift during training. Normalizing the target by a running mean and
  standard deviation stabilizes the regression; the difficulty is that naively re-normalizing also
  shifts every output of the network, including for inputs it had already fit, which is itself a source
  of non-stationarity. PopArt resolves this by rescaling the final linear layer's weights and bias
  whenever the running statistics change, so the network's *outputs are preserved exactly* across a
  normalization update while the *targets* are adaptively rescaled.

Several empirical facts about *existing* systems are on the table. In single-agent continuous control
the rise of off-policy SAC (Haarnoja et al. 2018) established a strong sample-efficient baseline;
benchmark studies in MARL (Papoudakis et al. 2021) reported that multi-agent policy-gradient methods
such as COMA are beaten by off-policy MADDPG and QMIX by a clear margin on the particle-world
environment and on the StarCraft Multi-Agent Challenge. A decentralized independent-PPO study
(de Witt et al. 2020) found that *purely local* PPO can reach high success rates on several hard SMAC
maps. Value targets in some of these tasks span orders of magnitude (in the particle-world `Spread`
task, episode returns range from below −200 up to 0). In StarCraft maps the environment-provided
global state and the agents' local observations carry overlapping but non-identical information: the
global state contains agent/enemy positions, health, shield, and weapon cooldown, while the local
observation additionally carries agent id, available movement and attack options, and relative
distances to allies/enemies within a sight radius.

## Baselines

These are the prior methods a new centralized value function would be measured against.

**MADDPG — per-agent centralized Q critic (Lowe et al. 2017).** Each agent `i` has a decentralized
deterministic actor `μ_i(o_i)` and a *centralized action-value critic* `Q_i(x, a_1,…,a_n)` that takes
the global information `x` and the joint action of all agents. Trained off-policy from a replay buffer.
The centralized `Q_i` stabilizes learning because, from agent `i`'s viewpoint, the environment is
stationary once the other agents' actions are given.

**COMA — centralized critic with a counterfactual baseline (Foerster et al. 2018).** One centralized
action-value critic `Q(s, A)` for the whole team, used to compute a per-agent *counterfactual
advantage*: hold the other agents' actions fixed and marginalize agent `i`'s own action under its
current policy, `A_i = Q(s,A) − Σ_{a'_i} π_i(a'_i|o_i) Q(s, (A_{−i}, a'_i))`. This isolates agent `i`'s
contribution to the team return and so addresses the multi-agent credit-assignment problem directly.

**VDN — additive value decomposition (Sunehag et al. 2018).** Represent the team action-value as a sum
of per-agent utilities, `Q_tot(τ, A) = Σ_i Q_i(τ_i, a_i)`, each `Q_i` conditioning only on local
information. The sum makes the team `argmax` decompose into per-agent `argmax`, giving decentralized
greedy execution, and gradients flow into each `Q_i` from the shared team TD error.

**QMIX — monotonic value decomposition (Rashid et al. 2018).** Relax VDN's sum to a learned *monotonic*
mixing of the per-agent `Q_i`, where the mixing weights are produced by a hypernetwork conditioned on
the global state and constrained non-negative. Monotonicity (`∂Q_tot/∂Q_i ≥ 0`) is enough to keep the
per-agent and team `argmax` consistent while allowing a richer, state-dependent combination than a
plain sum. This is the off-policy state of the art on SMAC.

**IPPO — independent PPO with a local value function (de Witt et al. 2020).** The fully decentralized
point of the design space: every agent runs its own PPO with a value function `V(o_i)` that sees only
the local observation, no global state and no other agents. Simple, scalable, and empirically strong on
several hard SMAC maps.

**Existing global-state conventions.** Two ways of building a *centralized* value input were already in
use. The *concatenation of local observations* (CL; used by Lowe et al. 2017) forms the global input by
stacking all agents' local observations `(o_1,…,o_n)`; its dimensionality grows with the number of
agents. The *environment-provided global state* (EP; used in much SMAC work following Foerster et al.
2017) feeds a single agent-agnostic global vector supplied by the environment. Each of these is a
concrete centralized input choice a method can adopt.

## Evaluation settings

The natural yardsticks for a cooperative-MARL method, all pre-existing:

- **Multi-agent Particle-World Environment (MPE; Lowe et al. 2017).** Cooperative 2D navigation /
  communication tasks: `Spread` (cover all landmarks), `Reference` and `Comm` (speaker–listener
  communication). Discrete actions; small agent counts; `Comm` has heterogeneous agents. MPE provides
  no native global state, so a global input is formed by concatenating local observations.
- **StarCraft Multi-Agent Challenge (SMAC; Samvelyan/Rashid et al. 2019).** Decentralized micromanage-
  ment of unit teams against scripted bots across maps with 2–27 agents and Easy / Hard / Super-Hard
  difficulty tiers. Provides an environment global state distinct from per-agent observations, which is
  the one benchmark that exposes multiple options for the centralized value input. Agents can die
  mid-episode, leaving some agents active and others inactive within the same trajectory.
- **Google Research Football (GRF; Kurach et al. 2020) and the Hanabi challenge (Bard et al. 2020).**
  Cooperative team football academy scenarios and a fully cooperative turn-based card game requiring
  reasoning over partners' hidden information.
- **Protocol.** Parameter sharing across homogeneous agents (shared policy and value networks);
  Adam optimizer (`ε = 1e-5`), `γ = 0.99`, GAE `λ = 0.95`, gradient-norm clipping at 10, orthogonal
  initialization, recurrent (GRU) or MLP networks of width 64 with two fully connected layers, value
  loss as a Huber loss with `δ = 10`. Hyperparameters (learning rate, epochs, minibatches, clip `ε`,
  entropy coefficient, activation) are selected by grid search of a fixed size, matched across the
  compared methods for fairness. The natural performance metric on SMAC is the median evaluation win
  rate over a set of evaluation games per seed; on MPE the episode return; on GRF the success rate.

## Code framework

A policy-gradient learner for this setting already exists end to end; the only thing not yet settled is
the value function's architecture. The harness owns the data pipeline (an episode buffer with
`batch["state"]` of shape `(B, T, state_dim)` and `batch["obs"]` of shape `(B, T, n_agents, obs_dim)`),
the optimizer, the PPO clip surrogate for the actor, the advantage / return computation, and the value
regression loop. The learner instantiates a critic module, calls it on the batch to get a value tensor,
bootstraps return targets from it, regresses the critic to those targets, and feeds the resulting
advantage into the actor update. The single empty slot is the critic `nn.Module` itself — what it
conditions on and how it maps that to a per-agent value.

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """The team's value function for the policy-gradient learner.

    Receives a batch with:
        batch["state"]          : (B, T, state_dim)              global state per timestep
        batch["obs"]            : (B, T, n_agents, obs_dim)      per-agent local observations
        batch["actions_onehot"] : (B, T, n_agents, n_actions)    previous joint action if requested
        batch.batch_size, batch.max_seq_length, batch.device
    Must return a per-agent value tensor of shape (B, T, n_agents, 1); the learner
    later does .squeeze(3). Only a state-dependent baseline is needed by the learner;
    it never asks the critic for an action-value.

    The whole design problem lives here: what does the value function look at, and
    how does it turn that into a per-agent value?
    """

    def __init__(self, scheme, args):
        super().__init__()
        self.n_agents = args.n_agents
        self.output_type = "v"
        # available pieces of the input space, sizes only:
        #   scheme["state"]["vshape"] : global state dim
        #   scheme["obs"]["vshape"]   : per-agent observation dim
        #   scheme["actions_onehot"]["vshape"][0], args.n_agents, args.hidden_dim
        #   args.obs_individual_obs, args.obs_last_action
        # TODO: choose the value function's inputs and build its network here.
        pass

    def forward(self, batch, t=None):
        # t is None -> whole sequence; else a single timestep index
        # TODO: assemble the chosen inputs from batch and produce q of shape (B, T, n_agents, 1)
        raise NotImplementedError


# existing policy-gradient training step the critic plugs into
def train_step(critic, target_critic, actor, batch, rewards, mask, optimizer):
    with th.no_grad():
        boot = target_critic(batch).squeeze(3)                  # bootstrap values for the return target
    returns = compute_returns(rewards, mask, boot)              # n-step / GAE return targets (existing)
    v = critic(batch)[:, :-1].squeeze(3)                        # current value estimates
    critic_loss = ((returns.detach() - v) ** 2 * mask).sum() / mask.sum()
    optimizer.zero_grad(); critic_loss.backward(); optimizer.step()
    advantages = (returns.detach() - v).detach()               # baseline subtracted -> advantage
    update_actor_with_ppo_clip(actor, batch, advantages, mask)  # existing PPO clip surrogate
```

The training loop supplies the global state and every agent's observation to the critic and consumes a
per-agent value; how the critic uses what it is given is the open slot.
