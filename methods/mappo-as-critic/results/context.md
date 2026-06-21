# Context: what should a centralized value function condition on in cooperative MARL (circa 2018-2021)

## Research question

A team of `n` agents acts in a partially observable cooperative task and shares a single scalar reward.
Formally a decentralized partially observable Markov decision process (DEC-POMDP; Oliehoek & Amato
2016) `⟨S, A, O, R, P, n, γ⟩`: there is a global state `s ∈ S`, but at execution time agent `i` sees
only a local observation `o_i = O(s; i)`, each agent draws its action from a policy `π_θ(a_i | o_i)`
that may condition only on `o_i`, and all agents receive the same reward `R(s, A)` for the joint action
`A = (a_1,…,a_n)`. The team objective is the discounted return `J(θ) = E[Σ_t γ^t R(s^t, A^t)]`.

When such a team is trained with policy gradients, a centralized value function is used as the
variance-reducing baseline. Centralized-Training-with-Decentralized-Execution (CTDE) makes this
legitimate: the partial-observability constraint binds only at *execution*, so a baseline that is used
only during training is free to look at more than the actor can — in particular the global state `s` and
every agent's observation. The precise question this context addresses is narrower than "should the
critic be centralized": **given that the critic will be centralized, exactly which features should enter
its input, so that the advantage it produces is low in variance and bias and the resulting on-policy
update generalizes across cooperative maps of varying difficulty and agent count?** This is a question
about the critic's *input representation*, holding the actor, the learner, the optimizer, and the GAE
settings fixed.

## Background

The on-policy machinery a policy-gradient method in this setting reuses is mature in single-agent RL:

- **The baseline identity.** `∇_θ J = E[∇_θ log π(a|s) · A^π(s,a)]` for any baseline subtracted inside
  the advantage, because `E[∇_θ log π(a|s) · b(s)] = 0` whenever `b` does not depend on the action. The
  baseline is a free design choice constrained by variance, not correctness — it may condition on
  anything that is action-independent given the state.
- **Generalized Advantage Estimation, GAE (Schulman et al. 2016).** With `δ_t = r_t + γ V(s_{t+1}) −
  V(s_t)`, `Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}`. Every term is an error of `V`, so the *quality of `V`*
  controls the variance of the advantage and hence of the gradient. `λ ≈ 0.95` is the usual sweet spot.
- **PPO (Schulman et al. 2017).** Reuse a batch over several epochs with the clipped surrogate
  `L^{CLIP}(θ) = E[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]`, `r_t = π_θ(a_t|s_t)/π_{θ_old}(a_t|s_t)`.
  The clip makes the multi-epoch reuse safe by removing the sampled objective's incentive to push a
  useful ratio move past `[1−ε, 1+ε]`. PPO carries implementation lore — orthogonal init, advantage
  normalization, value clipping — known to be load-bearing (Engstrom et al. 2020; Andrychowicz et al.
  2021).

The motivating empirical facts already on the table by this time:

- The folklore that on-policy policy gradients are sample-inefficient relative to off-policy methods
  (hardened by SAC, Haarnoja et al. 2018), and benchmark studies reporting that multi-agent PG methods
  such as COMA are beaten by off-policy MADDPG and QMIX on the particle world and on the StarCraft
  Multi-Agent Challenge (SMAC; Samvelyan et al. 2019; Papoudakis et al. 2021).
- Pulling the other way: independent PPO can reach high success on several hard SMAC maps (de Witt et
  al. 2020) — yet *centralized* PPO using a particular global-state input is reported to do **worse**
  than the purely local critic on some maps. That reversal is the diagnostic clue, because a centralized
  critic sees strictly more than a local one and so its baseline should be at least as good. If it is
  worse, the centralized critic is being fed the *wrong* features.

## The candidate critic inputs

The prior centralized actor-critics (COMA, MADDPG) condition the critic on the joint *action* and the
global state. But a GAE baseline only needs a *state* value `V`, never `Q`, so the joint action is
unnecessary; the remaining and decisive choice is which *state* features the value reads. Four input
representations bracket the space (the EP/AS/FP/CL taxonomy of Yu et al. 2021/2022, "The Surprising
Effectiveness of PPO in Cooperative, Multi-Agent Games," arXiv:2103.01955):

- **EP — Environment-Provided global state.** The single agent-agnostic vector the simulator hands you
  (in SMAC: positions, health, shield, weapon cooldown for all units). Compact and truly global, but the
  *same* vector for every agent, so it drops agent-specific local features: the agent's own id, its
  available actions, its relative distances to allies and enemies in its sight range. Because a per-agent
  value often depends on exactly those features, an EP critic is systematically wrong in a way that
  varies by agent — which is the explanation for the "centralized worse than local" reversal.
- **CL — Concatenated Local observations.** Stack `(o_1,…,o_n)` into one big vector for every agent's
  critic. No local feature is missing, but the input grows linearly with `n` and with each `o_i`, so on
  large maps value learning becomes high-dimensional and sample-hungry; and concatenated observations are
  still not the true global state if some information is observed by no agent.
- **AS — Agent-Specific global state.** For agent `i`, concatenate the environment global state `s` with
  agent `i`'s own observation `o_i`. This gives the critic the comprehensive global picture of EP *and*
  the agent-specific local detail EP drops, without paying CL's full all-observations concatenation.
  Because each agent feeds in a *different* `o_i`, one shared (parameter-sharing) critic network produces
  *different* per-agent values — the agent-specific component is what lets a single shared net be
  agent-specific.
- **FP — Feature-Pruned agent-specific global state.** AS, but with the features that are duplicated
  between `s` and `o_i` removed: keep `s`, append only the parts of `o_i` not already in `s`. Same
  information content as AS at a smaller input dimension; requires a per-environment overlap map.

## The finding this method records

The headline ablation of arXiv:2103.01955 is that the *critic input representation* matters as much as
the algorithm: incorporating **both global and agent-specific features** (AS or, where the overlap map
is available, FP) substantially outperforms the bare environment-provided global state (EP) on SMAC, and
is what closes the gap between centralized PPO and strong off-policy baselines. The agent-specific input
is also what makes one shared centralized critic emit per-agent values. The lean identity form of the
central-V critic — global state plus an agent one-hot — is the EP design point; the recommended,
stronger design point is AS/FP, the central-V critic that additionally conditions on each agent's own
observation.

## Code framework

```python
import torch
import torch.nn as nn


class MAPPOCritic(nn.Module):
    """Minimal scaffold for the AS central-V critic used as a PPO baseline."""

    def __init__(self, state_dim, obs_dim, n_agents, hidden_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(state_dim + obs_dim + n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, obs):
        return self.net(torch.cat([state, obs], dim=-1))

    def update(self, obs, actions, returns, advantages):
        """PPO-style value update stub; fill with clipped MSE on returns."""
        raise NotImplementedError
```
