## Research question

Cooperative multi-agent reinforcement learning under partial observability: a team of `n` agents each
sees only its own local observation, every agent shares one scalar team reward, and the policies must be
**decentralizable** — at execution each agent acts on `o_i` alone. Training, though, happens in a
simulator, so a value function used only as a baseline may condition on anything available at training
time (the global state, every agent's observation). This is Centralized-Training-with-Decentralized-
Execution (CTDE). The single thing being designed is the **centralized critic architecture** for MAPPO
(Multi-Agent PPO): what the value function conditions on and how it mixes per-agent features. Everything
else — the actor (an RNN policy over individual agents), the EPyMARL `ppo_learner`, the optimizer, the
GAE/`n`-step settings, the smaclite SMAC environment interface — is fixed. The critic's input set and
network are the only free variables, and by the GAE identity `Â_t = Σ_l (γλ)^l δ_{t+l}` (every term an
error of `V`), that choice sets the bias-variance of every per-agent advantage and hence whether MAPPO
scales to hard cooperation maps.

## Prior art before the first rung (centralized-critic lineage)

The first rung reacts to the centralized-critic methods that preceded MAPPO. These are the baselines the
ladder climbs out of; each is named with the gap it leaves.

- **COMA (Foerster et al. 2018).** A single centralized *action-value* critic `Q(s, A)` for the team
  plus a counterfactual baseline that marginalizes agent `i`'s own action with the others held fixed,
  `A_i = Q(s,A) − Σ_{a'_i} π_i(a'_i|o_i) Q(s,(A_{−i},a'_i))` — a clean answer to credit assignment. Gap:
  it conditions the critic on the *joint action*, so the input scales with `n` and the critic is coupled
  to every agent's current policy; and it must evaluate `Q` once per candidate action to form the
  baseline.
- **MADDPG (Lowe et al. 2017).** A separate centralized action-value critic `Q_i(x, a_1,…,a_n)` per
  agent — global info plus the joint action — so that, conditioned on the joint action, each agent's
  world looks stationary. Gap: one critic per agent, joint-action input that scales with `n`, off-policy
  with replay and target networks.
- **Value decomposition (VDN, Sunehag et al. 2018; QMIX, Rashid et al. 2018).** Factorize the joint
  `Q_tot` into per-agent `Q_i` (a sum, or a monotonic state-conditioned mixing) for decentralizable
  argmax. Gap: still action-value critics, off-policy, and the factorization restricts the representable
  class (VDN cannot represent interactions where one agent's best action depends on another's; QMIX's
  monotonicity is a softer but real constraint).
- **Independent learning (IQL, Tan 1993; IAC).** Each agent learns from its own stream with no
  centralization. Gap: co-learners change one another's dynamics, so value targets go stale and the
  classic non-stationarity instability appears — yet, with a proximal per-agent step, independent PPO is
  reported to be surprisingly competitive, which is itself a clue that the critic's *input* (not merely
  "is it centralized") is what decides things.

The common thread: all condition the critic on *actions*, pay for it in `n`-scaling, action enumeration,
or a restricted class — and a GAE baseline only ever needs a *state* value `V`. The ladder below holds
the actor and learner fixed and varies exactly the centralized critic's input and architecture.

## The fixed substrate

A MAPPO loop on cooperative SMAC maps (via **smaclite**, a pure-Python SMAC reimplementation needing no
StarCraft II binary) is frozen and must not be touched. EPyMARL's `ppo_learner` with the Yu et al. 2022
MAPPO defaults: an RNN actor over individual agents with the clipped surrogate
`min(r^a A^a, clip(r^a, 1−ε, 1+ε) A^a)` on each agent's own ratio `r^a = π_θ(a^a|o^a)/π_{θ_old}(a^a|o^a)`
(`eps_clip=0.2`), an entropy bonus (`entropy_coef=0.001`), and `epochs=4` of reuse per batch. The critic
is trained by **masked mean-squared regression** to a `q_nstep=5` bootstrapped return target produced by
a **target copy** of the critic (so the regression does not chase its own tail); the advantage handed to
the actor is the masked TD error `R_t^{a,(n)} − V(x^a_t)`. The shared team reward (`common_reward=True`)
is broadcast to all agents; rewards are standardized (`standardise_rewards=True`), returns are not
(`standardise_returns=False`). `γ=0.99`, `lr=3e-4`, Adam, gradient-norm clipping, `hidden_dim=128`. The
config exposes `args.n_agents`, `args.n_actions`, `args.hidden_dim`, `args.obs_last_action=False`,
`args.obs_individual_obs=False`, `args.obs_agent_id=True`. Each map trains ~5M environment steps.

## The editable interface

Exactly one region is editable: the `CustomCritic` class (and a custom-imports line) in
`custom_critic.py`. The harness wires it into `ppo_learner` as `critic_type: "custom_critic"`. The
contract the class MUST satisfy:

- Inherit from `nn.Module`; accept `(scheme, args)` in `__init__`, where `scheme["state"]["vshape"]` is
  the global state dim and `scheme["obs"]["vshape"]` is the per-agent observation dim.
- Set `self.output_type = "v"` in `__init__`.
- Implement `forward(self, batch, t=None)` where `batch["state"]` is `(B, T, state_dim)`,
  `batch["obs"]` is `(B, T, n_agents, obs_dim)`, and `batch.batch_size`, `batch.max_seq_length`,
  `batch.device` are available. `t=None` means the whole sequence; otherwise `t` is an integer time
  index (a length-1 window). Return `q` of shape **`(B, T, n_agents, 1)`** — the learner does
  `.squeeze(3)`, so the trailing singleton is mandatory.

The starting point is the scaffold default: a lean **central-V** critic over `(state ⊕ agent-one-hot)`.
Each method on the ladder replaces exactly this class.

```python
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ── Custom imports (editable) ────────────────────────────────────────────


# ======================================================================
# EDITABLE — Custom centralized critic for MAPPO
# ======================================================================
class CustomCritic(nn.Module):
    """Centralized critic for MAPPO on SMAC (via smaclite).

    Plugged into epymarl's ppo_learner via ``critic_type: "custom_critic"``.
    The learner calls ``critic(batch)`` without ``t`` and later does
    ``.squeeze(3)``, so the output MUST be ``(B, T, n_agents, 1)``.
    """

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        # Default: simple state + agent-id MLP (central V baseline).
        self.state_dim = int(scheme["state"]["vshape"])
        input_shape = self.state_dim + self.n_agents
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        state = batch["state"][:, ts]                                    # (B, T, state_dim)
        state = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)     # (B, T, n, state_dim)
        agent_id = th.eye(self.n_agents, device=batch.device)
        agent_id = agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        inputs = th.cat([state, agent_id], dim=-1)                       # (B, T, n, state+n)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)                                                  # (B, T, n, 1)
        return q
```

## Evaluation settings

Three cooperative SMAC maps spanning difficulty and team composition — **MMM** (heterogeneous 10-agent:
1 Medivac + 2 Marauder + 7 Marine), **2s3z** (medium heterogeneous symmetric), and **3s5z** (hard
heterogeneous symmetric, team of 8) — each over three seeds {42, 123, 456}. Performance is the **test win
rate** `test_battle_won_mean_<map>` (primary, higher is better) under the greedy policy, with
`test_return_mean_<map>` as the secondary metric (higher is better), evaluated over 32 test episodes. A
strong centralized critic should generalize across the maps rather than specialize to one scenario.
