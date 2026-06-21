## Research question

Cooperative multi-agent reinforcement learning under partial observability: `n` agents each see only a local observation, the team receives one shared scalar reward, and execution must be decentralizable — each agent acts on `o_i` alone. Training happens in a simulator, so a baseline value function may use anything available centrally (the global state, all observations). This is Centralized-Training-with-Decentralized-Execution (CTDE). The design problem here is the **centralized critic architecture for MAPPO**: what the value function conditions on and how it mixes per-agent features. The actor, the EPyMARL `ppo_learner`, optimizer, GAE/`n`-step settings, and the smaclite SMAC interface are fixed. Only the critic's input set and network are free, and that choice sets the bias-variance of the per-agent advantages.

## Prior art / Background / Baselines

The starting baselines are the main centralized-critic methods available for CTDE:

- **COMA (Foerster et al., 2018).** A single centralized action-value critic `Q(s, A)` for the team plus a counterfactual baseline that marginalizes agent `i`'s own action while holding the others fixed. Gap: the critic takes the joint action as input, so its input dimension scales with `n`; forming the baseline requires evaluating `Q` once per candidate action; and the critic is coupled to every agent's current policy.
- **MADDPG (Lowe et al., 2017).** A separate centralized action-value critic `Q_i(x, a_1,…,a_n)` per agent, conditioning on global information and the joint action so that each agent's world looks stationary when the joint action is fixed. Gap: one critic per agent, joint-action input that scales with `n`, and an off-policy replay/target-network setup.
- **Value decomposition (VDN, Sunehag et al., 2018; QMIX, Rashid et al., 2018).** Factorize the joint action-value into per-agent values — a sum for VDN, or a monotonic state-conditioned mixing for QMIX — so the argmax is decentralizable. Gap: still action-value critics and off-policy, and the factorization restricts the representable class (VDN cannot represent interactions where one agent's best action depends on another's; QMIX's monotonicity is a softer but real constraint).
- **Independent learning (IQL / IAC).** Each agent learns from its own stream with no centralization. Gap: co-learners change one another's dynamics, so value targets go stale and the classic non-stationarity instability appears.

## Fixed substrate / Code framework

A MAPPO loop on cooperative SMAC maps via **smaclite** (a pure-Python SMAC reimplementation that needs no StarCraft II binary) is frozen. EPyMARL's `ppo_learner` runs an RNN actor over individual agents with the clipped surrogate `min(r^a A^a, clip(r^a, 1−ε, 1+ε) A^a)` on each agent's own ratio `r^a = π_θ(a^a|o^a)/π_{θ_old}(a^a|o^a)` (`eps_clip=0.2`), an entropy bonus (`entropy_coef=0.001`), and `epochs=4` of reuse per batch. The critic is trained by masked mean-squared regression to a `q_nstep=5` bootstrapped return target produced by a target copy of the critic; the advantage handed to the actor is the masked TD error `R_t^{a,(n)} − V(x^a_t)`. Shared team reward (`common_reward=True`) is broadcast to all agents; rewards are standardized (`standardise_rewards=True`), returns are not (`standardise_returns=False`). `γ=0.99`, `lr=3e-4`, Adam, gradient-norm clipping, `hidden_dim=128`. The config exposes `args.n_agents`, `args.n_actions`, `args.hidden_dim`, `args.obs_last_action=False`, `args.obs_individual_obs=False`, `args.obs_agent_id=True`. Each map trains ~5M environment steps.

## Editable interface

Exactly one region is editable: the `CustomCritic` class (and a custom-imports line) in `custom_critic.py`. The harness wires it into `ppo_learner` as `critic_type: "custom_critic"`. The contract the class MUST satisfy:

- Inherit from `nn.Module`; accept `(scheme, args)` in `__init__`, where `scheme["state"]["vshape"]` is the global state dim and `scheme["obs"]["vshape"]` is the per-agent observation dim.
- Set `self.output_type = "v"` in `__init__`.
- Implement `forward(self, batch, t=None)` where `batch["state"]` is `(B, T, state_dim)`, `batch["obs"]` is `(B, T, n_agents, obs_dim)`, and `batch.batch_size`, `batch.max_seq_length`, `batch.device` are available. `t=None` means the whole sequence; otherwise `t` is an integer time index (a length-1 window). Return `q` of shape **`(B, T, n_agents, 1)`** — the learner does `.squeeze(3)`, so the trailing singleton is mandatory.

The starting scaffold is the default `CustomCritic` shown below. Each method on the ladder replaces exactly this class.

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

Three cooperative SMAC maps spanning difficulty and team composition — **MMM** (heterogeneous 10-agent: 1 Medivac + 2 Marauder + 7 Marine), **2s3z** (medium heterogeneous symmetric), and **3s5z** (hard heterogeneous symmetric, team of 8) — each over three seeds {42, 123, 456}. Performance is the **test win rate** `test_battle_won_mean_<map>` (primary, higher is better) under the greedy policy, with `test_return_mean_<map>` as the secondary metric (higher is better), evaluated over 32 test episodes. A strong centralized critic should generalize across the maps rather than specialize to one scenario.
