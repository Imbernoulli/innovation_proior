The local critic did the first thing I asked of it and confirmed the second thing I worried about. The bimodal collapse is gone — no seed stuck at exactly 0.0 on a map the others win, no analogue of the 4.9-return disaster — and the means jumped on exactly the maps where the attention critic had died: MMM from 0.135 to 0.635, 3s5z from 0.115 to 0.635, with 3s5z now *tight* ({0.81, 0.47, 0.62}), the seed-to-seed consistency a robust value regression buys. So the diagnosis held: the binding constraint was value-learning robustness, and a plain local MLP fixed it. But the local critic is *capped* on 2s3z — mean 0.448, wide spread {0.125, 0.906, 0.313}, and the returns track the win rates sensibly (12.96, 19.65, 14.64), so this is a ceiling, not a collapse. 2s3z is a symmetric map where coordinated positioning of the two stalkers and three zealots is the whole game — exactly the global, cross-agent structure a critic that sees only $o^a$ is blind to. That is the "robust but capped" signal that the next move is to bring the global state back, now that value learning is no longer the bottleneck.

I propose the **MAPPO central-V critic** in its lean environment-provided (EP) form — the standard centralized state-value critic of Yu et al. 2022, matching EPyMARL's `CentralVCritic`. I have to be careful, because the attention critic *was* centralized (its tokens carried the state) and it collapsed; the lesson was not "centralization is bad" but "cross-agent *mixing* in the critic, under this loop's value-learning conditions, is fragile." So the clean way to bring back global information *without* the mixing that blew up is to feed the global state into the same plain robust MLP that just worked: a shared $V_\phi([s, e_a])$ over the global state plus an agent one-hot, no attention, no interaction layer, nothing that can land in a degenerate basin. It is the smallest possible change from the local critic — swap $o^a$ for $s$, keep the identical 128–128–1 MLP and the identical agent one-hot.

I want to justify the state-value choice from the ground up rather than default to it. Why a *state* value and not the action-value critics of the centralized prior art — COMA's $Q(s, A)$ with a counterfactual baseline, MADDPG's per-agent $Q_i(x, a_1, \dots, a_n)$? Because the GAE baseline I am subtracting only ever needs a *state* value. The advantage identity says $\nabla J = \mathbb{E}[\nabla \log \pi \cdot A]$ for any baseline independent of the action given the state, and GAE forms
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
from $V$, never $Q$. Conditioning the critic on the joint action would make the input scale with $n$, couple the critic to every agent's current policy, and — for COMA — force an action enumeration to build the counterfactual: all machinery I do not need for variance reduction. A centralized $V(s)$ buys the variance reduction and sheds every one of those costs. So the strongest-baseline critic is single-agent PPO with one thing changed from the local critic: the value function reads the global state.

I have to be honest about what *this* harness exposes versus the richer forms the centralized-V idea can take, because the difference is load-bearing. The fully developed agent-specific construction would feed the critic the global state *and* each agent's own observation $[s, o^a]$, and there is a real argument it is better — the bare global state is *agent-agnostic* and drops the local features a per-agent value depends on. But the config this task fixes is explicit: `obs_individual_obs=False`, `obs_last_action=False`, `obs_agent_id=True`. So the centralized critic the task defines is the lean **environment-provided** form: global state plus an agent one-hot, and only that, $V_\phi([s, e_a])$. I note the gap explicitly because it is exactly where a stronger critic would go next: this critic sees the global picture but, being agent-agnostic in its state input, leans entirely on the one-hot to distinguish agents — and the one-hot carries an agent's *identity* but none of its current local detail (what it can fire, how exposed it is). If that omission caps this critic the way blindness to global structure capped the local one, the agent-specific input is the obvious later lever.

The one-hot earns its place even in the lean form because I am sharing one critic across all agents — the same robustness-and-data argument as before, homogeneous agents, $n$-fold data per step. A shared network handed the *same* $s$ for every agent could only output one value identical for all agents, which is fine only if the true per-agent value is genuinely agent-agnostic; on a heterogeneous map like MMM (a Medivac that heals, Marauders, Marines) it is not — a healer's value of a state differs from a frontline marine's. The one-hot is the minimal thing that lets the shared network emit a *different* value per agent, routing different agents to different value functions while still pooling all the data. So $[s, e_a]$ is the leanest centralized critic that is both data-efficient (shared) and able to distinguish agents (the id): the local critic's architecture with the information swapped from local to global, plus the same id it already carried.

The shapes mirror the local critic with one inverse difference. The global state is one vector per $(B, T)$, shape $(B, T, \texttt{state\_dim})$, so to put it on the agent axis I must *broadcast* it across the $n$ agents — `state.unsqueeze(2).expand(-1, -1, n, -1)` to $(B, T, n, \texttt{state\_dim})$. The local critic needed no broadcast because `obs` already lived on the agent axis; the state critic does, because the state is shared. The agent id is the $n\times n$ identity broadcast to $(B, T, n, n)$; concatenate to $(B, T, n, \texttt{state\_dim}+n)$, run the identical 128–128–1 ReLU MLP, and the head gives $(B, T, n, 1)$ — the trailing singleton the learner squeezes. `batch["obs"]` is now never read; the critic conditions purely on the shared global state and the constant id. Everything downstream is the same fixed loop the previous two rungs used.

I should knock down the worry that this re-creates the attention critic's failure with a state input. The attention critic collapsed because of cross-agent *mixing* fit to a moving bootstrapped target — the softmax over agents could find a degenerate pattern and per-agent values could smear toward a team mean. The state-V critic has *no* mixing: each agent's value is a deterministic MLP of $[s, e_a]$, the agents do not attend to each other, and the only per-agent variation comes from the fixed one-hot. There is no degenerate basin for attention to fall into because there is no attention. So it should inherit the local critic's robustness while adding the global information the local critic lacked — the best of both prior rungs.

Stated to be falsified: against the attention critic, the seed collapse stays gone (no mixing to break), so no dead seeds anywhere — as robust as the local critic. Against the local critic, the *ceiling* should lift exactly where global structure matters: I expect 2s3z — the local critic's weakest map (0.448), where coordinated positioning is everything — to improve the most, clearing 0.7; and MMM to rise, because its heterogeneity is precisely where a critic that sees the whole team's state should produce better-aligned advantages and where the one-hot lets the shared net give the Medivac a different value than a Marine. 3s5z is the interesting case: the local critic was already tight at 0.635, and on the largest team (8 agents) the agent-agnostic EP state drops the most local detail, so I expect the most modest gain there. If the pattern is "centralization clearly helps on 2s3z and MMM, helps least on the biggest team 3s5z," that signature both vindicates bringing the global state back *and* fingers the remaining gap as the agent-agnostic state input — the agent-specific $[s, o^a]$ lever I am deliberately leaving on the table.

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
    """MAPPO critic — shared MLP over (state + agent one-hot).

    Standard centralized V from Yu et al. 2022.
    Matches epymarl's CentralVCritic. All agents share the same network;
    the agent one-hot lets the shared network produce agent-specific
    value estimates while still conditioning on the full global state.
    """

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        state_dim = int(scheme["state"]["vshape"])
        input_shape = state_dim + self.n_agents
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
