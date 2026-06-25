**Problem (from step 2).** The local critic fixed the attention critic's seed collapse and lifted MMM
(0.135→0.635) and 3s5z (0.115→0.635), but is *capped* on 2s3z (0.448, spread {0.125, 0.906, 0.313}) — a
ceiling, not a collapse (returns track win rates). 2s3z's coordinated positioning is exactly the global,
cross-agent structure a critic that sees only `o^a` is blind to. With value learning no longer the
bottleneck, bring the global state back.

**Key idea (centralized state-V critic).** Feed the *global state* into the same robust plain MLP that
worked: a shared `V_φ([s, e_a])` over the global state plus an agent one-hot — no attention, no
cross-agent mixing. This is the standard MAPPO central-V (matching EPyMARL's
`CentralVCritic`), in its lean environment-provided (EP) form. The smallest change from the local critic:
swap `o^a` for `s`, keep the identical 128–128–1 MLP and agent one-hot.

**Why a state value (not Q).** A GAE baseline only needs a *state* value (`δ_t = r_t + γV(s_{t+1}) −
V(s_t)`, never `Q`). The action-value centralized critics (COMA's `Q(s,A)` + counterfactual, MADDPG's
per-agent `Q_i`) pay an `n`-scaling joint-action input, policy coupling, and action enumeration that a
state-V critic sheds while keeping the variance reduction.

**Why no collapse this time.** Unlike the attention critic, there is *no* cross-agent mixing — each
agent's value is a deterministic MLP of `[s, e_a]`, no softmax over agents to fall into a degenerate
basin — so it inherits the local critic's robustness and adds global information.

**Same-named ≠ paper (what the harness exposes).** The task config fixes `obs_individual_obs=False`,
`obs_last_action=False`, so this baseline is the lean EP critic — global state + one-hot *only*. It is
*not* the agent-specific `[s, o^a]` form (AS/FP) that the central-V idea can take; that omission (an
agent-agnostic state input leaning entirely on the one-hot for per-agent detail) is the lever a stronger
critic would pull next.

**Hyperparameters.** Input `state_dim + n_agents`; MLP 128–128–1, ReLU; reads only `batch["state"]`
(broadcast across the agent axis — the inverse of IPPO, which needed no broadcast); `batch["obs"]` never
read. One shared critic; the one-hot lets it emit per-agent values on heterogeneous maps.

**What to watch.** No seed collapse (no mixing). The ceiling should lift where global structure matters:
2s3z should improve most (expect it to clear 0.7), MMM should rise (heterogeneity favors a whole-team
state), 3s5z the most modest gain (bigger team, agent-agnostic EP drops more local detail). "Helps least
on the biggest team" would flag the agent-specific input as the next lever.

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

    Standard centralized V critic, matching epymarl's CentralVCritic.
    All agents share the same network;
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
