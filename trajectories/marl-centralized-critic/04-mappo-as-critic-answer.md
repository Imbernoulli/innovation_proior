**Problem (from step 3).** The centralized EP critic (global state + one-hot) was the strongest baseline
— MMM 0.927, 2s3z 0.833 — but weakest on **3s5z** (0.740), the largest team. The diagnosis: the
environment-provided state is *agent-agnostic*, so on the biggest team it drops the most agent-specific
local detail and leans entirely on the constant one-hot to distinguish agents.

**Key idea (agent-specific central-V, AS/FP).** Feed the critic *both* the global state `s` and the
agent's own observation `o^a`: `V_φ([s, o^a, e_a])`. This is the recommended critic input of Yu et al.
2022 (arXiv:2103.01955) — the agent-specific global state (AS), and where an overlap map exists its
feature-pruned form (FP). It carries EP's global picture *and* the local detail EP drops (what the agent
can fire, how exposed it is, its relative distances), at the cost of a single `o^a` — not CL's full
all-observations concatenation that scales with `n`.

**Why it is the natural next move.** No cross-agent mixing (unlike the attention critic that collapsed) —
each agent's value is a deterministic MLP, so it keeps the plain-MLP robustness while adding only local
information. The agent-specific component is also what makes one shared (parameter-sharing) critic emit
genuinely per-agent values grounded in each agent's situation, not just its index — the per-agent
specialization the EP one-hot could only gesture at.

**Faithful form here.** The pruned FP form needs a per-feature `s`/`o^a` overlap map, which smaclite does
not expose (dense `state`/`obs` vectors). So the runnable form is the un-pruned AS input `[s, o^a, e_a]`:
it loses no information, only carries some redundant dimensions a width-128 MLP absorbs.

**Hyperparameters.** Input `state_dim + obs_dim + n_agents`; identical 128–128–1 ReLU MLP; reads both
`batch["state"]` (broadcast across the agent axis) and `batch["obs"]` (already on the agent axis, no
broadcast); append the agent one-hot. The only change from the EP critic is `+obs_dim` and reading
`obs`. Nothing in the actor/learner/GAE/optimizer moves.

**Bar to clear (vs the strongest baseline's real numbers).** EP means: MMM 0.927, 2s3z 0.833, 3s5z 0.740.
No seed should collapse (no mixing). The falsifiable prediction is an *ordering*: the gain over EP should
be largest on 3s5z (where the agent-agnostic state dropped the most local detail), smallest on the near-
saturated MMM — 3s5z closing toward the other two maps. If instead it helped MMM most or some seed died,
the agent-specific story is wrong.

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
    """Agent-specific central-V critic — shared MLP over (state + obs_i + agent one-hot).

    The agent-specific (AS) global-state critic input of Yu et al. 2022
    (arXiv 2103.01955): condition the centralized value on both the global
    state and the agent's own observation, so one shared network produces
    per-agent values that see the agent-specific local features the bare
    global state drops. Returns (B, T, n_agents, 1).
    """

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        state_dim = int(scheme["state"]["vshape"])
        obs_dim = int(scheme["obs"]["vshape"])
        input_shape = state_dim + obs_dim + self.n_agents      # AS: state + o_i + agent one-hot
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        state = batch["state"][:, ts]                                    # (B, T, state_dim)
        state = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)     # (B, T, n, state_dim)
        obs = batch["obs"][:, ts]                                        # (B, T, n, obs_dim)
        agent_id = th.eye(self.n_agents, device=batch.device)
        agent_id = agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        inputs = th.cat([state, obs, agent_id], dim=-1)                  # (B, T, n, state+obs+n)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)                                                  # (B, T, n, 1)
        return q
```
