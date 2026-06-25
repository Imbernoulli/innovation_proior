**Problem (from step 1).** The attention critic collapsed in the value-learning-fragility pattern, not
the helps-on-hard-maps pattern: bimodal per-seed outcomes (two dead seeds on MMM and 3s5z; a 4.9-return
0.0-win seed on 2s3z where others hit 18.8/0.81). Cross-agent mixing in the critic, fit to a
bootstrapped target with unnormalized returns and no warmup, lands in degenerate basins that poison the
advantages. The binding constraint is value-learning robustness, not representational power.

**Key idea (local critic = IPPO).** Strip the critic to the most robust thing — a plain MLP — and
condition it on *exactly what the actor sees*: agent `a`'s own observation `o^a` (plus an agent one-hot),
`V_φ([o^a, e_a])`. No global state, no peer information, no attention. This is the local-input value
function (the IPPO ablation, matching EPyMARL's `ACCritic`).

**Why it works.** A local MLP cannot collapse the way the attention critic did. The non-stationarity that
condemns naive independent learners is reassigned to PPO's ratio clip (which restrains each agent's
policy step), not to the critic. A coordinate-ascent view justifies it: holding the others fixed, each
agent faces an induced single-agent POMDP, and the natural baseline for a policy on `o^a` is a value
function on that same information set. The local critic and PPO's clip are two halves of "restrained
single-agent improvement on the right per-agent information."

**Parameter sharing + id.** Homogeneous agents share one critic (`n`× data per step, faster/stabler
value learning); a one-hot agent id restores per-agent specialization without adding peer/global info.

**Hyperparameters.** Input `obs_dim + n_agents`; MLP 128–128–1, ReLU; reads only `batch["obs"]` (which
already lives on the agent axis — no broadcast); `batch["state"]` never read. Everything downstream is
the fixed MAPPO loop.

**What to watch.** The bimodal seed collapse should disappear — every seed should at least train, per-
seed variance should shrink, and the means on MMM (mat 0.135) and 3s5z (mat 0.115) should rise as the
"dead seed" floor is removed. Open risk: the local critic may be *robust but capped* on the hard
coordination maps, blind to global structure a stable centralized critic could exploit — which is the
hypothesis the next rung tests.

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
    """IPPO critic — per-agent MLP over local obs + agent one-hot.

    Matches epymarl's ACCritic. No centralization: each agent's value
    depends only on its own observation. Serves as the "no centralization"
    floor baseline.
    """

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        obs_dim = int(scheme["obs"]["vshape"])
        input_shape = obs_dim + self.n_agents   # obs + agent-one-hot
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        obs = batch["obs"][:, ts]                                        # (B, T, n, obs_dim)
        agent_id = th.eye(self.n_agents, device=batch.device)
        agent_id = agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        inputs = th.cat([obs, agent_id], dim=-1)                         # (B, T, n, obs+n)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)                                                  # (B, T, n, 1)
        return q
```
