# Agent-Specific (AS/FP) central-V critic, distilled

The AS critic is the recommended centralized value-function *input representation* from Yu et al. 2022
("The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games," arXiv:2103.01955): condition
the centralized state value on **both** the environment global state `s` **and** the agent's own
observation `o_i` (the *agent-specific global state*, AS), pruning the `s`/`o_i` overlap when an overlap
map is available (the *feature-pruned* form, FP). It is the same parameter-shared central-V actor-critic
as the lean (EP) MAPPO critic, with one change: the value input also reads each agent's observation.

## Problem it solves

Cooperative MARL in a DEC-POMDP with a shared reward and partial observability, trained under CTDE. The
actor (`π_θ(a_i|o_i)`), the learner, GAE, and the optimizer are fixed; the open design choice is which
features enter the centralized value `V`. By the GAE identity `Â_t = Σ_l (γλ)^l δ_{t+l}`, every term is
an error of `V`, so the input set controls the variance of every per-agent actor update.

## Key idea

The reflex — centralize the critic on the bare global state (EP) — can be **worse** than a purely local
critic, because EP is agent-agnostic and drops the agent-specific local features (the agent's id,
available actions, relative distances) that a per-agent value depends on. The fix keeps both halves:

- **AS — Agent-Specific global state.** For agent `i`, concatenate the global state `s` with the agent's
  own observation `o_i`. This carries EP's comprehensive global picture *and* the agent-specific local
  detail EP drops, at the cost of one `o_i` (not CL's full all-observations concatenation).
- **FP — Feature-Pruned.** AS with the `s`/`o_i` duplicate features removed: same information, smaller
  input. Requires a per-environment overlap map; where unavailable, AS is the faithful fallback.

The agent-specific component is also what lets a single **parameter-shared** critic emit *per-agent*
values: feeding each agent its own `o_i` makes one shared network produce different values per agent,
recovering the per-agent specialization that a shared EP critic (same `s` for all) cannot express.

## Final form

Per agent `a` (parameters shared across `a`):

- **Critic:** `V_φ([s ; o^a ; one-hot(a)])` — MLP, two hidden layers + ReLU, scalar head. FP replaces
  `o^a` with its overlap-pruned part.
- **Advantage:** GAE `A^a_t = Σ_l (γλ)^l δ^a_{t+l}`, `δ^a_t = r_t + γ V_φ(x^a_{t+1}) − V_φ(x^a_t)`,
  `x^a_t = [s_t, o^a_t, e_a]`, team reward `r_t` shared. The EPyMARL learner uses a fixed `q_nstep=5`
  target `R_t^{a,(n)} = Σ_{l=0}^{n-1} γ^l r_{t+l} + γ^n V_{\bar φ}(x^a_{t+n})` and passes the masked TD
  error `R_t^{a,(n)} − V_φ(x^a_t)` to the actor.
- **Actor loss:** clipped surrogate on each agent's own ratio `r^a = π_θ(a^a|o^a)/π_{θ_old}(a^a|o^a)`,
  plus an entropy bonus. Unchanged from the lean MAPPO critic.
- **Defaults:** `γ = 0.99`, GAE `λ = 0.95`, clip `ε = 0.2`, advantage normalization, Adam, reward
  standardization and gradient-norm clipping in the EPyMARL learner, `hidden_dim` from the config.

## Working code

The critic module, filling the single open slot in the CTDE actor-critic harness. It returns
`(B, T, n_agents, 1)` (the learner does `.squeeze(3)`) and sets `output_type = "v"`. It reads **both**
`batch["state"]` and `batch["obs"]`; the only change from the EP central-V critic is the extra `obs_i`.

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MAPPOASCritic(nn.Module):
    """Agent-Specific (AS) central-V critic: shared MLP over (state + obs_i + agent one-hot)."""

    def __init__(self, scheme, args):
        super(MAPPOASCritic, self).__init__()
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

## Relation to the alternatives

- **EP (lean MAPPO central-V):** same actor and learner, but the critic reads only `[s, e_a]`. EP is
  agent-agnostic and drops local features, so under a shared network it cannot produce per-agent values
  and can lose to a purely local critic on maps where local detail matters.
- **CL (concatenated observations):** keeps all local detail but the input grows linearly with `n` and
  can still miss truly-global structure; AS gets the local detail at the cost of a single `o_i`.
- **IPPO (local critic):** reads only `o_i`; AS adds the global state on top, so it cannot be blind to
  team-level structure the way the local critic is.
