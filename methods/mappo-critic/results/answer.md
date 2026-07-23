# MAPPO's centralized critic, distilled

MAPPO (Multi-Agent PPO) is single-agent PPO applied to a cooperative DEC-POMDP with one change that
carries the method: the value function is a **centralized state-value critic** `V_φ(s)`. The
decentralized actor `π_θ(a_i | o_i)` conditions only on the local observation (so it can be deployed
under partial observability), but the critic — used *only during training* as the GAE baseline — is
free to condition on the global state the actor cannot see. The design problem MAPPO solves is what that
critic conditions on and how it is built; the rest is the standard PPO recipe.

## Problem it solves

Cooperative multi-agent RL under partial observability and shared reward (CTDE: centralized training,
decentralized execution). Policy gradients need a low-variance baseline; GAE's advantage variance is,
term by term, controlled by the value function's error. The question is how to build a value function
whose advantage estimate is low in variance/bias and that scales across cooperative tasks of varying
difficulty and agent count, so that a simple on-policy method is competitive with off-policy
value-decomposition (QMIX) and centralized-Q actor-critics (MADDPG, COMA).

## Key idea

1. **Centralized state-value, not centralized Q.** A GAE baseline needs only `V(s)`, never an
   action-value. Using a centralized `V(s)` gives the variance reduction while avoiding what made the
   centralized-`Q` baselines costly: no joint-action input (so no scaling with `n`, no coupling of the
   critic to every agent's policy, no action enumeration for a counterfactual), and no
   additivity/monotonicity restriction on a factorized joint `Q`.

2. **Agent-specific global state input (AS / FP).** The critic's *input representation* is the lever.
   The environment-provided global state (EP) is agent-agnostic and omits agent-specific local features
   (agent id, available actions, relative distances); a concatenation of all local observations (CL) is
   high-dimensional and can miss truly-global information; a local-only critic (IPPO) misses global
   information. The **Agent-Specific (AS)** global state feeds the critic *both*: for agent `i`,
   `concat(global state s, o_i)`. A minimal identity form appends an **agent one-hot** to `s`; EPyMARL's
   `cv_critic` uses this state-plus-agent-id construction by default and can additionally append the
   flattened joint observation or previous joint action through feature flags. The agent-specific
   component lets one **shared** (parameter-sharing) network emit a *different* value per agent. The
   **Feature-Pruned (FP)** variant removes features that overlap between `s` and `o_i`, keeping the same
   information at lower input dimension.

3. **Standard PPO everywhere else, retuned for MARL non-stationarity.** Clipped policy surrogate, GAE,
   entropy bonus, orthogonal init, Adam; plus value-target normalization, clipped value loss, small clip
   `ε`, few epochs, few minibatches, large batch, and death masking on environments where agents die.

## The critic

For agent `i`, input `x_i`:

- **EP (environment-provided):** `x_i = s`
- **CL (concat local obs):** `x_i = (o_1, …, o_n)`
- **AS (agent-specific):** `x_i = concat(s, o_i)` — global + local
- **FP (feature-pruned AS):** `x_i = concat(s, prune(o_i))` — overlap with `s` removed
- **minimal agent-specific identity:** `x_i = concat(s, e_i)`, `e_i` the agent one-hot

`V_φ` is a shared MLP (two hidden layers, width 64, ReLU, orthogonal init) ending in a scalar value
head. The same shared network produces a per-agent value because each agent's input differs in its
agent-specific component.

## Objectives

Actor (decentralized, on `o_i`), over valid batch-time-agent positions `(b,t,i)`, with
`r_{θ,bti} = π_θ(a_{bti}|o_{bti})/π_{θ_old}(a_{bti}|o_{bti})`, GAE advantage `Â`, entropy `S`,
coefficient `σ`:

```
L(θ) = (1/|M|) Σ_(b,t,i)∈M min( r_{θ,bti} Â_bti,
                                clip(r_{θ,bti}, 1−ε, 1+ε) Â_bti )
       + σ (1/|M|) Σ_(b,t,i)∈M S[π_θ(o_bti)]
```

Critic (centralized, on value input `x_bti`), clipped value loss with reward-to-go `R̂_bti`:

```
V_clip = V_old(x_bti) + clamp(V_φ(x_bti) − V_old(x_bti), −ε, ε)
L(φ) = (1/|M|) Σ_(b,t,i)∈M max( (V_φ(x_bti) − R̂_bti)^2,
                                (V_clip − R̂_bti)^2 )
```

The value loss is written here with squared errors; the full on-policy trainer applies the same
clipped-error structure with Huber loss (`δ = 10`) and PopArt-normalized returns. GAE:
`δ_t = r_t + γ V(s_{t+1}) − V(s_t)`, `A_t = Σ_{l≥0} (γλ)^l δ_{t+l}`, with `V` denormalized before forming
`δ_t`. Recurrent (GRU) variants additionally sum over time and train with BPTT on fixed-length chunks.

## Key design choices and why

- **V not Q:** a baseline only needs a state value; avoids joint-action scaling and value-decomposition
  representational limits.
- **Agent-specific input (global + local):** EP misses local features (biased value), CL is high-dim
  and can miss global info, local-only misses global info; AS/FP has both; the agent-specific part lets a
  shared net produce per-agent values.
- **Feature pruning:** the AS concatenation double-counts features common to `s` and `o_i`, inflating
  input dim for no information gain; FP removes the overlap.
- **Value normalization (PopArt):** returns span orders of magnitude (e.g. `Spread`: −200…0) and drift;
  normalize the target, but rescale the output layer's weight/bias on each statistics update so outputs
  are preserved — avoiding the non-stationarity of naive renormalization.
- **Small clip `ε`<0.2, ≤2 minibatches, 5–15 epochs, large batch, clipped value loss:** MARL is
  non-stationary (teammates' policies shift), so limit per-update policy/value change; more data per
  gradient lowers gradient variance.
- **Death masking:** replace a dead agent's critic input with a fixed `0_a` (zeros + agent id) so the
  rare out-of-distribution dead states collapse to one input the critic can fit; keeping the id matters
  on role-asymmetric maps. The alternative — skipping dead states and folding `R_d = Σ_{t≥d} γ^{t−d}r_t`
  into the death timestep — turns the 1-step GAE return at death into a `(T−d)`-step Monte-Carlo
  estimate, defeating GAE's variance reduction.

## Working code

The EPyMARL central-V critic builds the `(B, T, n_agents, 1)` value tensor directly from the episode batch.
With the default MAPPO config it uses global state plus agent one-hot; optional flags append the flattened
joint observation and previous joint action.

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralVCritic(nn.Module):
    """Shared central-V critic returning (B, T, n_agents, 1)."""

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        input_shape = scheme["state"]["vshape"]
        if args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        if args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = []
        state = batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        inputs.append(state)

        if self.args.obs_individual_obs:
            obs = batch["obs"][:, ts].view(bs, max_t, -1)
            inputs.append(obs.unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        if self.args.obs_last_action:
            if t == 0:
                last_actions = th.zeros_like(batch["actions_onehot"][:, 0:1])
            elif isinstance(t, int):
                last_actions = batch["actions_onehot"][:, slice(t - 1, t)]
            else:
                last_actions = th.cat(
                    [th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
                    dim=1,
                )
            inputs.append(last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))

        agent_id = th.eye(self.n_agents, device=batch.device)
        inputs.append(agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat(inputs, dim=-1)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

In the full on-policy stack, AS/FP/death-masked value inputs are already assembled as `share_obs` /
`cent_obs`; the critic maps that centralized observation to one scalar value and the replay buffer stores
`(T+1, rollout_threads, n_agents, 1)` value and return tensors.

```python
import torch.nn as nn
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_shape_from_obs_space


class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device):
        super().__init__()
        self._use_popart = args.use_popart
        self.base = MLPBase(args, get_shape_from_obs_space(cent_obs_space))
        self.rnn = RNNLayer(args.hidden_size, args.hidden_size, args.recurrent_N, args.use_orthogonal) \
            if args.use_recurrent_policy or args.use_naive_recurrent_policy else None
        self.v_out = PopArt(args.hidden_size, 1, device=device) if self._use_popart \
            else nn.Linear(args.hidden_size, 1)

    def forward(self, cent_obs, rnn_states, masks):
        features = self.base(check(cent_obs))
        if self.rnn is not None:
            features, rnn_states = self.rnn(features, check(rnn_states), check(masks))
        values = self.v_out(features)
        return values, rnn_states
```

Clipped value loss with PopArt value normalization, mirroring the canonical MAPPO trainer (`V_old` are
the value predictions from the data batch; `popart_head` is the value output layer):

```python
def cal_value_loss(values, value_preds_old, returns, clip_param, huber_delta, popart_head):
    value_pred_clipped = value_preds_old + (values - value_preds_old).clamp(-clip_param, clip_param)
    popart_head.update(returns)                            # W *= sigma_old/sigma_new
                                                           # b = (sigma_old*b + mu_old - mu_new)/sigma_new
    err_clipped  = popart_head.normalize(returns) - value_pred_clipped
    err_original = popart_head.normalize(returns) - values
    loss_clipped  = huber_loss(err_clipped,  huber_delta)
    loss_original = huber_loss(err_original, huber_delta)
    return th.max(loss_original, loss_clipped).mean()      # clipped value loss
```
