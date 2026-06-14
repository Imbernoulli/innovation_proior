# IPPO critic, distilled

The IPPO (Independent PPO) critic is a **per-agent decentralised value function**: each agent `a`
estimates its own value `V_φ(z^a)` from its **local observation only** (plus an agent one-hot id), with
no global state and no peer information. One critic is parameter-shared across all agents. It is the
local-input value-function variant for cooperative multi-agent PPO.

## Problem it solves

Cooperative MARL in a Dec-POMDP with a shared team reward and partial observability, trained under CTDE
(centralised training, decentralised execution). The actor is fixed: a decentralised policy
`π^a(u^a|τ^a)` on each agent's local history. The open design choice is the critic's input set, which
fixes the bias-variance of every advantage and whether the method survives multi-agent non-stationarity.

## Key idea

Condition the critic on **exactly the information the actor uses** — the agent's own local observation —
rather than on the global state. Two reasons make this a good default for a partially observable
problem:

- **Lower variance.** A centralised critic that conditions on other agents' information requires the
  actor update to average over variables outside one agent's actor input. That marginalisation is often
  estimated through samples and can inject variance into the per-agent update. A decentralised critic
  already removes peer information from its input.
- **Lower bias.** A critic keyed on the bare global state `s` is biased relative to the actor's
  history-conditioned objective: within one state many histories map to it, so `V(s) = E_{h|s}[V(h)]`
  averages over them and cannot represent the value rising as evidence accumulates (information-gathering
  looks worthless). A critic on the agent's own observation/history keeps the information set the actor
  conditions on. (The unbiased centralised alternative would be a history-state critic `V(h,s)`, since
  `V^π(h) = E_{s|h}[V^π(h,s)]`; a state-only critic is the biased one.)

The non-stationarity that usually condemns independent learning (co-learners change the dynamics; value
targets go stale) is addressed mainly by **PPO's ratio clipping**, which removes the sampled objective's
incentive to push useful probability-ratio moves past `[1-ε,1+ε]`. A coordinate-ascent view gives the
right sufficient condition: telescoping the joint return into single-agent changes improves the joint
return if every coordinate term is non-negative. A trust-region single-agent step can supply such a
term on the induced POMDP with other agents held fixed; practical PPO is an approximation of that
logic, while vanilla actor-critic / Q-learning lack even this local improvement story. The local
`V_φ(z^a)` is the natural baseline for that induced POMDP.

**Parameter sharing + agent id.** With homogeneous agents, one shared critic pools all agents' data
(`N`× the samples per step, faster value learning); a one-hot agent id appended to the input restores
per-agent specialisation without breaking the local information constraint.

## Final form

Per agent `a` (parameters shared across `a`):

- **Critic:** `V_φ(z^a)` — MLP on `[z^a ; one-hot(a)]`, two hidden layers + ReLU, scalar head.
- **Advantage:** GAE `A^a_t = Σ_{l=0}^{T-t-1} (γλ)^l δ^a_{t+l}`,
  `δ^a_t = r_t + γ V_φ(z^a_{t+1}) - V_φ(z^a_t)`, with team reward `r_t` shared across agents. At
  `λ=1`, the truncated residual sum telescopes to a bootstrapped return minus `V_φ(z^a_t)`. The
  practical learner uses a fixed `q_nstep=5` target
  `R_t^{a,(n)} = Σ_{l=0}^{n-1} γ^l r_{t+l} + γ^n V_{\bar φ}(z^a_{t+n})` and passes the masked TD error
  `R_t^{a,(n)} - V_φ(z^a_t)` to the actor.
- **Actor loss:** clipped surrogate on the agent's own ratio
  `r^a = π_θ(u^a|τ^a)/π_θold(u^a|τ^a)`,
  `L^a(θ) = E[ min(r^a A^a, clip(r^a, 1-ε, 1+ε) A^a) ]`, plus an entropy bonus.
- **Critic loss:** mean-squared regression of `V_φ(z^a)` to the return target in the EPyMARL learner.
- **Defaults:** `γ = 0.99`, GAE `λ = 0.95` for the GAE estimator, clip `ε = 0.2`, advantage
  normalisation for the GAE setup, Adam, reward standardisation and gradient-norm clipping in the
  EPyMARL learner, and `hidden_dim` supplied by the config.

## Working code

The critic module, filling the single open slot in the CTDE actor-critic harness. It returns
`(B, T, n_agents, 1)` (the learner does `.squeeze(3)`) and sets `output_type = "v"`. It reads only
`batch["obs"]` — the global state in `batch["state"]` is never used.

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        inputs.append(batch["obs"][:, ts])
        inputs.append(
            th.eye(self.n_agents, device=batch.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, max_t, -1, -1)
        )
        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        input_shape += self.n_agents
        return input_shape
```

How it plugs into the PPO-in-MARL learner (fixed): the target critic's values produce fixed `n`-step
returns, the actor is updated by the clipped surrogate on each agent's own ratio, and the critic is fit
by masked regression onto those returns.

```python
# advantage from the local critic; PPO clipped actor update on each agent's own ratio
target_vals = target_critic(batch).squeeze(3)                            # V_φ(z^a) bootstrap
target_returns = nstep_returns(rewards, mask, target_vals, n=q_nstep)
v = critic(batch)[:, :-1].squeeze(3)
advantages = (target_returns.detach() - v) * mask                        # per-agent advantage

ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())                # r^a, per agent
surr1 = ratios * advantages
surr2 = th.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
pg_loss = -((th.min(surr1, surr2) + entropy_coef * entropy) * mask).sum() / mask.sum()

critic_loss = ((target_returns.detach() - v) ** 2 * mask).sum() / mask.sum()
```

## Relation to the alternatives

- **State-conditioned central critic (MAPPO central-V):** same actor and clipped loss, but the critic
  reads a global value input rather than local `obs`. A history-state value can be unbiased, but a bare
  state critic is biased under partial observability and central inputs can raise actor-update variance
  and input dimensionality.
- **IPPO vs IAC:** IPPO is independent actor-critic with PPO's clipped policy objective. Removing policy
  and value clipping gives an IAC-like variant; lowering the effective learning rate is not an
  equivalent substitute for clipping.
