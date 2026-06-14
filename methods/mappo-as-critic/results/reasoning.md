Let me fix what I am actually free to change. The team is `n` agents in a partially observable cooperative task, one shared reward `R(s, A)`, and the actor is settled: a decentralized policy `π_θ(a_i | o_i)` on each agent's local observation, trained by PPO's clipped surrogate. I am training in a simulator, so a value function used only as a baseline can see anything I have at training time, and I have already convinced myself the critic should be *centralized* — a state value `V` rather than the joint-action `Q` that COMA and MADDPG carry, because GAE only ever needs `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`, never `Q`. So the actor is fixed, the learner is fixed, GAE is fixed, and the one free variable left is the critic's *input set*: which features go into `V`. The reason that single choice is worth obsessing over is the GAE identity itself — `Â_t = Σ_l (γλ)^l δ_{t+l}`, and every term is an error of `V`, so the variance of the whole policy gradient is, term by term, set by how good `V` is. A bad input set poisons every actor update. That is why I refuse to default the input.

The first thing I do is take seriously a result that should not happen. Independent, purely local PPO — each agent's critic seeing only its own `o_i` — is reported to be *strong* on hard cooperative maps, sometimes stronger than a centralized critic that reads the simulator's global state. That is backwards on its face. A centralized critic sees strictly more than a local one; its value function is at least as expressive; its baseline should be at least as good, and the advantage at least as low-variance. If the centralized critic loses to the local one, the extra information it was handed must be the *wrong* information, or the *incomplete* information, in a way that actively hurts. So I am not going to reach for "the global state" as though it were obviously the right centralized input. I am going to figure out what the local critic has that the global-state critic threw away, and put it back.

Start with the global state the environment hands me — call it the environment-provided state, EP. In SMAC it is a single agent-agnostic vector: positions, health, shield, weapon cooldown of every unit. It is compact and genuinely global, and it is the *same vector for every agent*. That last clause is the whole problem. Because EP is agent-agnostic, it does not contain agent-specific local features: the agent's own identity, the set of actions currently available to it, its relative distances to the allies and enemies inside its own sight range. Now ask what the value of a situation is *for agent `i`*. It very often turns on exactly those features — whether `i` can fire right now, how exposed `i` is, where `i` stands relative to the enemy it is engaging. A critic that never sees them computes a `V` that is systematically wrong, and wrong differently for different agents, because the missing features differ by agent. That error feeds straight into the per-agent GAE residuals as bias and variance, and it corrupts the actor update for the agent whose local situation EP cannot represent. So an EP critic can be genuinely *worse* than a local critic that at least sees `o_i`. The reversal is explained: it was never "centralization fails," it was "EP drops the local features the per-agent value depends on." That is the wall. The reflex move — centralize on the bare state — is the move that loses.

So the local critic's hidden strength is precisely that it keeps `o_i`. The obvious next thought is: do not throw away the global picture either; keep *both*. Before I commit to "both," let me push the alternative that keeps all local information — concatenate every agent's observation into one big vector `(o_1,…,o_n)` and feed that to every critic. Call it CL. Now no local feature is missing for anyone. But two costs appear. The input dimension grows linearly with `n` and with each `o_i`, so on a large map the critic input is enormous, the value surface is high-dimensional, and value learning becomes sample-hungry exactly when I can least afford it. And — subtler — a concatenation of *observations* is still not the true global state: if some piece of information is observed by no agent at all, CL does not contain it, so CL can simultaneously pay the full dimensionality price *and* miss genuinely global structure. CL fixes EP's omission, but expensively and incompletely.

So I am caught between EP (compact, global, missing local detail) and CL (has local detail, high-dimensional, can still miss global). Each leaves something on the floor. The resolution is to stop choosing and concatenate the two *good* halves: for agent `i`, form the critic input as the environment global state `s` together with that one agent's own observation `o_i`. Call it the agent-specific global state, AS — agent-specific because each agent gets a *different* critic input (`s` is shared, `o_i` is not). This hands the critic EP's comprehensive global picture *and* the agent-specific local features EP was dropping, without paying CL's all-observations concatenation: AS adds one `o_i` of width `obs_dim`, not `n` of them.

Now I want to chase a second payoff of AS that I almost missed, because it settles a design choice I have not yet made: do I share critic parameters across agents? The agents here are homogeneous — same observation and action spaces — and parameter sharing is the thing that makes value learning sample-efficient, because every agent's experience trains the same weights, `n`-fold data per gradient step. But here is the trap if I share *and* feed EP: a single shared network handed the *same* `s` for every agent can only output *one* value, identical for all agents. That is fine only if the true per-agent value is genuinely agent-agnostic; it cannot express a value that differs by an agent's role or local situation. AS dissolves the trap. Because each agent feeds in its *own* `o_i` — which carries the agent's identity and its local features — one shared network produces *different* values for different agents. So "agent-specific input" and "share parameters" are the same insight from two sides: I want a single cheap, sample-efficient network that nevertheless distinguishes agents, and routing each agent's own observation into the input buys me both at once. The agent-specific component is the mechanism that lets a shared critic be agent-specific.

There is a wrinkle in AS I should not paper over. If I concatenate `s` and `o_i` and they *overlap* — both encode, say, the enemy's health — the input double-counts those features. That adds no information; it only inflates the input dimension, which is exactly the cost I just held against CL. In SMAC the overlap is real: the environment state and a local observation share a lot. So the honest, fully-optimized version is to take AS and then *prune the repeated features* — keep `s`, append only the parts of `o_i` not already in `s`. Same information content as AS, smaller input. Call it the feature-pruned agent-specific state, FP. FP strictly dominates AS on input dimensionality at no information cost — *when* I have the per-feature overlap map that tells me which features to drop. That map is per-environment bookkeeping; where it is unavailable, AS is the faithful fallback that keeps all the information at a modestly larger input. So I have a small family, AS and its pruned form FP, and the one sentence behind both is the design principle I would actually keep: *give the value function both the global picture and the agent-specific local features, and do not let it carry redundant dimensions.*

Let me now reason about how this critic is *called*, because the learner is particular about shapes and the whole point is to land AS in that exact call. The learner hands me a batch with `state` of shape `(B, T, state_dim)` and `obs` of shape `(B, T, n_agents, obs_dim)`, calls `critic(batch)` over the whole sequence (or `critic(batch, t)` for one timestep), and afterward does `.squeeze(3)` — so I must return `(B, T, n_agents, 1)`, the trailing singleton mandatory. So inside `forward` I compute the time extent (`max_seq_length` if `t` is `None`, else 1), take the time slice, and assemble the per-agent input. The global state is one vector per `(B, T)`; I broadcast it across the agent axis so every agent sees the same `s` — `state.unsqueeze(2).expand(-1, -1, n_agents, -1)`, giving `(B, T, n, state_dim)`. The agent observations already live on the agent axis — `obs` is `(B, T, n, obs_dim)` — so they line up directly; no broadcast needed, each agent's slot already carries its own `o_i`. I append the agent one-hot — the `n×n` identity broadcast to `(B, T, n, n)` — because even with AS the explicit id is a cheap, unambiguous role signal and costs only `n` dimensions. Concatenate `[state, obs, agent_id]` along the last axis to get `(B, T, n, state_dim + obs_dim + n)`, push through a small MLP, scalar head, out comes `(B, T, n, 1)`. Every operation is along the feature axis with the agent axis as a batched dimension, so the same shared weights apply independently to each agent's input vector — exactly the parameter-shared, agent-specific critic I argued for. This is a deliberately small architectural change from the lean central-V critic: that one fed `[state, agent_id]` (the EP design point); AS additionally threads in `obs` (and FP would prune the overlap). The MLP itself stays generic — two hidden layers, ReLU, a linear head to one — because the whole thesis is that the *input* is the design, not the network.

Let me write the module that fills the one open slot, the centralized AS critic:

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MAPPOASCritic(nn.Module):
    """Agent-Specific (AS) central-V critic: shared MLP over (state + obs_i + agent one-hot).

    The AS/FP design point of Yu et al. 2022 (arXiv:2103.01955): condition the
    centralized value on both the global state and the agent's own observation, so
    one shared network produces per-agent values that see the agent-specific local
    features the bare global state drops. Returns (B, T, n_agents, 1)."""

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

Now the training side, so I am honest about how AS plugs into the learner I am not changing. The shared team reward `r_t` is broadcast to all `n` agents. The critic is trained by masked mean-squared regression to a bootstrapped return target — in the EPyMARL PPO learner, a `q_nstep` target `R_t^{a,(n)} = Σ_{l=0}^{n-1} γ^l r_{t+l} + γ^n V_{\bar φ}(x^a_{t+n})` produced by a *target* copy of the critic so the regression does not chase its own tail, with episode masks handling termination. The advantage handed to the actor is the masked TD error `R_t^{a,(n)} − V_φ(x^a_t)`, where `x^a_t` is now the AS input `[s_t, o^a_t, e_a]`. The actor update is the clipped surrogate on each agent's *own* ratio `r^a = π_θ(a^a|o^a)/π_{θ_old}(a^a|o^a)`, `min(r^a A^a, clip(r^a, 1−ε, 1+ε) A^a)`, plus an entropy bonus. Adam, reward standardization, gradient-norm clipping, `γ = 0.99` are the learner's ordinary machinery. The *only* thing AS changes relative to the lean EP critic is the value input: it additionally reads `batch["obs"]` and concatenates each agent's own observation, so `input_shape` grows by `obs_dim`. Nothing in the actor, the learner, or the optimizer moves.

```python
# AS critic plugs into the fixed PPO-in-MARL learner exactly like the lean central-V critic;
# only the value input changed (it now reads obs_i as well as the global state).
target_vals = target_critic(batch).squeeze(3)                            # V_φ([s, o^a, e_a]) bootstrap
target_returns = nstep_returns(rewards, mask, target_vals, n=q_nstep)
v = critic(batch)[:, :-1].squeeze(3)
advantages = (target_returns.detach() - v) * mask                        # per-agent advantage

ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())                # r^a, per agent
surr1 = ratios * advantages
surr2 = th.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
pg_loss = -((th.min(surr1, surr2) + entropy_coef * entropy) * mask).sum() / mask.sum()

critic_loss = ((target_returns.detach() - v) ** 2 * mask).sum() / mask.sum()
```

Let me retrace the causal chain so I trust it. The actor is decentralized on `o_i` and the learner is fixed, so the one free choice is the critic's input set, and the GAE identity makes that choice control the variance of every actor update. The reflex — centralize on the bare global state EP — is the move that loses, because EP is agent-agnostic and drops the agent-specific local features (id, available actions, relative distances) that a per-agent value depends on; that is exactly why centralized PPO on EP can do *worse* than purely local PPO, the reversal that flagged the input as the suspect. The local critic's strength is that it keeps `o_i`; the all-observations concatenation CL keeps every `o_i` but pays a dimension linear in `n` and can still miss truly-global structure. Concatenating the two good halves — global state plus the agent's own observation — is the agent-specific state AS, which carries EP's global picture and the local detail EP dropped at the cost of a single `o_i`; pruning the `s`/`o_i` overlap gives FP, the same information at a smaller input where the overlap map is available. AS doubles as the thing that lets one shared, parameter-sharing critic emit per-agent values, because each agent feeds in a different `o_i`. The module is a small change to the lean central-V critic: `input_shape = state_dim + obs_dim + n_agents`, read both `batch["state"]` and `batch["obs"]`, append the agent one-hot, return `(B, T, n, 1)`, and let the fixed learner do everything else.
