A team of $n$ homogeneous agents acts in a partially observable cooperative task — a DEC-POMDP with one shared scalar reward $R(s, A)$ for the joint action $A = (a_1,\dots,a_n)$ — and is trained under centralized training with decentralized execution. The actor is settled: a decentralized policy $\pi_\theta(a_i \mid o_i)$ on each agent's local observation, optimized by PPO's clipped surrogate. The learner, GAE, and the optimizer are fixed too. I have already convinced myself the baseline should be a *centralized state value* $V$ rather than the joint-action $Q$ that COMA and MADDPG carry, because GAE only ever needs the residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, never $Q$. That leaves exactly one free variable: the critic's *input set* — which features go into $V$. The reason this single choice is worth obsessing over is the GAE identity itself,
$$\hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l \, \delta_{t+l},$$
where every term is an error of $V$, so the variance of the entire policy gradient is, term by term, set by how good $V$ is. A poor input set poisons every actor update, which is why I refuse to default the input to "the global state."

What forces the issue is a result that should not happen: independent, purely local PPO — each agent's critic seeing only its own $o_i$ — is reported strong on hard cooperative maps, sometimes stronger than a centralized critic fed the simulator's global state. That is backwards on its face, since a centralized critic sees strictly more than a local one and its baseline should be at least as good. If the centralized critic loses, the extra information it was handed must be the *wrong* or *incomplete* information. Take the environment-provided state, EP: in SMAC a single agent-agnostic vector of positions, health, shield, and weapon cooldown for every unit — compact, genuinely global, and the *same* vector for every agent. That last clause is the whole problem. Being agent-agnostic, EP omits the agent-specific local features: the agent's own identity, the actions currently available to it, its relative distances to allies and enemies inside its own sight range. But the value of a situation *for agent $i$* very often turns on exactly those features — whether $i$ can fire now, how exposed it is, where it stands relative to the enemy it engages. A critic that never sees them computes a $V$ that is systematically wrong, and wrong *differently* for different agents, and that error feeds straight into the per-agent residuals as bias and variance. So an EP critic can genuinely lose to a local critic that at least keeps $o_i$. The reversal was never "centralization fails"; it was "EP drops the local features the per-agent value depends on." The reflex move — centralize on the bare state — is the move that loses. The natural over-correction is to keep *every* local observation: concatenate $(o_1,\dots,o_n)$ and feed it to every critic (call it CL). Now nothing local is missing, but the input grows linearly in $n$ and in each $o_i$, so on a large map value learning becomes high-dimensional and sample-hungry exactly when I can least afford it — and a concatenation of observations is still not the true global state, since anything observed by no agent is absent from CL too. CL fixes EP's omission expensively and incompletely.

I propose the **Agent-Specific (AS) central-V critic** — and its feature-pruned refinement FP. The resolution is to stop choosing between EP and CL and instead concatenate the two *good* halves: for agent $i$, form the critic input from the environment global state $s$ together with that one agent's own observation $o_i$, so the value is $V_\phi([\,s \,;\, o^a \,;\, \text{one-hot}(a)\,])$. It is agent-specific because each agent gets a different input ($s$ is shared, $o_i$ is not). This hands the critic EP's comprehensive global picture *and* the agent-specific local detail EP dropped, while paying only for a single $o_i$ of width $\text{obs\_dim}$ rather than CL's $n$ of them. The defining update around it is unchanged from the lean MAPPO critic: per-agent advantage $A^a_t = \sum_l (\gamma\lambda)^l \delta^a_{t+l}$ with $\delta^a_t = r_t + \gamma V_\phi(x^a_{t+1}) - V_\phi(x^a_t)$ and $x^a_t = [s_t, o^a_t, e_a]$, the team reward $r_t$ broadcast to all agents; the EPyMARL learner regresses the critic by masked mean-squared error to a bootstrapped $q\text{-}n$step target
$$R_t^{a,(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V_{\bar\phi}(x^a_{t+n}), \qquad n = 5,$$
produced by a *target* copy $\bar\phi$ of the critic so the regression does not chase its own tail, and passes the masked TD error $R_t^{a,(n)} - V_\phi(x^a_t)$ to the actor. The actor itself takes the clipped surrogate on each agent's own ratio $r^a = \pi_\theta(a^a\mid o^a)/\pi_{\theta_{\text{old}}}(a^a\mid o^a)$,
$$\min\!\big(r^a A^a,\ \mathrm{clip}(r^a, 1-\varepsilon, 1+\varepsilon)\, A^a\big),$$
plus an entropy bonus, with $\gamma = 0.99$, $\lambda = 0.95$, $\varepsilon = 0.2$, advantage normalization, Adam, reward standardization and gradient-norm clipping. *None* of that moves; the only thing AS changes is that the value input additionally reads $o^a$, so $\text{input\_shape}$ grows by $\text{obs\_dim}$.

Two design choices carry the method, and both beat the obvious alternative. First, why thread in each agent's *own* $o_i$ rather than share one global value — because the agents are homogeneous, I want to *share critic parameters* for $n$-fold data per gradient step, but a single shared network handed the same $s$ for every agent can only emit *one* value, identical for all agents, which cannot express a value that differs by role or local situation. Routing each agent's own $o_i$ into the input dissolves that trap: one shared network now produces *different* values for different agents because each feeds a different observation. "Agent-specific input" and "share parameters" are the same insight from two sides — a single cheap, sample-efficient network that nevertheless distinguishes agents. The explicit agent one-hot is kept on top as a cheap, unambiguous role signal costing only $n$ dimensions. Second, why FP — because if $s$ and $o_i$ *overlap* (in SMAC they share a great deal, e.g. both encode an enemy's health), naive AS double-counts those features, adding no information while inflating the input dimension, the very cost I held against CL. FP keeps $s$ and appends only the parts of $o_i$ not already in $s$: identical information content at a smaller input, strictly dominating AS on dimensionality — *when* a per-environment overlap map is available. Where it is not, AS is the faithful fallback that keeps all the information at a modestly larger input. The one sentence behind both is the principle I would actually keep: give the value function both the global picture and the agent-specific local features, and do not let it carry redundant dimensions. Note that the network itself stays deliberately generic — two hidden layers, ReLU, a scalar head — because the whole thesis is that the *input* is the design, not the architecture.

The module fills the single open slot in the CTDE actor-critic harness. The learner hands a batch with `state` of shape $(B, T, \text{state\_dim})$ and `obs` of shape $(B, T, n, \text{obs\_dim})$ and afterward calls `.squeeze(3)`, so `forward` must return $(B, T, n, 1)$ with that trailing singleton mandatory. Inside it I take the time slice, broadcast the one-per-$(B,T)$ global state across the agent axis with `state.unsqueeze(2).expand(...)` so every agent sees the same $s$, leave `obs` as is since each agent's slot already carries its own $o_i$, append the $n\times n$ identity broadcast as the agent one-hot, concatenate `[state, obs, agent_id]` along the feature axis, and push through the MLP. Every operation runs along the feature axis with the agent axis as a batched dimension, so the same shared weights apply independently to each agent's input vector — exactly the parameter-shared, agent-specific critic argued for above.

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
