The attention critic told me which constraint binds, and it told me in the per-seed split. On MMM two of three seeds sat at *exactly* 0.0 win rate while one reached 0.41 (mean 0.135); on 3s5z two seeds were 0.0 and one limped to 0.34 (mean 0.115); and the clinching tell was seed 42 on 2s3z — 0.0 win rate with a *return* of 4.92, when the other two seeds on the same map hit 0.81 with returns near 18.8. A return of 4.9 where 18.8 is reachable is not a narrow loss; it is the policy never finding the fight, because the per-agent advantage it was fed came from a critic that landed in a degenerate basin. That is value-learning fragility, not a capacity ceiling: a high-capacity attention critic fit to a bootstrapped target via a target copy with unnormalized returns and masked MSE, with nothing in the fixed loop to stabilize it — no return normalization, no warmup, a cold transformer under `lr=3e-4`. The cross-agent *mixing* that the encoder added over a flat MLP turned out to be the liability. So the lesson points *down* the complexity ladder: establish a critic whose value regression simply *works* before I reconsider what information it should see.

I propose the **IPPO critic** — the local-input value function (the IPPO ablation of Yu et al. 2022, matching EPyMARL's `ACCritic`). Strip the critic to the most robust thing, a plain MLP, and condition it on *exactly what the actor conditions on*: agent $a$'s own observation $o^a$ and nothing else, $V_\phi([o^a, e_a])$. No global state, no peer information, no attention. After the attention critic's collapse the appeal is direct — a per-agent MLP over $o^a$ is the lowest-variance, most robust value regression I can write, and it removes the cross-agent mixing that just blew up.

My reflex is that this should be obviously worse: it is "independent learning," the thing that is supposed to fail. The classic objection is non-stationarity — if agent $a$'s critic sees only $o^a$ while the other agents are simultaneously learning, then from $a$'s view the dynamics drift, the critic's targets go stale, and the convergence story evaporates (Tan 1993); a second objection is that an independent learner cannot always separate environment stochasticity from a peer's exploration, which provably blocks optimal play in some coordination games (Claus & Boutilier 1998). Those are real. But the right question is *whose job* stability is. The critic's job is to be a low-variance, low-bias baseline; stability of the *policy update* is a different job, and PPO attacks it with the clipped ratio objective. The non-stationarity disaster in independent learning is specifically that one agent, reacting to a co-learner's shift, takes a catastrophic policy step that shifts the dynamics for everyone and spirals. PPO's clip removes the objective's incentive to push a sampled probability ratio past $[1-\varepsilon, 1+\varepsilon]$ once it has crossed the useful side. That is exactly the restraint I need against peer-induced surprises — and it is the *inverse* of what the attention critic lacked: there, nothing restrained the *value* function; here, the actor's clip restrains the *policy*, and the value function is so simple it cannot misbehave. If clipping keeps every agent's effective step modest, then from any one agent's view the others drift slowly enough that a local critic can track them.

There is a principled — not merely "robust by being dumb" — argument that the local critic is the *right* baseline. Take the agents and telescope the joint return into single-agent changes: hold $\pi^1, \pi^2$ fixed and improve only $\pi^3$, then hold $\pi^1, \pi^3_{\text{new}}$ and improve $\pi^2$, and so on; if every coordinate term is non-negative the joint return rises. Look at one term — change only $\pi^3$, others fixed. Holding the others fixed, the world agent 3 faces is a single-agent decision process — a POMDP in which $\pi^1, \pi^2$ are absorbed into the dynamics — and agent 3's policy conditions only on $o^3$, a POMDP policy. So "improve $\pi^3$ holding the rest fixed" is just "improve a single-agent policy on the induced POMDP," and a restrained single-agent step can make that term non-negative. PPO is a first-order clipped approximation rather than the exact trust-region step, and simultaneous updates without recollection are a further approximation, so this is not a theorem for the practical update — but it tells me the *structure* I want: each agent takes a restrained single-agent improvement step against the POMDP the others induce. And the natural baseline for a POMDP policy that conditions on $o^3$ is a value function on that *same* information set, $V_\phi(o^3)$. The local critic is not a compromise forced by execution; it is the baseline aligned with the coordinate step's information set, and the local critic and PPO's clip are the two halves of one idea: restrained single-agent improvement on the right per-agent information. That is precisely the claim the attention critic had no right to — it mixed information across agents that the actor's coordinate step does not condition on, so its advantages could be both high-variance and misaligned.

For the module, I share one critic across all agents rather than learn $n$ separate nets. The agents are homogeneous within a type and share observation/action spaces, so a separate net per agent wastes data — each net learns from one agent's stream — whereas one shared critic lets every agent's experience update the same parameters, $n$-fold more data per gradient step, faster and more stable value learning, which is exactly what I want after watching the attention critic struggle to fit. The only thing sharing strictly removes is index-based specialization, and I hand that back cheaply by appending a one-hot agent id, so the input is $[o^a \oplus \text{one-hot}(a)]$ of width $\texttt{obs\_dim} + n$. The architecture is the plain PPO MLP I trust: `fc1: (obs_dim+n) → 128`, ReLU; `fc2: 128 → 128`, ReLU; `fc3: 128 → 1`. No attention, no cross-agent mixing, nothing that can collapse — the deliberate inverse of the step-1 critic.

The shapes have one structural difference from the centralized critics, and it is the load-bearing detail here: the local critic reads `batch["obs"]`, which already lives on the agent axis as $(B, T, n, \texttt{obs\_dim})$, so each agent's slot already carries its own $o^a$ and there is *no* broadcast — unlike a state critic, which must expand a single $(B, T, \texttt{state\_dim})$ vector across $n$ agents. The agent id is the $n\times n$ identity broadcast to $(B, T, n, n)$; concatenate along the last axis to $(B, T, n, \texttt{obs\_dim}+n)$, run the MLP, and the head gives $(B, T, n, 1)$ — the trailing singleton the learner squeezes. `batch["state"]` is simply never read, so the critic carries no peer or global information by accident: agent $a$'s value depends only on $o^a$ and the constant $a$, which makes it fully decentralizable — it would compute the identical value at execution from local information alone. Everything downstream is the fixed loop: the `q_nstep=5` target from a target copy of this critic, masked MSE regression, the actor's clipped surrogate on each agent's own ratio, the entropy bonus, reward standardization, Adam, grad-clip.

Stated to be falsified: first, the bimodal seed collapse should *disappear* — a plain local MLP cannot land in a degenerate attention basin, so I expect every seed to at least train (no seed stuck at exactly 0.0 on a map the others win, no 4.9-return where 18.8 is reachable) and the per-seed variance to shrink sharply. Second, the means should rise on the maps where mat collapsed — MMM (0.135) and 3s5z (0.115) — as the dead-seed floor is removed. Third, and where I am genuinely unsure: on 2s3z the local critic's *ceiling* could be lower than a working centralized critic's, because it is blind to the global, cross-agent structure a state-conditioned critic could exploit. If the pattern is "robust everywhere but capped on the harder coordination maps," the next move writes itself — bring the global state back, but in the lean robust MLP form, once value learning is no longer the bottleneck.

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
    floor baseline from Yu et al. 2022 (arXiv 2103.01955).
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
