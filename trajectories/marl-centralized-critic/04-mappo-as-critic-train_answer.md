The centralized state-V critic confirmed both halves of what I predicted, and the half it confirmed about its own weakness is exactly the lever I had left on the table. Bringing the global state back into the robust MLP did what I expected where global structure matters: 2s3z jumped from the local critic's capped 0.448 to 0.833 — the single biggest gain on the ladder, clearing the 0.7 bar — and MMM rose from 0.635 to 0.927, the heterogeneity payoff where a whole-team state lets the shared net give the Medivac a different value than a Marine. No seed collapse anywhere, as there could not be without cross-agent mixing. But the EP critic is *weakest* on 3s5z, mean 0.740 ({0.719, 0.656, 0.844}) — the lowest of its three maps. I called this in advance: 3s5z is the largest team (8 agents), the global state is highest-dimensional there, and the environment-provided state is *agent-agnostic*, so on the biggest team it drops the most agent-specific local detail and the critic has to lean hardest on the one-hot — which carries an agent's identity but none of its current local situation. The signature is precisely "centralization helps most on 2s3z and MMM, helps least on the biggest team 3s5z," and that signature is the diagnosis: the remaining gap is the agent-agnostic state input.

I propose the **agent-specific (AS) central-V critic** — the recommended critic input of Yu et al. 2022. The EP critic leans entirely on the one-hot to make the shared network agent-specific, and the one-hot is a *constant*: it tells the network *which* agent this is but nothing about *what that agent is currently seeing*. The per-agent value of a SMAC situation depends heavily on local features the agent-agnostic state drops — whether this agent can fire right now, how exposed it is, its relative distances to the allies and enemies inside its own sight range — and those features live in the agent's own observation $o^a$, which the critic is currently not reading at all (I deliberately did not, because the EP baseline's config fixed `obs_individual_obs=False`). The fix is to stop choosing between global and local and feed the critic *both*: for agent $a$, the global state $s$ together with that agent's own observation $o^a$, $V_\phi([s, o^a, e_a])$. It is *agent-specific* because each agent now gets a different critic input ($s$ shared, $o^a$ not), carrying the EP critic's comprehensive global picture *and* the local detail EP was dropping, at the cost of a single $o^a$ rather than the full all-observations concatenation.

I have to make sure I am not re-deriving a failure mode I already climbed past. Is this the attention critic's mistake in disguise? No — and the distinction is the whole point of the ladder. The attention critic collapsed because of *cross-agent mixing*: a softmax over agents fit to a moving bootstrapped target found degenerate basins. The agent-specific critic has *no* mixing — each agent's value is a deterministic MLP of $[s, o^a, e_a]$, agents do not attend to or read each other, and the only cross-agent information is the global state every agent already shared in the EP critic. So I keep the robustness the plain-MLP critics earned and add only the *local* information the EP form dropped. Is this CL, the all-observations concatenation I rejected when I derived the local critic? Also no — CL stacks every agent's observation $(o_1, \dots, o_n)$ into every critic's input, so the dimension grows linearly with $n$ and the value learning turns sample-hungry exactly on the big maps. The agent-specific input adds only *one* $o^a$ per agent, of width $\texttt{obs\_dim}$, not $n$ of them. It threads the needle CL and EP each missed: EP had the global picture but no local detail; CL had all local detail but at $n\times$ the cost and could still miss truly-global structure; AS has the global state *and* this agent's local detail at the cost of one observation.

There is a second payoff that closes a loop from the earlier rungs. I have been sharing one critic across all agents since the local critic, for the data-efficiency and robustness it buys. The EP critic handed that shared network the *same* $s$ for every agent, so it could only distinguish agents through the one-hot — a thin, identity-only signal. With the agent-specific input, each agent feeds in its own $o^a$, so one shared network produces *different* values for different agents *grounded in their actual local situation*, not just their index. The agent-specific component is the mechanism that makes a shared critic genuinely agent-specific; the one-hot was the impoverished stand-in for it. On MMM's mixed unit types and 3s5z's stalkers and zealots, that is exactly the per-agent specialization the EP critic could only gesture at through the id.

I should be honest about the fully optimized version and why I land where I do. The agent-specific state $[s, o^a]$ can double-count features: in SMAC the environment state and a local observation overlap (both encode, say, enemy health), so concatenating them inflates the input with redundant dimensions. The fully pruned form (FP) removes the duplicated features — keep $s$, append only the part of $o^a$ not already in $s$ — for the same information content at a smaller input. But that pruning needs a per-feature overlap map, and smaclite gives `state` and `obs` as dense vectors without an exposed correspondence between their components, so it is not runnable here. The faithful form is the un-pruned AS input $[s, o^a, e_a]$: it loses no information; it only carries some redundant dimensions, and a width-128 MLP absorbs that easily. So I keep all the information and accept the modest redundancy.

The module is the EP critic with the obs threaded back in, combining the two structural shape patterns I have already used. The global state is one vector per $(B, T)$, so it must be *broadcast* across the agent axis — `state.unsqueeze(2).expand(-1, -1, n, -1)` to $(B, T, n, \texttt{state\_dim})$ — the same broadcast the EP critic used. The observations already live on the agent axis as $(B, T, n, \texttt{obs\_dim})$, so they line up directly with *no* broadcast — the same non-broadcast the local critic used; each agent's slot already carries its own $o^a$. The agent one-hot is the $n\times n$ identity broadcast to $(B, T, n, n)$. Concatenate $[\text{state}, \text{obs}, \text{agent\_id}]$ along the last axis to $(B, T, n, \texttt{state\_dim} + \texttt{obs\_dim} + n)$, run the identical 128–128–1 ReLU MLP, scalar head, out comes $(B, T, n, 1)$ — the trailing singleton the learner squeezes. The only change from the EP critic is `input_shape` growing by $\texttt{obs\_dim}$ and the critic now reading `batch["obs"]` as well as `batch["state"]`. Nothing in the actor, learner, GAE, or optimizer moves: same `q_nstep=5` target from a target copy, same masked MSE regression, same clipped actor surrogate on each agent's own ratio, same reward standardization and grad-clip. This is single-agent PPO with one thing changed from the strongest baseline — the value input now sees both the global state and the agent's own observation.

The bar this has to clear, against the EP critic's real numbers: MMM 0.927, 2s3z 0.833, 3s5z 0.740. With no cross-agent mixing I expect no seed collapse and a spread no worse than EP's on any map — a dead seed would falsify the "no mixing, no degenerate basin" claim and I would stop. The gain I expect is concentrated exactly where the EP critic was weakest and for the reason I diagnosed: **3s5z**, the largest team, where the agent-agnostic state dropped the most local detail — adding $o^a$ should lift it the most, and the cleanest confirmation is 3s5z closing toward the other two maps rather than trailing at 0.740. MMM, near-saturated at 0.927, should gain the least for lack of headroom, and 2s3z modestly. So the falsifiable prediction is an *ordering*: the improvement over EP should be largest on 3s5z, smallest on the near-saturated MMM — the inverse of where the EP critic gained over the local critic. If instead the agent-specific input helped uniformly or helped MMM most, that would tell me the 3s5z weakness was not the agent-agnostic state but something else, and the agent-specific story would be wrong. The thing I watch hardest is whether the per-agent values become genuinely agent-specific on the heterogeneous maps — whether the Medivac and the Marines get distinguishably different baselines now that the critic sees each one's local situation rather than only its id.

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
