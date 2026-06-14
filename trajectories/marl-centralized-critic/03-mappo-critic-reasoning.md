The local critic did the first thing I asked of it and confirmed the second thing I worried about. The
bimodal collapse is gone: no seed is stuck at exactly 0.0 on a map the others win, and the 4.9-return
disaster from the attention critic has no analogue here. Every seed trains. The means jumped on exactly
the maps where the attention critic had died — MMM went from 0.135 to 0.635, 3s5z from 0.115 to 0.635 —
which is the "remove the dead-seed floor" effect I predicted, and 3s5z in particular is now *tight*
({0.81, 0.47, 0.62}), the kind of seed-to-seed consistency a robust value regression buys. So the
diagnosis from step 1 was right: the binding constraint was value-learning robustness, and a plain local
MLP fixed it. But look at where the local critic is *capped*. On 2s3z the mean is only 0.448, and the
per-seed spread is wide — {0.125, 0.906, 0.313}. That is not a value-learning collapse (the returns
track the win rates sensibly: 12.96, 19.65, 14.64); it is the local critic's *ceiling*. Seed 42 reaches
0.125 and seed 456 only 0.313 on a map where seed 123 shows 0.906 is attainable. The local critic is
robust but it is missing something on 2s3z — and 2s3z is a symmetric map where coordinated positioning of
the two stalkers and three zealots matters, exactly the kind of *global, cross-agent* structure a critic
that sees only `o^a` is blind to. That is the "robust but capped" pattern I said I would watch for, and
it is the cleanest possible signal that the next move is to bring the global state back — now that the
value learning is no longer the bottleneck.

So I return to centralization, but I have to be careful, because the attention critic *was* centralized
(its tokens carried the state) and it collapsed. The lesson was not "centralization is bad"; it was
"cross-agent *mixing* in the critic, under this loop's value-learning conditions, is fragile." The clean
way to bring back global information *without* reintroducing the mixing that blew up is to feed the global
state into the same plain, robust MLP that just worked — a shared `V_φ([s, e_a])` over the global state
plus an agent one-hot, no attention, no cross-agent interaction layer, nothing that can land in a
degenerate basin. This is the standard centralized state-value critic, and it is the smallest possible
change from the local critic: swap `o^a` for `s` in the input, keep the identical 128–128–1 MLP and the
identical agent one-hot. I am deliberately keeping everything the local critic got right (the robust MLP,
parameter sharing, the id) and changing only the *information* the critic conditions on.

Let me justify the state-value choice from the ground up rather than defaulting to it. Why a *state*
value and not the action-value critics of the centralized prior art (COMA's `Q(s,A)` with a
counterfactual baseline, MADDPG's per-agent `Q_i(x, a_1,…,a_n)`)? Because the GAE baseline I am
subtracting only ever needs a *state* value. The advantage identity says `∇J = E[∇log π · A]` for any
baseline independent of the action given the state, and GAE forms `δ_t = r_t + γV(s_{t+1}) − V(s_t)` from
`V`, never `Q`. Conditioning the critic on the joint action would make the input scale with `n`, couple
the critic to every agent's current policy, and (for COMA) force an action enumeration to build the
counterfactual — all machinery I do not need for variance reduction. A centralized `V(s)` buys the
variance reduction and sheds every one of those costs. So the strongest-baseline critic is single-agent
PPO with one thing changed from the local critic: the value function reads the global state.

Now the input itself, and here I have to be honest about what *this* harness exposes versus the richer
forms the centralized-V idea can take, because the difference is load-bearing. The fully developed
agent-specific construction would feed the critic the global state *and* each agent's own observation
`[s, o^a]` (pruning the overlap), and there is a real argument that this is better — the bare global
state is agent-agnostic and drops the local features a per-agent value depends on. But the config this
task fixes is explicit: `obs_individual_obs=False` and `obs_last_action=False`, with `obs_agent_id=True`.
So the centralized critic the *task* defines is the lean **environment-provided** form: global state plus
an agent one-hot, and *only* that. I am not going to import the agent-specific `[s, o^a]` machinery into
this step, because the baseline I am deriving is the one the harness actually runs — `V_φ([s, e_a])`. I
note the gap explicitly, because it is exactly where a stronger critic would go next: this critic sees
the global picture but, being agent-agnostic in its state input, it leans entirely on the one-hot to
distinguish agents, and the one-hot carries an agent's *identity* but none of its current local detail
(what it can fire, how exposed it is). If that omission caps this critic the way blindness to global
structure capped the local one, the agent-specific input is the obvious lever — but that is a later
move; here I derive the EP critic the config specifies.

Why does the one-hot earn its place even in the lean form? Because I am sharing one critic across all
agents (the same robustness-and-data argument as before: homogeneous agents, `n`-fold data per step). A
shared network handed the *same* `s` for every agent can only output one value, identical for all
agents — which is fine only if the true per-agent value is genuinely agent-agnostic, and on a
heterogeneous map like MMM (a Medivac that heals, Marauders, Marines) it is not: a healer's value of a
state differs from a frontline marine's. The one-hot is the minimal thing that lets the shared network
emit a *different* value per agent — it can route different agents to different value functions while
still pooling all the data. So `[s, e_a]` is the leanest centralized critic that is both data-efficient
(shared) and able to distinguish agents (the id). It is exactly the local critic's architecture with the
information swapped from local to global, plus the same id it already carried.

The shapes mirror the local critic with one structural difference, and it is the inverse of the IPPO
case. The global state is one vector per `(B, T)`, shape `(B, T, state_dim)`, so to put it on the agent
axis I must *broadcast* it across the `n` agents — `state.unsqueeze(2).expand(-1, -1, n, -1)` to
`(B, T, n, state_dim)`. (The local critic needed no broadcast because `obs` already lived on the agent
axis; the state critic does, because the state is shared.) The agent id is the `n×n` identity broadcast
to `(B, T, n, n)`. Concatenate to `(B, T, n, state_dim + n)`, run the identical 128–128–1 ReLU MLP, and
the head gives `(B, T, n, 1)` — the trailing singleton the learner squeezes. `batch["obs"]` is now never
read; the critic conditions purely on the shared global state and the constant id. Everything downstream
is the same fixed loop the previous two rungs used: the `q_nstep=5` target from a target copy of this
critic, masked MSE regression, the actor's clipped surrogate on each agent's own ratio, the entropy
bonus, reward standardization, Adam, grad-clip.

Let me also knock down the worry that this just re-creates the attention critic's failure with a state
input. The attention critic collapsed because of *cross-agent mixing* fit to a moving bootstrapped
target — the softmax over agents could find a degenerate pattern, and per-agent values could smear
toward a team mean. The state-V critic has *no* mixing: each agent's value is a deterministic MLP of
`[s, e_a]`, the agents do not attend to each other, and the only per-agent variation comes from the
fixed one-hot. There is no degenerate basin for attention to fall into because there is no attention. So
this critic should inherit the local critic's robustness (same MLP, same fit) while adding the global
information the local critic lacked — the best of both prior rungs.

Here is what I expect against the two baselines I have measured, stated to be falsified. Against the
attention critic: the seed collapse stays gone (no mixing to break), so no dead seeds anywhere — this
should be as robust as the local critic. Against the local critic: the *ceiling* should lift on exactly
the maps where global structure matters and the local critic was capped. I expect 2s3z — the local
critic's weakest map (0.448), where coordinated positioning is the whole game — to improve the most,
because that is where seeing the global state should most help the per-agent advantage; I would be
surprised if 2s3z did not clear 0.7. I also expect MMM to rise, because its heterogeneity (a healer plus
two unit types) is precisely where a critic that sees the *whole* team's state, rather than each agent's
local slice, should produce better-aligned advantages — and where the one-hot lets the shared net give
the Medivac a different value than a Marine. 3s5z is the interesting case: the local critic was already
tight at 0.635, and 3s5z is a larger team (8 agents) where the global state is higher-dimensional and the
agent-agnostic EP form drops more local detail — so I expect an improvement but a more modest one, and I
would not be shocked if 3s5z is where the EP critic's lack of agent-specific local features shows up as
the smallest gain. If the pattern is "centralization clearly helps on 2s3z and MMM, helps least on the
biggest team 3s5z," that is the signature that the global state is worth bringing back *and* that the
remaining gap is the agent-agnostic state input — the agent-specific `[s, o^a]` lever I deliberately left
on the table. The distilled lean central-V module — global state plus agent one-hot, plain MLP, no
attention, no obs — is in the answer.
