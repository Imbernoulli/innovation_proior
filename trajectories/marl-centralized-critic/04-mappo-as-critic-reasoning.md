The centralized state-V critic confirmed both halves of what I predicted, and the half it confirmed
about its *own* weakness is exactly the lever I left on the table. Bringing the global state back into the
robust MLP did what I expected on the maps where global structure matters: 2s3z jumped from the local
critic's capped 0.448 to 0.833 — the single biggest gain on the ladder, and it cleared the 0.7 bar I
guessed — and MMM rose from 0.635 to 0.927, the heterogeneity payoff where a whole-team state lets the
shared net give the Medivac a different value than a Marine. No seed collapse anywhere, as I argued there
could not be without cross-agent mixing. But look at where the EP critic is *weakest*: 3s5z, mean 0.740,
the lowest of its three maps — {0.719, 0.656, 0.844}. I called this in advance: 3s5z is the largest team
(8 agents), the global state is highest-dimensional there, and the environment-provided state is
*agent-agnostic*, so on the biggest team it drops the most agent-specific local detail and the critic has
to lean hardest on the one-hot — which carries an agent's identity but none of its current local
situation. The signature is precisely "centralization helps most on 2s3z and MMM, helps least on the
biggest team 3s5z," and that signature is the diagnosis: the remaining gap is the *agent-agnostic state
input*. The critic sees the whole team's state but not what each agent is locally facing, and on a
crowded 8-agent map that omission is where the per-agent advantage goes soft.

So the next move is forced. The EP critic leans entirely on the one-hot to make the shared network
agent-specific, and the one-hot is a *constant* — it tells the network *which* agent this is but nothing
about *what that agent is currently seeing*. The per-agent value of a situation in a SMAC fight depends
heavily on local features the agent-agnostic state drops: whether this agent can fire right now, how
exposed it is, its relative distances to the allies and enemies inside its own sight range. Those
features live in the agent's own observation `o^a`, which the critic is currently not reading at all
(I deliberately did not, because the EP baseline's config fixed `obs_individual_obs=False`). The fix is
to stop choosing between global and local and feed the critic *both*: for agent `a`, the global state `s`
together with that agent's own observation `o^a`. This is the agent-specific global state — agent-specific
because each agent now gets a *different* critic input (`s` shared, `o^a` not). It carries the EP
critic's comprehensive global picture *and* the local detail EP was dropping, and it does so at the cost
of a single `o^a`, not the full all-observations concatenation.

Let me make sure I am not just re-deriving the failure modes I already climbed past. First, is this the
attention critic's mistake in disguise? No — and the distinction is the whole point of the ladder. The
attention critic collapsed because of *cross-agent mixing*: a softmax over agents fit to a moving
bootstrapped target found degenerate basins. The agent-specific critic has *no* mixing. Each agent's
value is a deterministic MLP of `[s, o^a, e_a]`; agents do not attend to or read each other; the only
cross-agent information is the global state every agent already shared in the EP critic. So I keep the
robustness that the plain-MLP critics earned — no attention, no degenerate basin — and add only the
*local* information the EP form dropped. Second, is this CL, the all-observations concatenation that I
rejected back when I derived the local critic? No — CL stacks every agent's observation `(o_1,…,o_n)`
into every critic's input, so the dimension grows linearly with `n` and the value learning gets
sample-hungry exactly on the big maps. The agent-specific input adds only *one* `o^a` per agent, of width
`obs_dim`, not `n` of them. It threads the needle CL and EP each missed: EP had the global picture but no
local detail; CL had all local detail but at `n`× the cost and could still miss truly-global structure;
agent-specific has the global state *and* this agent's local detail at the cost of one observation.

There is a second payoff in the agent-specific input that closes a loop from the earlier rungs. I have
been sharing one critic across all agents since the local critic, for the data-efficiency reason
(homogeneous agents, `n`-fold data per gradient step, and the robustness that bought). The EP critic
handed that shared network the *same* `s` for every agent, so it could only distinguish agents through
the one-hot — a thin, identity-only signal. With the agent-specific input, each agent feeds in its *own*
`o^a`, so one shared network produces *different* values for different agents *grounded in their actual
local situation*, not just their index. The agent-specific component is the mechanism that makes a shared
critic genuinely agent-specific; the one-hot was the impoverished stand-in for it. On the heterogeneous
maps — MMM's mixed unit types, 3s5z's stalkers and zealots — that is exactly the per-agent specialization
the EP critic could only gesture at through the id.

I should be honest about the fully optimized version and why I am landing where I am. The agent-specific
state `[s, o^a]` can double-count features: in SMAC the environment state and a local observation overlap
(both encode, say, enemy health), so concatenating them inflates the input with redundant dimensions. The
fully pruned form removes the duplicated features — keep `s`, append only the part of `o^a` not already in
`s` — for the same information content at a smaller input. That pruning needs a per-feature overlap map,
which this harness does not hand me cleanly: smaclite gives `state` and `obs` as dense vectors without an
exposed correspondence between their components. So the faithful, runnable form here is the un-pruned
agent-specific input `[s, o^a, e_a]`. I note that the pruned form would be strictly better on input
dimensionality at no information cost *if* the overlap map were available — but the un-pruned form loses
no information; it only carries some redundant dimensions, and a width-128 MLP absorbs that easily. So I
keep all the information and accept the modest redundancy.

The module is the EP critic with the obs threaded back in, and the shapes combine the two structural
patterns I have already used. The global state is one vector per `(B, T)`, so it must be *broadcast*
across the agent axis — `state.unsqueeze(2).expand(-1, -1, n, -1)` to `(B, T, n, state_dim)` — the same
broadcast the EP critic used. The observations already live on the agent axis as `(B, T, n, obs_dim)`, so
they line up directly with *no* broadcast — the same non-broadcast the local critic used; each agent's
slot already carries its own `o^a`. The agent one-hot is the `n×n` identity broadcast to `(B, T, n, n)`.
Concatenate `[state, obs, agent_id]` along the last axis to `(B, T, n, state_dim + obs_dim + n)`, run the
identical 128–128–1 ReLU MLP, scalar head, out comes `(B, T, n, 1)` — the trailing singleton the learner
squeezes. The only change from the EP critic is `input_shape` growing by `obs_dim` and the critic now
reading `batch["obs"]` as well as `batch["state"]`. Nothing in the actor, the learner, GAE, or the
optimizer moves — same `q_nstep=5` target from a target copy, same masked MSE regression, same clipped
actor surrogate on each agent's own ratio, same reward standardization and grad-clip. This is single-agent
PPO with one thing changed from the strongest baseline: the value input now sees both the global state
and the agent's own observation.

Here is the bar this has to clear and what I would validate, stated against the EP critic's real numbers
with no invented ones. The EP critic landed MMM 0.927, 2s3z 0.833, 3s5z 0.740 (means). The agent-specific
critic must not regress the seed robustness — there is no cross-agent mixing, so I expect no seed collapse
and a spread no worse than the EP critic's on any map; if a seed died, that would falsify the "no mixing,
no degenerate basin" claim and I would stop. The gain I expect is concentrated exactly where the EP
critic was weakest and for the reason I diagnosed: **3s5z**, the largest team, where the agent-agnostic
state dropped the most local detail — adding `o^a` should lift 3s5z the most, and the cleanest
confirmation would be 3s5z closing toward the other two maps rather than trailing at 0.740. I expect MMM,
already near-saturated at 0.927, to gain the least simply because there is little headroom left, and 2s3z
to gain modestly. So the falsifiable prediction is an ordering: the improvement over EP should be
*largest on 3s5z*, smallest on the near-saturated MMM — the inverse of where the EP critic gained over the
local critic. If instead the agent-specific input helped uniformly or helped MMM most, that would tell me
the 3s5z weakness was not the agent-agnostic state but something else (the team size straining the value
regression, say), and the agent-specific story would be wrong. The thing I would watch hardest is whether
the per-agent values become genuinely agent-specific on the heterogeneous maps — whether the Medivac and
the Marines get distinguishably different baselines now that the critic sees each one's local situation
rather than only its id. The distilled agent-specific critic — global state plus the agent's own
observation plus the one-hot, plain MLP, no attention — is in the answer.
