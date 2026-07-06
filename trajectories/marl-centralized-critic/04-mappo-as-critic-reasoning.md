The centralized state-V critic confirmed both halves of what I predicted, and I can put the arithmetic
on it. Bringing the global state back into the robust MLP lifted exactly the maps where I said global
structure was the missing quantity. Measure the gain over the local critic map by map: 2s3z went from
0.448 to 0.833, a `+0.385` jump — the single biggest gain on the ladder, and it cleared the 0.7 bar I
guessed; MMM went from 0.635 to 0.927, `+0.292`; 3s5z went from 0.635 to 0.740, only `+0.104`. So the
ordering of gains is 2s3z `> ` MMM `>` 3s5z, the exact ranking I committed to at the last rung — "helps
most on 2s3z and MMM, helps least on the biggest team 3s5z." And the paired variance prediction landed
too: 2s3z's local-critic spread of 0.78 (`{0.125, 0.906, 0.313}`) compressed to 0.156
(`{0.9375, 0.7812, 0.7812}`) once every seed got the same global coordination signal. No seed collapse
anywhere, as I argued there could not be without cross-agent mixing.

But the interesting part is not that the state critic is strong — it is *where* it is weakest, because
the weakness is a fingerprint. The EP critic's worst map is 3s5z at 0.740, below MMM's 0.927 and 2s3z's
0.833 by 0.19 and 0.09 respectively. I called this in advance from the information asymmetry: the
environment-provided state is *agent-agnostic*, a third-person board description identical for every agent,
so it trades away each agent's egocentric local detail for whole-team global structure, and on the biggest
team (8 agents, where a single agent's slice of the fight is its own micro-battle) that trade gives back
the least. Now here is the tell that turns the diagnosis from plausible to sharp — the per-seed
comparison between the local critic and the state critic *on 3s5z specifically*. The local critic's 3s5z
seeds were `{42: 0.8125, 123: 0.4688, 456: 0.6250}`; the state critic's are `{42: 0.7188, 123: 0.6562,
456: 0.8438}`. Two seeds rose sharply when I added the state (123 by `+0.187`, 456 by `+0.219`), but seed
42 *fell*, from 0.8125 down to 0.7188, a `−0.094` regression. On the biggest team, one seed's local-obs
critic actually *beat* the global-state critic. That is the one place on the entire ladder where seeing
less-but-local outscored seeing global, and it is exactly on the map where the diagnosis says the
agent-agnostic state dropped the most egocentric detail. The state critic sees the whole team's board but
not what each agent is locally facing, and on a crowded 8-agent map that omission is where the per-agent
advantage goes soft — soft enough that a purely local critic beat it on one seed. The signature is
precise, and it names the remaining gap: the agent-agnostic state input.

Let me step back and read the whole ladder as the controlled sweep it turned into, because three rungs of
numbers now constrain the fourth. Rung one to two changed *two* things at once — it removed the attention
mixing and the global state together — and the result was that the exact-0.0 collapses vanished but 2s3z
stayed capped; the lesson was that value-learning robustness is necessary but a local critic's information
is short where coordination matters. Rung two to three changed exactly *one* thing — swap `o^a` for `s`,
same MLP — and the result was clean: `+0.385` on 2s3z, `+0.292` on MMM, `+0.104` on 3s5z; the lesson was
that global structure is worth bringing back, and it pays in proportion to how much cross-agent
coordination the map demands. Reading the per-map winners across all three baselines, the state critic is
the best mean on every map (MMM 0.927 vs 0.635 vs 0.135; 2s3z 0.833 vs 0.448 vs 0.542; 3s5z 0.740 vs 0.635
vs 0.115) — but the *margin* is the tell. On MMM and 2s3z the state critic dominates the local one by
0.29 and 0.39. On 3s5z it leads by only 0.105, and inside that thin lead sits the seed-42 inversion where
local beat global. So the ladder has isolated the residual with unusual precision: robustness is handled
(no rung since the first has collapsed), global structure is handled (the state critic banked it), and the
one thing still unaccounted for is per-agent egocentric local detail — visible only on the biggest team,
where it is both most needed and most missing. That is a single, named deficit, and the fourth rung should
be the minimal edit that closes it and nothing more.

So the next move is forced, and it is not a new idea so much as the union of the two the ladder already
proved. The EP critic leans entirely on the one-hot to make the shared network agent-specific, and the
one-hot is a *constant* — it tells the network *which* agent this is but nothing about *what that agent is
currently seeing*. The per-agent value of a situation in a SMAC fight depends heavily on local features the
agent-agnostic state drops: whether this agent can fire right now, how exposed it is, its relative
distances to the allies and enemies inside its own sight range. Those features live in the agent's own
observation `o^a`, which the critic is currently not reading at all — I deliberately did not, because the
EP baseline's config fixed `obs_individual_obs=False`. And the 3s5z seed-42 result is the empirical proof
that this detail was load-bearing: a critic that had `o^a` (the local one) beat a critic that had only `s`
right there. The fix is to stop choosing between global and local and feed the critic *both*: for agent
`a`, the global state `s` together with that agent's own observation `o^a`. This is the agent-specific
global state — agent-specific because each agent now gets a *different* critic input (`s` shared, `o^a`
not). It carries the EP critic's comprehensive global picture *and* the local detail EP was dropping, and
it does so at the cost of a single `o^a`, not the full all-observations concatenation.

Let me lay the alternatives out and let the cost arithmetic do the eliminating, because "add observations
to the state critic" has more than one form and only one of them is right for the map I am trying to fix.
There is the concatenation-of-local form (CL): stack *every* agent's observation `(o_1, …, o_n)` into
every critic's input, so each critic sees the whole team's local views. And there is the agent-specific
form (AS): add only *this* agent's own `o^a`. The information CL adds over AS is the *other* agents' local
views — but their global-relevant content is already in `s`, which the critic reads, and their
agent-specific egocentric content is exactly what agent `a`'s *own* value does not need (agent `a`'s
baseline should condition on agent `a`'s situation, per the coordinate-step argument from two rungs ago).
And the cost is decisive: CL's input grows as `n · obs_dim`, so on MMM it adds `10 · obs_dim`, on 3s5z
`8 · obs_dim`, versus AS's single `1 · obs_dim` — CL is `n` times the extra input width, i.e. eight to ten
times the added dimensionality, and it piles that on precisely the big maps where the value regression is
already the hardest and 3s5z is the map I am trying to rescue. A high-dimensional, sample-hungry input on
the biggest team is the opposite of what the diagnosis calls for. So CL adds cost that scales with the
team size to carry information a single agent's value does not use, and it does so most on the map that can
least afford it. AS adds one observation. That is the form that threads the needle: EP had the global
picture but no local detail; CL had all local detail but at `n×` the cost and could still miss truly-
global structure; AS has the global state *and* this agent's local detail at the cost of one observation.

There is a principled reason, not just a cost reason, that `o^a` is the right observation to add and the
other agents' are not, and it comes straight from the coordinate-ascent argument I used to justify the
local critic two rungs ago. Holding the other agents' policies fixed, agent `a` faces an induced
single-agent POMDP, and the correct baseline for a policy that acts on `o^a` is a value function on that
*same* information set — plus, since I am training centrally, whatever global context sharpens the target,
which is `s`. So agent `a`'s ideal critic input is precisely `[s, o^a]`: the global state that reduces the
target's variance and the agent's own observation that defines its coordinate-step information set. The
other agents' observations `o^{b≠a}` are not part of agent `a`'s coordinate-step information set — their
policies are what the induced POMDP holds fixed, and their globally-relevant state is already inside `s`.
So CL does not merely cost more; it feeds agent `a`'s baseline information that the coordinate step says is
not agent `a`'s to condition on, which is one more way its advantages could be noisier without being
better. AS is the input the principle picks out, and the cost arithmetic just confirms it is also the
cheap one.

I also have to make sure I am not quietly re-deriving a failure mode I already climbed past, so let me
check the two I know. Is this the attention critic's mistake in disguise? No — and the distinction is the
whole point of the ladder. The attention critic collapsed because of *cross-agent mixing*: a softmax over
agents fit to a moving bootstrapped target found degenerate basins. The agent-specific critic has *no*
mixing. Each agent's value is a deterministic MLP of `[s, o^a, e_a]`; agents do not attend to or read each
other; the only cross-agent information is the global state every agent already shared in the EP critic.
So I keep the robustness that the plain-MLP critics earned — no attention, no `~198k`-parameter
interaction core, no degenerate basin — and add only the *local* information the EP form dropped. And is
this CL by another name? Also no, and the cost arithmetic above is exactly why: AS adds one `o^a` of width
`obs_dim`, not `n` of them.

There is a second payoff in the agent-specific input that closes a loop I opened two rungs back. I have
been sharing one critic across all agents since the local critic, for the data-efficiency reason
(homogeneous agents, `n`-fold data per gradient step, and the robustness that bought). The EP critic
handed that shared network the *same* `s` for every agent, so it could only distinguish agents through the
one-hot — and I noted at the time that this made the id do *all* the per-agent work, a thin, constant
signal. With the agent-specific input, each agent feeds in its *own* `o^a`, so one shared network produces
*different* values for different agents *grounded in their actual local situation*, not just their index.
The agent-specific component is the mechanism that makes a shared critic genuinely agent-specific; the
one-hot was the impoverished stand-in for it. On the heterogeneous maps — MMM's mixed unit types, 3s5z's
stalkers and zealots — that is exactly the per-agent specialization the EP critic could only gesture at
through the id, and it is why I expect the payoff concentrated on the big, mixed team.

Let me verify the capacity claim rather than assert it, because it is the cleanest argument that AS should
not *regress* the EP baseline. The AS input `[s, o^a, e_a]` is a strict *superset* of the EP input
`[s, e_a]` — it is the same vector with `o^a` appended. A width-128 `fc1` can represent any EP function
exactly by assigning zero weight to the `o^a` block of its input; the rest of the network is byte-for-byte
the EP critic. So the AS hypothesis class *contains* the EP hypothesis class: whatever value function the
EP critic learned, the AS critic can represent. That means AS cannot be worse than EP for a
*representational* reason — if it underperforms, the cause must be optimization or overfitting on the extra
input, not a missing capability. I should be honest that this is a nesting argument about representable
functions, not a training guarantee: more input dimensions can make the fit harder or invite overfitting,
and the un-pruned `[s, o^a]` carries redundant dimensions because in SMAC the state and a local
observation overlap (both encode, say, an enemy's health), so concatenating them duplicates features. But
the redundancy costs input *width*, not information: a 128-wide `fc1` can learn to down-weight the
duplicated dimensions, and the fully pruned form (keep `s`, append only the part of `o^a` not already in
`s`) would need a per-feature overlap map that smaclite does not expose — it hands me `state` and `obs` as
dense vectors with no correspondence between their components. So the faithful, runnable form here is the
un-pruned agent-specific input `[s, o^a, e_a]`: it loses no information, only carries some redundant
dimensions that a width-128 MLP absorbs, and by the nesting argument its floor is the EP critic's
behavior.

It is worth naming that this is where the central-V design space actually terminates under what the
harness exposes, so the choice reads as an endpoint and not an arbitrary stop. The input can be local only
(the IPPO critic, robust but blind to global structure — the rung-two floor), global only (the EP critic,
the config-defined baseline, agent-agnostic — the rung-three strongest baseline), or global-plus-local.
Global-plus-local has exactly three forms: the full concatenation CL, dominated on both cost (it scales as
`n · obs_dim`) and principle (it feeds agent `a` the other agents' egocentric detail its coordinate step
does not condition on); the feature-pruned AS, which is strictly the best on input dimensionality but needs
a per-feature `s`/`o^a` overlap map that smaclite does not hand me; and the un-pruned AS, which loses no
information and pays only in redundant width a 128-wide MLP absorbs. Re-introducing cross-agent mixing on
top of any of these is the one move the first rung already falsified — the loop has no return
normalization and no warmup to stabilize a softmax over agents, and it collapsed. So among the inputs I
can actually run, un-pruned AS is the maximal one that stays inside the robust plain-MLP regime: it is the
last rung because every richer input either needs machinery smaclite withholds or reintroduces the
instability the ladder climbed past.

The module is the EP critic with the obs threaded back in, and the shapes combine the two structural
patterns I have already used — this is where the local critic's non-broadcast and the state critic's
broadcast meet in one forward pass. The global state is one vector per `(B, T)`, so it must be *broadcast*
across the agent axis — `state.unsqueeze(2).expand(-1, -1, n, -1)` to `(B, T, n, state_dim)` — the same
broadcast the EP critic used. The observations already live on the agent axis as `(B, T, n, obs_dim)`, so
they line up directly with *no* broadcast — the same non-broadcast the local critic used; each agent's
slot already carries its own `o^a`. The agent one-hot is the `n×n` identity broadcast to `(B, T, n, n)`.
Trace it on 3s5z with `n = 8` to be sure the axes align: `state` `(B, T, state_dim)` expands to
`(B, T, 8, state_dim)`; `obs` is already `(B, T, 8, obs_dim)`; `eye(8)` broadcasts to `(B, T, 8, 8)`;
concatenate along the last axis to `(B, T, 8, state_dim + obs_dim + 8)`, run the identical 128–128–1 ReLU
MLP, scalar head, out comes `(B, T, 8, 1)` — the trailing singleton the learner squeezes. The one thing
I must not get wrong is that `obs` is *not* broadcast: if I accidentally expanded a single agent's obs
across the axis I would feed every agent the same `o`, destroying the per-agent grounding that is the whole
point; the check is that `obs` enters `forward` already shaped `(B, T, n, obs_dim)` and I concatenate it
untouched. The only change from the EP critic is `input_shape` growing by `obs_dim` and the critic now
reading `batch["obs"]` as well as `batch["state"]`. Nothing in the actor, the learner, GAE, or the
optimizer moves — same `q_nstep=5` target from a target copy, same masked MSE regression, same clipped
actor surrogate on each agent's own ratio, same reward standardization and grad-clip. This is single-agent
PPO with one thing changed from the strongest baseline: the value input now sees both the global state and
the agent's own observation.

Here is the bar this has to clear and what I would validate, stated against the EP critic's real numbers
with no invented ones. The EP critic landed MMM 0.927, 2s3z 0.833, 3s5z 0.740 (means). The agent-specific
critic must not regress the seed robustness — there is no cross-agent mixing, so I expect no seed collapse
and a spread no worse than the EP critic's on any map; if a seed died, that would falsify the "no mixing,
no degenerate basin" claim and I would stop. The gain I expect is concentrated exactly where the EP
critic was weakest and for the reason I diagnosed: **3s5z**, the largest team, where the agent-agnostic
state dropped the most local detail and where the local critic already beat the state critic on seed 42.
Adding `o^a` should lift 3s5z the most, and it has both the diagnosis and the headroom on its side — MMM
at 0.927 has only 0.073 of win rate left to gain, 2s3z at 0.833 has 0.167, but 3s5z at 0.740 has the most
room (0.26) *and* is where the mechanism points. So the falsifiable prediction is an *ordering*: the
improvement over EP should be *largest on 3s5z*, smallest on the near-saturated MMM — the exact inverse of
where the EP critic gained over the local critic (there the gain was largest on 2s3z, smallest on 3s5z).
The cleanest single confirmation would be 3s5z closing toward the other two maps rather than trailing at
0.740, and a sharper one still would be seed 42 on 3s5z recovering the ground it lost when I swapped its
`o^a` for `s` — since AS hands that observation back, seed 42 should climb back above the 0.7188 the state
critic left it at, toward the 0.8125 it reached with local obs alone. If instead the agent-specific input
helped uniformly, or helped MMM most, or a seed died, that would tell me the 3s5z weakness was not the
agent-agnostic state but something else — the team size straining the value regression, say — and the
agent-specific story would be wrong. The thing I would watch hardest, beyond the win rates, is whether the
per-agent values become genuinely agent-specific on the heterogeneous maps — whether the Medivac and the
Marines get distinguishably different baselines now that the critic sees each one's local situation rather
than only its id. The distilled agent-specific critic — global state plus the agent's own observation plus
the one-hot, plain MLP, no attention — is in the answer.
