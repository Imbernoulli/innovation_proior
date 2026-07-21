The centralized state-V critic confirmed both halves of what I predicted. Bringing the global state back
into the robust MLP lifted exactly the maps where I said global structure was the missing quantity: 2s3z
went from the local critic's 0.448 to 0.833, `+0.385` — the single biggest gain on the ladder, and it
cleared the 0.7 bar I guessed; MMM from 0.635 to 0.927, `+0.292`; 3s5z from 0.635 to 0.740, only `+0.104`.
So the ordering of gains is 2s3z `>` MMM `>` 3s5z, the ranking I committed to. And the paired variance
prediction landed: 2s3z's local-critic spread of 0.78 (`{0.125, 0.906, 0.313}`) compressed to 0.156
(`{0.9375, 0.7812, 0.7812}`) once every seed got the same global coordination signal. No seed collapse
anywhere, as I argued there could not be without cross-agent mixing.

The interesting part is not that the state critic is strong but *where* it is weakest, because the
weakness is a fingerprint. Its worst map is 3s5z at 0.740, below MMM's 0.927 and 2s3z's 0.833. I called
this in advance from the information asymmetry: the environment-provided state is *agent-agnostic*, so on
the biggest team (8 agents, where a single agent's slice of the fight is its own micro-battle) it gives
back the least of the egocentric local detail it traded away. And the per-seed comparison on 3s5z turns
the diagnosis from plausible to sharp. The local critic's 3s5z seeds were `{42: 0.8125, 123: 0.4688, 456:
0.6250}`; the state critic's are `{42: 0.7188, 123: 0.6562, 456: 0.8438}`. Two seeds rose sharply when I
added the state (123 by `+0.187`, 456 by `+0.219`) — but seed 42 *fell*, from 0.8125 to 0.7188, a
`−0.094` regression. On the biggest team, one seed's local-obs critic actually *beat* the global-state
critic. That is the one place on the entire ladder where seeing less-but-local outscored seeing global,
and it is exactly on the map where the diagnosis says the agent-agnostic state dropped the most egocentric
detail. The signature names the remaining gap: the agent-agnostic state input.

Reading the whole ladder, the residual is isolated with unusual precision. Robustness is handled — no rung
since the first has collapsed. Global structure is handled — the state critic banked it, and it is the
best mean on every map. The one thing still unaccounted for is per-agent egocentric local detail, visible
only on the biggest team, where it is both most needed and most missing: 3s5z is where the state critic's
lead over the local critic is thinnest (0.105) and where the seed-42 inversion sits inside that thin lead.
That is a single named deficit, and the finale should be the minimal edit that closes it and nothing more.

So the next move is not a new idea but the union of the two the ladder already proved. The EP critic leans
entirely on the one-hot to make the shared network agent-specific, and the one-hot is a *constant* — it
tells the network *which* agent this is but nothing about *what that agent is currently seeing*. The
per-agent value in a SMAC fight depends heavily on local features the agent-agnostic state drops: whether
this agent can fire right now, how exposed it is, its relative distances to allies and enemies inside its
own sight range. Those features live in the agent's own observation `o^a`, which the critic is not reading
at all — I deliberately did not, because the EP config fixed `obs_individual_obs=False`. The 3s5z seed-42
result is the empirical proof that the detail was load-bearing: a critic that had `o^a` beat one that had
only `s` right there. The fix is to stop choosing between global and local and feed the critic *both*: for
agent `a`, the global state `s` together with its own observation `o^a`. This is the agent-specific global
state — agent-specific because each agent now gets a *different* critic input (`s` shared, `o^a` not). It
carries EP's comprehensive global picture *and* the local detail EP was dropping, at the cost of a single
`o^a`.

There is more than one way to add observations, and only one is right for the map I am trying to fix.
There is the concatenation-of-local form (CL): stack *every* agent's observation `(o_1, …, o_n)` into
every critic's input. And there is the agent-specific form (AS): add only *this* agent's `o^a`. The
cost is decisive: CL's input grows as `n · obs_dim`, so on 3s5z it adds `8 · obs_dim` and piles that on
precisely the biggest map where the value regression is already hardest and which I am trying to rescue,
versus AS's single `obs_dim`. And there is a principled reason, not just a cost reason, that `o^a` is the
right observation and the others are not, straight from the coordinate-ascent argument I used two rungs
ago: holding the other agents' policies fixed, agent `a` faces an induced single-agent POMDP, and the
correct baseline for a policy acting on `o^a` is a value function on that same information set — plus, since
I train centrally, whatever global context sharpens the target, which is `s`. So agent `a`'s ideal input
is precisely `[s, o^a]`. The other agents' observations `o^{b≠a}` are not in agent `a`'s coordinate-step
information set — their policies are what the induced POMDP holds fixed, and their globally-relevant state
is already inside `s`. So CL does not merely cost more; it feeds agent `a`'s baseline information the
coordinate step says is not its to condition on, one more way its advantages could be noisier without being
better. AS is the input the principle picks out, and the cost arithmetic confirms it is also the cheap one.

I have to make sure I am not re-deriving a failure mode I already climbed past. Is this the attention
critic's mistake in disguise? No — the attention critic collapsed from *cross-agent mixing*, a softmax over
agents fit to a moving bootstrapped target finding degenerate basins. The agent-specific critic has *no*
mixing: each agent's value is a deterministic MLP of `[s, o^a, e_a]`, agents do not attend to or read each
other, and the only cross-agent information is the global state every agent already shared in the EP
critic. So I keep the plain-MLP robustness — no attention, no ~198k interaction core, no degenerate basin —
and add only the local information EP dropped.

There is a second payoff that closes a loop I opened two rungs back. I have shared one critic across all
agents since the local critic, for data efficiency (homogeneous agents, `n`-fold data per step). The EP
critic handed that shared network the *same* `s` for every agent, so it could only distinguish agents
through the one-hot — the id doing *all* the per-agent work, a thin constant signal. With the
agent-specific input each agent feeds in its *own* `o^a`, so one shared network produces different values
grounded in each agent's actual local situation, not just its index. The agent-specific component is the
mechanism that makes a shared critic genuinely agent-specific; the one-hot was the impoverished stand-in.
On the heterogeneous maps — MMM's mixed unit types, 3s5z's stalkers and zealots — that is exactly the
per-agent specialization the EP critic could only gesture at through the id.

There is a clean argument that AS should not *regress* the EP baseline. The AS input `[s, o^a, e_a]` is a
strict *superset* of the EP input `[s, e_a]` — the same vector with `o^a` appended. A width-128 `fc1` can
represent any EP function exactly by zeroing the weights on the `o^a` block, with the rest of the network
byte-for-byte the EP critic. So the AS hypothesis class *contains* the EP one: whatever value function EP
learned, AS can represent, and if AS underperforms the cause is optimization or overfitting on the extra
input, not a missing capability. This is a nesting argument about representable functions, not a training
guarantee: more input dimensions can make the fit harder, and the un-pruned `[s, o^a]` carries redundant
dimensions because in SMAC the state and a local observation overlap (both encode, say, an enemy's health),
so concatenating them duplicates features. But the redundancy costs input *width*, not information — a
128-wide `fc1` can down-weight the duplicated dimensions. The fully pruned form (keep `s`, append only the
part of `o^a` not already in `s`) would need a per-feature overlap map that smaclite does not expose; it
hands me `state` and `obs` as dense vectors with no correspondence between components. So the faithful,
runnable form is the un-pruned `[s, o^a, e_a]`: it loses no information, only carries some redundant
dimensions a width-128 MLP absorbs, and by the nesting argument its floor is the EP critic's behavior.
This is also where the central-V design space terminates under what the harness exposes — every richer
input either needs machinery smaclite withholds (feature-pruned AS) or reintroduces the cross-agent mixing
the first rung already falsified — so un-pruned AS is the maximal input that stays inside the robust
plain-MLP regime.

The module is the EP critic with the obs threaded back in, and the shapes combine the two structural
patterns I have already used. The global state is one vector per `(B, T)`, so it must be *broadcast* across
the agent axis — `state.unsqueeze(2).expand(-1, -1, n, -1)` to `(B, T, n, state_dim)` — the same broadcast
the EP critic used. The observations already live on the agent axis as `(B, T, n, obs_dim)`, so they line
up directly with *no* broadcast, each agent's slot carrying its own `o^a`. The one-hot is the `n×n`
identity broadcast to `(B, T, n, n)`; concatenate `[state, obs, agent_id]` to `(B, T, n, state_dim +
obs_dim + n)`, run the identical 128–128–1 ReLU MLP, scalar head, out comes `(B, T, n, 1)`. The one thing
I must not get wrong is that `obs` is *not* broadcast: if I expanded a single agent's obs across the axis I
would feed every agent the same `o`, destroying the per-agent grounding that is the whole point — the check
is that `obs` enters `forward` already shaped `(B, T, n, obs_dim)` and I concatenate it untouched. The only
change from the EP critic is `input_shape` growing by `obs_dim` and the critic now reading `batch["obs"]`
as well as `batch["state"]`. Nothing in the actor, learner, GAE, or optimizer moves. This is single-agent
PPO with one thing changed from the strongest baseline: the value input now sees both the global state and
the agent's own observation.

Here is the bar this has to clear, against the EP critic's real numbers with none invented. EP means: MMM
0.927, 2s3z 0.833, 3s5z 0.740. The agent-specific critic must not regress the seed robustness — no
cross-agent mixing, so I expect no collapse and a spread no worse than EP's; a dead seed would falsify the
"no mixing, no degenerate basin" claim and I would stop. The gain I expect is concentrated where EP was
weakest and for the reason I diagnosed: **3s5z**, the largest team, where the agent-agnostic state dropped
the most local detail and where the local critic already beat the state critic on seed 42. Adding `o^a`
should lift 3s5z the most — it has both the diagnosis and the headroom (MMM at 0.927 has only 0.073 left,
3s5z at 0.740 has 0.26). So the falsifiable prediction is an *ordering*: the improvement over EP should be
largest on 3s5z, smallest on the near-saturated MMM — the exact inverse of where EP gained over the local
critic. The cleanest single confirmation would be 3s5z closing toward the other two maps rather than
trailing at 0.740, and a sharper one still would be seed 42 on 3s5z recovering the ground it lost when I
swapped its `o^a` for `s`, climbing back above the 0.7188 the state critic left it at, toward the 0.8125 it
reached with local obs alone. If instead the agent-specific input helped uniformly, or helped MMM most, or
a seed died, that would tell me the 3s5z weakness was not the agent-agnostic state but something else — the
team size straining the value regression — and the agent-specific story would be wrong. The thing I would
watch hardest, beyond win rates, is whether the per-agent values become genuinely agent-specific on the
heterogeneous maps — whether the Medivac and the Marines get distinguishably different baselines now that
the critic sees each one's local situation rather than only its id. The distilled agent-specific critic —
global state plus the agent's own observation plus the one-hot, plain MLP, no attention — is in the answer.
