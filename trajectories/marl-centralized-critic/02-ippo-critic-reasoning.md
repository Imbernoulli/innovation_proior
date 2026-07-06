The attention critic told me exactly what I feared, and it told me in the per-seed split. The signature
is the bimodal collapse I flagged, not the "helps most on hard maps" story I hoped for. Let me read the
numbers as mechanism rather than as a scoreboard. Averaged across the three maps the attention critic
lands at `(0.1354 + 0.5417 + 0.1146)/3 = 0.264` win rate, but that average hides the shape of the
failure, and the shape is the whole story. On MMM the three seeds are `{0.0, 0.4062, 0.0}` — two of them
sit at *exactly* 0.0, not 0.02 or 0.05, and one reaches 0.41; a mean of 0.135 is really "one seed that
trained, two that didn't." On 3s5z it is worse: seeds 42 and 123 are both exactly 0.0, only 456 limps to
0.34, mean 0.115. Across the nine runs, five are pinned at exactly zero win rate. A policy that is merely
weak lands at 0.05 or 0.1; a value at *exactly* 0.0 across many seeds is the fingerprint of a degenerate
basin the critic fell into, which is precisely the diagnostic I said in advance would separate a
value-learning collapse from a capacity ceiling.

The single cleanest tell is seed 42 on 2s3z, and it is worth doing the arithmetic I promised myself I
would look for. On 2s3z the three seeds are `{0.0, 0.8125, 0.8125}` in win rate with returns
`{4.9203, 18.8608, 18.7663}`. Look at the two live seeds: their returns differ by only
`|18.8608 − 18.7663| = 0.094`, a spread of half a percent — when the value learning works, the two seeds
converge to essentially the same place. Now the dead seed: win 0.0 with return 4.92, a return that is
`4.92/18.86 ≈ 0.26` of what the working seeds reach, i.e. nearly four times lower. A return of 4.9 on a
map where the same critic elsewhere reaches 18.8 is not "the team played and narrowly lost" — it is the
policy never finding the fight, because the per-agent advantage it was fed came from a critic that landed
in a degenerate basin. That is exactly the fingerprint I named at the previous rung: a win rate pinned at
0.0 with a return far below what a working seed on the same map achieves means the policy never even
engaged, which only a broken baseline can cause.

There is a second, subtler thing in the tables that confirms the critic and not the task is the culprit,
and it is that the collapse *manifests differently on different maps* while always tracing back to the
value function. On 3s5z the two dead seeds (42, 123) have returns `13.6090` and `15.5036` — high, not
collapsed — yet win rate 0.0. That is a different failure texture from 2s3z's seed 42: on 3s5z the policy
does engage and rack up shaped reward (damage dealt) but converts none of it into a clean win, whereas on
2s3z the broken seed barely engages at all. If the *task* were simply too hard I would expect a consistent
failure texture; instead I see "never engages" on one map and "engages but cannot convert" on another,
both under the same critic. The variance in *how* it dies is itself evidence that the instability lives in
the value learning, not in the environment. And the pattern across maps is the exact inverse of the
"structure helps on hard maps" story: the hard, larger-team maps (MMM at 0.135, 3s5z at 0.115) are where
the attention critic collapses hardest, and the medium map (2s3z at 0.542) is where it survives best —
which is the "2s3z survives as the easiest value regression" call I made, now confirmed.

So the mechanism is settled: a high-capacity attention critic, fit to a bootstrapped target via a target
copy with unnormalized returns and masked MSE, collapsing onto an over-smoothed or otherwise broken
attention pattern that the masked MSE tolerates but that poisons the advantages. The mixing across
agents — the one thing the encoder-only critic added over a flat MLP — turned out to be a liability under
these value-learning conditions, because nothing in the fixed loop stabilizes it (no return
normalization, no warmup, a cold transformer under `lr=3e-4`). The lesson is sharp and it points *down*
the complexity ladder: architectural interaction modeling in the critic is not free, and the binding
constraint here is value-learning robustness, not representational power. Before I reconsider *what
information* the critic should see, I should establish a critic whose value regression simply *works* — a
plain MLP, no attention, the most robust thing I can fit — and find out what a stable critic alone buys.

Let me lay out the moves the diagnosis actually leaves open, because there is more than one way to step
down. Option one: keep the global state but drop the attention — a plain MLP over `[s, e_a]`, which
removes the cross-agent mixing that blew up while keeping the centralized information. Option two: go to
the *opposite extreme* from the attention critic — strip both the mixing *and* the global state, and
condition the critic on exactly what the actor conditions on, agent `a`'s own observation `o^a`,
`V_φ(o^a)`. Option three: something in between, say the state plus each agent's own obs. The single-
variable ablation of the mat failure is option one — it changes only the architecture (mix → no mix) and
holds the information constant. But I have a contrarian clue that makes option two the more *informative*
first step, not merely the more cautious one. The folklore says more centralized information is strictly
better — the critic is thrown away at execution, so feed it everything. Yet independent, purely local PPO
is reported to be *strong* on hard SMAC maps, sometimes beating centralized PPO that uses the global
state. That is backwards if centralization were free, and it means the question is not just "does removing
the mixing restore robustness" but "does the critic even *need* the global state." Testing the local
extreme first answers a bigger question: it establishes whether a robust critic on the actor's own
information set is not just stable but *sufficient*, and it sets up option one — bring the state back into
the same robust MLP — as the clean isolating comparison for the next rung. So I take option two now,
deliberately, and hold option one in reserve.

My reflex says this should be obviously worse — it is "independent learning," the thing that is supposed
to fail. Let me argue against it honestly. The classic objection is non-stationarity: if agent `a`'s
critic sees only `o^a` and the other agents are simultaneously learning and changing their policies, then
from `a`'s view the dynamics drift, the critic's targets go stale, and the convergence story evaporates
(Tan 1993). A second classic objection is that an independent learner cannot always separate environment
stochasticity from a peer's exploration, which provably blocks optimal play in some coordination games
(Claus & Boutilier 1998). Those are real, and they are why independent actor-critic and independent
Q-learning are observed to be unstable on cooperative benchmarks. I cannot wave them away. But let me ask
*whose job* stability is. The critic's job is to be a low-variance, low-bias baseline. Stability of the
*policy update* is a different job, and PPO attacks it with the clipped ratio objective. The
non-stationarity disaster in independent learning is specifically that one agent, reacting to a
co-learner's shift, takes a catastrophic policy step, which shifts the dynamics for everyone and spirals.
PPO's clip does not prove the realized ratio can never leave the band, but on the sampled actions it
removes the objective's incentive to push a probability ratio past `[1−ε, 1+ε]` (here `ε = 0.2`) once it
has crossed the useful side. That is exactly the restraint I need against peer-induced surprises — and
notice it is the *opposite* of what the attention critic lacked: there, nothing restrained the *value*
function; here, the actor's clip restrains the *policy*, and the value function is so simple it cannot
misbehave. If clipping keeps every agent's effective step modest, then from any one agent's view the
others drift slowly enough that a local critic can track them. The instability that condemns IAC/IQL may
be an artifact of unrestrained updates, not of the local critic per se.

I want one more argument before I trust a purely local critic as *principled* and not merely "robust by
being dumb," and I want to write the identity out rather than gesture at it. Take three agents and
telescope the joint return into single-agent changes:
`J(π¹′,π²′,π³′) − J(π¹,π²,π³) = [J(π¹′,π²′,π³′) − J(π¹′,π²′,π³)] + [J(π¹′,π²′,π³) − J(π¹′,π²,π³)]
+ [J(π¹′,π²,π³) − J(π¹,π²,π³)]`. Each bracket changes *exactly one* policy while holding the others
fixed, so if every one of the three coordinate terms is non-negative, the joint return rises. Look at one
term — say the first, change only `π³` from `π³` to `π³′` while `π¹′, π²′` are held. Holding the others
fixed, the world agent 3 faces is a single-agent decision process — a POMDP in which `π¹′, π²′` are
absorbed into the dynamics — and agent 3's policy conditions only on `o³`, which is a POMDP policy. So
"improve `π³` holding the rest fixed" is just "improve a single-agent policy on the induced POMDP," and a
restrained single-agent step can make that term non-negative. PPO is a first-order clipped approximation,
not the exact trust-region step, so this is not a theorem for the practical update, and simultaneous
updates without recollection are a further approximation — but the identity tells me the *structure* I
want: each agent takes a restrained single-agent improvement step against the POMDP the others induce.
And the natural baseline for a POMDP policy that conditions on `o³` is a value function on that *same*
information set — `V_φ(o³)`. The local critic is not a compromise forced by execution; it is the baseline
aligned with the coordinate step's information set. The local critic and PPO's clip are the two halves of
one idea: restrained single-agent improvement on the right per-agent information. That is the principle
the attention critic had no claim to — it mixed information across agents that the actor's coordinate step
does not condition on, which is one more reason its advantages could be both high-variance and misaligned.

Now the module, with each remaining choice motivated. Do I learn `n` separate critics or one shared
`V_φ`? The agents on these maps are homogeneous within a type and share observation/action spaces, so a
separate net per agent wastes data — each net learns from one agent's stream. One shared critic lets
every agent's experience update the same parameters, and the multiplier is concrete: on MMM `n = 10`, so
a shared critic sees ten times the samples per gradient step; on 3s5z `n = 8`, eight times; on 2s3z
`n = 5`, five times. That is `n`-fold more data per step, faster and more stable value learning — exactly
what I want after watching the attention critic struggle to fit. The cost is that a shared net cannot
specialize per agent by *index* — but the agents already differ in *behavior* because they see different
observations, so a shared function at different inputs still produces different values. The only thing
sharing strictly removes is index-based specialization, and I hand that back cheaply by appending a
one-hot agent id to the critic input, which adds only `n` dimensions (10 on MMM, 8 on 3s5z, 5 on 2s3z) —
negligible next to `obs_dim`. So: one shared critic, input `[o^a ⊕ one-hot(a)]` of width `obs_dim + n`.
The architecture is the plain PPO MLP I trust: `fc1: (obs_dim+n) → 128`, ReLU; `fc2: 128 → 128`, ReLU;
`fc3: 128 → 1`. Its processing path is `fc2: 128·128 = 16,384` parameters — roughly a *twelfth* of the
attention critic's `~198k`-parameter interaction core — no attention, no cross-agent mixing, nothing that
can collapse. It is the deliberate inverse of the step-1 critic: I traded an order of magnitude of
capacity for a regression that cannot fall into a degenerate basin.

Let me check the one-hot actually earns its `n` extra dimensions and is not cargo-culted from the
scaffold. Consider the limit where I drop it: a shared net on `o^a` alone gives two agents with *identical*
observations *identical* values. For a homogeneous critic that is not a bug — it is correct; two units in
the same local situation should be valued the same. So the one-hot does nothing on truly-symmetric states,
and that is fine. Where it earns its place is the heterogeneous case, and MMM is the sharp example: the
team is 1 Medivac, 2 Marauders, 7 Marines, and the agent index encodes the *type* (the units are ordered
by type in the scheme). A Medivac that heals has a genuinely different value function from a frontline
Marine even when their local observations look numerically similar — the healer's value of a state hinges
on whether allies need healing, the Marine's on whether it can trade fire. Without the one-hot the shared
net has no way to route the Medivac's index to a different value branch than a Marine's; with it, the
constant `e_a` lets one shared network hold `n` value functions that agree on their shared structure and
diverge on their type-specific part, while still pooling all `n` agents' data into every gradient step.
So the one-hot is not centralization sneaking back in — it carries an agent's *identity*, never any peer's
observation or the global state — it is the minimal signal that keeps a shared homogeneous critic from
being forced to value distinct unit types identically.

The shapes have to line up with how the learner calls this, and here is the one place the local critic
differs structurally from the centralized ones: it reads `batch["obs"]`, which already lives on the
agent axis as `(B, T, n, obs_dim)`, so each agent's slot already carries its own `o^a` and there is *no
broadcast* — unlike the state critic, which has to expand a single `(B, T, state_dim)` vector across `n`
agents. The agent id is the `n×n` identity broadcast to `(B, T, n, n)`. Concatenate along the last axis
to `(B, T, n, obs_dim+n)`, run the MLP, and the head gives `(B, T, n, 1)` — the trailing singleton the
learner squeezes. `batch["state"]` is simply never read; the critic carries no peer or global
information by accident, because the only inputs are `o^a` and the constant id of `a`. Let me verify the
decentralizability that the whole principled story rests on, concretely: agent `a`'s output is
`fc3(relu(fc2(relu(fc1([o^a, e_a])))))`, a function of `o^a` and the *constant* one-hot `e_a` and nothing
else — no `s`, no `o^{b≠a}`. So at execution, from local information alone, the critic would compute the
identical value; it respects exactly the same information constraint as the actor. That is the check that
tells me I have not accidentally smuggled centralization in through a broadcast or a shared-state read.
Everything downstream is the fixed loop: the `q_nstep=5` target from a target copy of this critic, masked
MSE regression, the actor's clipped surrogate on each agent's own ratio, an entropy bonus, reward
standardization, Adam, grad-clip. The only multi-agent choice I am making lives in *what the critic
sees* — and I am betting that seeing *less*, but fitting it robustly, beats seeing *more* through an
architecture that cannot be fit stably.

Here is what I expect against the attention critic's numbers, stated to be falsified. First and most
basic: the bimodal seed collapse should *disappear*. The mat critic's three-way split on MMM
(`{0.0, 0.41, 0.0}`) and the dead seeds on 3s5z (`{0.0, 0.0, 0.34}`) were value-learning failures; a
plain local MLP cannot land in a degenerate attention basin, so I expect every seed to at least *train* —
no seed stuck at exactly 0.0 win rate on a map the others win, and no return of 4.9 where 18.8 is
reachable. Concretely I expect the per-seed variance to shrink sharply on every map, and none of the nine
runs to sit at exactly 0.0. Second, the mean win rates should rise on the maps where mat collapsed: MMM
(mat 0.135) and 3s5z (mat 0.115) should both improve substantially, because the floor of "two dead seeds"
is removed — even three mediocre-but-live seeds would clear those means easily. Third — and this is the
place I am genuinely unsure — on 2s3z mat already managed 0.542, and it got there with two *strong* live
seeds at 0.8125 each (plus one dead seed dragging the mean down). The local critic's *ceiling* could be
lower than a working centralized critic's, because the local critic is blind to global structure a
state-conditioned critic could exploit, and 2s3z is a symmetric map where coordinated positioning of the
two stalkers and three zealots is the whole game. So I would not be surprised if the local critic improves
the *robustness* everywhere — no dead seeds — but on 2s3z leaves a ceiling that a working centralized
critic would clear; I genuinely cannot predict whether its 2s3z *mean* will beat mat's 0.542 or fall
short of it, because removing the dead seed helps the floor while the missing global structure may cap the
top. There is a cleaner way to state the robustness prediction than "means go up," and it is in the *shape*
of the change rather than its level. What the local critic removes is the dead-seed floor, so the seed
that should move most is the *worst* seed on each map, not the best: mat's collapsed seeds (the exact-0.0
runs on MMM and 3s5z, the 4.9-return seed on 2s3z) should lift toward their live siblings, compressing the
per-seed spread, while the seeds that already trained under mat have less room to move. So the falsifiable
signature is variance-compression — the gap between a map's best and worst seed should shrink markedly —
more than a uniform ceiling-raise. And there is a secondary-metric fingerprint I can check independently
of win rate: under mat, `test_return_mean` carried the tell-tale outliers (return 4.9 sitting next to
18.8 on 2s3z). Under a robust local critic those outliers should vanish and the returns should track the
win rates *monotonically* within each map — a seed with a higher win rate should have a higher return, no
inversions, no 4-times-lower return on a live map. If instead I still see a return far below its map-mates
paired with a 0.0 win rate, the local critic did not fix the value learning and my whole diagnosis is
wrong. That honest uncertainty is itself the setup for the next rung: if the pattern is "robust but capped
on 2s3z," the diagnosis writes itself — bring the global state back, but in the lean, robust MLP form that
fits stably, and find out whether centralization helps *once the value learning is no longer the
bottleneck*. The distilled local-critic module — obs plus agent one-hot, plain MLP, no state — is in the
answer.
