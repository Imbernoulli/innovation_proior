The attention critic told me exactly what I feared, and it told me in the per-seed split — the bimodal
collapse I flagged, not the "helps most on hard maps" story I hoped for. Averaged across the three maps
it lands at 0.264 win rate, but that average hides the shape of the failure, and the shape is the whole
story. On MMM the three seeds are `{0.0, 0.4062, 0.0}` — two sit at *exactly* 0.0, one reaches 0.41; a
mean of 0.135 is "one seed that trained, two that didn't." On 3s5z it is worse: seeds 42 and 123 are both
exactly 0.0, only 456 limps to 0.34. Across the nine runs, five are pinned at exactly zero. A merely weak
policy lands at 0.05 or 0.1; a value at *exactly* 0.0 across many seeds is the fingerprint of a degenerate
basin, which is precisely the diagnostic I said would separate value-learning collapse from a capacity
ceiling.

The single cleanest tell is seed 42 on 2s3z. The three seeds there are `{0.0, 0.8125, 0.8125}` in win
rate with returns `{4.9203, 18.8608, 18.7663}`. The two live seeds' returns differ by only 0.094 — when
the value learning works, the seeds converge to essentially the same place. The dead seed is win 0.0 with
return 4.92, roughly `0.26` of what the live seeds reach, nearly four times lower. A return of 4.9 on a
map where the same critic elsewhere reaches 18.8 is not "the team played and narrowly lost" — it is the
policy never finding the fight, because the per-agent advantage it was fed came from a critic that landed
in a degenerate basin. That is the fingerprint I named: a zero win rate paired with a return far below a
working seed's on the same map means the policy never engaged, which only a broken baseline can cause.

There is a second, subtler confirmation that the critic and not the task is the culprit: the collapse
*manifests differently on different maps* while always tracing back to the value function. On 3s5z the two
dead seeds (42, 123) have *high* returns (13.61, 15.50) yet win rate 0.0 — the policy engages and racks up
shaped reward but converts none of it into a clean win, a different texture from 2s3z seed 42, which
barely engages at all. If the *task* were simply too hard I would expect one consistent failure texture;
instead I see "never engages" on one map and "engages but cannot convert" on another, both under the same
critic. The variance in *how* it dies is itself evidence the instability lives in the value learning, not
the environment. And the pattern across maps is the exact inverse of the "structure helps on hard maps"
story: the hard, larger-team maps (MMM 0.135, 3s5z 0.115) collapse hardest, and the medium map (2s3z
0.542) survives best — the "2s3z survives as the easiest value regression" call, now confirmed.

So the mechanism is settled: a high-capacity attention critic, fit to a bootstrapped target via a target
copy with unnormalized returns and masked MSE, collapsing onto a broken attention pattern that the masked
MSE tolerates but that poisons the advantages. The mixing across agents — the one thing the encoder-only
critic added over a flat MLP — turned out to be a liability, because nothing in the fixed loop stabilizes
it (no return normalization, no warmup, a cold transformer under `lr=3e-4`). The lesson points *down* the
complexity ladder: architectural interaction modeling in the critic is not free, and the binding
constraint is value-learning robustness, not representational power. Before I reconsider *what information*
the critic should see, I should establish a critic whose value regression simply *works* — a plain MLP,
no attention — and find out what a stable critic alone buys.

There is more than one way to step down, and the choice is worth making deliberately. The single-variable
ablation of the mat failure would keep the global state and just drop the attention — a plain MLP over
`[s, e_a]`, changing only mix→no-mix. But I have a contrarian clue that makes the *opposite* extreme more
informative: strip both the mixing and the global state and condition the critic on exactly what the actor
conditions on, agent `a`'s own observation `o^a`, `V_φ(o^a)`. The folklore says more centralized
information is strictly better — the critic is thrown away at execution, so feed it everything. Yet
independent, purely local PPO is reported to be *strong* on hard SMAC maps, sometimes beating centralized
PPO that uses the global state. That is backwards if centralization were free, and it means the question
is not only "does removing the mixing restore robustness" but "does the critic even *need* the global
state." Testing the local extreme first answers the bigger question and sets up the state-critic — bring
the state back into the same robust MLP — as the clean isolating comparison afterward. So I take the local
critic now and hold the state critic in reserve.

My reflex says this should be obviously worse — it is "independent learning," the thing that is supposed to
fail. The classic objection is non-stationarity: if agent `a`'s critic sees only `o^a` and the others are
simultaneously learning, then from `a`'s view the dynamics drift, the targets go stale, and the
convergence story evaporates (Tan 1993). A second is that an independent learner cannot always separate
environment stochasticity from a peer's exploration, which provably blocks optimal play in some
coordination games (Claus & Boutilier 1998). Those are real. But let me ask *whose job* stability is. The
critic's job is to be a low-variance, low-bias baseline; stability of the *policy update* is a different
job, and PPO attacks it with the clipped ratio objective. The non-stationarity disaster is specifically
that one agent, reacting to a co-learner's shift, takes a catastrophic policy step that shifts the
dynamics for everyone and spirals. PPO's clip does not prove the realized ratio can never leave the band,
but on the sampled actions it removes the objective's incentive to push a ratio past `[1−ε, 1+ε]`
(`ε = 0.2`) once it has crossed the useful side — exactly the restraint I need against peer-induced
surprises, and the opposite of what the attention critic lacked. There, nothing restrained the *value*
function; here the actor's clip restrains the *policy* and the value function is so simple it cannot
misbehave. If clipping keeps every agent's effective step modest, then from any one agent's view the
others drift slowly enough that a local critic can track them. The instability that condemns IAC/IQL may
be an artifact of unrestrained updates, not of the local critic per se.

I want one more argument before I trust a local critic as *principled* rather than merely robust-by-being-
dumb. Telescope the joint return over three agents into single-agent changes: hold two policies fixed and
improve the third, then advance to the next, so the total improvement is a sum of coordinate terms each
changing *exactly one* policy. If every coordinate term is non-negative, the joint return rises. Look at
one term — improve only `π³`, the others held. Holding the others fixed, the world agent 3 faces is a
single-agent decision process — a POMDP in which the fixed policies are absorbed into the dynamics — and
agent 3's policy conditions only on `o³`. So "improve `π³` holding the rest fixed" is just "improve a
single-agent policy on the induced POMDP," and a restrained single-agent step can make that term
non-negative. PPO is a first-order clipped approximation, not the exact trust-region step, and
simultaneous updates without recollection are a further approximation — so this is not a theorem for the
practical update, but the identity tells me the *structure* I want: each agent takes a restrained
single-agent improvement step against the POMDP the others induce. And the natural baseline for a POMDP
policy that conditions on `o³` is a value function on that *same* information set, `V_φ(o³)`. The local
critic is not a compromise forced by execution; it is the baseline aligned with the coordinate step's
information set. The local critic and PPO's clip are two halves of one idea — restrained single-agent
improvement on the right per-agent information — the principle the attention critic had no claim to, since
it mixed information across agents the actor's coordinate step does not condition on.

Now the module. Do I learn `n` separate critics or one shared `V_φ`? The agents are homogeneous within a
type and share observation/action spaces, so a separate net per agent wastes data — each learns from one
agent's stream. One shared critic lets every agent's experience update the same parameters, giving `n`-
fold more data per gradient step (ten agents on MMM, eight on 3s5z, five on 2s3z), faster and more stable
value learning — exactly what I want after watching the attention critic struggle to fit. Sharing removes
only *index-based* specialization, and I hand that back cheaply with a one-hot agent id, `n` extra
dimensions. So: one shared critic, input `[o^a ⊕ one-hot(a)]` of width `obs_dim + n`, architecture the
plain PPO MLP I trust — `fc1: (obs_dim+n) → 128`, ReLU; `fc2: 128 → 128`, ReLU; `fc3: 128 → 1`. Its
processing path is `fc2: 128·128 = 16,384` parameters, roughly a *twelfth* of the attention critic's
~198k interaction core, with nothing that can collapse: the deliberate inverse of the step-1 critic.

The one-hot earns its `n` dimensions only in the heterogeneous case, and it is worth being clear that it
does *nothing* on truly-symmetric states — two agents with identical observations getting identical values
is correct for a homogeneous critic. Where it earns its place is a map like MMM: 1 Medivac, 2 Marauders, 7
Marines, with the agent index encoding *type* (units are ordered by type in the scheme). A Medivac that
heals has a genuinely different value function from a frontline Marine even when their observations look
numerically similar, and without the one-hot the shared net has no way to route the Medivac's index to a
different value branch. The one-hot is not centralization sneaking back in — it carries identity, never a
peer's observation or the global state — it is the minimal signal keeping a shared homogeneous critic from
valuing distinct unit types identically.

The shapes are where the local critic differs structurally from the centralized ones: it reads
`batch["obs"]`, which already lives on the agent axis as `(B, T, n, obs_dim)`, so each agent's slot
carries its own `o^a` and there is *no broadcast* — unlike a state critic, which must expand a single
`(B, T, state_dim)` vector across `n` agents. The agent id is the `n×n` identity broadcast to
`(B, T, n, n)`; concatenate to `(B, T, n, obs_dim+n)`, run the MLP, and the head gives `(B, T, n, 1)`.
`batch["state"]` is simply never read, so the critic carries no peer or global information by accident.
That is the check the whole principled story rests on: agent `a`'s output is a function of `o^a` and the
constant `e_a` and nothing else, so at execution it would compute the identical value from local
information alone — it respects exactly the same information constraint as the actor. Everything downstream
is the fixed loop: the `q_nstep=5` target from a target copy, masked MSE regression, the clipped surrogate
on each agent's own ratio, entropy bonus, reward standardization, Adam, grad-clip. The only multi-agent
choice I am making lives in *what the critic sees* — I am betting that seeing *less*, fit robustly, beats
seeing *more* through an architecture that cannot be fit stably.

Here is what I expect against the attention critic's numbers, to be falsified. First and most basic: the
bimodal seed collapse should *disappear*. A plain local MLP cannot land in a degenerate attention basin,
so I expect every seed to at least *train* — none stuck at exactly 0.0 on a map the others win, and no
return of 4.9 where 18.8 is reachable. Second, the means should rise on the maps where mat collapsed — MMM
(0.135) and 3s5z (0.115) — because the floor of "two dead seeds" is removed; even three mediocre-but-live
seeds clear those means. Third — and here I am genuinely unsure — mat managed 0.542 on 2s3z, and it got
there with two *strong* live seeds at 0.8125 plus one dead seed dragging the mean down. The local critic's
*ceiling* could be lower than a working centralized critic's, because it is blind to global structure a
state-conditioned critic could exploit, and 2s3z is a symmetric map where coordinated positioning of the
two stalkers and three zealots is the whole game. So I would not be surprised if the local critic improves
robustness everywhere but leaves a 2s3z ceiling a working centralized critic would clear; I genuinely
cannot predict whether its 2s3z mean beats or falls short of mat's 0.542, because removing the dead seed
helps the floor while the missing global structure may cap the top. The cleaner way to state the
robustness prediction is in the *shape* of the change: what the local critic removes is the dead-seed
floor, so the seed that should move most is the *worst* seed on each map, compressing the per-seed spread,
while the seeds that already trained have less room to move. And a secondary-metric check independent of
win rate: under a robust critic the returns should track win rates *monotonically* within each map, with
no seed showing a 4-times-lower return next to a live one. If I still see such an inversion paired with a
0.0 win rate, the local critic did not fix the value learning and my diagnosis is wrong. That honest
uncertainty sets up the next move: if the pattern is "robust but capped on 2s3z," bring the global state
back in the lean robust MLP and find out whether centralization helps once value learning is no longer the
bottleneck. The distilled local-critic module — obs plus agent one-hot, plain MLP, no state — is in the
answer.
