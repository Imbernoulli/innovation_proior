The local critic did the first thing I asked of it and confirmed the second thing I worried about. The
bimodal collapse is gone: not one of the nine runs sits at exactly 0.0, the minima are MMM 0.281, 2s3z
0.125, 3s5z 0.469, and the 4.9-return disaster has no analogue — every seed trains. I predicted a
secondary-metric fingerprint last rung, that returns should track win rates *monotonically* within each
map with no inversions, and that is what I see: on 2s3z `{0.125, 0.313, 0.906}` lines up with
`{12.96, 14.64, 19.65}`, on 3s5z `{0.469, 0.625, 0.813}` with `{17.08, 18.35, 19.13}`, on MMM
`{0.281, 0.719, 0.906}` with `{19.66, 22.01, 23.50}`. No 4-times-lower return next to a live seed
anywhere. That is a value regression that is actually fitting, and it settles the step-1 diagnosis: the
binding constraint was value-learning robustness, and a plain local MLP fixed it. The means moved where I
said the dead-seed floor would lift them — MMM from mat's 0.135 to 0.635, 3s5z from 0.115 to 0.635, both
maps where mat had two seeds pinned at zero.

But 2s3z is the case I flagged as genuinely uncertain, and the arithmetic resolves it in the direction I
feared. The local critic's 2s3z mean is 0.448 — mat's was 0.542, so the robust critic landed *below* the
fragile one on this one map, a delta of `−0.094`. That decomposes cleanly: mat's 0.542 came from two
*strong* live seeds at 0.8125 plus one dead seed dragging the mean down, whereas the local critic's three
seeds are `{0.125, 0.906, 0.313}` — one strong and two weak-but-training. So it removed the dead seed
(good) but could not push seeds 42 and 456 above 0.125 and 0.313, while mat's two working seeds had
reached 0.8125. This is *not* a value-learning collapse — the returns `{12.96, 19.65, 14.64}` track the
win rates, no exact-0.0, no 4.9-outlier — the two weak seeds train, they just plateau low. That is the
"robust but capped" pattern, and 2s3z is exactly where it should appear: a symmetric map where coordinated
positioning of the two stalkers and three zealots is the whole game, global cross-agent structure a critic
seeing only `o^a` is blind to. The local critic's regression is sound; its *information* is short — `o^a`
does not contain "where is the rest of my team relative to the enemy line," and on 2s3z that is precisely
what the per-agent advantage needs. The spread staying wide on 2s3z (a 0.78 gap) while it compressed on
3s5z is the same story from the variance angle. That is the cleanest signal to bring the global state back,
now that value learning is no longer the bottleneck.

One cross-check before I move, to be sure the 2s3z cap is about *information* and not a cursed seed. If
seed 42 were simply a bad-luck initialization, it would be the weak seed everywhere. It is not: seed 42 is
weakest on MMM (0.281) and 2s3z (0.125) but the local critic's *best* seed on 3s5z at 0.8125, where the
weak one is seed 123 (0.469). And under the attention critic the dead seeds were a different set again. So
there is no globally-cursed seed; the weakness moves with the map and the critic, not the random draw —
exactly what I would expect if the binding factor is *what the critic can see on that map*, which is what
makes 2s3z's cap an information deficit I can fix by changing the critic's input.

So I return to centralization, but carefully, because the attention critic *was* centralized (its tokens
carried the state) and it collapsed. The lesson was not "centralization is bad" but "cross-agent *mixing*
in the critic, under this loop's value-learning conditions, is fragile." Re-adding attention with a
stabilizer is not an option — the loop offers none (no return normalization, no warmup, a cold transformer
under a fixed `lr`), which is exactly why it collapsed. The action-value critics (COMA's `Q(s,A)`,
MADDPG's `Q_i`) the contract already ruled out at the first rung: `output_type="v"` and the learner forms
`R^{a,(n)} − V(x^a)`, so a `Q` in that slot is not a valid baseline. The clean way to bring back global
information *without* the mixing is to feed the global state into the same plain MLP that just worked — a
shared `V_φ([s, e_a])` over the global state plus an agent one-hot. This is the smallest possible change
from the local critic: swap `o^a` for `s`, keep the identical 128–128–1 MLP and one-hot. That
single-variable change is exactly the isolating comparison I set up last rung — hold the robust
architecture fixed, change only the information from local to global — and it is now safe because value
learning is no longer at risk.

Let me justify the state-value choice rather than default to it. Why a *state* value and not the
action-value critics of the centralized prior art? Because the GAE baseline only ever needs a *state*
value. The estimator `∇J = E[∇log π · A]` stays unbiased for any baseline that does not depend on the
action given the state, and GAE builds its advantage from the TD residual `δ_t = r_t + γV(s_{t+1}) −
V(s_t)`, which reads `V` and never evaluates a `Q`. Conditioning on the joint action would scale the input
with `n`, couple the critic to every agent's current policy so the target drifts as the policy moves, and
— for COMA — force an action enumeration to build the counterfactual: all machinery I do not need for
variance reduction and cannot host in the fixed learner anyway. A centralized `V(s)` buys the variance
reduction and sheds every one of those costs.

Now the input, and here I have to be honest about what *this* harness exposes versus the richer forms the
central-V idea can take. The fully developed agent-specific construction would feed the critic the global
state *and* each agent's own observation `[s, o^a]`, and there is a real argument that this is better — the
bare global state is agent-agnostic and drops the local features a per-agent value depends on. But the
config this task fixes is explicit: `obs_individual_obs=False`, `obs_last_action=False`, with
`obs_agent_id=True`. So the centralized critic the *task* defines is the lean environment-provided form:
global state plus an agent one-hot, and *only* that. I am not importing the agent-specific `[s, o^a]`
machinery here, because the baseline I am deriving is the one that actually runs — `V_φ([s, e_a])`. I note
the gap explicitly, because it is exactly where a stronger critic would go next: this critic sees the
global picture but, being agent-agnostic in its state input, leans entirely on the one-hot to distinguish
agents, and the one-hot carries identity but none of an agent's current local detail (what it can fire,
how exposed it is). If that omission caps this critic the way blindness to global structure capped the
local one, the agent-specific input is the obvious lever — but that is a later move.

The one-hot matters *more* here than it did for the local critic, and it is worth the degenerate check.
Now the shared network is handed the *same* `s` for every agent, so drop the one-hot and the input is
identical across all `n` agents at a given `(B, T)` — the network can only output one value per state. On
a heterogeneous map like MMM (Medivac, Marauders, Marines) the true per-agent value is genuinely not
agent-agnostic — the healer's value of a state differs from a frontline Marine's — so a critic emitting
one value per state would be systematically wrong for at least one type. Here `s` is shared, so the
one-hot is now the *only* thing distinguishing agents; it lets the shared net route different agents (by
index, which encodes type) to different value functions while pooling all `n` agents' data. So `[s, e_a]`
is the leanest centralized critic that is both data-efficient and able to distinguish agents — but I note,
and file away, that here the id is doing *all* the per-agent work, and an id is a thin, constant signal.

It is worth naming what the swap gains and loses, because the asymmetry is the whole reason to expect a
map-dependent result. The global state `s` is *not* egocentric — it is the same third-person description
of the board for every agent — so it answers "what is the global situation" (which `o^a` could not, since
`o^a` spans only one agent's sight) but *not* "what is agent `a` locally facing right now," and the
one-hot only answers "which agent is `a`" as a constant. The local critic had that egocentric detail for
free in `o^a`, and the state swap throws it away. So the EP critic trades per-agent egocentric detail for
whole-team global structure. On a map where the missing quantity was global structure (2s3z's coordinated
positioning) that is a clear win; on a map where each agent's own local situation was carrying the
per-agent value — the larger teams, where a single agent's slice of an 8- or 10-unit fight is its own
micro-battle — the trade could be closer to neutral. That asymmetry is the mechanism behind the ordering I
predict below.

The shapes mirror the local critic with one inverse difference. The global state is one vector per
`(B, T)`, so to put it on the agent axis I must *broadcast* it — `state.unsqueeze(2).expand(-1, -1, n,
-1)` to `(B, T, n, state_dim)` — whereas the local critic needed none because `obs` already lived on the
agent axis. The one-hot is the `n×n` identity broadcast to `(B, T, n, n)`; concatenate to
`(B, T, n, state_dim + n)`, run the 128–128–1 ReLU MLP, head gives `(B, T, n, 1)`. `batch["obs"]` is now
never read. And this really is the single-variable change: line the critics up and the *only* thing that
moves is `fc1`'s input width, from `obs_dim + n` to `state_dim + n`; `fc2` and `fc3` are byte-for-byte
identical, same depth, width, activation, sharing, one-hot, and downstream loop. I have changed what the
critic *reads* and nothing about how it *computes* — which is what makes the comparison against the local
critic a clean read on the value of centralization, rather than the confound of architecture and
information moving together that the mat→IPPO jump could not avoid.

One more worry to knock down: this is not the attention critic's failure with a state input. That critic
collapsed because of *cross-agent mixing* fit to a moving bootstrapped target — the softmax over agents
could find a degenerate pattern and smear per-agent values toward a shared term. The state-V critic has
*no* mixing: each agent's value is a deterministic MLP of `[s, e_a]`, no softmax over agents anywhere in
the graph, the only per-agent variation coming from the fixed one-hot. Its processing path is the same
~16.5k-parameter hidden layer the robust local critic used, not the ~198k interaction core. So it should
inherit the local critic's robustness while adding the global information the local critic lacked.

Here is what I expect against the two measured baselines, to be falsified. Against the attention critic:
the seed collapse stays gone (no mixing to break), no seed at exactly 0.0, returns tracking win rates
monotonically as they did for IPPO. Against the local critic: the *ceiling* should lift on exactly the
maps where global structure matters. I expect 2s3z — the local critic's weakest at 0.448, where it even
fell below mat, and where coordinated positioning is the whole game — to improve the most, and I would be
surprised if it did not clear 0.7. Paired with that level prediction is a variance one: 2s3z's spread was
the widest anywhere (a 0.78 gap), and if the cause was a local-only baseline swinging with how lucky each
seed's local estimates happen to be, then handing every seed the *same* global coordination signal should
compress that spread as well as lift the mean. I also expect MMM to rise, its heterogeneity being exactly
where a critic seeing the *whole* team's state should produce better-aligned advantages and where the
one-hot lets the shared net give the Medivac a different value than a Marine. 3s5z is the interesting case:
the local critic was already at 0.635 and reasonably tight, and 3s5z is the larger team where the state is
higher-dimensional and the agent-agnostic EP form drops more local detail — so I expect an improvement but
a more modest one, possibly the smallest gain. If the pattern is "centralization helps most on 2s3z and
MMM, least on the biggest team 3s5z," that ordering is the signature that global state is worth bringing
back *and* that the remaining gap is the agent-agnostic state input — the agent-specific `[s, o^a]` lever
I deliberately left on the table. The distilled lean central-V module — global state plus agent one-hot,
plain MLP, no attention, no obs — is in the answer.
