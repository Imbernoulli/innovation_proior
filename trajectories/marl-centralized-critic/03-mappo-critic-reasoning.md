The local critic did the first thing I asked of it and confirmed the second thing I worried about, and
both are in the numbers if I read them carefully. The bimodal collapse is gone: not one of the nine runs
sits at exactly 0.0, the minima are MMM 0.281, 2s3z 0.125, 3s5z 0.469, and the 4.9-return disaster from
the attention critic has no analogue — every seed trains. I predicted a secondary-metric fingerprint at
the last rung, that the returns should now track the win rates *monotonically* within each map with no
inversions, and that is exactly what I see: on 2s3z `{0.125, 0.313, 0.906}` in win rate lines up with
`{12.96, 14.64, 19.65}` in return; on 3s5z `{0.469, 0.625, 0.813}` lines up with `{17.08, 18.35, 19.13}`;
on MMM `{0.281, 0.719, 0.906}` lines up with `{19.66, 22.01, 23.50}`. No 4-times-lower return next to a
live seed anywhere. That is the signature of a value regression that is actually fitting, and it settles
the step-1 diagnosis: the binding constraint was value-learning robustness, and a plain local MLP fixed
it.

The means moved exactly where I said the dead-seed floor would lift them. MMM went from mat's 0.135 to
0.635, a `+0.500` jump; 3s5z from 0.115 to 0.635, `+0.521`; both of these are the maps where mat had two
seeds pinned at zero, and removing that floor is worth half a win-rate point on its own. And 3s5z in
particular is now reasonably tight and live — `{0.813, 0.469, 0.625}`, spread 0.344 with no dead seed —
the kind of seed-to-seed consistency a robust value regression buys.

But now look at 2s3z, because this is the case I flagged as genuinely uncertain, and the arithmetic
resolves it in the direction I feared. The local critic's 2s3z mean is 0.448 — and mat's was 0.542. The
robust local critic actually landed *below* the fragile attention critic on this one map, a delta of
`0.448 − 0.542 = −0.094`. That looks paradoxical until I decompose it: mat's 0.542 came from two *strong*
live seeds at 0.8125 each plus one dead seed dragging the mean down, whereas the local critic's three
seeds are `{0.125, 0.906, 0.313}` — one strong seed and two weak-but-training ones. So the local critic
removed the dead seed (good) but could not push seeds 42 and 456 above 0.125 and 0.313 (the ceiling),
while mat's two working seeds had reached 0.8125. Crucially this is *not* a value-learning collapse: the
returns `{12.96, 19.65, 14.64}` track the win rates sensibly, there is no exact-0.0 and no 4.9-outlier, so
the two weak seeds are training — they just plateau low. That is the "robust but capped" pattern I said I
would watch for, and 2s3z is exactly where it should appear: a symmetric map where coordinated positioning
of the two stalkers and three zealots is the whole game — global, cross-agent structure that a critic
seeing only `o^a` is blind to. The local critic gives each agent a low-variance baseline on its own
information, but that information does not contain "where is the rest of my team relative to the enemy
line," and on 2s3z that is precisely the quantity the per-agent advantage needs. So the local critic's own
regression is sound; its *information* is short. The spread staying wide on 2s3z (`{0.125, 0.906, 0.313}`,
a 0.78 gap) while it compressed on 3s5z is the same story from the variance angle: where global structure
matters, a local critic's quality swings with how lucky a seed's local-only value estimates happen to be.
That is the cleanest possible signal that the next move is to bring the global state back — now that the
value learning is no longer the bottleneck.

One more cross-check before I move, because I want to be sure the 2s3z cap is about *information* and not
a cursed seed. If seed 42 were simply a bad-luck initialization, I would expect it to be the weak seed
everywhere. It is not. Seed 42 is the weakest on MMM (0.281) and on 2s3z (0.125) for the local critic —
but on 3s5z seed 42 is the local critic's *best* seed at 0.8125, while there the weak one is seed 123
(0.469). And under the attention critic the dead seeds were a different set again (42 and 456 on MMM, 42
and 123 on 3s5z). So there is no globally-cursed seed; the weakness moves with the map and the critic, not
with the random draw. That is what I would expect if the binding factor is *what the critic can see on
that map* rather than a seed lottery — which is precisely the case for treating 2s3z's cap as an
information deficit I can fix by changing the critic's input.

So I return to centralization, but I have to be careful, because the attention critic *was* centralized
(its tokens carried the state) and it collapsed. The lesson was not "centralization is bad"; it was
"cross-agent *mixing* in the critic, under this loop's value-learning conditions, is fragile." Let me
lay the options out so the choice is deliberate. I could re-add attention but try to stabilize it — but
the loop offers me no stabilizer (no return normalization, no warmup, a cold transformer under a fixed
`lr=3e-4`), which is exactly why it collapsed, and I have nothing new to change that, so reintroducing
the mixing is reintroducing the failure. I could reach again for an action-value critic (COMA's `Q(s,A)`
with a counterfactual, MADDPG's per-agent `Q_i`), but the contract already ruled those out at the first
rung: `output_type = "v"` and the learner forms `R^{a,(n)} − V(x^a)`, a state-value baseline, so a `Q` in
that slot is not a valid baseline and the counterfactual marginalization lives in the frozen learner. I
could jump straight to the richer agent-specific form that also feeds each agent its own observation — but
the config this task fixes is explicit about that, and I will come to it in a moment. The clean way to
bring back global information *without* reintroducing the mixing that blew up is to feed the global state
into the same plain, robust MLP that just worked — a shared `V_φ([s, e_a])` over the global state plus an
agent one-hot, no attention, no cross-agent interaction layer, nothing that can land in a degenerate
basin. This is the standard centralized state-value critic, and it is the smallest possible change from
the local critic: swap `o^a` for `s` in the input, keep the identical 128–128–1 MLP and the identical
agent one-hot. That single-variable change is exactly the isolating comparison I set up last rung — hold
the robust architecture fixed, change only the *information* from local to global — and it is now safe to
run because the value learning is no longer the thing at risk.

Let me justify the state-value choice from the ground up rather than defaulting to it, and let me write
the identity that forces it. Why a *state* value and not the action-value critics of the centralized prior
art? Because the GAE baseline I am subtracting only ever needs a *state* value. The policy-gradient
estimator is `∇J = E[∇log π · A]` and stays unbiased for any baseline that does not depend on the action
given the state; GAE builds its advantage from the TD residual `δ_t = r_t + γV(s_{t+1}) − V(s_t)`, which
reads `V` at `s_t` and `s_{t+1}` and *never* evaluates a `Q`. Conditioning the critic on the joint action
would make the input scale with `n`, couple the critic to every agent's current policy (so the target
drifts as the policy moves), and — for COMA — force an action enumeration to build the counterfactual, all
machinery I do not need for variance reduction and cannot host in the fixed learner anyway. A centralized
`V(s)` buys the variance reduction and sheds every one of those costs. So the strongest-baseline critic is
single-agent PPO with one thing changed from the local critic: the value function reads the global state.

Now the input itself, and here I have to be honest about what *this* harness exposes versus the richer
forms the centralized-V idea can take, because the difference is load-bearing. The fully developed
agent-specific construction would feed the critic the global state *and* each agent's own observation
`[s, o^a]`, and there is a real argument that this is better — the bare global state is agent-agnostic and
drops the local features a per-agent value depends on. But the config this task fixes is explicit:
`obs_individual_obs=False` and `obs_last_action=False`, with `obs_agent_id=True`. So the centralized critic
the *task* defines is the lean **environment-provided** form: global state plus an agent one-hot, and
*only* that. I am not going to import the agent-specific `[s, o^a]` machinery into this step, because the
baseline I am deriving is the one the harness actually runs — `V_φ([s, e_a])`. I note the gap explicitly,
because it is exactly where a stronger critic would go next: this critic sees the global picture but,
being agent-agnostic in its state input, it leans entirely on the one-hot to distinguish agents, and the
one-hot carries an agent's *identity* but none of its current local detail (what it can fire, how exposed
it is). If that omission caps this critic the way blindness to global structure capped the local one, the
agent-specific input is the obvious lever — but that is a later move; here I derive the EP critic the
config specifies.

Why does the one-hot earn its place in the lean form, and why does it matter *more* here than it did for
the local critic? Because I am sharing one critic across all agents (the same robustness-and-data argument
as before: homogeneous agents, `n`-fold data per step), and now the shared network is handed the *same*
`s` for every agent. That is a sharper situation than the local critic, and it is worth making the
degenerate check concrete. Drop the one-hot: the input to the shared net is `s`, identical for all `n`
agents at a given `(B, T)`, so the network can only output *one* value, the same for every agent. With the
local critic that would not have mattered much — the agents differed through their own `o^a` — but here
`s` is shared, so the one-hot is now the *only* thing distinguishing agents. On a heterogeneous map like
MMM (a Medivac that heals, Marauders, Marines) the true per-agent value is genuinely not agent-agnostic —
the healer's value of a state differs from a frontline marine's — so a critic that could only emit one
value per state would be systematically wrong for at least one type. The one-hot is the minimal thing that
lets the shared network route different agents (by index, which encodes type) to different value functions
while still pooling all the data. So `[s, e_a]` is the leanest centralized critic that is both
data-efficient (shared) and able to distinguish agents (the id). It is exactly the local critic's
architecture with the information swapped from local to global, plus the same id it already carried — but
I notice, and file away, that here the id is doing *all* the per-agent work, and an id is a thin, constant
signal.

It is worth naming precisely what information this swap gains and what it loses, because the two are not
symmetric and the asymmetry is the whole reason to expect a map-dependent result. The global state `s` is
*not* egocentric — it is the same vector for every agent, a third-person description of the whole board.
So it answers "what is the global situation" (which the local `o^a` could not, since `o^a` only spans one
agent's sight range), and that is the gain. But `s` does *not* answer "what is agent `a` locally facing
right now," and the one-hot only answers "which agent is `a`" as a constant index, never "what does `a`
see this timestep." The local critic had that egocentric detail for free — `o^a` is exactly agent `a`'s
current local situation — and the state swap throws it away. So the EP critic trades per-agent egocentric
detail for whole-team global structure. On a map where the missing quantity was global structure (2s3z's
coordinated positioning), that is a clear win; on a map where each agent's own local situation was
carrying the per-agent value (the larger teams, where a single agent's slice of an 8- or 10-unit fight is
its own micro-battle), the trade could be closer to neutral, because I gave up the very egocentric detail
that was doing the work. That asymmetry is the mechanism behind the ordering I will predict below.

The shapes mirror the local critic with one structural difference, and it is the inverse of the IPPO
case. The global state is one vector per `(B, T)`, shape `(B, T, state_dim)`, so to put it on the agent
axis I must *broadcast* it across the `n` agents — `state.unsqueeze(2).expand(-1, -1, n, -1)` to
`(B, T, n, state_dim)`. Take 3s5z with `n = 8` to trace it concretely: `state` comes in `(B, T, state_dim)`,
expands to `(B, T, 8, state_dim)`; the one-hot identity `eye(8)` broadcasts to `(B, T, 8, 8)`; concatenate
to `(B, T, 8, state_dim + 8)`, run the 128–128–1 ReLU MLP, and the head gives `(B, T, 8, 1)` — the trailing
singleton the learner squeezes. Contrast the broadcast with the local critic, which needed *none* because
`obs` already lived on the agent axis as `(B, T, 8, obs_dim)`; here the shared state must be replicated
onto every agent slot precisely because it is shared. `batch["obs"]` is now never read; the critic
conditions purely on the shared global state and the constant id. Everything downstream is the same fixed
loop the previous two rungs used: the `q_nstep=5` target from a target copy of this critic, masked MSE
regression, the actor's clipped surrogate on each agent's own ratio, the entropy bonus, reward
standardization, Adam, grad-clip.

I want to confirm this really is the single-variable change I claimed and not a quiet redesign. Line the
two critics up parameter by parameter: the local critic was `fc1: (obs_dim + n) → 128`, `fc2: 128 → 128`,
`fc3: 128 → 1`; the state critic is `fc1: (state_dim + n) → 128`, `fc2: 128 → 128`, `fc3: 128 → 1`. The
*only* thing that moves is the input width of `fc1`, from `obs_dim + n` to `state_dim + n` — the global
state in SMAC is a third-person description of every unit, so `state_dim` is generally the larger of the
two, and `fc1` grows accordingly — but `fc2` and `fc3` are byte-for-byte identical, so the network's
processing capacity past the first layer is unchanged. Same depth, same width, same activation, same
sharing, same one-hot, same downstream loop; I have changed what the critic *reads* and nothing about how
it *computes*. That is what makes the comparison against the local critic a clean read on the value of
centralization rather than a confound of architecture and information moving together — the mistake the
jump from mat to IPPO could not avoid (it changed both mixing and information at once), and the one I can
avoid here.

Let me also knock down the worry that this just re-creates the attention critic's failure with a state
input, because superficially both are "centralized critics" and one of them collapsed. The attention
critic collapsed because of *cross-agent mixing* fit to a moving bootstrapped target — the softmax over
agents could find a degenerate pattern, and per-agent values could smear toward a shared term. The
state-V critic has *no* mixing: each agent's value is a deterministic MLP of `[s, e_a]`, the agents do not
attend to each other, the only per-agent variation comes from the fixed one-hot, and there is no softmax
over agents anywhere in the graph. There is no degenerate basin for attention to fall into because there
is no attention. Concretely, its processing path is the same `~16.5k`-parameter hidden layer the robust
local critic used, not the `~198k`-parameter interaction core of the attention critic. So this critic
should inherit the local critic's robustness (same MLP, same fit) while adding the global information the
local critic lacked — the best of both prior rungs.

Here is what I expect against the two baselines I have measured, stated to be falsified. Against the
attention critic: the seed collapse stays gone (no mixing to break), so no seed at exactly 0.0 anywhere —
this should be as robust as the local critic, with returns tracking win rates monotonically as they did
for IPPO. Against the local critic: the *ceiling* should lift on exactly the maps where global structure
matters and the local critic was capped. I expect 2s3z — the local critic's weakest map at 0.448, where
it even fell below mat, and where coordinated positioning is the whole game — to improve the most, because
that is where seeing the global state should most help the per-agent advantage; I would be surprised if
2s3z did not clear 0.7, which would be a `+0.25` swing or more. There is a variance prediction paired with
the level one: 2s3z's local-critic spread was the widest anywhere (`{0.125, 0.906, 0.313}`, a 0.78 gap),
and if the cause was that a local-only baseline swings with how lucky each seed's local value estimates
happen to be, then handing every seed the *same* global coordination signal should compress that spread as
well as lift the mean — I expect 2s3z's three seeds to cluster far tighter than 0.78. I also expect MMM to
rise from 0.635,
because its heterogeneity (a healer plus two unit types) is precisely where a critic that sees the *whole*
team's state, rather than each agent's local slice, should produce better-aligned advantages — and where
the one-hot lets the shared net give the Medivac a different value than a Marine. 3s5z is the interesting
case: the local critic was already at 0.635 and reasonably tight, and 3s5z is the larger team (8 agents)
where the global state is higher-dimensional and the agent-agnostic EP form drops more local detail — so I
expect an improvement but a more modest one, and I would not be shocked if 3s5z is where the EP critic's
lack of agent-specific local features shows up as the smallest gain, or even where the state critic barely
clears the local critic that at least carried each agent's own observation. If the pattern is
"centralization clearly helps on 2s3z and MMM, helps least on the biggest team 3s5z," that ordering is the
signature that the global state is worth bringing back *and* that the remaining gap is the agent-agnostic
state input — the agent-specific `[s, o^a]` lever I deliberately left on the table. The distilled lean
central-V module — global state plus agent one-hot, plain MLP, no attention, no obs — is in the answer.
