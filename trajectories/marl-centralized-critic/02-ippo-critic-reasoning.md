The attention critic told me exactly what I feared, and it told me in the per-seed split. The signature
is the bimodal collapse I flagged, not the "helps most on hard maps" story I hoped for. On MMM two of
three seeds sit at *exactly* 0.0 win rate (seed 42 and 456) while seed 123 reaches 0.41 — mean 0.135 is
really "one seed that trained, two that didn't." On 3s5z it is worse: seeds 42 and 123 are both 0.0,
only 456 limps to 0.34, mean 0.115. And the tell that this is value-learning fragility and not a
capacity ceiling is seed 42 on 2s3z: win rate 0.0 with a *return* of 4.92, when the other two seeds on
the same map hit 0.81 win rate with returns near 18.8. A return of 4.9 on a map where the same critic
elsewhere reaches 18.8 is not "the team played and narrowly lost" — it is the policy never finding the
fight, because the per-agent advantage it was fed came from a critic that landed in a degenerate basin.
That is precisely the failure mode I named: a high-capacity attention critic, fit to a bootstrapped
target via a target copy with unnormalized returns and masked MSE, collapsing onto an over-smoothed or
otherwise broken attention pattern that the masked MSE tolerates but that poisons the advantages. The
mixing across agents — the one thing the encoder-only critic added over a flat MLP — turned out to be a
liability under these value-learning conditions, because nothing in the fixed loop stabilizes it (no
return normalization, no warmup, a cold transformer under `lr=3e-4`). So the lesson is sharp and it
points *down* the complexity ladder: architectural interaction modeling in the critic is not free, and
the binding constraint here is value-learning robustness, not representational power. Before I reconsider
*what information* the critic should see, I should establish a critic whose value regression simply
*works* — a plain MLP, no attention, the most robust thing I can fit — and find out what a stable
critic alone buys.

So let me strip the critic to a plain MLP and, while I am at it, question the input the default reaches
for. The scaffold default and the attention token both lean on the global state `s`. The folklore says
more centralized information is strictly better — the critic is thrown away at execution, so feed it
everything. But I have a contrarian clue I cannot ignore: independent, purely local PPO is reported to
be *strong* on hard SMAC maps, sometimes beating centralized PPO that uses the global state. That is
backwards if centralization were free, and it makes me want to try the opposite extreme deliberately:
condition the critic on *exactly what the actor conditions on* — agent `a`'s own observation `o^a`,
nothing else, `V_φ(o^a)`. No global state, no peer information. Each agent gets its own local value
function as its baseline, from the same local information its policy uses. After the attention critic's
collapse, the appeal is obvious: a per-agent MLP over `o^a` is the lowest-variance, most robust value
regression I can write, and it removes the cross-agent mixing that just blew up.

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
removes the objective's incentive to push a probability ratio past `[1−ε, 1+ε]` once it has crossed the
useful side. That is exactly the restraint I need against peer-induced surprises — and notice it is the
*opposite* of what the attention critic lacked: there, nothing restrained the *value* function; here,
the actor's clip restrains the *policy*, and the value function is so simple it cannot misbehave. If
clipping keeps every agent's effective step modest, then from any one agent's view the others drift
slowly enough that a local critic can track them. The instability that condemns IAC/IQL may be an
artifact of unrestrained updates, not of the local critic per se.

I want one more argument before I trust a purely local critic as *principled* and not merely "robust by
being dumb." Take three agents and telescope the joint return into single-agent changes: hold `π¹, π²`
fixed and improve only `π³`, then hold `π¹, π³_{new}` and improve `π²`, and so on; if every such
coordinate term is non-negative the joint return rises. Look at one term — change only `π³`, others
fixed. Holding the others fixed, the world agent 3 faces is a single-agent decision process — a POMDP in
which `π¹, π²` are absorbed into the dynamics — and agent 3's policy conditions only on `o³`, which is a
POMDP policy. So "improve `π³` holding the rest fixed" is just "improve a single-agent policy on the
induced POMDP," and a restrained single-agent step can make that term non-negative. PPO is a first-order
clipped approximation, not the exact trust-region step, so this is not a theorem for the practical
update, and simultaneous updates without recollection are a further approximation — but the identity
tells me the *structure* I want: each agent takes a restrained single-agent improvement step against the
POMDP the others induce. And the natural baseline for a POMDP policy that conditions on `o³` is a value
function on that *same* information set — `V_φ(o³)`. The local critic is not a compromise forced by
execution; it is the baseline aligned with the coordinate step's information set. The local critic and
PPO's clip are the two halves of one idea: restrained single-agent improvement on the right per-agent
information. That is the principle the attention critic had no claim to — it mixed information across
agents that the actor's coordinate step does not condition on, which is one more reason its advantages
could be both high-variance and misaligned.

Now the module, with each remaining choice motivated. Do I learn `n` separate critics or one shared
`V_φ`? The agents on these maps are homogeneous within a type and share observation/action spaces, so a
separate net per agent wastes data — each net learns from one agent's stream. One shared critic lets
every agent's experience update the same parameters: `n`-fold more data per gradient step, faster and
more stable value learning, which is exactly what I want after watching the attention critic struggle to
fit. The cost is that a shared net cannot specialize per agent by *index* — but the agents already differ
in *behavior* because they see different observations, so a shared function at different inputs still
produces different values. The only thing sharing strictly removes is index-based specialization, and I
hand that back cheaply by appending a one-hot agent id to the critic input. So: one shared critic, input
`[o^a ⊕ one-hot(a)]` of width `obs_dim + n`. The architecture is the plain PPO MLP I trust: `fc1:
(obs_dim+n) → 128`, ReLU; `fc2: 128 → 128`, ReLU; `fc3: 128 → 1`. No attention, no cross-agent mixing,
nothing that can collapse — the deliberate inverse of the step-1 critic.

The shapes have to line up with how the learner calls this, and here is the one place the local critic
differs structurally from the centralized ones: it reads `batch["obs"]`, which already lives on the
agent axis as `(B, T, n, obs_dim)`, so each agent's slot already carries its own `o^a` and there is *no
broadcast* — unlike the state critic, which has to expand a single `(B, T, state_dim)` vector across `n`
agents. The agent id is the `n×n` identity broadcast to `(B, T, n, n)`. Concatenate along the last axis
to `(B, T, n, obs_dim+n)`, run the MLP, and the head gives `(B, T, n, 1)` — the trailing singleton the
learner squeezes. `batch["state"]` is simply never read; the critic carries no peer or global
information by accident, because the only inputs are `o^a` and the constant id of `a`. I double-check
that: agent `a`'s value depends only on `o^a` and on the constant `a`, so it is fully decentralizable and
would compute the identical value at execution from local information alone. Good — the critic respects
the same information constraint as the actor. Everything downstream is the fixed loop: the `q_nstep=5`
target from a target copy of this critic, masked MSE regression, the actor's clipped surrogate on each
agent's own ratio, an entropy bonus, reward standardization, Adam, grad-clip. The only multi-agent
choice I am making lives in *what the critic sees* — and I am betting that seeing *less*, but fitting it
robustly, beats seeing *more* through an architecture that cannot be fit stably.

Here is what I expect against the attention critic's numbers, stated to be falsified. First and most
basic: the bimodal seed collapse should *disappear*. The mat critic's three-way split on MMM
({0.0, 0.41, 0.0}) and the dead seeds on 3s5z ({0.0, 0.0, 0.34}) were value-learning failures; a plain
local MLP cannot land in a degenerate attention basin, so I expect every seed to at least *train* — no
seed stuck at exactly 0.0 win rate on a map the others win, and no return of 4.9 where 18.8 is
reachable. Concretely I expect the per-seed variance to shrink sharply on every map. Second, the mean
win rates should rise on the maps where mat collapsed: MMM (mat 0.135) and 3s5z (mat 0.115) should both
improve substantially, because the floor of "two dead seeds" is removed. Third — and this is the place I
am genuinely unsure — on 2s3z mat already managed 0.54 (two good seeds, one dead), and the local critic's
ceiling could be *lower* than a working centralized critic's, because the local critic is blind to
global structure that a state-conditioned critic could exploit. So I would not be surprised if the local
critic improves the *robustness* everywhere but leaves a *ceiling* on the harder coordination maps that
a centralized critic — if I can make one train stably — would clear. If that is the pattern (robust but
capped), the diagnosis for the next rung writes itself: bring the global state back, but in the lean,
robust MLP form that fits stably, and find out whether centralization helps *once the value learning is
no longer the bottleneck*. The distilled local-critic module — obs plus agent one-hot, plain MLP, no
state — is in the answer.
