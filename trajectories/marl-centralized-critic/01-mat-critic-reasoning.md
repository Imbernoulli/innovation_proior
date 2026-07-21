I am starting the ladder, so there is no prior measurement to react to — only the scaffold and the
question of which critic to put in it first. The default fill is a lean central-V: a shared MLP over the
global state plus an agent one-hot, `V_φ([s, e_a])`. That is the plainest centralized critic there is,
and precisely because it is plain it leaves an obvious lever untouched — it never lets the agents'
representations *interact*. The whole reason cooperation is hard is that the value of the team situation
depends on how the agents relate to each other, and a critic that maps a flat `[s, e_a]` vector through
an MLP has to learn those relations implicitly inside its weights, with no architectural place to put
them. So the most ambitious thing I can drop in first is a critic whose *architecture* is built to model
agent-to-agent interaction. If structure helps, I want to see it help from the start.

But the wrong first move is to reach for a critic the fixed loop cannot host, so let the contract do the
eliminating. The prior art lists four families, and the editable region plus the learner accept almost
none of them. The contract fixes `self.output_type = "v"`, `forward` returns `(B, T, n_agents, 1)`, and
the learner forms the advantage as the masked TD error `R_t^{a,(n)} − V(x^a_t)` — a *state-value*
baseline. That single fact prunes the action-value families: COMA's `Q(s, A)` and MADDPG's `Q_i(x, a)`
both need the joint action as input (the batch handed to `forward` carries only `state` and `obs`), and
COMA's counterfactual marginalization lives in the *frozen* learner, which I have no hook to change — a
`Q` dropped in the `"v"` slot just makes the learner compute `R − Q(s,A)`, which is neither a valid
baseline nor COMA's advantage. VDN/QMIX exist to make an `argmax_a Q` decentralizable, but the actor is
a stochastic RNN policy-gradient learner with no argmax for a mixing network to serve. That leaves
IQL/IAC — a legitimate point, and the floor I can always fall back to, but the *least* ambitious. What
survives that both fits the `V`-baseline slot *and* gives interaction an architectural home is not on the
four-item list at all: a value critic whose network mixes agent features. I reached it by elimination,
not by taste.

There is a strategic choice underneath *which* critic goes first. I could start cheap — the plain state
MLP — and climb only if it leaves headroom; or start ambitious and descend if it fails. The two are not
symmetric in what they teach. Starting ambitious tests the strongest hypothesis (that architectural
interaction modeling helps) immediately, and a failure at the top is more informative than a success at
the bottom, because it converts the rest of the ladder into a controlled removal of complexity that
isolates what actually mattered. So I put the most structured critic first on purpose, accepting it is
the likeliest to fail.

Where should the structure live? Each agent has a feature vector — its own observation `o_i`, and I can
hand it the global state `s` too. The thing I want the critic to capture is which agents matter to which:
in a SMAC fight, a marine's value depends a lot on the allies and enemies near *it* and very little on a
unit across the map. A fixed MLP cannot express "agent `i`'s value should weight agent `j`'s features by
how relevant `j` is to `i`," because that weighting is data-dependent — it changes with the
configuration. What computes a data-dependent, content-addressed mixing across a set of tokens is exactly
self-attention: for each agent (query) it computes a softmax weighting over all agents (keys) and pulls
in their values, `softmax(QKᵀ/√d_k)V`. The `1/√d_k` is not decorative — with four heads on a width-128
model the per-head dimension is `d_k = 32`, so unit-variance query/key dot products have variance 32 and
the raw scores run large; without the scaling the softmax saturates toward one-hot and its gradient
nearly vanishes, and a cold-started attention block stops learning. Dividing by `√32` keeps the logits
order-1, where the softmax is soft and its gradient is alive. So the design I want first is: turn each
agent into a token, run a self-attention layer across the agent axis so every agent's representation is
informed by every other agent's, and read a per-agent value off the mixed representation. This is the
"attention over agents" family of centralized critics — the critic side of the multi-agent
attention/sequence-modeling line.

Here is where I have to be disciplined about what this task actually exposes, because the full
sequence-modeling method for cooperative MARL is much more than an attention critic, and the harness
gives me only the critic. The full method is an encoder-decoder over the agent axis: an *unmasked*
encoder builds interaction-aware representations and serves as the centralized value, and a *masked*
auto-regressive decoder generates each agent's action conditioned only on its predecessors. The decoder
is the engine of the whole thing — it is the literal implementation of a decomposition identity that
the joint advantage equals the sum of per-agent advantages *each conditioned on the agents before it*,
which is what gives the method its monotonic-improvement story and its Nash limit. But in this task the
**actor is fixed** — it is the standard MAPPO RNN policy, decentralized, with no cross-agent action
conditioning. I do not get to replace the actor; the editable region is `CustomCritic` only. So I cannot
build the auto-regressive decoder, cannot feed predecessor actions into action generation, and cannot
realize the advantage-decomposition guarantee. What I can build is the *encoder half* used as a critic:
the unmasked self-attention over agent tokens with a per-agent value head. I should name this honestly
to myself — this is the **critic-only, encoder-only** form of the attention method, stripped of the
decoder that is the method's actual source of power. I am keeping the part that mixes agent features and
throwing away the part that conditions the joint action. Whether the mixing *alone* helps is exactly the
open question, and it is the right first probe: it isolates "does an attention critic beat a flat MLP
critic," holding the MAPPO actor and learner fixed.

Now build the token. Each agent's token should carry both its local detail and the global picture, so I
form the per-agent input as `[o_i ⊕ s]` and project it to a model width `d_model`. A token of `[o_i]`
alone would make a *centralized* critic blind to the global structure no agent sees locally, throwing
away the one thing centralization buys; a token of `[s]` alone would be identical across all agents (the
state is shared), so the attention would mix `n` copies of the same vector and collapse to a function of
`s` with nothing agent-specific to interact. Only `[o_i ⊕ s]` gives each agent a *distinct* token
(through `o_i`) *and* the global picture (through `s`). I set `d_model = hidden_dim` (128) to stay inside
the config's width — but "comparable width" is not "comparable capacity," and the gap is worth pricing,
because it is the risk I am taking stated in advance. A single `TransformerEncoderLayer(d_model=128,
nhead=4, dim_feedforward=512)` carries on the order of **~198k** parameters in its attention-plus-
feed-forward core; the plain MLP baseline's hidden layer (`fc2: 128·128`) is **~16k**. So the attention
critic's interaction machinery is roughly *twelve times* the parameter mass of the MLP's hidden layer,
and every one of those parameters is being fit to a moving bootstrapped value target with only ~5M
environment steps, `epochs=4` of reuse, and unnormalized returns. Capacity that outruns its supervision
is capacity that memorizes early returns and then drifts — the divergence risk I am naming.

The remaining hyperparameters: 4 heads gives 32-dim per-head subspaces, enough to let different heads
specialize on different relational patterns (allies vs enemies, near vs far) without fragmenting a
width-128 model — 8 heads at `d_k = 16` gets each head too thin, 2 heads under-uses the separation. The
feed-forward width is `4·d_model` in the usual proportion, GELU, and — deliberately — **no dropout**,
because this is a value regression where I want a low-variance estimate and dropout would inject noise
into the very baseline whose job is to *reduce* variance. The attention must be *unmasked*: building an
interaction-aware representation of agent `i` should look at every other agent, not just earlier ones —
there is no causal ordering on *reading* observations, only (in the full method) on *emitting* actions,
and I have no emission stage. One encoder layer, not a deep stack: SMAC teams are small (8–10 units), the
relational structure is shallow, and a deeper stack on a value regression with only ~5M steps is likelier
to overfit than to help. After the encoder, a per-token linear head maps each agent's mixed
representation to a scalar.

The shapes have to line up with how the learner calls this, and one detail is load-bearing. The batch
gives `state` `(B, T, state_dim)` and `obs` `(B, T, n, obs_dim)`; the result is later `.squeeze(3)`-ed,
so I return `(B, T, n, 1)`. Inside `forward` I broadcast the state across the agent axis, concatenate
with `obs`, and project to `(B, T, n, 128)`. The transformer encoder expects `(batch, seq_len, d_model)`
and *my* sequence is the agent axis — so I flatten `(B, T)` into one batch dimension, giving `(B·T, n,
128)`, run the encoder so attention mixes across the `n` agents independently for each `(B, T)` position,
then reshape back and apply the value head. The flatten-then-restore is the point: if I left `T` on the
sequence axis, the softmax would mix a marine at time `t` with the same marine at `t+3`, i.e. attention
over *time*, which both leaks future timesteps into the value at `t` and breaks the per-agent output
shape. The reshape `(B·T, n, d)` is exactly what forbids that. Everything else — the actor's clipped
surrogate, the `q_nstep=5` target via a target critic, the masked MSE regression, reward standardization,
Adam, grad-clip — is the fixed MAPPO loop and I touch none of it.

One structural reassurance: `TransformerEncoderLayer` wraps attention and feed-forward in residual
connections, so in the limit where the value projection goes to zero, each agent's post-attention
representation is just `token_i` passed through, and the head reads a value off `proj([o_i ⊕ s])` alone —
a perfectly reasonable per-agent MLP over `[o_i, s]`. The attention is a *strict addition* on a sane
base: if the run goes badly, the diagnosis is that the attention found an *actively harmful* basin, not
that the base was unsound. That is the real shape of the risk — not that the encoder can't represent a
good value (the residual guarantees it can fall back to one), but that the extra ~198k parameters, fit to
a moving target with no return normalization and no warmup, settle into a pattern the masked MSE tolerates
but that poisons the per-agent advantages.

The value-learning conditions here are not gentle, and the risks compound. That high-capacity core is a
bootstrapped regression against a target copy of itself, masked, with returns *not* standardized — the
setup where a large critic can fit early returns sharply, let the target chase, and settle the attention
weights onto a pattern that produces biased advantages. The residual guards against a *total* collapse to
the team mean but not against a subtler smearing: if the softmax concentrates on one or two "carry"
agents, every query pulls in a shared value term, the per-agent values grow correlated, and the
agent-specific signal the advantage needs thins out. And transformers are initialization- and
warmup-sensitive; the fixed `lr=3e-4` with no schedule may not cooperate with a cold attention block, so
seed-to-seed the run could land in very different basins. Nothing in the fixed loop stabilizes any of it.

So here is what I will watch, stated to be falsified. If the attention critic's interaction modeling is
worth its complexity, it should help most where coordination is hardest — the larger, more heterogeneous
teams (MMM's 10 mixed units, 3s5z's team of 8) — and at least match the simpler critics on the medium
map 2s3z. If instead the value-learning fragility dominates, I expect the opposite: high seed variance
and *collapse* on the hard maps, some seeds near a zero win rate while another looks fine — a bimodal
per-seed outcome rather than a tight cluster. A seed pinned near zero win rate while its return sits far
below a working seed's on the same map would mean the policy never even found the fight, which only a
broken baseline can cause — the fingerprint that separates a value-learning collapse from a capacity
ceiling. 2s3z, with the shortest horizon and fewest agents, is where I would expect the attention critic
to survive even if it fails elsewhere. If the sophisticated critic underperforms a plain one, the lesson
is that architectural interaction modeling in the critic is not free — it buys representational power at
the cost of value-learning stability — and the right next move is to step *down* in complexity to
something whose value regression is robust, then reconsider what the critic should see. The distilled
module — token projection, one unmasked encoder layer, per-agent value head — is in the answer.
