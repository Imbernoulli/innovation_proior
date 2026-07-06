I am starting the ladder, so there is no prior measurement to react to — only the scaffold and the
question of which critic to put in it first. The default fill is a lean central-V: a shared MLP over the
global state plus an agent one-hot, `V_φ([s, e_a])`. That is the plainest centralized critic there is,
and precisely because it is plain it leaves an obvious lever untouched — it never lets the agents'
representations *interact*. The whole reason cooperation is hard is that the value of the team situation
depends on how the agents relate to each other, and a critic that maps a flat `[s, e_a]` vector through
an MLP has to learn those relations implicitly inside its weights, with no architectural place to put
them. So the most ambitious thing I can drop in first is a critic whose *architecture* is built to model
agent-to-agent interaction. If structure helps, I want to see it help from the start.

Before I commit to that, let me actually walk the menu the background hands me and let the *contract*
do the eliminating, because the wrong first move here is to reach for a centralized critic that the fixed
loop cannot host. The prior art lists four families. COMA is a single centralized *action*-value critic
`Q(s, A)` with a counterfactual baseline that marginalizes agent `i`'s own action. MADDPG is a per-agent
action-value critic `Q_i(x, a_1,…,a_n)` conditioned on the joint action. VDN/QMIX factorize a joint
action-value into per-agent pieces so the argmax stays decentralizable. And IQL/IAC is independent
learning with no centralization at all. Now look at what the editable region and the learner will accept.
The contract fixes `self.output_type = "v"`, `forward` returns `(B, T, n_agents, 1)`, and the learner
forms the advantage as the masked TD error `R_t^{a,(n)} − V(x^a_t)` — a *state-value* baseline fed
through GAE/`n`-step. That single fact is decisive. A COMA critic would need the joint action `A` as
input (the batch handed to `forward` carries `state` and `obs`, not the acting joint action in a form I
can condition on), and even granting the action, the counterfactual advantage
`A^i = Q(s,A) − Σ_{a'} π_i(a'|·) Q(s,(a', A^{-i}))` is *assembled in the learner*, which is frozen — I
have no hook to marginalize agent `i`'s action, so plugging a `Q` into the `"v"` slot would just make the
learner compute `R − Q(s,A)`, which is not COMA's advantage and not a valid baseline. MADDPG is worse
still: its `Q_i` is an off-policy deterministic-policy-gradient critic with a bootstable target of a
different type, alien to this on-policy GAE loop. VDN/QMIX exist to make a `argmax_a Q` decentralizable,
but the actor here is a stochastic RNN policy trained by policy gradient — there is no argmax over a mixed
`Q` for a mixing network to serve, so the whole decomposition apparatus has no consumer. That leaves
IQL/IAC, which is a legitimate point but the *least* ambitious one on the board — a local critic with no
centralization, the floor I can always fall back to. So the contract has already pruned the action-value
and decomposition families; what survives that both fits the `V`-baseline slot *and* gives interaction
an architectural home is not on the four-item list at all — it is a value critic whose network mixes
agent features. That is the design I want to probe first, and I reached it by elimination, not by taste.

There is a strategic question underneath the choice of *which* critic first, and it is worth settling
explicitly because it shapes the whole ladder. I could start cheap — drop in the plain state MLP the
scaffold already gives me — and climb up in complexity only if the plain critic leaves headroom. Or I
could start ambitious — put the highest-capacity, most-structured critic in first — and *descend* if it
fails. The two orderings are not symmetric in what they teach. Starting cheap tells me the floor but
leaves the strongest hypothesis (that architectural interaction modeling helps) untested until late, and
if the plain critic already does well I might never learn whether structure was worth anything. Starting
ambitious tests the strongest hypothesis immediately: if the attention critic *wins*, I have my answer
and the rest of the ladder is refinement; if it *loses*, the loss is diagnostic — it tells me the binding
constraint is not representational power, and every rung below it becomes a controlled removal of
complexity that isolates what actually mattered. A failure at the top is more informative than a success
at the bottom, because it converts the ladder into an ablation. So I put the most structured critic first
on purpose, accepting that it is the likeliest to fail, because its failure would be the most useful thing
the first measurement could hand me.

Let me reason about where the structure should live. I have `n` agents; each has a feature vector — its
own observation `o_i`, and I can hand each the global state `s` too. The thing I want the critic to
capture is which agents matter to which: in a SMAC fight, a marine's value depends a lot on the allies
and enemies near *it* and very little on a unit across the map. A fixed MLP cannot express "agent `i`'s
value should weight agent `j`'s features by how relevant `j` is to `i`," because that weighting is
data-dependent — it changes with the configuration. What computes a data-dependent, content-addressed
mixing across a set of tokens is exactly self-attention: for each agent (query) it computes a softmax
weighting over all agents (keys) and pulls in their values, `softmax(QKᵀ/√d_k)V`. The `1/√d_k` is not
decorative, and I can put a number on why. With four heads on a width-128 model the per-head key
dimension is `d_k = 128/4 = 32`. If the query and key components are roughly unit-variance, their dot
product over 32 dims has variance 32, i.e. a standard deviation of `√32 ≈ 5.66`, so the raw scores range
over roughly `±11` before the softmax. A softmax on logits that large is essentially one-hot, its
Jacobian is `diag(p) − ppᵀ` which vanishes as `p → e_j`, and the attention block stops learning. Dividing
by `√32 ≈ 5.66` pulls the logits back to order 1, where the softmax is soft and its gradient is alive.
That is the regime I need a *cold-started* attention layer to begin learning in, and it is the reason I
keep the standard scaling rather than folding it into the head count. So the design I want first is: turn
each agent into a token, run a self-attention layer across the agent axis so every agent's representation
is informed by every other agent's, and read a per-agent value off the mixed representation. This is the
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
form the per-agent input as `[o_i ⊕ s]` — the agent's observation concatenated with the broadcast
global state — and project it up to a model width `d_model` with a linear map. Why include `s` in every
token rather than just `o_i`? There are three token contents I could pick from, and it is worth being
concrete about the trade. A token of `[o_i]` alone makes the critic blind to any global structure no
agent sees locally, which for a *centralized* value function throws away the one advantage centralization
buys — I would be building an attention critic that is still only as informed as the local observations,
just mixed. A token of `[s]` alone makes every agent's token identical (the state is shared), so the
attention would attend over `n` copies of the same vector and collapse to a function of `s` with no
per-agent grounding beyond whatever the value head can read off a shared representation — the interaction
modeling would have nothing agent-specific to interact. The `[o_i ⊕ s]` token is the only one of the
three that gives each agent a *distinct* token (through `o_i`) *and* the global picture (through `s`), so
that the value of the situation can depend on global structure some agent may not see locally, and the
attention mixes representations that already know the global context. I set
`d_model = hidden_dim` (128) to stay inside the config's width and keep the critic comparable in
capacity to the MLP baselines — but I should be honest that "comparable width" is not "comparable
capacity," and it is worth counting. A single `TransformerEncoderLayer(d_model=128, nhead=4,
dim_feedforward=512)` carries, in its attention block, a fused `QKV` projection of `3·128·128 = 49,152`
weights plus an output projection of `128·128 = 16,384`, and in its feed-forward block
`128·512 + 512·128 = 131,072` weights, plus two LayerNorms — roughly `198k` parameters in the
interaction core alone. The plain MLP baseline's hidden processing is `fc2: 128·128 = 16,384`. So the
attention critic's interaction machinery is on the order of *twelve times* the parameter mass of the
MLP's hidden layer, and every one of those parameters is being fit to a moving bootstrapped value target.
That number is the risk stated in advance: I am not adding a little structure, I am adding an order of
magnitude of capacity to a regression that has to track a target chasing itself. And the data budget is
not generous relative to that capacity — each map trains ~5M environment steps, the batch is reused only
`epochs=4` times, and the returns feeding the target are unnormalized, so the effective supervision per
parameter is thin for a `~198k`-parameter interaction core compared to a `~16k` MLP hidden layer that
sees the same data. Capacity that outruns its supervision is capacity that memorizes early returns and
then drifts, which is precisely the divergence risk I am naming.

The head-count choice is a budget split, so I fix it deliberately: 4 heads gives 32-dim per-head
subspaces, enough to let different heads specialize on different relational patterns — allies versus
enemies, near versus far — without fragmenting a width-128 model. Eight heads would drop `d_k` to 16,
which starts to make each head too thin to hold a useful relation; two heads at `d_k = 64` under-uses the
ability to separate relation types. Four is the middle that keeps each per-head subspace real. The
feed-forward width is `4·d_model = 512` in the usual transformer proportion, GELU activation, and —
deliberately — **no dropout**, because this is a value regression where I want a deterministic,
low-variance estimate, and dropout would inject noise into the very baseline whose job is to *reduce*
variance. The attention here must be *unmasked*: building an interaction-aware representation of agent `i`
should be allowed to look at every other agent, not just earlier ones — there is no causal ordering on
*reading* observations, only (in the full method) on *emitting* actions, and I have no emission stage.
After the encoder layer, a per-token linear head maps each agent's mixed representation to a scalar value.
One encoder layer, not a deep stack: SMAC teams are small (8–10 units), the relational structure is
shallow, and a deeper stack on a value regression with only ~5M steps of data is more likely to overfit
and destabilize than to help.

I should pin the shapes against how the learner calls this, because the transformer needs the agent axis
isolated as its sequence dimension and the learner is strict about the trailing singleton. The batch
gives `state` of shape `(B, T, state_dim)` and `obs` of shape `(B, T, n, obs_dim)`; `forward(batch)` is
called over the whole sequence (or `forward(batch, t)` for one timestep) and the result is later
`.squeeze(3)`-ed, so I must return `(B, T, n, 1)`. Let me trace it on a concrete map to be sure — take
MMM with `n = 10`. Inside `forward` I compute the time extent, take the slice, broadcast the state across
the agent axis to `(B, T, 10, state_dim)`, concatenate with `obs` to `(B, T, 10, obs_dim+state_dim)`, and
project to `(B, T, 10, 128)`. The transformer encoder expects `(batch, seq_len, d_model)`, and *my*
sequence is the agent axis — so I flatten `(B, T)` into one batch dimension, giving `(B·T, 10, 128)`, run
the encoder so attention mixes across the 10 agents independently for each `(B, T)` position, then reshape
back to `(B, T, 10, 128)` and apply the value head to get `(B, T, 10, 1)`. The flatten-then-restore is
the load-bearing detail: it is what makes the self-attention operate over *agents* and not accidentally
over time. Let me check the failure I am guarding against — if instead I had left the tensor as
`(B, T·10, 128)` or fed `(B, T, ...)` with `T` as the sequence axis, the softmax would mix a marine at
time `t` with the same marine at time `t+3`, i.e. attention over the *time* axis, which is both wrong
(the critic would leak future timesteps into the value at `t`) and shape-incompatible with the per-agent
`(B, T, n, 1)` output. The reshape `(B·T, n, d)` is exactly the thing that forbids that. Everything else —
the actor's clipped surrogate on each agent's own ratio, the `q_nstep=5` target via a target critic, the
masked MSE critic regression, reward standardization, Adam, grad-clip — is the fixed MAPPO loop and I
touch none of it.

I want one sanity check that the architecture at least *nests* something sound, so that if it fails I know
the failure is in the interaction and not in a broken base. `TransformerEncoderLayer` wraps its
attention and feed-forward in residual connections. So take the limit where the attention learns nothing
— push the value projection toward zero. Then each agent's post-attention representation is
`token_i + (≈0)`, the residual passes `token_i` through, the feed-forward acts token-wise, and the head
reads a value off `token_i = proj([o_i ⊕ s])` alone. In that limit the critic degenerates to a per-agent
MLP over `[o_i, s]` with no cross-agent information beyond the state everyone already shares — a perfectly
reasonable non-interacting critic. So the attention is a *strict addition* on top of a sane base: if the
run goes badly, the diagnosis is that the attention found an *actively harmful* basin, not that the
scaffold was unsound. That reframes the whole risk. The danger is not that the encoder can't represent a
good value — the residual guarantees it can fall back to a decent one — the danger is that the extra
`~198k` parameters, fit to a moving target with no return normalization and no warmup, settle into a
pattern the masked MSE tolerates but that poisons the per-agent advantages.

So let me be honest about the risks I am walking into, because the value-learning conditions here are not
gentle. First, that high-capacity core is trained as a *bootstrapped* value regression against a target
copy of itself, masked, with returns *not* standardized (`standardise_returns=False`). A large critic fit
to a moving bootstrapped target with unnormalized returns is exactly the setup where value learning can
lock into a bad basin: the critic can fit early-training returns sharply, the target chases, and the
attention weights can settle onto a pattern that the regression is happy with but that produces biased
per-agent advantages. Second, the residual guards against a *total* collapse to the team mean, but it does
not guard against a subtler smearing — if the softmax concentrates on one or two "carry" agents, every
query pulls in a shared value term and the per-agent values become highly correlated, so that after the
head the advantages differ across agents mostly by their own return realizations and little by their
situation; the agent-specific signal the advantage needs thins out. Third, transformers are notoriously
initialization- and warmup-sensitive, and the MAPPO learner has no transformer-specific schedule; the
fixed `lr=3e-4` and a cold-started attention block may not cooperate, and seed-to-seed the run could land
in very different basins. Nothing in the fixed loop stabilizes any of this — no return normalization to
tame the unnormalized target, no warmup to ease the cold attention in, no per-parameter schedule.

So here is what I will watch when the numbers come back, stated as falsifiable expectations. If the
attention critic's interaction modeling is genuinely worth its complexity, it should *help most where
coordination is hardest* — the larger, more heterogeneous teams (MMM's 10 mixed units, 3s5z's team of 8)
— and at least match the simpler critics on the medium map (2s3z), so I would read a `test_battle_won_mean`
that is highest on MMM and 3s5z as the "structure pays" signature. If instead the value-learning
fragility dominates, I expect the opposite signature: high seed variance, and *collapse* on the hard
maps where the bootstrapped value target is hardest to track and the attention has the most room to find
a degenerate pattern — possibly some seeds at or near a zero win rate while one seed looks fine, i.e. a
bimodal per-seed outcome rather than a tight cluster. And there is a specific tell that would separate the
two stories cleanly: a seed whose `test_battle_won_mean` sits at exactly 0.0 while its `test_return_mean`
is far below what a working seed on the same map reaches would mean the policy never even *found* the
fight, which only a broken baseline can cause — that is the fingerprint of value-learning collapse, not
of a capacity ceiling. The medium map 2s3z, with the shortest horizon and fewest agents, is where I would
expect the attention critic to survive even if it fails elsewhere, because there the value regression is
easiest and the relational structure is small enough that even a mis-trained attention does little harm.
The diagnosis I am setting up for the next rung is precisely this: if the sophisticated critic
underperforms a plain one, the lesson is that *architectural interaction modeling in the critic is not
free* — the critic-only attention form, lacking the decoder that gave the full method its guarantee, buys
representational power at the cost of value-learning stability, and the right next move is to step *down*
in critic complexity to something whose value regression is robust, and only then reconsider what
information the critic should see. I expect the attention critic to be the riskiest point on the ladder,
and I am putting it first deliberately so the measurement tells me whether complexity or robustness is the
binding constraint here. The distilled module — the token projection, the single unmasked encoder layer,
and the per-agent value head, in the scaffold's `CustomCritic` shape — is in the answer.
