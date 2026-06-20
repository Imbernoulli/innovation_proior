Three minutes and change, val_loss 3.2770 at 1395 steps after putting the head on FP8 — that was the last
change that introduced a genuinely new mechanism. Since then the architecture has been tuned hard in ways I
won't relitigate here: a pile of systems and hyperparameter work has pushed the per-step time down toward
~99 ms and shifted the operating point to a longer, ~1750-step config. The recipe is fast and the body is
mature. So when I go looking for the next real method, I'm not hunting for raw throughput anymore — I'm
hunting for a structural inefficiency in how attention itself behaves, something the model is being forced
to do awkwardly because of a constraint I introduced earlier and never reconsidered.

And there is one. Ever since I switched attention to a sliding window — local attention, where each query
only sees a fixed-width window of recent keys rather than the whole sequence — the model has been quietly
leaning on a crutch. The classic problem with softmax attention is that the weights are forced to sum to
one. A head *must* distribute a full unit of attention probability across whatever keys are in its window,
even when none of them are relevant to the current query. There's no "none of the above" option built into
softmax; the head can't simply abstain. So what models do, and what mine has surely learned to do, is dump
the leftover probability mass onto some fixed, semantically-empty token — an attention sink. The
beginning-of-sequence token is the natural candidate: it's always there, it carries little content, and the
head can route "I have nothing to say here" mass onto it and effectively no-op. This is the well-known sink
behavior, and it's the mechanism by which softmax attention performs a soft no-op.

But look at what my own earlier choices have done to that crutch. With a *sliding window*, the BoS token is
frequently *outside the window* — once a query is more than a window-width past the start of the sequence,
it can't attend to BoS at all, so the sink it would like to use isn't even reachable. And it's worse than
that, because I'm using rotary position embeddings. Rotary encodes relative position into the dot product,
so the attention score between a query and the BoS key isn't a fixed thing — it depends on how far apart
they are, and that distance changes for every query as it slides along the sequence. So even when BoS *is*
in the window, its effectiveness as a sink shifts with position. The model is trying to use a fixed token as
a distance-independent dumping ground for unwanted probability mass, but I've put it in a regime where that
token is sometimes absent and always position-dependent. The sink is unreliable, and the model is paying for
that unreliability somewhere in its step count — it's spending capacity working around a no-op mechanism
that my windowing and rotary have half-broken.

So the insight is: what the model actually needs is a *distance-independent* way to perform a context-based
no-op. It needs, for each head, a way to say "given the current token's context, this head should output
roughly nothing" — and it needs that ability to not depend on the presence or relative position of any
particular sink token. The softmax-must-sum-to-one constraint forces the head to attend to *something*; fine,
let it attend to something, but then give me a separate, explicit valve downstream that can scale the head's
contribution to zero when the context says this head has nothing useful to add. Decouple "where does the
attention go" from "does this head fire at all." If I have that valve, the model no longer has to abuse a
token as a sink; it can let the softmax point wherever and just close the valve.

The cleanest realization of a valve is a learned gate on the attention output — a per-head, per-token scalar
in (0,1) that multiplies the head's output. When the gate is near 1 the head fires normally; when it's near
0 the head contributes nothing for that token, which is exactly the context-based no-op I want, and a gate
is by construction distance-independent — it's computed from the token's own representation, not from any
relationship to a sink. The question is what the gate should be a function of. It should be a function of the
current context — the residual stream at this token — so the gate is sigmoid of a small linear map of x. But
I don't want to feed all 768 dimensions of the residual into a gate; that's a fat extra matmul per head per
token, and it would let the gate depend on everything, which is more than this needs. The no-op decision is a
coarse, low-rank kind of thing — "is this head relevant right now or not" — and I suspect it can be read off
a tiny slice of the residual stream.

So I make the gate *sparse* in its inputs: feed it only the first 12 dimensions of x. Twelve, out of 768.
The gate for a head is sigmoid of a linear map from just those 12 residual dims — a [num_heads, 12] weight
per gated layer — and the result is a per-head multiplier I broadcast over the head's output. Twelve active
dimensions is almost free: the gate matmul is 12-wide instead of 768-wide, a rounding error against the
attention cost, and twelve dimensions are enough to carry the relevance signal. That's the "sparse" in sparse
attention gate — only 12 of 768 residual dims feed it. I keep the gate parameters in a single bank,
`attn_gate_bank` of shape `[num_gated_layers, num_heads, 12]`, and inside each gated layer's attention
forward, after I've computed the attention output y of shape (B, T, num_heads, head_dim), I multiply y by
sigmoid of the tiny linear map applied to x[..., :12], viewed to broadcast per head.

Initialization matters for the early dynamics, same discipline as everywhere else in this network. I
zero-init the gate bank. With zero weights, the linear map outputs 0, sigmoid(0) is exactly 0.5, so every
gate starts at 0.5 — every head's output is halved uniformly at step zero, a benign symmetric starting point
that asserts no opinion about which heads should fire, and the gate then *learns* away from 0.5 toward 0 or
1 as the data dictates. Nothing in the gate injects a confident random signal at init; it earns its values
from a neutral start, just like the zero-init head and the zero-init residual projections.

The cost-benefit is the usual speedrun calculus. The gate adds a tiny per-step cost — a 12-wide matmul and a
sigmoid per gated layer, negligible against attention. The benefit is that the model gets a proper
distance-independent no-op and stops paying the tax of a broken sink, which should let it reach the bar in
fewer steps. My estimate is on the order of 50 fewer steps. The risk is that 12 dimensions is too few to
carry a useful gating signal, or that gating heads off destabilizes attention and costs more steps than it
saves — but the zero-init keeps the early behavior gentle, and the mechanism is addressing a real
inefficiency I can name precisely: a sink crutch that my own windowing and rotary have made unreliable. I
expect the gate to claw back the steps that the unreliable sink was costing.

```python
# a per-head attention gate driven by only the first 12 residual dims (sparse):
# attn_gate_bank: one [num_heads, 12] weight per gated layer
self.attn_gate_bank = nn.Parameter(torch.zeros(num_gated_layers, num_heads, 12))

# inside the attention forward, after computing the attention output y of shape (B, T, num_heads, head_dim):
# sparse context-based gate: sigmoid of a tiny linear map from 12 residual dims, per head, no-op when ~0
y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
```

Here `attn_gate_w` is the per-layer `[num_heads, 12]` slice pulled from the bank, `x[..., :12]` is the
sparse 12-dim slice of the residual stream feeding the gate, the sigmoid gives a per-head, per-token
multiplier in (0,1), and the zero-init bank means every gate starts at sigmoid(0)=0.5 and learns from there.
The chain: softmax forces every head to attend to something, so heads learn to dump unwanted mass on a BoS
sink; sliding-window + rotary make that sink unreliable (often outside the window, position-dependent); the
model needs a distance-independent context-based no-op instead; a learned per-head gate, sigmoid of a sparse
12-dim slice of the residual, multiplies the attention output and closes to ~0 when the head should
abstain — cheap to compute, zero-initialized to start neutral at 0.5, and it removes the sink crutch, saving
roughly 50 steps.
