Three minutes and change, val_loss 3.2770 at 1395 steps after putting the head on FP8 — and reading that
record, the FP8 bet paid off exactly as a per-step-time win should: the step_avg fell 148.08 → 136.11 ms
(the halved head matmul) while the step count stayed flat, 1390 → 1395, and val_loss even improved to 3.2770.
On the product, 1395 × 136.11 ≈ 189k ms against the softcap record's 204k — an ~8% cut, all of it from the
cheaper step, which is exactly the fingerprint a pure step-time optimization should leave.
That was the last change that introduced a genuinely new *mechanism*. Since then the recipe has been tuned
hard in ways I won't relitigate here: a pile of systems and hyperparameter work has pushed the per-step time
down toward ~99 ms and shifted the operating point to a longer, ~1750-step config. The recipe is fast and the
body is mature. So when I go looking for the next real method, I'm not hunting for raw throughput anymore — the
FP8 head already took the fattest matmul, and the body matmuls aren't worth the FP8 risk — I'm hunting for a
structural inefficiency in how attention itself behaves, something the model is being forced to do awkwardly
because of a constraint I introduced earlier and never reconsidered.

And there is one, and it traces back to the sliding window I introduced for the long-context switch. Local
attention, where each query only sees a fixed-width window of recent keys rather than the whole sequence, has
a subtle interaction with a basic property of softmax that I've been quietly paying for. The classic problem
with softmax attention is that the weights are forced to sum to one. A head *must* distribute a full unit of
attention probability across whatever keys are in its window, even when none of them are relevant to the
current query. There's no "none of the above" option built into softmax; the head can't simply abstain. So
what models do, and what mine has surely learned to do, is dump the leftover probability mass onto some fixed,
semantically-empty token — an attention sink. The beginning-of-sequence token is the natural candidate: it's
always there, it carries little content, and the head can route "I have nothing to say here" mass onto it and
effectively no-op. This is the well-known sink behavior, and it's the mechanism by which softmax attention
performs a soft no-op.

Let me make the "no-op" concrete, because the magnitudes explain why the sink exists. Suppose a head's window
holds W keys and none is relevant to the current query. If the head spreads its unit of probability roughly
uniformly, its output is the *average* of the W value vectors, ≈ (1/W)·Σvᵢ, whose norm for roughly
independent unit-scale values is ~||v||/√W — for a window of ~1024 that's ~1/32 of a single value's
magnitude. Small, but not zero, and worse, it's *noise*: an average of irrelevant values, uncorrelated with
what this token needs, injected straight into the residual stream. The head would rather output *nothing*
than this noise, but softmax won't let it — the weights must sum to one, so it can't just turn off. The sink
is the escape: pour ~all the mass onto a token whose value is ~0 (BoS carries little content), and the output
becomes ~0·(that value) ≈ 0 — a genuine no-op instead of injected noise. So the sink isn't a quirk; it's the
only way a softmax head can achieve a near-zero output, and a head with nothing to say leans on it constantly.

But look at what my own earlier choices have done to that crutch. With a *sliding window*, the BoS token is
frequently *outside the window* — once a query is more than a window-width past the start of the sequence, it
can't attend to BoS at all, so the sink it would like to use isn't even reachable. And it's worse than that,
because I'm using rotary position embeddings. Rotary encodes relative position into the dot product, so the
attention score between a query and the BoS key isn't a fixed thing — it depends on how far apart they are,
and that distance changes for every query as it slides along the sequence. So even when BoS *is* in the
window, its effectiveness as a sink shifts with position: a query 50 tokens from BoS and a query 500 tokens
from BoS see different rotary phases on the BoS key, so a head that wants a stable, distance-independent
dumping ground can't get one out of a rotary-positioned token. The model is trying to use a fixed token as a
distance-independent no-op valve, but I've put it in a regime where that token is sometimes absent and always
position-dependent. The sink is unreliable, and the model is paying for that unreliability somewhere in its
step count — it's spending capacity working around a no-op mechanism that my windowing and rotary have
half-broken.

So what the model actually needs is a *distance-independent* way to perform a context-based no-op. It needs,
for each head, a way to say "given the current token's context, this head should output roughly nothing" — and
it needs that ability to not depend on the presence or relative position of any particular sink token. The
softmax-must-sum-to-one constraint forces the head to attend to *something*; fine, let it attend to something,
but then give me a separate, explicit valve downstream that can scale the head's contribution to zero when the
context says this head has nothing useful to add. Decouple "where does the attention go" from "does this head
fire at all." If I have that valve, the model no longer has to abuse a token as a sink; it can let the softmax
point wherever and just close the valve.

Let me consider the ways to build that valve, because there are a few and I want the one that's genuinely
distance-independent and context-dependent. One option is to add a dedicated learnable "register" or sink
token to the sequence — a slot every query can always attend to, carrying the overflow mass. But that's still
a token *in the sequence*, so it's subject to the very windowing and rotary that broke the BoS sink; I'd be
rebuilding the same fragile mechanism with a nicer name. A second option is to change the softmax itself — the
"off-by-one" softmax that adds a constant 1 to the denominator, giving the head an implicit escape route for
probability mass without forcing it onto a key. That's cleaner, but the escape is a *fixed* constant, the same
for every token and context; it lets a head leak mass but not *decide, per token, based on context* whether to
fire. What I want is context-dependent: the no-op decision should depend on what this token is and what's
around it. The third option — a learned gate on the attention output, a per-head, per-token scalar in (0, 1)
that multiplies the head's output — is the one that's both. When the gate is near 1 the head fires normally;
when it's near 0 the head contributes nothing for that token, which is exactly the context-based no-op I want,
and a gate is by construction distance-independent — it's computed from the token's own representation, not
from any relationship to a sink. The *per-token* granularity is the crux: the gate is evaluated at every
position from that position's own residual state, so a single head can fire on the tokens where it has
something to contribute and abstain on the tokens where it doesn't, within the same sequence. That's strictly
what the sink was faking — a head "no-op-ing" on some queries and not others — but done cleanly and locally
instead of by routing mass to a distant token whose reachability changes with position. A per-head-*global*
gate (one scalar per head for the whole sequence) couldn't do this; it could only turn a head off entirely or
leave it on. The per-token gate is what makes it a genuine context-based no-op rather than a head on/off
switch. So the output gate strictly dominates the register (not sequence-bound) and
the off-by-one softmax (context-dependent, not a fixed constant). That's the valve.

The question is what the gate should be a function of. It should be a function of the current context — the
residual stream at this token — so the gate is sigmoid of a small linear map of x. But I don't want to feed
all 768 dimensions of the residual into a gate; that's a fat extra matmul per head per token, and it would let
the gate depend on everything, which is more than this needs. The no-op decision is a coarse, low-rank kind of
thing — "is this head relevant right now or not" — a roughly binary judgment, and I suspect it can be read off
a tiny slice of the residual stream rather than needing the whole 768-dim state. So I make the gate *sparse*
in its inputs: feed it only the first 12 dimensions of x. Twelve, out of 768. The gate for a head is sigmoid
of a linear map from just those 12 residual dims — a [num_heads, 12] weight per gated layer — and the result
is a per-head multiplier I broadcast over the head's output. The placement is the natural one: the gate
multiplies y *after* the attention aggregation (the softmax-weighted sum of values) but before the output
projection c_proj, so it scales the whole head's contribution as a unit. Gating there means "this head's
output, whatever it computed, counts for this much" — which is precisely the no-op knob I want, cleanly
separated from the softmax that decided *where* the attention pointed. Gating the values or the query instead
would tangle the no-op decision back into the attention computation itself; scaling the finished head output
keeps the two concerns — where attention goes, and whether the head fires — fully decoupled. Let me check the cost is really negligible: the
gate matmul is tokens × 12 × num_heads per gated layer, against the attention's tokens × window × head_dim ×
num_heads — with the window in the hundreds-to-thousands and head_dim 128, the 12-wide gate is smaller than
the attention by a factor of roughly (window × 128)/12 ≈ thousands, a genuine rounding error. The parameter
count is tiny too: with 6 heads (head_dim 128 at width 768), each gated layer's gate weight is a [6, 12] = 72-
number matrix, so the whole bank across a handful of gated layers is a few hundred parameters — nothing next
to the ~200M+ of the rest of the model. And there's the same safety floor the U-net had: if every gate
learned its way to 1, `sigmoid(...)·y = y` and I recover the exact ungated attention I have now, so the gate
strictly *contains* the current model as the all-gates-open corner and can't do worse than baseline except
through the optimization it enables. The worst case is that training drives all gates to 1 and I've spent a
few hundred parameters to rediscover the ungated net. That's the
"sparse" in sparse attention gate — only 12 of 768 residual dims feed it, so it's almost free, and the bet is
that twelve dimensions carry enough of the relevance signal.

Why the *raw first 12 dims* of x, rather than a learned projection down to 12? Because a fixed slice lets the
residual stream self-organize: the model can learn to *write* the gate-relevance signal into those first 12
coordinates during training, treating them as a dedicated low-bandwidth channel for "which heads should fire
here." A learned projection would be a full 768→12 matmul per gated layer, reintroducing a matmul I'm trying
to avoid, whereas the raw slice is a zero-cost gather of 12 numbers and lets the network decide what to put
there. Because the residual stream has 768 dimensions to spare, dedicating 12 of them to a coarse binary
relevance signal costs almost nothing in representational room, and the optimizer fills them as needed.

And which layers get gated? The bank is `[num_gated_layers, num_heads, 12]` — a subset of layers, not all
twelve. That's deliberate: the sink abuse is worst in the layers whose attention *concentrates*, which the
value-residual work identified as the deeper layers. Those are exactly the layers that most often have
"nothing relevant in the window" for a given query and so most often reach for the sink, so gating them is
where the no-op valve pays off; the early broad-attention layers rarely need to abstain. So the gate is
applied where the mechanism predicts the sink tax is largest, not sprayed uniformly — which also keeps the
already-negligible cost confined to the layers that actually benefit.

Initialization matters for the early dynamics, same discipline as everywhere else in this network, and I want
to verify it lands where I intend. I zero-init the gate bank. With zero weights, the linear map outputs 0 for
every token, sigmoid(0) is exactly 0.5, so every gate starts at 0.5 — every head's output is halved uniformly
at step zero. Is a uniform 0.5 a benign start? It's a symmetric point that asserts no opinion about which
heads should fire — no head is preferentially opened or closed — and the constant 0.5 factor on every head is
just a global 2× rescale of the attention contribution, which the surrounding norms and residual scales absorb
without trouble. The gate then *learns* away from 0.5 toward 0 or 1 as the data dictates. Nothing in the gate
injects a confident random signal at init; it earns its values from a neutral start, just like the zero-init
head and the zero-init residual projections. (Contrast a random-init gate, which would open some heads and
close others arbitrarily at step zero, a random perturbation the early steps would have to undo.)

The sigmoid is load-bearing in a second way beyond giving a (0,1) range: it's smooth. A hard gate — a 0/1
indicator of "fire or not" — would be non-differentiable at its threshold and pass no gradient, so the model
couldn't learn *when* to gate. The sigmoid is differentiable everywhere, asymptotes to 0 and 1 without ever
reaching them (so a head is never *hard*-killed, only pushed arbitrarily close to off), and its gradient is
largest near 0.5 — exactly where the gates start — so the early training gets the strongest signal about which
direction each gate should move. A head that should abstain drifts its gate toward 0; one that should fire
drifts toward 1; and the smoothness means the optimizer sees a clean gradient the whole way. That's why the
valve is a sigmoid and not a threshold.

The cost-benefit is the usual speedrun calculus. The gate adds a tiny per-step cost — a 12-wide matmul and a
sigmoid per gated layer, negligible against attention as the arithmetic showed. The benefit is that the model
gets a proper distance-independent no-op and stops paying the tax of a broken sink, which should let it reach
the bar in fewer steps. My estimate is on the order of 50 fewer steps — small in absolute terms, but at a
~1750-step operating point that's ~3%, and it's ~free. Why only ~50 and not a dramatic cut? Because the sink
tax is real but bounded: the model already *has* a workaround (the unreliable BoS sink), so I'm not enabling
a capability it lacked, I'm removing the friction of a workaround that half-works. The gain is the wasted
capacity the broken sink was consuming, not a whole new mechanism — a few percent, the size of a friction
removal, which is exactly the magnitude I'd expect and the honest thing to predict rather than overselling it. The risk is that 12 dimensions is too few to carry a
useful gating signal, or that gating heads off destabilizes attention and costs more steps than it saves — but
the zero-init keeps the early behavior gentle, and the mechanism is addressing a real inefficiency I can name
precisely: a sink crutch that my own windowing and rotary have made unreliable. So the falsifiable signature is
a modest *step-count* drop (~50 steps) at essentially unchanged step_avg (the gate is a rounding error), with
val_loss holding under the bar. If the step count *doesn't* move, then either 12 dims was too few or the sink
wasn't actually costing what I think, and I'd try widening the gate's input slice (say to 24 or 32 dims) before
abandoning the mechanism, since the input-sparsity is the cheapest thing to relax and the first suspect.

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
sink; my own sliding-window + rotary make that sink unreliable (often outside the window, and rotary makes its
score position-dependent); so the model needs a distance-independent context-based no-op instead, and of the
options — a register token (still sequence-bound, still fragile), an off-by-one softmax (a fixed, non-context
escape), or an output gate — only the learned per-head gate is both distance-independent and per-token
context-dependent; the gate is sigmoid of a *sparse* 12-of-768-dim slice of the residual (cost a rounding
error against attention, verified), multiplies the attention output and closes to ~0 when the head should
abstain, and is zero-initialized so every gate starts at the neutral sigmoid(0)=0.5 and learns from there —
removing the sink crutch and saving roughly 50 steps at essentially flat step_avg.
