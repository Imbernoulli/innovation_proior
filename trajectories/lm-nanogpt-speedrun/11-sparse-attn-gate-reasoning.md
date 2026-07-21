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

Of the ways to build that valve, two fail the test. A dedicated learnable "register" or sink token is still a
token *in the sequence*, subject to the very windowing and rotary that broke the BoS sink — the same fragile
mechanism with a nicer name. An "off-by-one" softmax (a constant 1 added to the denominator) gives the head an
escape route for probability mass, but a *fixed* one, the same for every token; it can't decide per token,
based on context, whether to fire. The third option — a learned gate on the attention output, a per-head
scalar in (0, 1) that multiplies the head's output — is both distance-independent (computed from the token's
own representation, not from any relationship to a sink) and per-token (evaluated at each position from that
position's residual state, so one head can fire where it has something to contribute and abstain where it
doesn't, within the same sequence). That per-token granularity is exactly what the sink was faking, done
cleanly and locally; a per-head-*global* gate could only switch a head on or off entirely. So the output gate
dominates both alternatives. That's the valve.

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
keeps the two concerns — where attention goes, and whether the head fires — fully decoupled. The cost is
negligible: the gate matmul is tokens × 12 × num_heads against attention's tokens × window × head_dim ×
num_heads, smaller by roughly (window × 128)/12 ≈ thousands, and the whole bank across a handful of gated
layers is a few hundred parameters. And it has the U-net's safety floor: if every gate learns its way to 1,
`sigmoid(...)·y = y` recovers the exact ungated attention, so the gate strictly *contains* the current model
and the worst case is spending a few hundred parameters to rediscover it. That's the "sparse" in sparse
attention gate — only 12 of 768 residual dims feed it — and the bet is that twelve carry enough of the
relevance signal.

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

I zero-init the gate bank, same discipline as everywhere else in this network. With zero weights the linear
map outputs 0, sigmoid(0) = 0.5, so every gate starts at 0.5 — every head's output uniformly halved at step
zero. That's benign: a symmetric point asserting no opinion about which heads fire, and the constant 0.5 is
just a global 2× rescale the surrounding norms absorb. The gate then learns away from 0.5 toward 0 or 1 as the
data dictates — no confident random signal at init, unlike a random-init gate that would open and close heads
arbitrarily for the early steps to undo.

The sigmoid is load-bearing in a second way beyond giving a (0,1) range: it's smooth. A hard gate — a 0/1
indicator of "fire or not" — would be non-differentiable at its threshold and pass no gradient, so the model
couldn't learn *when* to gate. The sigmoid is differentiable everywhere, asymptotes to 0 and 1 without ever
reaching them (so a head is never *hard*-killed, only pushed arbitrarily close to off), and its gradient is
largest near 0.5 — exactly where the gates start — so the early training gets the strongest signal about which
direction each gate should move. A head that should abstain drifts its gate toward 0; one that should fire
drifts toward 1; and the smoothness means the optimizer sees a clean gradient the whole way. That's why the
valve is a sigmoid and not a threshold.

The gate's cost is a 12-wide matmul and a sigmoid per gated layer, negligible against attention; the benefit
is a proper distance-independent no-op that stops paying the broken-sink tax. My estimate is on the order of
50 fewer steps — ~3% at a ~1750-step operating point, and ~free. Not a dramatic cut, because the sink tax is
bounded: the model already *has* a workaround (the unreliable BoS sink), so I'm removing the friction of a
half-working mechanism, not enabling a capability it lacked — the gain is the wasted capacity the broken sink
was consuming, a friction-removal magnitude, the honest thing to predict rather than oversell. So the
falsifiable signature is a modest *step-count* drop (~50 steps) at unchanged step_avg, with val_loss holding
under the bar. If the step count *doesn't* move, then either 12 dims was too few or the sink
wasn't actually costing what I think, and I'd try widening the gate's input slice (say to 24 or 32 dims) before
abandoning the mechanism, since the input-sparsity is the cheapest thing to relax and the first suspect. The zero-init `attn_gate_bank` and
the `y * sigmoid(F.linear(x[..., :12], attn_gate_w))` multiply are in the answer.
