Learned-scaling did beat both earlier steps, and the *pattern* of how it won is the tell I was watching
for — so let me read the table hard before I decide what the next rule has to be. Validation loss came in
at 2.2680, under prores's 2.2707 and vanilla's 2.2763. But look at the *sequence* of val-loss steps:
vanilla→prores was `−0.0056`, prores→learned-scaling is `−0.0027`. The second scalar refinement bought
about half of what the first did; the cumulative gain over the floor is `−0.0083`, and the marginal
returns on "add another scalar knob" are visibly shrinking. That deceleration is itself a datum — it says
I am scraping the bottom of what scalar conditioning can give. Now the secondary columns, because they say
*why*. WikiText-2 finally moved, `44.11 → 43.91` (`−0.20`), exactly the "token identity preserved at
depth" signature I predicted for the `x0` injection. ARC-Easy extended its gain to 55.85 (`+0.50`), the
best downstream number yet, again consistent with a route that re-anchors token identity for knowledge
recall. And LAMBADA did precisely the split-channel thing I flagged: it *rose* to 68.76, `+1.55` *above*
prores's 67.21, even as the overall val_loss improved. That is the load-bearing observation. ProRes helped
LAMBADA (the deep-layer-conditioning channel) and left WikiText-2/ARC flat; learned-scaling helped
WikiText-2/ARC (the token-identity channel) and gave LAMBADA back. Neither scalar scheme could hold both
channels at once — each one *traded* a channel to buy the other. Read together with the decelerating
val-loss, the message is not "scalars are exhausted for lack of tuning" but "a single scalar per layer is
the wrong *shape* of knob," because a scalar cannot simultaneously do deep-layer conditioning and
token-identity preservation — those need different mixings, and one number per layer picks one.

Let me make the limitation algebraically exact, because it tells me what the next rule has to be. Any
depth-flow rule can be written `h_l = Σ_{i<l} M_{i→l} v_i`, where the sources are the embedding `v_0` and
each earlier transformation output `v_i`, and `M` is the depth-mixing matrix. For the plain residual,
every valid causal entry of `M` is exactly 1 — a lower-triangular all-ones kernel, which is the outer
product `1·1ᵀ` and therefore *rank one*. ProRes multiplies those entries by a scheduled scalar per row;
learned-scaling by a learned per-row scalar on the carry plus a learned `x0` term — still, at heart,
scalar coefficients folded through a single running stream, a low-rank semiseparable kernel whose rank is
capped by that scalar structure. None of them can realize an *arbitrary* `M`: they cannot make layer 9
weight layer 3's output heavily and layer 8's output lightly for one token and reverse it for the next.
The sequence axis escaped exactly this trap long ago — it stopped using fixed or separable mixing and
moved to *softmax attention over the whole prefix*, which produces a dense, full-rank, content-dependent
mixing matrix, a different row per token. The depth axis can attempt the same jump. I want layer `l` to
form its input as `h_l = Σ_i α_{i→l} v_i` with `Σ_i α_{i→l} = 1`, where the weights `α` are computed by
softmax attention over the depth sources — a learned, per-token choice of which earlier representations to
combine. That turns the depth-mixing matrix from rank-one (vanilla) or low-rank-scalar (the last two
steps) into generically full rank: each destination can read any subset of the past, differently for each
token. That is the move left on the table, and it is the largest one.

Now the design choices, derived rather than assumed. What are the values, the keys, the queries? The
*values* have to be the *stored representations themselves* — if I transform them into some new learned
state before mixing, I am back to summaries of summaries, the very compression I am trying to escape. The
same vectors serve as *keys*, because the source *content* is what should make the weight vary across
tokens. For the *query* I have a real choice, and it is worth costing both. Option one is to project the
query from the current hidden state, `q_l = W_l h`: maximally expressive, but it adds a `d×d` matrix per
destination — `1024² ≈ 1.05M` parameters *each* — and it couples the depth-scoring to the sequential
forward state, so I cannot compute any score until its destination has executed. Option two is a single
learned *pseudo-query* `w_l ∈ R^d` per destination: `1024` parameters, fixed per layer, but because the
*keys* are content-dependent the resulting weights still vary token to token and example to example, and
the queries are all known *before* their destinations run, which leaves room to batch the scoring. The
pseudo-query is the right default — the `d×d`-per-destination version spends a million parameters and a
serialization constraint to buy a query that varies with the forward state, and the content-dependence I
actually need already lives in the keys. The scores must not use raw source magnitudes: the whole problem
started with depth-dependent norm growth, so a raw `exp(w_l^T v_i)` would let a large-norm deep source win
on scale alone, re-importing the disease through the routing path. RMS-normalizing the *keys* (not the
values) gives me source *direction* at comparable scale while still mixing the *raw* representation once a
source is selected. And the query init has to be exact: a random `w_l` injects an arbitrary depth
preference before training knows what the sources mean, whereas `w_l = 0` makes every logit `exp(0) = 1`,
so the softmax is uniform — the model starts from an equal-weight average over available sources and
learns deviations from that neutral prior. Softmax rather than independent per-source gates, finally,
because I want a *fixed probability budget* over sources: emphasizing one source takes mass from the
others, which is the retrieval behavior — "which past layer do I read" — the depth axis actually wants.

Here is where I have to be honest about *this task's* budget, because the full version of "attend over
every earlier sublayer output" does not fit the run I am given, and I want the numbers, not a vibe. A
24-layer stack has two sublayers each, so attending over *all* sublayer outputs means up to 48 stored
source tensors, each of shape `(B, T, D)`. At micro-batch 32, sequence length 1024, `d = 1024`, in bf16,
one such tensor is `32·1024·1024·2 bytes ≈ 67 MB`. Holding all 48 resident — and I must, because every
one is a source that the backward pass differentiates through — costs `48·67 ≈ 3.2 GB` of activation
*per* scoring site, plus an attention at every one of the 48 destinations, `O(L²d)` compute. Stack that on
top of what the 355M model already spends: parameters in bf16 are `≈ 0.71 GB`, gradients another
`0.71 GB`, and AdamW's two fp32 moments `355M·2·4 ≈ 2.84 GB` — roughly `4.3 GB` of fixed footprint before
any activations, on a 2-GPU split. Bolting a persistent `3.2 GB` source list onto that, under `torch.
compile` which likes to keep activations live for fusion, is exactly the overhead that does not survive a
fixed 2-GPU, micro-batch-32 budget. So the full sublayer-granular version is out on memory, not on
principle — and the method's own structure points at the fix: the dense source list can be *grouped* into
summaries and the attention run over those summaries, a scaling variant, not a different idea.

So for this task I partition the 24 layers into **blocks of 4**, giving 6 blocks. *Within* a block I use
ordinary residual connections — the cheap, well-conditioned default — and I run the depth-attention only
at **block boundaries**, attending over the *block* outputs rather than the sublayer outputs. That
collapses the source list from 48 sublayer tensors to at most 7 (the embedding plus 6 block outputs):
`7·67 ≈ 0.47 GB` instead of `3.2 GB`, a `48/7 ≈ 6.9×` memory reduction (an `8×` reduction counting
block-vs-sublayer sources), while keeping the one thing that matters — dynamic, content-dependent
aggregation along depth. I keep the embedding as the first source: it is the one representation every
later block may need to recover directly, the same `x0` route learned-scaling found, now available to the
attention itself rather than through a dedicated scalar. At each boundary (every block after the first) a
dedicated pseudo-query attends over all preceding block outputs and picks the input to the next block; the
first block reads the embedding directly; after the last block a final readout query attends over all 6
block outputs plus the embedding to produce the input to the final LayerNorm. The parameter cost is one
`d`-vector per boundary except the first — `n_blocks − 1 = 5` boundary queries — plus one output query,
each zero-initialized: `6·1024 ≈ 6k` parameters against 355M, negligible, and far fewer than the ~49
queries the full-sublayer version would need. The deliberate cost of the coarsening is that I give up
*sublayer*-granularity routing — I cannot have block 5 read block 2's attention output but not its MLP
output — and the bet is that block-level dynamic aggregation captures most of the depth-routing benefit at
a fraction of the memory.

Before I trust this I should trace what the zero-init actually produces on the first forward, because it
is *not* the same clean start the last step had and I do not want to pretend otherwise. With every
`w_l = 0`, each boundary logit is `w_l^T RMSNorm(v_i) = 0` for all sources, so the softmax is uniform:
`1/k` over the `k` available sources. So on step zero, block boundary `b` sets its input to the *mean* of
the embedding and all preceding block outputs — not to the running stream, not to the latest block output.
That is genuinely different from vanilla and from learned-scaling, both of which started bit-for-bit at the
Pre-LN residual; block AttnRes starts at a *uniform block-average* operating point and has to learn its
way toward whatever concentration the data prefers. I flag this honestly as a cost: it is a departure from
the "start at exactly the working floor" discipline that made learned-scaling safe, and it is a plausible
place for the step to lose ground if the averaging start is a worse basin than the residual start. The
mitigating read is that a uniform average is still well-conditioned — every source gets equal forward
weight and equal gradient at init, nothing is amplified by scale — and the softmax can concentrate
quickly. But it is a real difference and I will watch for it.

The gradient story changes with it. The plain additive recurrence has a unit-coefficient `I` in the
backward product `∏(I + ∂f/∂h)`; a normalized softmax mixture does *not* preserve that exact unit `I` but
gives direct, differentiable, weighted paths from the loss to every block output with nonzero attention
weight, plus the score-gradient path through the keys. At zero init every block source has uniform `1/k`
weight, so early gradients are spread across all earlier block outputs rather than forced through the
immediate predecessor — arguably better conditioning for deep routing, since every block gets a direct
gradient from the readout rather than one attenuated through the whole stack above it. And the coarsening
keeps both scales honest: the within-block residuals preserve the clean local identity path (inside each
block of 4 I run exactly the vanilla highway) while the boundary attention adds the global
content-dependent routing on the coarse scale.

There is one optimizer wrinkle, and it is the only thing this step changes about training. The
pseudo-queries are zero-initialized and sit at a *leveraged* point — they decide the *entire* mixing at
each boundary, so a query that moves too fast can swing which block the next block reads and destabilize
everything downstream of it. This is a sharper lever than learned-scaling's scalars, and the contrast is
instructive: those scalars started at `resid_lambda = 1`, *on* the working vanilla point, so a bad step
merely perturbs a good configuration and they safely ran at the base learning rate. The queries start at
the uniform-average point — *not* the vanilla point — and their gradient is leveraged over the whole
downstream stack, so an over-eager early step can lurch the boundary mix to a bad basin the lower layers
then have to accommodate. So I give the query parameters their own optimizer group at a *reduced* learning
rate, `0.1×` the base rate, with no weight decay (they are not weight matrices, and decay would just pull
them back toward the uniform init they started at). The `0.1×` is a deliberate order-of-magnitude caution,
not a value I can derive precisely, and it is bracketed by two failure modes I will read off the numbers:
too hot and the boundary mixing destabilizes and I lose the clean within-block highways; too cold and the
queries barely leave their uniform init and the step collapses toward a fixed block-averaging model only
marginally above learned-scaling. The main model matrices stay in the decayed group, the other 1-D
parameters (LayerNorm gains and the like) in the standard no-decay group; the cosine schedule and
`CONFIG_OVERRIDES` stay default.

Now place it in the edit surface. The `Block` stays the vanilla Pre-LN block — within a block I just call
`block(x)` as usual. In `GPT.__init__` I add `attnres_block_size = 4`, compute `n_blocks = n_layer // 4`,
and register the two query parameters: `attnres_queries` of shape `(n_blocks − 1, n_embd)` (zero) and
`attnres_query_out` of shape `(n_embd,)` (zero). In the `GPT.forward` block loop I keep a `block_outputs`
list seeded with the embedding `x`; for each of the 6 blocks, if it is not the first I stack the previous
block outputs, RMS-norm the keys, score with that boundary's query (`einsum('d, n b t d -> n b t', ...)`),
softmax over the source axis, and mix the *raw* stacked values to form the block input; then I run the 4
layers of the block with ordinary residuals and append the result. After the loop I do the same attention
once more with `attnres_query_out` over all block outputs to produce `x` for `ln_f`. In
`configure_optimizers` I split the two query tensors into their own `0.1×`-LR, no-decay group. The
distilled module and the literal scaffold fill are in the answer.

So the delta from step 3 is precise: learned-scaling gave each layer two *scalar* knobs on a single
running stream, and the data showed a scalar cannot hold both the deep-layer-conditioning and
token-identity channels at once; full_attnres (here, block-partitioned AttnRes) replaces the scalar
accumulator at block boundaries with *content-dependent softmax attention over block outputs* — a dense,
full-rank depth-mixing at the block scale, with the embedding as a first-class source, zero-initialized to
start at uniform averaging. Here is what I expect against the 2.2680 learned-scaling number, and where the
bet is. The mechanism predicts the largest val_loss drop so far: moving from a rank-one scalar mix
to a full-rank content-dependent mix is the first step that adds genuine *capacity* to the depth flow
rather than just conditioning it, so unlike the decelerating scalar steps I expect a clear step *up* in
size, plausibly into the mid-2.25s — the first sub-2.26 number. The dynamic depth aggregation should
sharpen long-range prediction most, because choosing which deep block to read per token is exactly what
helps completion: I expect LAMBADA to finally drop below *both* prores (67.21) and learned-scaling (68.76)
— since the attention can do the deep-layer-conditioning *and* the token-identity job in one mechanism,
not trade them — and WikiText-2 to fall below 43.91. Downstream accuracy should at least hold its gains;
HellaSwag, flat all along, might finally tick up if the richer depth flow helps commonsense composition.
The honest risks are two. First, the coarsening: if sublayer-granularity routing was where the real
benefit lived, block-level attention captures only part of it and the win is smaller than the full version
would give — and inside each block of 4 I am, admittedly, right back to the rigid unit-weight accumulator
I have been fighting. Second, the leveraged queries and the non-vanilla start: if the `0.1×` LR is too
aggressive the boundary mixing could destabilize and cost me the clean within-block highways, and if too
gentle the queries barely leave their uniform init and I get a fixed block-average only marginally above
learned-scaling. Either way, this is the strongest depth-flow rule I have reached,
and the number it lands is the bar any further idea has to clear.
