Learned-scaling did beat both earlier rungs, and the *pattern* of how it won is the tell I was watching
for. Validation loss came in at 2.2680 — under prores's 2.2707 and vanilla's 2.2763, the cleanest
val_loss improvement on the ladder so far. WikiText-2 perplexity finally moved, 44.11 → 43.91, exactly
the "token identity preserved at depth" signature I predicted for the x0 injection. ARC-Easy extended its
gain to 55.85, the best downstream number yet, again consistent with a route that re-anchors token
identity (knowledge recall). And LAMBADA did precisely the split-channel thing I flagged: it rose to
68.76, *above* prores's 67.21, even as the overall val_loss improved — so learned-scaling and prores are
helping through different mechanisms, and learned-scaling's gentle init-at-1 carry did *not* reproduce
prores's deep-layer-conditioning win on the long-range task. Read all four numbers together and the
ceiling is now visible. Every rung so far — the prores schedule, the learned `resid_lambda`, even the x0
injection — is a *scalar* knob on the depth flow. ProRes scaled the branch by one prescribed number;
learned-scaling scaled the carry and the embedding by two learned numbers. But a scalar is a rank-one
operation on the depth axis: at each layer the stream is still, fundamentally, an *unweighted-or-scalar
sum* of past outputs, mixed the same way for every token. The thing I have never given the network is the
ability to *choose which past layer's output to read*, per token, with a content-dependent weighting.
That is the move left on the table, and it is the largest one.

Let me make the limitation algebraically exact, because it tells me what the next rule has to be. Any
depth-flow rule can be written `h_l = Σ_{i<l} M_{i→l} v_i`, where the sources are the embedding `v_0` and
each earlier transformation output `v_i`, and `M` is the depth-mixing matrix. For the plain residual,
every valid causal entry is 1 — an all-ones kernel, rank-one (the value factors as `1·1`). ProRes
multiplies those entries by a scheduled scalar; learned-scaling by a learned scalar plus a learned x0
term — still, at heart, scalar coefficients folded through a single running stream, a low-rank
semiseparable kernel. The sequence axis escaped this long ago: it stopped using fixed or separable
mixing and moved to *softmax attention over the whole prefix*, which gives a dense, full-rank,
content-dependent mixing matrix. The depth axis can try the same jump. I want layer `l` to form its input
as `h_l = Σ_i α_{i→l} v_i` with `Σ_i α_{i→l} = 1` and the weights `α` computed by softmax attention over
the depth sources — a learned, per-token choice of which earlier representations to combine. That turns
the depth-mixing matrix from rank-one (vanilla) or low-rank-scalar (the last two rungs) into generically
full rank: each destination can read any subset of the past, differently for each token.

Now the design choices, derived rather than assumed. What are the sources, the keys, the queries? The
values have to be the *stored representations themselves* — if I transform them into some new learned
state I am back to summaries of summaries, the very compression I am trying to escape. The same vectors
serve as keys, because the source *content* is what should make the weight vary across tokens. For the
query I have a choice: project it from the current hidden state (expressive, but couples scoring to the
sequential forward state and adds a `d×d` matrix per destination) or use a single learned *pseudo-query*
`w_l ∈ R^d` per destination (cheap, fixed per layer, but the keys are still content-dependent so the
weights still vary by token and example). The pseudo-query is the right default — and it has a structural
bonus: the queries for a group of destinations are known before those destinations execute, which leaves
room to batch the scoring. The scores must not use raw source magnitudes — the whole problem started with
depth-dependent norm growth, so `exp(w_l^T v_i)` would let a large-norm source win on scale alone. RMS-
normalizing the *keys* (not the values) gives me source direction at comparable scale while still mixing
the *raw* representation once a source is selected. And the query init has to be exact: random `w_l`
injects an arbitrary depth preference before training knows what the sources mean, whereas `w_l = 0`
makes every logit `exp(0) = 1`, so the softmax is uniform — the model starts from an equal-weight average
over available sources and learns deviations from that neutral prior. Softmax (not independent gates)
because I want a *fixed probability budget* over sources: emphasizing one source takes mass from the
others, which is the retrieval behavior — "which past layer do I read" — that the depth axis wants.

Here is where I have to be honest about *this task's* constraints, because the full version of "attend
over every earlier output" does not fit the run I am given. A 24-layer block has two sublayers each, so
attending over *all* sublayer outputs means up to 48 stored source tensors, each `(B, T, D)`, and an
attention at every one of the 48 destinations — `O(L²d)` compute and `O(Ld)` source memory per token,
with 48 queries plus a readout. At GPT-2 Medium scale under bf16 with `torch.compile`, holding 48 full
activation streams as an attention source list and scoring against them at every sublayer is exactly the
kind of memory-and-compute overhead that does not survive a fixed 2-GPU, micro-batch-32 budget. The
method's own structure points at the fix: the dense source list can be *grouped* into summaries and the
attention run over those summaries — a scaling variant, not a different idea. So for this task I partition
the 24 layers into **blocks of 4**, giving 6 blocks. *Within* a block I use ordinary residual
connections — the cheap, well-conditioned default — and I run the depth-attention only at **block
boundaries**, attending over the *block* outputs (not the sublayer outputs). That collapses the source
list from 48 sublayer tensors to ~6 block tensors, roughly an 8× memory reduction, while keeping the one
thing that matters: dynamic, content-dependent aggregation along depth. I keep the embedding as the
first source — it is the one representation every later block may need to recover directly, the same x0
route learned-scaling found, now available to the attention itself. At each boundary (every block after
the first) a dedicated pseudo-query attends over all preceding block outputs and picks the input to the
next block; the first block reads the embedding directly. After the last block a final readout query
attends over all 6 block outputs (plus the embedding) to produce the input to the final LayerNorm.

Count the parameters this needs. With `n_blocks = 6` I need a query at each boundary except the first —
`n_blocks − 1 = 5` boundary queries — plus one output query, each a `d`-vector, all zero-initialized so
the model starts from uniform depth-averaging. That is `6 × 1024 ≈ 6k` parameters, negligible against
355M, and far fewer than the 49 queries the full-sublayer version would need. The within-block residuals
add nothing new — they are the standard `block(x)` the scaffold already runs. So the whole method is: run
4 vanilla blocks, take a learned softmax average over block outputs, run the next 4, and so on, with a
final learned average feeding the head. This is genuinely more expressive than the scalar rungs — the
block-boundary mixing matrix is dense and content-dependent, rank up to 6 along the (coarsened) depth
axis instead of rank one — while staying inside the budget. The cost of the coarsening is that I have
given up *sublayer*-granularity depth routing (I cannot, say, have block 5 read block 2's attention
output but not its MLP output); the block boundary is the resolution. That is the deliberate trade this
task forces, and the bet is that block-level dynamic aggregation captures most of the depth-routing
benefit at a fraction of the cost.

The gradient story changes, and I should keep it exact. The plain additive recurrence has a unit-
coefficient identity term in the backward product `∏(I + ∂f/∂h)` — gradient routes straight through
depth with coefficient one. A normalized softmax mixture does not preserve that exact unit `I` term;
instead it gives *direct, differentiable, weighted paths from the loss to every block output with nonzero
attention weight*, plus the score-gradient path through the keys. At zero init every block source has
nonzero (uniform) weight, so at the start of training gradients are spread across all earlier block
outputs rather than forced through the immediate predecessor — a different, and for deep routing
arguably better, conditioning than the strict identity highway. The within-block residuals keep the
clean local identity path on the fine scale; the boundary attention adds the global routing on the coarse
scale.

There is one optimizer wrinkle, and it is the only thing this rung changes about training. The
pseudo-queries are zero-initialized and sit at a leveraged point — they decide the *entire* mixing at
each boundary — so a query that moves too fast can swing which block the next block reads, destabilizing
everything downstream. So I give the query parameters their own optimizer group at a *reduced* learning
rate, `0.1×` the base rate, with no weight decay (they are not weight matrices and decay would just pull
them back toward the uniform-init they started at). The main model matrices stay in the decayed group,
the other 1-D parameters (LayerNorm gains, etc.) in the standard no-decay group. That `0.1×` query LR is
the one deliberate departure from the base schedule; everything else — the cosine schedule,
`CONFIG_OVERRIDES` — stays default.

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
running stream; full_attnres (here, block-partitioned AttnRes) replaces the scalar accumulator at block
boundaries with *content-dependent softmax attention over block outputs* — a dense, full-rank depth-
mixing at the block scale, with the embedding as a first-class source, zero-initialized to start at
uniform averaging. Here is what I expect against the 2.2680 learned-scaling number, and where the bet is.
The mechanism predicts the largest val_loss drop on the ladder: moving from a rank-one scalar mix to a
full-rank content-dependent mix is the first rung that adds genuine *capacity* to the depth flow rather
than just conditioning it, so I expect a clear win, plausibly into the mid-2.25s — the first sub-2.26
number. The dynamic depth aggregation should sharpen long-range prediction most, because choosing which
deep block to read per token is exactly what helps completion: I expect LAMBADA to finally drop below
*both* prores (67.21) and learned-scaling (68.76), and WikiText-2 to fall below 43.91. Downstream
accuracy should at least hold its gains; HellaSwag, flat all ladder, might finally tick up if the richer
depth flow helps commonsense composition. The honest risks are two. First, the coarsening: if
sublayer-granularity routing was where the real benefit lived, block-level attention will capture only
part of it and the win will be smaller than the full version would give. Second, the queries are
leveraged — if the `0.1×` LR is too aggressive the boundary mixing could destabilize and cost me the
clean within-block highways; if too gentle, the queries barely leave their uniform init and the rung
collapses toward a fixed block-averaging model only marginally above learned-scaling. Either way, this is
the strongest depth-flow rule available inside the four-baseline ladder, and the number it lands is the
bar any further idea has to clear.
