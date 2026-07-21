I want to train a small GPT to ~3.8 val loss on WikiText-103 on one A100, and I want every later speedup to
be measured against something I actually trust. So before I optimize anything, I need a baseline that is
*known-correct* — a GPT that reaches the bar by the book — so that when I start cutting corners I can tell
whether a change is a real speedup or a silent regression hidden by the noise. The temptation is to start
clever; the discipline is to start boring, then earn every deviation.

The target itself is worth pinning down, because that number is the whole contract. Cross-entropy of 3.8 nats
is a perplexity of e^3.8 ≈ 44.7 — the two forms of the bar are the same statement. At initialization, over a
padded vocabulary of 50304 with no information, the best the model can do is the uniform distribution, whose
cross-entropy is ln(50304) ≈ 10.83 nats. So the run has to carry the model about 7 nats — a factor of ~1130
in perplexity — down to the bar and then *stop* the instant it crosses 3.8, because every nat past the bar is
wall-clock I'm not being paid for. And the corpus is small: WikiText-103 is ~100M tokens, stored as GPT-2 BPE
ids in 16-bit integers (~200MB) that sit resident on the 40GB A100 alongside a 30M-param model, so batches
draw by random-offset sampling straight from GPU memory with no host↔device transfer in the loop. That
~100M tokens against ~30M parameters is a few tokens per parameter, far under the ~20 a compute-optimal run
would want — firmly a step-limited regime where the model revisits the corpus rather than running out of it.
That is exactly where a little weight decay earns its place, so I keep AdamW's decay on.

The most boring correct GPT is the decoder-only transformer the setting describes: embed tokens, add a learned
absolute-position embedding (attention is permutation-equivariant, so order has to be injected), then a stack
of pre-norm residual blocks each with causal self-attention and a 4×-expansion GELU MLP, a final norm, and a
weight-tied linear projection to vocabulary logits under next-token cross-entropy. Before I decide which of
those defaults are load-bearing, I want to know where the parameters and FLOPs actually live, because that
tells me which "defaults" are doing real work.

The token embedding is vocab × d = 50304 × 384 ≈ 19.3M parameters — one matrix, by a wide margin the largest
tensor in the model, about 64% of the ~30M total (each block carries ~1.77M across its attention and 4× MLP,
~10.6M over six; the position table 256 × 384 ≈ 0.098M is a rounding error). That single fact justifies two
defaults on the spot. Weight-tying the input embedding and the output projection stores and trains that 19.3M
matrix once instead of twice — untied, a separate output head would push the model to ~49.3M — and it is a
known aid to small models besides. And rounding the vocabulary up to 50304, a multiple of 64, keeps the
biggest matmul in the model tensor-core-aligned. The vocab head dominates the *arithmetic* too: per token the
final projection is d × vocab ≈ 19.3M multiply-adds (~38.6M FLOPs), more than the entire six-layer stack
combined (~23.6M FLOPs/token), and the B × L × 50304 logit tensor is the largest activation the model ever
holds — so weight-tying and vocab alignment sit directly on the hottest matmul I have. Worth remembering for
later: of the ~3.93M per-layer compute, the quadratic attention mixing at L=256 is only 0.39M — 10% — so at
this width "attention is quadratic" is a *minority* of a block's arithmetic, a fact I'll want in mind before I
get excited about attacking sequence length.

One initialization detail I'll fix the way deep residual nets do, because it's the difference between a stream
that stays calm and one that's already several times too hot by the top block. Every sublayer writes
x ← x + sublayer(LN(x)), so with six blocks and two sublayers each, twelve outputs accumulate into the
residual stream; if each carried variance σ² at init the twelve independent additions would leave the stream
at ~12σ², growing linearly with depth, and the late blocks would read an input dominated by the pile-up rather
than their own signal. Scaling the residual-projection init std by 1/√(2·num_blocks) = 1/√12 scales each
output's variance by 1/12, so the twelve additions contribute ~σ² total and the stream stays O(1) with depth.
Concretely the base std of 0.02 becomes 0.02/√12 ≈ 0.00577 on the MLP-project and attention-out weights.

Optimizer and schedule are the transformer defaults: AdamW with weight decay, and a warmup-then-decay LR
because a cold transformer hit with a large LR diverges. I take the native one-cycle form — max_lr 2e-3,
div_factor 1e2 so it starts at 2e-5, ramps to the peak over the warmup fraction, then cosine-anneals to a
floor of initial_lr/final_div_factor = 2e-5/0.05 = 4e-4 rather than zero — because the total step count and
the warmup percentage are the two knobs I know I'll be re-fitting as the dynamics change, and it's the
scheduler already wired in.

Now the choices specific to *this* task, chosen so a run finishes in minutes. The model is small — width 384,
six heads, six blocks — for a short experiment cycle. Six heads over 384 gives head dim 64, a multiple of 8 so
the flash-attention kernel takes its fast path, and 6·64 = 384 divides evenly (get that wrong and nothing
runs). Sequence length 256 keeps the quadratic cost down at baseline. And I run in bf16 on the tensor cores —
the one place I deliberately trade numerics for speed. What matters is the *exponent* width: bf16 keeps fp32's
full 8 exponent bits, so it covers the same dynamic range and gives up only mantissa (7 bits, ~2 decimal
digits). fp16, with 5 exponent bits, tops out near 65504 and needs loss-scaling to keep small gradients from
underflowing; bf16 doesn't, so I skip loss-scaling entirely, dodging a whole class of tuning that isn't
correctness-neutral. On the A100 the bf16 tensor cores run matmuls at roughly double the fp32 rate — close to
a free 2× on the heavy matmuls, with only mantissa precision at risk. I turn on TF32 for the fp32 fallback
paths and wrap the forward in a bf16 autocast, so the heavy matmuls run in bf16 while the reductions — softmax,
layernorm, cross-entropy — stay in fp32.

Two more correctness-neutral A100 levers I take for free. Call the fused scaled-dot-product-attention path
rather than a hand-rolled softmax(QKᵀ) that materializes the full B × heads × L × L score tensor; the win is
memory, not FLOPs — the flash path tiles the softmax online and never writes that tensor to HBM. And
`torch.compile` the network, compiled *inside* the timed region so the compilation cost is counted honestly
against the wall-clock rather than hidden.

The one choice I trust least is the batching. A width-384 six-block model is tiny, so a single microbatch fits
comfortably; where I want a larger *effective* batch I accumulate a fixed number of microbatches before
stepping. I'm setting that count to a constant now because I want the baseline boring, but I want to flag it,
because gradient statistics are not stationary. Early on the gradient is large and points obviously downhill,
so even one noisy microbatch already points the right way; late on the gradient is small and a large fraction
of it is sampling noise, so averaging many microbatches is what keeps a step from being dominated by that
noise. A single constant has to serve both regimes and is wrong somewhere — too much averaging early, too
little late. That's the first place I'd look for time later.

One shape I do want right before I trust any loss number, because a causal LM is only as honest as its mask:
the mask must forbid position i from attending to any key j > i. Built as `logical_not(triu(ones(L,L))).T` it
is True exactly where j > i, which in the boolean-mask convention marks the forbidden (future) entries. If
that ever broke, the model would peek one token into the future and the val loss would collapse to something
implausibly low — that discrepancy is the canary if a later refactor touches the mask.

That fixes the two block internals — pre-norm causal attention over the framework's multi-head attention, and
a pre-norm 4× GELU MLP — and the net assembly: token + learned-position embeddings, the interleaved blocks, a
final norm, the weight-tied output head, the 1/√(2·num_blocks) residual-projection init, AdamW, one-cycle, and
`torch.compile` in the timed region; the full modules are in the answer.

This is the floor: a textbook GPT, bf16 on the A100, fused attention, compiled — ~30M parameters, two-thirds
of them the tied vocab matrix, carrying the model from perplexity ~50304 to the 44.7 bar. It should reach the
bar; that's the point of a known-correct recipe, and I won't pretend to know the wall-clock in advance — I
expect single-digit minutes at this scale, and whatever it turns out to be is the clock every later rung has
to beat. What I'm watching is *where* that time goes, because every assumption I made the boring way is a
place I deliberately did not optimize: the fixed accumulation count I flagged; the fixed 256 sequence length
paying full attention from the first step, though that quadratic cost is only ~10% of a block here so I
shouldn't expect the world from touching it; the fixed GELU gate — the same pointwise curve everywhere, with
only the surrounding linear maps learned, so I've done nothing to ask whether a richer learned gate would earn
its keep; the learned absolute positions tied to index slots rather than distances; and the plain autocast
policy carrying an fp32 safety margin bf16's range may not actually need. The baseline's job is to be
trustworthy and set the clock; beating that clock is the rest of the ladder.
