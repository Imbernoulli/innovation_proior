I want to start the ladder from the architecture that, on paper, has the cleanest shot at the structure
of this language, and then let the measured numbers tell me where it actually breaks. Dyck-(k,m) is a
language about *matching*: the token I must predict at position `t` is determined entirely by the single
open bracket sitting on top of the stack, which could have been written many positions back. So the core
computational primitive the model needs is the ability to reach back to an arbitrary earlier position and
read it. That is exactly what self-attention does in one hop — every position can attend to every earlier
position directly, with no recurrence in between — so before I reach for anything with an explicit memory,
I want to know how far a pure attention stack gets on a language whose whole content is long-range
matching. If attention's `O(1)` reach were enough, I would not need a stack at all.

Let me first lay out the architectures actually on the table, because the choice at this rung sets up
every contrast the ladder will draw. Four families can plausibly fit `build_model`: a causal Transformer
with learned absolute positions; a causal Transformer with *relative* or rotary positions; a recurrent
network that streams the tokens with no position table at all; and a stacked causal convolution with a
receptive field wide enough to span the training lengths. The convolution I can rule out immediately on an
arithmetic ground: to reach back across a nested span of length 64 with a dilated stack I need a receptive
field of at least 64, which at kernel size 3 costs `log2(64) ≈ 6` dilated layers just to *see* the matching
bracket, and even then the field is fixed at build time and cannot cover the 128–256 span of the hardest
OOD environment — the receptive field is a hard-coded length ceiling, the exact failure I am trying to
diagnose, baked in structurally rather than discovered empirically. So the real contest is attention
versus recurrence, and the interesting question is *positions*.

The relative/rotary option is the tempting one, and I want to walk a few steps into it before I set it
down, because it looks like it preempts the whole length problem. Relative attention scores a query-key
pair by their *offset* `i − j`, not their absolute indices, so a model that learned "attend to the opener
`d` steps back" would, in principle, apply the same learned offset kernel at position 200 that it learned
at position 60. That is genuinely more length-robust in the limit. But two things make it the wrong move
*at this rung specifically*. First, it is a fix, and I have not yet measured the failure it fixes — if I
open with the length-robust variant and it works, I will not know whether the plain absolute-position
Transformer would have failed, and the entire diagnostic value of the ladder's first step evaporates; I
would be conflating "was there a problem" with "does my chosen cure work". Second, even relative positions
only interpolate the offset kernel over the *range of offsets seen in training*: a matching span of 180
brackets at OOD time involves offsets larger than any trained offset in an env whose training tops out at
64, so the relative kernel is itself extrapolating past its trained support — it softens the failure but
does not obviously abolish it, and I would rather see the raw failure cleanly than half-fix it and squint.
So I set relative positions down deliberately, not because it is bad, but because the canonical
absolute-position Transformer is the correct *first* measurement: it is the strongest-in-distribution,
most-standard baseline, and its OOD collapse (if it collapses) names the problem for everything downstream.

Let me build the model as a fill of the `build_model` slot and reason about each piece against the task. The
scaffold hands me `vocab_size(config.k) = 2k + 2`, `config.hidden_dim = 64`, and the language parameters,
and it will train whatever `DyckModel` I return with token-level cross-entropy at every non-PAD position.
The sequence is `[BOS, x1, …, xn]`, scored left-to-right, so the model must be **causal**: the prediction
at position `t` may depend only on tokens at positions `≤ t`, because at every position it is predicting
the *next* token from the prefix it has seen. A bidirectional encoder would be cheating — it would let
position `t` peek at the token it is supposed to predict. So I need a decoder-only / causal Transformer.

The first piece is the token embedding: `nn.Embedding(vocab, hidden)` maps each of the `2k + 2` symbols
to a `hidden`-dimensional vector. That is uncontroversial. The second piece is where the first real
design decision lives, and it is the decision the whole length-generalization question turns on:
**positional information**. Self-attention is permutation-equivariant — `softmax(QKᵀ)V` is a sum of dot
products and weighted averages over the *set* of positions, with no term anywhere that knows which
position is which. Let me make that concrete rather than assert it: if I feed the two-token prefixes `( )`
and `) (` through a positionless attention block, the query at the second position forms the same set of
key-value dot products in both cases — `{(·(, (·)}` as an unordered bag — so its output is identical, and
the head cannot distinguish a well-formed prefix from an ill-formed one. For a language where `( )` is
valid and `) (` is not, a model that cannot tell positions apart is a bag of brackets and will fail
outright. So I must inject position. The standard, most direct way is a **learned absolute positional
embedding**: a second `nn.Embedding(max_len, hidden)` indexed by the integer position, added to the token
embedding at the bottom of the stack. Each position `0, 1, 2, …` gets its own learned vector, and the model
is free to use those vectors however it likes.

I want to be honest with myself about what learned absolute positions do and do not buy, because this is
where I expect the architecture to struggle and I would rather predict the failure than be surprised by
it. During training the model only ever sees positions up to the training maximum — length 64 in two of
the three environments. The positional embedding table has rows for every position, but only the rows for
positions seen in training receive a gradient. The rows for positions beyond the training maximum — the
ones the model will need at OOD evaluation, lengths 65–96, 97–128, and the brutal 128–256 of the
length-OOD probe — are never trained. They sit at their random initialization. So at OOD time the model
is asked to attend over positions whose positional codes it has literally never optimized. Whatever
position-dependent computation it learned in-distribution — "look back to the matching open bracket using
the relationship between my position and its position" — is expressed in terms of position vectors that, in
the OOD range, are noise.

Let me quantify how much of an OOD sequence is poisoned this way, because the fraction tells me how badly
to expect each environment to break. Take the length-OOD environment: training strings top out at 64
tokens, and a mid-range OOD string runs, say, 192 tokens. Positions 0–64 carry trained codes; positions
65–191 — that is 127 of the 192 positions, about 66% — carry untrained random codes. The query at the tail
of the string, exactly where the deepest still-open brackets must be matched, sits in the untrained regime
and forms its attention scores against a mix of trained and untrained keys. For the `dyck-k2-m3`
environment the gap is far gentler: training to 64, OOD 65–96, so an 80-token OOD string has only positions
65–79 untrained — 15 of 80, under 20%. This is the structural reason absolute positions are expected to
fail to length-generalize on Dyck, *and* the reason I expect the damage to scale with the train→test gap:
the mechanism that lets attention locate the matching bracket is anchored to absolute coordinates, and the
fraction of the sequence living in untrained coordinates is roughly `(L − L_train)/L`, largest exactly on
`dyck-length-ood`. I will cap `max_len` generously (1024) so the model never indexes out of bounds and the
harness does not crash, but a large table does not help — the untrained rows are still untrained, and
padding the table with more never-visited rows changes nothing about the ones the OOD evaluation lands on.

Now the attention stack itself. I use `nn.TransformerEncoderLayer` with a causal mask, which is the
clean way to express a decoder-only block in PyTorch: it is multi-head self-attention followed by a
position-wise feed-forward, with residual connections and layer norm around each. I set `norm_first=True`
(pre-norm), because pre-norm Transformers are markedly more stable to train than post-norm at small depth
and few steps, and I have only 100 gradient steps per environment to work with — I cannot afford a warmup
schedule to tame post-norm. The distinction is not cosmetic: post-norm puts the LayerNorm *after* the
residual add, so the identity path is renormalized every block and early gradients through a deep stack can
blow up without a warmup ramp; pre-norm leaves a clean residual highway and lets the very first of my 100
steps move the weights usefully instead of being spent stabilizing. With a budget this small, every step
that goes to taming the optimizer is a step not spent learning the matching rule. I use `dropout=0.0`: the
training set is large (8 000 strings) relative to the tiny vocabulary and the model is small, and with only
100 steps I am nowhere near the regime where I need regularization against overfitting; dropout would just
add noise to an already short optimization, and worse, it would inject variance into exactly the
long-range attention paths I need to be crisp. I use `activation="gelu"` as the standard smooth
nonlinearity, and `dim_feedforward = 4 * hidden = 256`, the usual 4× expansion in the feed-forward block.

The depth and head count I fix by the representational budget, not by taste. The number of heads is 4 and
the number of layers is 4. The layer count is justified by the representational result for self-attention
on bounded-depth Dyck: a logarithmic number of layers suffices to track depth-`m` nesting, and `m` here is
at most 5, so `log2 m ≈ log2 5 ≈ 2.3` — 4 layers is comfortably above that floor, with margin for the
finite-precision, finite-steps reality that the clean construction ignores. The hidden width is
`config.hidden_dim = 64`, giving head dimension `64 / 4 = 16`. Sixteen dimensions per head is enough to
carry a one-hot-ish key that distinguishes the `2k + 2 ≤ 18` symbol types with room to spare, so I do not
expect the head dimension to be the bottleneck — the positions will be.

Let me sketch how a single head would actually solve the matching, because tracing the mechanism tells me
precisely which piece the untrained positions corrupt. To predict the token after position `t`, the model
needs the identity of the open bracket on top of the stack — the most recent opener whose closer has not
yet been written. A natural attention implementation is a "find the matching opener" head: the query at `t`
carries the current *depth* `d` (how many brackets are still open), and it wants to attend to the earlier
position that opened the bracket now on top. But "the position that opened the current top" is not a fixed
offset — it depends on the whole intervening structure — so the head cannot key on content alone; it keys
on a *positional* relationship computed from the depth counter, which is itself accumulated by summing
`+1` for openers and `−1` for closers over the prefix. Every arithmetic step in that chain — the depth
counter, the "which earlier position sits at the matching depth" lookup — is expressed through the position
embeddings. When those embeddings go to noise past length 64, the depth arithmetic and the matching lookup
both degrade at once, which is why I expect the collapse to be a cliff rather than a gentle slope: it is
not one mechanism failing but the shared coordinate system that two mechanisms are built on.

The optimization budget deserves its own arithmetic, because 100 steps is severe and I want to know whether
the model even *sees* enough data to fit in-distribution. Batch size 64 over 100 gradient steps touches
`64 × 100 = 6 400` example-slots against a training pool of 8 000 strings — so the model makes barely under
one pass over the data, not the tens of epochs a from-scratch Transformer usually wants. That reinforces the
pre-norm, zero-dropout, no-warmup choices: with under one epoch there is no overfitting to regularize
against, and every stabilization trick I skip is a step returned to actual learning. It also sharpens the
prediction — if in-distribution token accuracy comes out high despite under one epoch, the attention prior
is doing real work; if it comes out low, the failure is a *fitting* failure from too few steps, not a
length failure, and I would have to revisit the budget before blaming positions.

I have to build the causal mask myself and hand it to the encoder. For a sequence of length `T`, the mask
is the strictly-upper-triangular matrix of `−inf` (every entry where query `i` looks at key `j > i`) and
`0` elsewhere; added to the pre-softmax scores, the `−inf` entries become `exp(−inf) = 0` after the
softmax, so position `i` attends only to positions `≤ i`. Let me trace it on `T = 3` to be sure I have the
triangle on the correct side: `torch.triu(full((3,3), −inf), diagonal=1)` gives row 0 = `[0, −inf, −inf]`,
row 1 = `[0, 0, −inf]`, row 2 = `[0, 0, 0]`. Reading row `i` as "what query `i` may see", query 0 sees only
key 0, query 1 sees keys 0–1, query 2 sees keys 0–2 — monotone growing prefixes, exactly the
autoregressive contract. If I had used `diagonal=0` instead I would have masked the diagonal too and
forbidden every position from seeing itself, which would break the model; the `diagonal=1` is load-bearing.
I construct the mask with `torch.triu(torch.full((T, T), -inf), diagonal=1)` and pass `is_causal=True` so
the kernel can take the fast path. This is the one piece of plumbing the scaffold leaves to me, and it is
the piece that enforces the autoregressive contract the loss assumes.

A subtlety worth pinning down: the positions buffer. I register
`torch.arange(max_len).unsqueeze(0)` as a non-persistent buffer and, at forward time, slice it to the
actual `T` and expand to the batch. Non-persistent so it does not bloat the state dict; sliced to `T` so I
only ever add as many positional codes as there are tokens. Let me walk the shapes through the forward pass
once to confirm the contract end to end: `input_ids` is `[B, T]`; `self.embed(input_ids)` is `[B, T, 64]`;
`self.positions[:, :T].expand(B, T)` is `[B, T]` of integers, and `self.pos(...)` lifts it to `[B, T, 64]`,
so the sum `h = embed + pos` is `[B, T, 64]`, dimensionally clean. The encoder preserves `[B, T, 64]`, and
the head projects the last axis `64 → vocab`, yielding `[B, T, vocab]` — exactly the shape
`DyckModel.forward` is contracted to return, and exactly what the harness's per-position cross-entropy and
next-valid-set evaluator consume. No transpose, no `batch_first` mismatch, because I set `batch_first=True`
on the layer so `[B, T, C]` flows straight through.

Let me check the parameter budget, because the harness rejects anything over 500 000 trainable params, and
I want a real number, not a hand-wave. Per encoder layer: the attention block has the fused input
projection `3 × 64 × 64 = 12 288` weights plus `3 × 64 = 192` biases, and the output projection
`64 × 64 = 4 096` plus 64 — that is `16 640` for attention. The feed-forward is `64 × 256 + 256 = 16 640`
up and `256 × 64 + 64 = 16 448` down, `33 088` together. Two LayerNorms add `2 × 128 = 256`. So one layer
is `16 640 + 33 088 + 256 = 49 984`, and four layers are `199 936` — call it 200k, the dominant cost. On
top: the position table at `max_len = 1024` is `1024 × 64 = 65 536`, the single biggest non-layer term; the
token embedding is `vocab × 64` (for `k = 2`, `6 × 64 = 384`); the head is `64 × vocab + vocab` (for
`k = 2`, `390`). Summing for the `dyck-k2-m3` config: `199 936 + 65 536 + 384 + 390 = 266 246`. So I predict
the leaderboard will report right around 266k parameters — comfortably under 500 000, and the position
table alone, not the attention stack, is what nudges it up. The budget is not the binding constraint here;
the binding constraint is whether the absolute-position recipe generalizes, and the arithmetic just
confirms I have room to spend on depth and width if the diagnosis says I need it later.

Why start the ladder *here*, with the architecture I half-expect to fail on the headline metric? Because
the failure is the most informative one to see first. The Transformer is the model with the strongest
in-distribution prior — its `O(1)` reach should let it fit the training lengths to high token accuracy
quickly, even in 100 steps. If it then collapses on the OOD split, that collapse cleanly isolates the
length-generalization problem from the fitting problem: the model can represent the matching, it just
cannot carry the representation past the positions it was trained on. That is a sharper diagnosis than
starting from a model that fails *both* in-distribution and OOD, because it tells me the next rung needs to
fix *generalization across length*, specifically, and not raw capacity. It also sets up the contrast I
care about: a recurrent model has no absolute-position table at all — it processes one token at a time and
its state is updated by the same transition regardless of absolute index — so if the recurrent baseline
length-generalizes where the Transformer does not, the absolute-position table is implicated as the
culprit, exactly as the prior art on self-attention for hierarchical languages would predict.

What do I expect to watch in the numbers, stated sharply enough to be wrong? In-distribution token accuracy
should be high but not perfect — the Transformer fits the training-length distribution well, but Dyck's
per-string accuracy is unforgiving: if a string of length `L` has per-position error `ε`, the chance the
whole string is correct is roughly `(1 − ε)^L`, so even a token accuracy of 0.96 on an 80-token string
gives `0.96^80 ≈ 0.04` expected string accuracy — a couple of percent. That single inequality tells me to
expect strong `id_token_acc` sitting alongside weak `id_string_acc`, and it is not a contradiction but the
arithmetic of a product of many near-one factors. On the OOD split I expect a visible drop in
`ood_token_acc`, worst on the `dyck-length-ood` environment where the untrained-position fraction I
computed (~66%) is largest, and I expect `ood_string_acc` pinned near zero across the board, because a
model that is even slightly noisy per position cannot survive the product over a 128–256-token string. The
`dyck-k8-m5` environment, with 8 bracket types and depth 5, is the hardest *content* — more distinct
closers to disambiguate, deeper stacks — so I expect the lowest accuracies there even in-distribution. The
falsifiable claim is concrete: if absolute positions are the bottleneck, `ood_token_acc` should sag
furthest exactly where the length extrapolation is largest, while `id_token_acc` stays high — a
fitting-fine, extrapolating-poorly signature. If instead both in-distribution and OOD are poor, my read of
the architecture is wrong and the next rung is about capacity, not position. The full scaffold module — the
causal Transformer with learned absolute positional embeddings, capped at `max_len = 1024` — is in the
answer; this rung is the strongest in-distribution prior on the board, deliberately placed first so its
length-OOD collapse names the problem the rest of the ladder has to solve.
