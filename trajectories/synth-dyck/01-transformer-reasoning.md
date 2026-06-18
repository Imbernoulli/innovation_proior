I want to start the ladder from the architecture that, on paper, has the cleanest shot at the structure
of this language, and then let the measured numbers tell me where it actually breaks. Dyck-(k,m) is a
language about *matching*: the token I must predict at position `t` is determined entirely by the single
open bracket sitting on top of the stack, which could have been written many positions back. So the core
computational primitive the model needs is the ability to reach back to an arbitrary earlier position and
read it. That is exactly what self-attention does in one hop — every position can attend to every earlier
position directly, with no recurrence in between — so before I reach for anything with an explicit memory,
I want to know how far a pure attention stack gets on a language whose whole content is long-range
matching. If attention's `O(1)` reach were enough, I would not need a stack at all.

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
position is which. If I shuffle the input rows, the output rows shuffle identically and nothing else
changes. For a language where order is everything — `( )` is valid, `) (` is not — a model that cannot
tell positions apart is a bag of brackets and will fail outright. So I must inject position. The standard,
most direct way is a **learned absolute positional embedding**: a second `nn.Embedding(max_len, hidden)`
indexed by the integer position, added to the token embedding at the bottom of the stack. Each position
`0, 1, 2, …` gets its own learned vector, and the model is free to use those vectors however it likes.

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
the OOD range, are noise. This is the structural reason absolute positions are expected to fail to
length-generalize on Dyck: the mechanism that lets attention locate the matching bracket is anchored to
absolute coordinates, and those coordinates run off the end of what was trained. I will cap `max_len`
generously (1024) so the model never indexes out of bounds and the harness does not crash, but a large
table does not help — the untrained rows are still untrained.

Now the attention stack itself. I use `nn.TransformerEncoderLayer` with a causal mask, which is the
clean way to express a decoder-only block in PyTorch: it is multi-head self-attention followed by a
position-wise feed-forward, with residual connections and layer norm around each. I set `norm_first=True`
(pre-norm), because pre-norm Transformers are markedly more stable to train than post-norm at small depth
and few steps, and I have only 100 gradient steps per environment to work with — I cannot afford a warmup
schedule to tame post-norm. I use `dropout=0.0`: the training set is large (8 000 strings) relative to
the tiny vocabulary and the model is small, and with only 100 steps I am nowhere near the regime where I
need regularization against overfitting; dropout would just add noise to an already short optimization. I
use `activation="gelu"` as the standard smooth nonlinearity, and `dim_feedforward = 4 * hidden`, the usual
4× expansion in the feed-forward block. The number of heads is 4 and the number of layers is 4 — enough
depth that, per the representational result for self-attention on bounded-depth Dyck, a logarithmic number
of layers suffices to track depth-`m` nesting, and `m` here is at most 5, so 4 layers is comfortably
above `log m`. The hidden width is `config.hidden_dim = 64`, giving head dimension 16.

I have to build the causal mask myself and hand it to the encoder. For a sequence of length `T`, the mask
is the strictly-upper-triangular matrix of `−inf` (every entry where query `i` looks at key `j > i`) and
`0` elsewhere; added to the pre-softmax scores, the `−inf` entries become `exp(−inf) = 0` after the
softmax, so position `i` attends only to positions `≤ i`. I construct it with
`torch.triu(torch.full((T, T), -inf), diagonal=1)` and pass `is_causal=True` so the kernel can take the
fast path. This is the one piece of plumbing the scaffold leaves to me, and it is the piece that enforces
the autoregressive contract the loss assumes.

A subtlety worth pinning down: the positions buffer. I register
`torch.arange(max_len).unsqueeze(0)` as a non-persistent buffer and, at forward time, slice it to the
actual `T` and expand to the batch. Non-persistent so it does not bloat the state dict; sliced to `T` so I
only ever add as many positional codes as there are tokens. The forward pass is then: embed the tokens,
add the (sliced) positional embeddings, run the causal encoder, and project to logits with a linear head
of shape `hidden → vocab`. The output is `[B, T, vocab]`, exactly the contract `DyckModel.forward`
requires, and the harness's cross-entropy and next-valid-set evaluator do the rest.

Let me check the parameter budget, because the harness rejects anything over 500 000 trainable params. The
dominant cost is the four encoder layers. Each layer has the four attention projections (`Q, K, V, O`),
each `hidden × hidden = 64 × 64`, and the feed-forward two linears `64 × 256` and `256 × 64`, plus norms
and biases. That is roughly `4 · 64² + 2 · 64 · 256 ≈ 16 384 + 32 768 ≈ 49 000` per layer, so ~200 000
across four layers, plus the embeddings (`vocab · 64` and `1024 · 64 ≈ 66 000` for the position table) and
the head. The position table at `max_len = 1024` is the single biggest non-layer cost, but it still keeps
me well under 500 000 — the leaderboard will report params in the mid-200-thousands, comfortably inside
budget. So the budget is not the binding constraint here; the binding constraint is whether the absolute-
position recipe generalizes.

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

What do I expect to watch in the numbers? In-distribution token accuracy should be high but not perfect —
the Transformer fits the training-length distribution well, but Dyck's per-string accuracy is
unforgiving (one wrong position fails the whole string), so I expect strong `id_token_acc` and weak
`id_string_acc`. On the OOD split I expect a visible drop in `ood_token_acc`, worst on the
`dyck-length-ood` environment where the gap between training (≤64) and test (128–256) is the largest and
the untrained positional rows dominate the sequence. The `dyck-k8-m5` environment, with 8 bracket types
and depth 5, is the hardest *content* (more distinct closers to disambiguate, deeper stacks), so I expect
the lowest accuracies there even in-distribution. The falsifiable claim is concrete: if absolute positions
are the bottleneck, `ood_token_acc` should sag furthest exactly where the length extrapolation is largest,
while `id_token_acc` stays high — a fitting-fine, extrapolating-poorly signature. If instead both
in-distribution and OOD are poor, my read of the architecture is wrong and the next rung is about capacity,
not position. The full scaffold module — the causal Transformer with learned absolute positional
embeddings, capped at `max_len = 1024` — is in the answer; this rung is the strongest in-distribution
prior on the board, deliberately placed first so its length-OOD collapse names the problem the rest of the
ladder has to solve.
