I want to start from the architecture that, on paper, has the cleanest shot at the structure
of this language, and then let the measured numbers tell me where it actually breaks. Dyck-(k,m) is a
language about *matching*: the token I must predict at position `t` is determined entirely by the single
open bracket sitting on top of the stack, which could have been written many positions back. So the core
computational primitive the model needs is the ability to reach back to an arbitrary earlier position and
read it. That is exactly what self-attention does in one hop — every position can attend to every earlier
position directly, with no recurrence in between — so before I reach for anything with an explicit memory,
I want to know how far a pure attention stack gets on a language whose whole content is long-range
matching. If attention's `O(1)` reach were enough, I would not need a stack at all.

Of the families that fit `build_model`, a stacked causal convolution rules itself out on arithmetic: to
reach back across a nested span of length 64 with a dilated stack I need a receptive field of at least 64,
about `log2(64) ≈ 6` dilated layers at kernel size 3 just to *see* the matching bracket, and the field is
fixed at build time — it cannot cover the 128–256 span of the hardest OOD environment, baking in exactly
the length ceiling I am trying to diagnose. So the real contest is attention versus recurrence, and the
interesting question is *positions*.

The tempting variant is relative or rotary positions, which score a query-key pair by their offset `i − j`
rather than absolute index, so a "attend to the opener `d` steps back" head could reuse its learned kernel
at any absolute position. But it is the wrong *first* move for two reasons. It is a fix, and I have not yet
measured the failure it fixes: open with the length-robust variant and, if it works, I will never know
whether the plain absolute-position Transformer would have failed — I would conflate "was there a problem"
with "does my cure work," and the diagnostic value of the first step evaporates. And even relative positions
only interpolate the offset kernel over the *range of offsets seen in training*; a matching span of 180
brackets at OOD time involves offsets larger than any trained offset when training tops out at 64, so the
relative kernel is itself extrapolating past its support. It softens the failure without obviously
abolishing it. So I set relative positions down deliberately: the canonical absolute-position Transformer
is the strongest-in-distribution, most-standard baseline, and its OOD collapse (if it collapses) names the
problem for everything downstream.

Now build the model as a fill of `build_model`. The sequence is `[BOS, x1, …, xn]`, scored left-to-right,
so the model must be **causal**: the prediction at position `t` may depend only on tokens at positions `≤ t`,
because at every position it predicts the *next* token from the prefix it has seen. A bidirectional encoder
would let position `t` peek at the token it is supposed to predict, so I need a decoder-only Transformer.

The token embedding `nn.Embedding(vocab, hidden)` is uncontroversial. The first real design decision is
**positional information**, and it is the one the whole length-generalization question turns on. Self-
attention is permutation-equivariant: feed the two-token prefixes `( )` and `) (` through a positionless
attention block and the query at the second position forms the same unordered bag of key-value dot products
in both cases, so its output is identical and the head cannot tell a well-formed prefix from an ill-formed
one. For a language where `( )` is valid and `) (` is not, that model is a bag of brackets and fails
outright. So I inject position with a **learned absolute positional embedding**: a second
`nn.Embedding(max_len, hidden)` indexed by integer position, added to the token embedding.

Here is where I expect the architecture to struggle. During training the model only ever sees positions up to the training maximum — length 64
in two of the three environments. Only the position rows seen in training receive gradient; the rows for
positions beyond it — 65–96, 97–128, and the brutal 128–256 of the length-OOD probe — sit at random
initialization. So at OOD time the model attends over positions whose codes it never optimized, and
whatever position-dependent computation it learned in-distribution ("look back to the matching opener using
the relationship between my position and its position") is, in the OOD range, expressed in terms of noise.

The fraction of an OOD string living in untrained coordinates tells me how badly to expect each environment
to break. In the length-OOD environment (train ≤64), a mid-range 192-token string has positions 65–191 —
about 127 of 192, ~66% — carrying untrained random codes, and the query at the tail, exactly where the
deepest still-open brackets must be matched, sits in that untrained regime. For `dyck-k2-m3` (train ≤64,
OOD 65–96) an 80-token string has only positions 65–79 untrained, under 20%. So the damage should scale
with the train→test gap, roughly `(L − L_train)/L`, largest exactly on `dyck-length-ood`. I cap `max_len`
at 1024 so the model never indexes out of bounds, but a large table does not help — padding it with more
never-visited rows changes nothing about the ones OOD evaluation lands on.

For the attention stack I use `nn.TransformerEncoderLayer` with a causal mask — multi-head self-attention,
a position-wise feed-forward, residuals and layer norm — and set `norm_first=True` (pre-norm). Post-norm
renormalizes the identity path every block and needs a warmup ramp to keep early gradients through a deep
stack from blowing up; with only 100 gradient steps per environment I cannot spend steps taming the
optimizer, and pre-norm leaves a clean residual highway so the very first steps move the weights usefully.
`dropout=0.0`: with 8000 strings, a tiny vocabulary, and under one epoch (below), there is no overfitting to
regularize against, and dropout would only inject variance into the long-range attention paths I need crisp.
`activation="gelu"`, `dim_feedforward = 4 * hidden = 256`.

Depth and heads I fix by representational budget. The self-attention result for bounded-depth Dyck says a
logarithmic number of layers tracks depth-`m` nesting; `m ≤ 5` here, so `log2 5 ≈ 2.3` — 4 layers is
comfortably above the floor with margin for the finite-precision, finite-steps reality the clean
construction ignores. Hidden width is `config.hidden_dim = 64`, head dimension `64 / 4 = 16`, enough to
carry a key distinguishing the `2k + 2 ≤ 18` symbol types with room to spare — the positions will be the
bottleneck, not the heads.

The optimization budget is severe enough to matter: batch 64 over 100 steps touches `64 × 100 = 6 400`
example-slots against 8 000 strings, barely under one pass. That reinforces the pre-norm, zero-dropout,
no-warmup choices — under one epoch there is nothing to overfit — and it sharpens the read: if
in-distribution accuracy comes out high despite under one epoch, the attention prior is doing real work; if
it comes out low, the failure is a *fitting* failure from too few steps, not a length failure.

I build the causal mask myself: `torch.triu(torch.full((T, T), -inf), diagonal=1)` puts `−inf` on every
entry where query `i` looks at key `j > i`, so after softmax each position attends only to positions `≤ i`.
The `diagonal=1` is load-bearing — `diagonal=0` would mask the diagonal too and forbid every position from
seeing itself. I pass `is_causal=True` for the fast path. With `batch_first=True` the `[B, T, 64]` tensor
flows straight through embedding → encoder → head without any transpose, and the head projects `64 → vocab`
to yield the `[B, T, vocab]` logits the harness's per-position cross-entropy and next-valid-set evaluator
consume.

The parameter budget matters only against the 500 000 ceiling. Four encoder layers dominate at ~200k
(each layer ≈50k: ~16.6k attention projections, ~33k feed-forward, LayerNorms); the `max_len = 1024`
position table adds `1024 × 64 = 65 536`, the single biggest non-layer term; embedding and head are a few
hundred each. For `dyck-k2-m3` that sums to about 266k — comfortably under 500k, and it is the position
table, not the attention stack, that nudges it up.

This is why the Transformer is the right thing to run first: it carries the strongest in-distribution prior,
so an OOD collapse isolates *generalization across length* from *fitting* — the model can represent the
matching, it just cannot carry the representation past the positions it was trained on. And it sets up the
contrast for what follows: a recurrent model has no position table at all, updating state by the same
transition regardless of absolute index, so if recurrence length-generalizes where this does not, the
position table is implicated.

What I expect to watch. In-distribution token accuracy should be high but per-string accuracy weak — not a
contradiction but arithmetic: a length-`L` string with per-position error `ε` survives at roughly
`(1 − ε)^L`, so even 0.96 token accuracy on an 80-token string gives `0.96^80 ≈ 0.04` strings correct. On
the OOD split `ood_token_acc` should sag, worst on `dyck-length-ood` where the untrained-position fraction
is largest, and `ood_string_acc` should pin near zero everywhere. `dyck-k8-m5`, with 8 bracket types and
depth 5, is the hardest *content* — more closers to disambiguate, deeper stacks — so I expect its accuracies
lowest even in-distribution. If absolute positions are the bottleneck the signature is clean: `ood_token_acc`
sags furthest where extrapolation is largest while `id_token_acc` stays high. Should both come out poor
instead, the read is wrong and the next move is about capacity, not position. The full module is in the answer.
