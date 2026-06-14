The untied run did exactly what I expected of a capacity/regularization move and no more, and the number
tells me precisely what the next rung has to be. Untied landed at validation loss 2.3058, with WikiText-2
perplexity 45.70 and LAMBADA 71.11, ARC-Easy 54.80, HellaSwag 33.05. Reading that against the prediction
I closed on: I bet untying would *free* output-head capacity in an abundant-data regime and give a modest
drop, and what I got is a clean, well-behaved floor — but a floor, not a leap. The decisive line in my
own step-1 close was the falsification clause: "if the untied model only matches or barely beats tied,
that itself is the diagnosis that the next move must *add* signal rather than merely *free* existing
parameters." That is the regime I am in. Untying decoupled the two ends and let the classifier specialize,
but it added *no new information* to the representation — every token still enters as the same vector
regardless of what preceded it. The 2.3058 is the cost of an order-blind input. So the next move is not
another capacity reshuffle; it is to inject a signal the token table structurally cannot carry.

What signal? The cheapest piece of context a token embedding throws away is local order. A plain token
embedding gives the *same* vector for a token no matter what came before it; the body has to reconstruct
"the previous token was X" from the residual stream every time. Adding `n`-gram features — at minimum the
bigram, the (previous token, current token) pair — puts cheap left-context back into the embedding, and
that has historically bought a few points. That is exactly the kind of new signal untied could not
provide: untied was about *which* matrix scores the output; this is about *what* the input vector knows.
So the bigram is the natural step-2 target. But the moment I try to embed pairs, I hit a wall the
single-token table never had.

The wall is combinatorial. A literal bigram embedding table needs one row per distinct ordered pair, and
the number of pairs is the square of the vocabulary: for `vocab_size ≈ 50304` that is about
`2.5 × 10^9` rows. At `n_embd = 1024` that is a table on the order of `10^12` parameters — not large,
unstorable. So unlike the unigram table, where I could in principle afford one row per token, the bigram
feature *forces* me off the "one row per feature" model entirely. That is the real constraint, and it is
the one to design around: I need a representation whose parameter cost is bounded and roughly
*independent* of how big the true feature space is, even when that space is `vocab_size²`.

The tool for "shrink a huge index space into a bounded one" is feature hashing — the hashing trick. Fix
a hash `h` from the feature space into `{0, …, B−1}`, give the table `B` rows, and a feature `w` (here a
bigram) gets the row `E[h(w)]`. No dictionary: I compute the hash on the fly, so I never enumerate the
`vocab_size²` pairs. Memory is `B · n_embd` regardless of how many distinct pairs exist; `B` is now a
knob I control instead of a number the data forces on me. That solves the storability problem outright.

But it buys that with collisions, and I should be honest about how bad they are. Take one feature `w_0`
hashed to some bucket. Any other feature avoids that bucket with probability `(B−1)/B = 1 − 1/B`; the
other features are roughly independent, so `w_0` collides with nobody with probability `(1 − 1/B)^{T−1}`
where `T` is the number of distinct features, and the collision probability is `p_col = 1 − (1 − 1/B)^{T−1}
≈ 1 − exp(−T/B)`. Feel what that means for bigrams. Even being generous and saying only `T = 10^8`
distinct bigrams actually occur, with `B = 10^6` buckets I get `p_col ≈ 1 − exp(−100) ≈ 1`: essentially
every feature collides with something. To push collisions down toward zero I would have to push `B` back
toward `T`, throwing away the whole memory win. So a single hash is a genuine tension: bounded memory and
low collisions pull against each other, and one hash function cannot have both. And there is a subtler
problem in the unsigned sum — when several features land in one bucket their contributions add in the same
direction, so a busy bucket is systematically inflated (biased, not just noisy). The standard remedy is a
random `±1` sign per feature so collision terms cancel in expectation: the signed kernel is unbiased
(`E[⟨x,x'⟩_φ] = ⟨x,x'⟩`, the diagonal `ξ(j)² = 1` giving the true inner product, the off-diagonal
`E[ξ(j)ξ(l)] = 0`), with variance `O(1/B)`. Random signs fix the *bias* of collisions, not the *fact* of
them — two genuinely important pairs still share one row.

Here is the part that decides the design. The general cure for "important features collide" would be to
*learn a better hash* — but the hash codomain is discrete, the assignment is frozen, and there is no
gradient that can nudge a feature out of a bad bucket. That route is dead. What I actually need is not a
learned hash but the *effect* of one: important pairs kept distinct, unimportant ones allowed to share,
obtained without differentiating through a discrete map. Two moves give it. First, use `k` independent
hashes into `{0, …, B−1}` and describe a feature by the *tuple* of buckets; two features truly collide
only if they agree on all `k`, which behaves like one hash with range `B^k`. Re-run the birthday number
with `B^k`: at `T = 10^8`, `B = 10^6`, going from `k = 1` to `k = 2` drops the per-feature collision
probability from `≈ 1` to `1 − exp(−10^8/10^{12}) ≈ 10^{−4}`, paid not in `B^k` parameters but in `k`
lookups into the same shared pool. Second, combine the `k` looked-up rows by a *learnable* per-feature
weight vector `p_w ∈ R^k`, `ê_w = Σ_i p_w^i H_i(w)`: these scalars are continuous, so features that
collide on one hash are pulled apart through the others, and `p_w → 0` for an unimportant feature both
mutes it and removes it from the effective collision set (implicit feature selection). The total cost is
`B · n_embd + vocab_size · k`, against `vocab_size² · n_embd` for the literal table, and both the
standard table (`B = T`, identity hash) and the plain hashing trick (`k = 1`, unit weights) fall out as
corners — so this is the parameterized object that has both as special cases, with the interesting
interior at small `k` and `B ≪ vocab_size²`.

Now specialize to *this* harness, because that is where it has to run, and the harness is narrower than
the full construction. It gives me one embedding module — `forward`, the tied/untied head hook, the
position-count hook, and the per-layer injection hook `get_value_embed(layer_idx)`, whose return is added
to the *residual stream* before each block. It does not give me a `vocab_size × k` per-feature weight
matrix or a place to combine several component rows per feature. So I collapse the general construction
onto the degrees of freedom the harness actually exposes. The component pool becomes a single bigram
table; the `k`-hash collision resistance becomes a sufficiently large `B`; and the per-feature importance
weights `p_w` collapse to the one exposed knob — *how much* bigram signal to mix in, and at *which
depth*.

Pick the table size first, which is the birthday tradeoff again. The number of bigrams that *actually
occur* in FineWeb is far below `vocab_size²` and is itself heavily Zipfian — a relatively small set of
frequent bigrams carries most of the mass — so I do not need `B` near `vocab_size²`; I need `B` large
enough that the *frequent* bigrams rarely collide with each other. A table of `B = 5 · vocab_size` rows —
five times the unigram vocabulary — is the working choice: a small constant multiple of the token table I
already have, and with the frequent-bigram count well below `B`, meaningful collisions are rare precisely
because of the Zipfian token distribution. So `bigram_vocab_size = 5 · vocab_size`.

The hash itself must be cheap and GPU-friendly — it runs on every position of every batch — so no
modular-exponentiation niceties, just integer arithmetic the tensor engine likes. Take the current token
id `c` and the previous token id `p`, both int32. A bare `c XOR p` would collide structurally (the XOR of
two small ids is small, so adjacent bigrams pile into low buckets), so first spread each id across the
int32 range by multiplying by a large odd constant — *different* constants for the two positions so the
pair is order-sensitive, `(c, p)` hashing differently from `(p, c)` — then XOR, then reduce by a modulus:
`index = (r1 · c XOR r2 · p) mod (B − 1)`, with `r1 = 36313`, `r2 = 27191`. Position 0 of a sequence has
no previous token, so I send it to a single reserved bucket — the index `B − 1` that the modulus never
produces — so the no-context case gets a clean slot instead of being faked from whatever id happened to
precede the start.

Now combine and inject. The full construction's `k`-row weighted sum and `vocab_size × k` importance
matrix are not expressible here, so I use one hashed bigram row per position and learn *where in depth* to
trust it. I zero-initialize the bigram table and gate its injection with a small learnable scalar per
layer (`bigram_lambdas`, one per layer, init 0.1). This is the harness-sized analogue of the per-feature
importance weight, collapsed to the exposed degree of freedom: how much bigram signal to mix at each
depth. The zero-init plus small initial gate is the crucial safety property and it directly answers the
worry the untied run left me with — I do not want to *risk* the clean 2.3058 floor with a new component
that could be net-harmful early. At step 0 the bigram contribution is identically zero, so the
augmentation cannot hurt the already-tuned token+position+untied base; training then grows the table and
the per-layer gates only where gradients ask for the signal. That is the same conservative discipline I
used for untied's zero-init output matrix, applied to a *new* additive signal rather than a reshuffled
one.

One harness-specific point I have to be honest about, because it differs from the clean story. In the
general construction (and in the value-residual lineage) one would inject a per-token signal into the
*value* path inside attention, so it rides the layer's learned attention matrix. This harness does not
expose the value path — `get_value_embed(layer_idx)` is added to the *residual stream* `x` before the
block (`x = x + ve`). So the bigram vector here is a residual-stream additive bias at each layer, not a
value injection; it changes what content flows into the block, not how attention routes. That is a weaker
insertion point than the value path, but it is the only one the harness gives, and it is enough: a gated,
per-position bigram residual added before every block lets each layer read local-order information it
otherwise would have to reconstruct. I keep the output projection — here untied from step 1's reasoning,
but in the literal step-2 fill the head is tied back to `wte` because this rung's question is the bigram
signal, not the tie — and I leave `forward`'s token+position stream itself unchanged; the bigram lives
purely in the per-layer injection. (The signed-hash trick would make collisions unbiased, but I
deliberately keep the simpler unsigned table; the collision budget here is the `5 · vocab_size` table, the
Zipfian active-bigram distribution, the zero init, and the learned per-layer gates.)

So the delta from untied is concrete: where untied changed only *which* matrix scores the output and left
every input vector order-blind, this rung adds a hashed (previous, current) bigram row, gated per layer
over a zero-initialized `5 · vocab_size` table, injected on the residual stream before each block. The
full scaffold module is in the answer. Now the falsifiable expectations against untied's 2.3058. The bet
is that local order is real, free information the token table structurally cannot carry, so the bigram
signal should *lower* validation loss below 2.3058 — a real drop, because this *adds* signal rather than
freeing parameters, which is exactly the move my step-1 diagnosis said was needed. I expect the
downstream accuracies to move *up* more visibly than untied's did, because bigram context is the kind of
local cue that helps tasks like ARC-Easy and HellaSwag where short-range co-occurrence matters — I would
watch ARC-Easy in particular, since untied sat at 54.80. The risk, and the thing I am genuinely unsure
about, is the *injection point*: because the bigram enters the residual stream rather than the value path,
its benefit may be blunted, and the per-layer gates may learn to trust it only at a few depths — if the
learned `bigram_lambdas` stay near their 0.1 init everywhere, that is the tell that the residual-stream
insertion is too weak to exploit the signal, and the next rung should look for a richer per-layer
injection rather than a richer feature. The other risk is that hash collisions among frequent bigrams add
noise that the gates cannot fully suppress, in which case LAMBADA (long-range, where a noisy local signal
can mislead) could fail to improve even if WikiText-2 and validation loss do. So I expect validation loss
below untied's 2.3058 with the perplexities tracking it and the downstream numbers nudging up; if instead
the gain is thin or the downstream is flat, the diagnosis for the next rung is that the *content* of the
injected signal matters less than *how richly it is injected per layer* — which points straight at a
per-layer, depth-aware value-style embedding rather than a single hashed feature.
