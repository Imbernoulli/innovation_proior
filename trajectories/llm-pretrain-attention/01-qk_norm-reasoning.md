I want to start from the cheapest intervention in the attention layer, because the rest of the run is
frozen and I want to understand the substrate before I touch position encoding. But "cheapest" only means
something once the expensive options are laid out. Everything a transformer does with order and relevance
passes through this one editable module, and there are really only a few distinct levers. I could change
*how order enters* the score — the learned absolute `wpe` is the default, and I could swap it for something
relative that makes the logit depend on the offset between two tokens rather than their absolute slots. I
could change *how the score is scaled* — the fixed `1/sqrt(d_k)` between the dot product and the softmax. I
could change *the shape of the distribution itself* — the single causal softmax that turns logits into a
convex average of values. Or I could restructure the heads: their count, dimension, how q, k, v are split
and recombined. Four axes — position, score scale, distribution shape, head structure — and they are
neither equally cheap nor equally isolated.

I start with score scale for a specific reason: of the four it is the only one that touches neither
position nor head structure nor the distributional form, so whatever it buys is a *clean* measurement of a
single effect. There is an asymmetry that fixes the order. The score-scale fix is position-agnostic — it
touches only the magnitudes feeding the softmax, nothing about which offset is scored — whereas a position
fix is *not* scale-agnostic: it still feeds the same softmax whose range I have not pinned. So stabilizing
the score first cannot spoil a later position measurement, but touching position first would leave a scale
confound sitting under every subsequent step. If I had ripped out `wpe` at step 1 and the position change
helped, I could not tell how much of the gain a drifting score was masking or inflating; if it did not
help, I could not tell whether the scheme was wrong or a saturated softmax simply ate the signal. Fix the
axis that is clean and independent, then read the entangled ones against a substrate I understand.

So, the score. There is a quiet problem in it, independent of position: the logit can drift in magnitude
over training, and the softmax does not forgive that. The `1/sqrt(d_k)` factor is a variance argument — if
q and k have zero-mean unit-variance entries, `q^T k = sum_i q_i k_i` is a sum of `d_k` zero-mean terms
with variance `d_k` and standard deviation `sqrt(d_k)`. Here `d_k = 1024/16 = 64`, so `sqrt(d_k) = 8`: the
raw logit spreads about ±8 before any learning. That matters because of what the softmax does with it. Over
`T = 1024` keys with logit standard deviation σ, the largest sits about `σ·sqrt(2 ln T) = 3.72σ` above the
mean; at σ = 8 that gap is ~30 units, `exp(30)` is astronomical, and the softmax dumps essentially all its
mass on one key from the start. Dividing by 8 drops σ to 1, the top-vs-mean gap to ~3.7, and the softmax
begins broad and responsive. Correct — *at initialization*.

But unit variance is an init statement, and the block is pre-norm: `x <- x + Attn(LN(x))`. The LayerNorm
normalizes the *input* to attention; what it does not normalize is `q = W_q·LN(x)`. `W_q` is free and
trainable — weight-decayed but not norm-bounded — and there is a concrete incentive to grow it: a sharper
attention pattern lowers loss on the many tokens whose prediction hinges on one earlier token, and the
cheapest way to sharpen a softmax is to scale up its inputs, i.e. q and k. Weight decay pulls `W_q` toward
zero with force `0.1·W_q`; the loss gradient pulls it up whenever sharper attention helps; the run settles
where those balance, at a norm set by how much sharper attention happens to help *this data* — an
uncontrolled quantity with no reason to match the unit-variance value `1/sqrt(d_k)` assumes. Decay bounds
the growth; it does not pin the output scale. So per-head q and k norms creep up. Write the growth as
`c_q, c_k`: the logit scales by `c_q c_k` while the divisor stays at `1/8`. If both norms merely double, the
effective σ quadruples — from ~1 back toward ~4, dragging the softmax into the saturated regime the
`1/sqrt(d_k)` was designed to avoid, and arriving *late* in training, exactly when I want clean gradients to
keep refining the pattern.

What does "saturated" cost in gradient terms? Two keys with logits `s` and `0`; the weight on the first is
`p = sigmoid(s)` and its gradient with respect to the logit is `p(1-p)`, the softmax Jacobian diagonal. At
`s = 2` (healthy, post-scaling) `p = 0.881`, `p(1-p) = 0.105`. At `s = 8` (the same key after a ×4 scale
drift) `p = 0.99966`, `p(1-p) = 0.000335` — a collapse of ~313×. So merely doubling `||q||, ||k||` costs
more than two orders of magnitude of gradient on the attention logits: the score freezes, can no longer
reorder which key wins, and does so silently — loss keeps dropping because the MLPs and value path keep
learning while the attention pattern has stopped being trainable. This is the slow failure, and it bites
hardest late, when I most want attention to keep sharpening on the right tokens.

So the score scale is a moving target the fixed `1/sqrt(d_k)` cannot track. I want the logit to depend only
on the *directions* of q and k — the angle attention is supposed to measure — not on how big the
projections have grown. The clean way to make a dot product direction-only is to normalize before taking
it: `q_hat = q/||q||`, `k_hat = k/||k||`, so `q_hat^T k_hat = cos(angle)` in `[-1, 1]` no matter how
`W_q, W_k` evolve — `c_q, c_k` cancel exactly. Saturation-creep removed at the root.

But that over-corrects, and the arithmetic says so. A cosine is in `[-1, 1]`, so the whole logit vector
spans at most 2; through a softmax the sharpest two-position contrast is `exp(1)/exp(-1) = 7.39`, capping
mass on one key against a single opposite competitor at ~88%, and far lower against 1023. I have traded
saturation for a softmax that *cannot concentrate* even when it should. The fix is to reintroduce a *single
deliberate* scale `g` replacing `1/sqrt(d_k)`: logit `= g·q_hat^T k_hat`, so sharpness is set on purpose by
one number per head rather than leaking in through weight growth. That is query-key normalization in its
cleanest form: L2-normalize q and k along the head dimension, then scale by a learned `g`.

That is the blank-page derivation. The edit surface gives me only the `CausalSelfAttention` body, and two
constraints bind me. First, the forward path runs the fused `scaled_dot_product_attention`, which applies
its own `1/sqrt(d_k)` and softmax internally; I cannot slip a learned `g` inside the fused kernel without
abandoning it and paying an explicit `T×T` map on every one of the 24 layers. Second, the per-vector
rescale I want on q and k along the head dimension is exactly RMSNorm, which the framework already exposes.

There is a fork there worth resolving, since there is also a mean-subtracting LayerNorm. LayerNorm is
`(x - mean)/std` — it removes the projection onto the all-ones direction before rescaling; RMSNorm is
`x/rms(x)`, a pure rescale that leaves direction untouched. I derived the target as "make the logit depend
only on the direction of q and k," meaning the direction the projection actually produced. RMSNorm
preserves that exactly and only fixes length; LayerNorm would additionally rotate q and k by projecting out
their mean, measuring the angle between *centered* vectors — a different similarity with no clean tie to the
score I set out to stabilize. Centering solves a problem I do not have at the cost of distorting the very
angle I want the logit to depend on. So RMSNorm is the faithful primitive, not just the convenient one.

Now line up what RMSNorm realizes against the L2-plus-`g` I derived. RMSNorm is `x/rms(x)` with
`rms(x) = ||x||/sqrt(d)`, i.e. `sqrt(d)·x/||x||` — L2 normalization times the fixed constant `sqrt(d)`.
Apply it to q and k (`d = 64`): the pre-scale dot product is
`(sqrt(d) q_hat)^T (sqrt(d) k_hat) = d·cos(angle) = 64·cos`. The fused kernel then divides by
`sqrt(d_k) = 8`, so the realized logit is `8·cos(angle)` — a cosine scaled by the *constant* 8. Does ±8 give
back usable sharpness? The extreme contrast is `exp(8)/exp(-8) = exp(16) ≈ 8.9×10^6`, plenty to concentrate
mass when the cosine genuinely separates a key, and nominally the same ±8 range `1/sqrt(d_k)` produced at
init. So the fixed `sqrt(d_k)` quietly plays the role my learned `g` would have, pinned at 8 rather than
tuned. The version that fits: RMSNorm q and k along the head dimension, and leave the fused kernel, causal
mask, and `wpe` exactly as is. Two lines added, no new parameters, no learnable scale, no change to
position.

That pins the character of the edit. At init the projected q, k entries are ~unit-variance, so
`||q_head|| ≈ 8`, and the vanilla realized logit is already ≈ `8·cos`; RMSNorm forces `||q_head|| = 8`
exactly and gives `8·cos` by construction. The two coincide at step zero up to the fluctuation of
`||q||, ||k||` around 8 — the edit is nearly a no-op at init and only begins to bite as the norms drift,
which is exactly the signature a stability fix should have: it does not disturb the carefully-tuned starting
point, it clamps the drift the fixed scaling silently allows.

I should be honest about what I give up against the clean derivation. The learned per-head `g` is gone; the
sharpness ceiling is pinned at `sqrt(d_k) = 8` rather than tuned, so a head that would genuinely want a
sharper-than-±8 range cannot get it, and I cannot tell from this step alone whether any head wants that. But
the property I came for holds: the logit scale is invariant to q/k magnitude drift, because RMSNorm strips
the magnitude before the product. Saturation-creep is fixed; only the deliberate-sharpening upside is left
on the table. And keeping `use_pos_emb = True` is forced, not chosen: this intervention says nothing about
position, so the learned `wpe` stays exactly as in the default. I am changing one thing — the score's
robustness to q/k norm drift — and nothing else.

A couple of details decide faithfulness. RMSNorm must be per head, along `head_dim`, *after* the reshape to
`(B, n_head, T, head_dim)` — normalizing across the full `n_embd = 1024` before splitting would mix all 16
heads' coordinates into one normalization and destroy the per-head direction the cosine is supposed to
measure. And it goes on q and k only, never v: v carries the content averaged into the output, the
saturation problem lives entirely in the logit, and normalizing v would distort what the head reads out. The
overhead is nil — the two extra normalizations touch `2·B·n_head·T·head_dim` elements against
`B·n_head·T^2·head_dim` for the attention product, a ratio of `2/T`, under a quarter of a percent — which
matters because the loop is frozen and I cannot afford to change the training budget.

So step 1 is the minimal, position-agnostic edit: two `F.rms_norm` calls on the per-head q and k right
before the attention product, everything else untouched. It is a *stability* fix, not a capacity or
inductive-bias fix — it removes a failure mode without adding any information the model did not already
have. So I expect a modest improvement over vanilla: cleaner late-training gradients, a slightly lower and
steadier `val_loss`, downstream roughly tracking the language-modeling gain rather than jumping on its own.
What it cannot touch is the thing the background kept circling: position is still absolute and additive, so
the relative structure of language — that a dependency is about the offset `m - n`, not the absolute slots —
is still left for the network to reconstruct indirectly through `wpe` cross terms. That is a
*representational* limitation, and RMSNorm-on-q-k does nothing about it. So this should land as the
*weakest* of the interventions I have in view — a real but small win over vanilla, beaten by anything that
fixes how order enters. A `val_loss` in the low 2.3s is an in-domain perplexity around ten
(`exp(2.3) ≈ 10`), and I would expect WikiText-2 and especially LAMBADA to sit well above that, since
LAMBADA's last-word prediction leans on exactly the long-range relative dependency this step leaves
untouched. The next move is already set: leave the score where I have stabilized it, and fix *position* —
make the logit depend on the offset `m - n`, not the absolute index.
