I am going to start from the cheapest possible intervention in the attention layer, because the rest of
the run is frozen and I want to understand the substrate before I touch position encoding. The default
attention is the textbook one: project to q, k, v, form the logit `q_m^T k_n / sqrt(d_k)`, causal-mask,
softmax, weight the values. Position comes from the learned `wpe` table added before the stack. Before I
go reorganize how order enters, there is a quieter problem in the score itself that I keep running into
when I pretrain at this scale, and it is independent of position: the logit can drift in magnitude over
training, and the softmax does not forgive that.

Let me be precise about why the scale matters. The whole point of the `1/sqrt(d_k)` factor is a
variance argument. If the entries of q and k are roughly independent with unit variance, the dot product
`q^T k = sum_{i=1}^{d_k} q_i k_i` is a sum of `d_k` zero-mean terms, so its variance is about `d_k` and
its standard deviation about `sqrt(d_k)`. At `d_k = 64` here (1024 / 16 heads) that is a spread of ±8
before any learning. Feed logits of that size into a softmax and you are already near the saturated
regime: one logit a few units above the others takes essentially all the mass, and the gradient through
the softmax — which scales like `p_i(1 - p_i)` on the diagonal — collapses toward zero. Dividing by
`sqrt(d_k)` rescales the logit back to unit standard deviation so the softmax starts in its responsive
region. That is the textbook justification, and it is correct *at initialization*.

But it assumes q and k have unit-variance entries, and that assumption only holds at initialization.
The block here is pre-norm: `x <- x + Attn(LN(x))`. The LayerNorm sits on the *input* to attention, so
the activations entering `c_attn` are normalized. What is *not* normalized is what `c_attn` does to them.
`q = W_q · LN(x)` — and `W_q` is a free, trainable, weight-decayed-but-not-norm-bounded matrix. As
training proceeds the optimizer is free to grow the rows of `W_q` and `W_k`, and there is a real
incentive to: a sharper, more confident attention pattern lowers loss on many tokens, and the cheapest
way to sharpen the softmax is to scale up the logits, which means scaling up q and k. So the per-head q
and k norms creep upward over the run. The fixed `1/sqrt(d_k)` was tuned for unit-variance q, k; once
their norms have, say, doubled, the effective logit standard deviation has quadrupled, and the softmax
is back in the saturated, low-gradient regime I was trying to avoid — except now it happens *late* in
training, exactly when I want clean gradients to keep refining. This is a slow, silent failure: the
network trains, the loss goes down, but the attention logits are drifting toward saturation and the late
gradients get noisier and the run gets less stable, especially if I were ever to push the learning rate.

So the score scale is not a one-time constant; it is a moving target that the fixed `1/sqrt(d_k)` cannot
track. The question is how to make the logit scale *invariant* to the magnitude of q and k, so it
depends only on their *directions* — on the angle between query and key, which is what attention is
supposed to be measuring — and not on how big the projection matrices have grown.

The clean way to make a dot product depend only on direction is to normalize the vectors before taking
it. If I divide q and k each by their length, `q_hat = q / ||q||`, `k_hat = k / ||k||`, then
`q_hat^T k_hat = cos(angle)` lives in `[-1, 1]` no matter how `W_q`, `W_k` evolve. The logit is now a
pure cosine similarity per head, bounded, and completely insensitive to the magnitude drift I was
worried about. That removes the saturation-creep failure at the root: the softmax input range can no
longer blow up just because the weights grew.

Now I have to be careful, because I have over-corrected. A cosine similarity is in `[-1, 1]`, so the
*entire* logit vector, before masking, spans at most a range of 2. Push that through a softmax and the
attention can never sharpen: `exp(1)/exp(-1) ≈ 7.4` is the most contrast two positions can have, so the
model literally cannot place 95% of its mass on one token no matter how confident it should be. I have
traded saturation for the opposite pathology — a softmax that can't concentrate. The network needs the
*range* of the logits to be a learnable quantity, decoupled from the *drift* of the q/k norms. So after
normalizing direction, I reintroduce a scale, but a *single learned* one, `g`, replacing the fixed
`1/sqrt(d_k)`: logit `= g · q_hat^T k_hat`. Now `g` controls how sharp attention is allowed to get, the
optimizer can raise it if confident attention helps, and crucially `g` is one number per head (or per
layer), not a side effect of the full `W_q`/`W_k` magnitudes — so the sharpness is learned deliberately
instead of leaking in through weight growth. This is the query-key normalization idea in its cleanest
form: L2-normalize q and k along the head dimension, then scale by a learned `g`.

That is what I would *derive*. But I should reconstruct what the task's edit surface actually lets me do,
because the harness fixes the loop and I only get the `CausalSelfAttention` body, and there is a subtle
gap between the clean derivation above and the form that fits cleanly into this scaffold. Two things
constrain me. First, the forward path here goes through PyTorch's fused
`scaled_dot_product_attention` (SDPA) for speed — that kernel internally applies its own `1/sqrt(d_k)`
scaling and does its own softmax; I do not get to insert a custom `g` *inside* the fused kernel without
giving up the fused path. Second, the normalization I want on q and k is a per-vector rescale along the
head dimension, and the loop already exposes a primitive for exactly that: RMSNorm.

Let me look at what RMSNorm does versus the L2-normalize-plus-`g` I derived. RMSNorm of a vector
`x ∈ R^{d}` is `x / rms(x)` where `rms(x) = sqrt(mean(x_i^2)) = ||x|| / sqrt(d)`. So RMSNorm is
`sqrt(d) · x / ||x||` — it is L2 normalization up to the *fixed* factor `sqrt(d)`. Applying RMSNorm to q
and k both, the logit becomes `(sqrt(d) q_hat)^T (sqrt(d) k_hat) = d · q_hat^T k_hat = d · cos(angle)`,
and then SDPA still divides by `sqrt(d)`, giving `sqrt(d) · cos(angle)`. So the realized logit is
`sqrt(d_k) · cos(angle)` — a cosine similarity scaled by the *constant* `sqrt(d_k) = 8`. That recovers a
*usable* logit range (±8, the same nominal range the original `1/sqrt(d_k)` was designed to produce at
init) without any learned `g` at all. The constant `sqrt(d_k)` plays the role `g` would have played,
fixed rather than learned. So the version that fits this scaffold is: **RMSNorm q and k along the head
dimension, leave everything else (SDPA, causal mask, `wpe`) exactly as is.** Two lines added to the
forward pass; no new parameters, no learnable scale, no change to position encoding.

I want to be honest with myself about what I am giving up relative to the clean derivation, because this
is the difference between the idea and the thing I can actually run here. The learned per-head `g` is
gone — the sharpness ceiling is pinned at `sqrt(d_k)` instead of being tuned by the optimizer. That is a
real omission: if confident attention would benefit from a sharper-than-±8 logit range, this version
cannot provide it. But it keeps the property I actually came for — the logit scale is now invariant to
the *drift* of the q/k magnitudes, because RMSNorm strips the magnitude out before the product. The
saturation-creep failure is fixed; only the deliberate-sharpening upside is left on the table. And the
keep-`use_pos_emb = True` decision is forced: this intervention says nothing about position, so the
learned `wpe` table stays exactly as in the default. I am changing *one thing* — the score's robustness
to q/k norm drift — and nothing else.

A couple of details worth pinning down so the edit is faithful. RMSNorm must be applied **per head**,
along the `head_dim` axis, *after* the reshape to `(B, n_head, T, head_dim)` — normalizing across the
full `n_embd` before splitting heads would mix heads together and destroy the per-head direction I want
to preserve. So the order is: project, split into heads, RMSNorm q, RMSNorm k, then attention. And I
apply it to q and k only, never to v — v carries the content that gets averaged into the output;
normalizing it would distort the values, and the saturation problem lives entirely in the *logit*, which
v is not part of. The dropout, the causal masking, the output projection all stay untouched.

So at step 1 the edit is the minimal, position-agnostic one: leave the entire loop and the learned
position table alone, and insert two `F.rms_norm` calls on the per-head q and k right before the
attention product (the distilled module is in the answer). It is the floor I want to start the ladder
from for a reason — it is the *only* one of the candidate attention changes that touches neither the
position scheme nor the head structure, so whatever it buys is purely the score-stability effect, with
nothing else confounded in.

Now reason about what this floor should do, because that is the entire point of running it first. The
intervention is a *stability* fix, not a *capacity* or *inductive-bias* fix. It removes a failure mode
(late-training logit-scale drift toward softmax saturation) without adding any new information the model
didn't have — position is still absolute-additive via `wpe`, exactly as in the default. So I expect a
*modest* improvement over an unmodified GPT-2 attention: cleaner late-training gradients, a slightly
lower and more stable validation loss, with the downstream metrics roughly tracking the language-modeling
gain. What I do *not* expect it to fix is the thing the prior-art survey kept circling: the position
signal is still absolute and additive, so the relative structure of language — that a dependency is about
the *offset* `m - n`, not the absolute slots — is still left for the network to discover indirectly
through the `wpe` cross terms. That is a *representational* limitation, not a stability one, and
RMSNorm-on-q-k does nothing about it. So my falsifiable expectation is sharp: this rung should land as
the *weakest* of the attention interventions, beaten by anything that actually fixes how order enters,
because it only stabilizes the score and leaves the absolute-position handicap fully in place. If the
later rungs that touch position do not beat this one, then the position story is wrong and the real bottleneck
was stability all along — but I do not expect that. The diagnosis is already pointed at the next step:
fix *position*, make the logit depend on the relative offset, not the absolute index.
