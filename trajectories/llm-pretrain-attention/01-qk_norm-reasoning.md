I am going to start from the cheapest possible intervention in the attention layer, because the rest of
the run is frozen and I want to understand the substrate before I touch position encoding. But "cheapest"
only means something once I have laid out what the expensive options even are, so let me map the surface
first. The one editable module is `CausalSelfAttention`, and everything a transformer does with order and
relevance passes through it, which means there are really only a few distinct levers I could pull. I could
change *how order enters* the score — the learned absolute `wpe` is the default, and I could swap it for a
sinusoidal table or for something relative that makes the logit depend on the offset between two tokens
rather than their absolute slots. I could change *how the score is scaled* — the fixed `1/sqrt(d_k)` factor
that sits between the dot product and the softmax. I could change *the shape of the attention distribution
itself* — the single causal softmax that turns logits into a convex average of values. Or I could
restructure the heads: their count, their dimension, how q, k, v are split and recombined. Four axes —
position, score scale, distribution shape, head structure — and they are neither equally cheap nor equally
isolated. The reason to start with score scale is that it is the only one of the four that touches neither
position nor head structure nor the distributional form, so whatever it buys is a *clean* measurement of a
single effect, uncontaminated by the others, and it hands me a stable floor to build the ladder on before I
start moving the more entangled parts.

The tempting alternative is to skip straight to position, because the background survey already flags the
absolute-additive `wpe` as the obvious weak point and I am fairly sure that is where the biggest single gain
lives. Let me walk that a few steps and see why it is the wrong *first* move rather than the wrong move.
Suppose at step 1 I ripped out `wpe` and dropped in a relative scheme. Whatever I measured, I would be
measuring it *through* a score path whose scale I have not characterized — the same logit that a drifting
`||q||, ||k||` can quietly push toward saturation. If the position change helped, I would not know how much
of the gain the score drift was masking or inflating; if it did not help, I would not know whether the
scheme was wrong or whether a saturated softmax was simply eating the signal before it could matter. The
two effects are confounded, and there is a concrete asymmetry that decides the order: the score-scale fix is
provably position-agnostic (it touches only the magnitudes feeding the softmax, nothing about which offset
is being scored), whereas a position fix is *not* scale-agnostic (it still feeds the same softmax whose
range I have not pinned). So stabilizing the score first cannot spoil a later position measurement, but
touching position first would leave a scale confound sitting under every subsequent rung. That asymmetry is
the whole argument for the ordering: fix the axis that is clean and independent, then read the entangled
ones against a substrate I understand.

The default attention is the textbook one: project to q, k, v, form the logit `q_m^T k_n / sqrt(d_k)`,
causal-mask, softmax, weight the values. Position comes from the learned `wpe` table added before the
stack. Before I reorganize how order enters, there is a quieter problem in the score itself that I keep
running into when I pretrain at this scale, and it is independent of position: the logit can drift in
magnitude over training, and the softmax does not forgive that.

Let me be precise about why the scale matters, because the whole intervention lives or dies on this
argument. The point of the `1/sqrt(d_k)` factor is a variance argument. If the entries of q and k are
roughly independent with zero mean and unit variance, the dot product `q^T k = sum_{i=1}^{d_k} q_i k_i` is a
sum of `d_k` independent zero-mean terms each with variance 1, so the sum has variance `d_k` and standard
deviation `sqrt(d_k)`. Here `d_k = n_embd / n_head = 1024 / 16 = 64`, so `sqrt(d_k) = 8`: the raw logit has
a spread of about ±8 before any learning. That number matters because of what the softmax does with it.
Over a context of `T = 1024` keys, if the logits are drawn with standard deviation σ, the largest of them
sits, in expectation, about `σ · sqrt(2 ln T)` above the mean, and `sqrt(2 ln 1024) = 3.72`, so the top
logit is roughly `3.72 σ` above the pack. At σ = 8 that is a gap of about thirty units to the mean; `exp(30)`
is astronomical, so the softmax dumps essentially all its mass on one or two keys and the distribution is
one-hot from the very start. Divide by `sqrt(d_k)` and σ drops back to 1, the top-vs-mean gap becomes ~3.7
units, and the softmax begins life as a broad, responsive distribution. That is the textbook justification,
and it is correct — *at initialization*.

But the unit-variance assumption is a statement about init, and the block here is pre-norm:
`x <- x + Attn(LN(x))`. The LayerNorm sits on the *input* to attention, so the activations entering `c_attn`
are normalized to unit scale. What is *not* normalized is what `c_attn` does to them. `q = W_q · LN(x)`, and
`W_q` is a free, trainable matrix — weight-decayed, yes, but not norm-bounded in any way that pins the
output scale. As training proceeds the optimizer is free to grow the rows of `W_q` and `W_k`, and there is a
concrete incentive to do exactly that: a sharper, more confident attention pattern lowers loss on the many
tokens whose prediction hinges on one earlier token, and the cheapest way to sharpen a softmax is to scale
up its inputs, which means scaling up q and k. One might hope weight decay stops this, but it does not stop
it where I need it to. Decay pulls `W_q` toward zero with a force proportional to
`weight_decay · W_q = 0.1 · W_q`, while the loss gradient pulls it *up* whenever a sharper attention lowers
the next-token cross-entropy; the run settles at the norm where those two forces balance, and that
equilibrium norm is set by *how much sharper attention happens to help this data* — an uncontrolled,
data-dependent quantity with no reason to coincide with the unit-variance value the `1/sqrt(d_k)` was tuned
for. Decay bounds the growth; it does not pin the output scale to the one value the score-scaling assumes.
So the per-head q and k norms creep upward over the run, toward whatever balance the data dictates.
Write the growth factors as `c_q` and `c_k`: the logit `q^T k` scales by `c_q · c_k`, while the `1/sqrt(d_k)`
divisor stays pinned at `1/8`. If both norms merely double, `c_q c_k = 4` and the effective logit standard
deviation quadruples — from σ ≈ 1 back up toward σ ≈ 4, dragging the softmax into the saturated regime the
`1/sqrt(d_k)` was designed to avoid, except now it arrives *late* in training, exactly when I want clean
gradients to keep refining the attention pattern.

Let me actually check what "saturated" costs in gradient terms, on the smallest example that shows it,
because I want a number and not a worry. Take two keys with logits `s` and `0`; the softmax weight on the
first is `p = sigmoid(s)`, and the gradient of that weight with respect to the logit is `p(1-p)` — the
softmax Jacobian on the diagonal. At `s = 2` (the healthy, near-init regime after the `1/sqrt(d_k)`
scaling), `p = 0.881` and `p(1-p) = 0.105`. Push the logit to `s = 8` (the same key after a ×4 drift in
effective scale), and `p = 0.99966`, so `p(1-p) = 0.000335`. The gradient has collapsed by a factor of
`0.105 / 0.000335 ≈ 313`. So a drift that merely doubles `||q||` and `||k||` costs more than two orders of
magnitude of gradient signal on the attention logits: the score effectively freezes, it can no longer
reorder which key wins, and it does so silently — the loss still goes down because the MLPs and the value
path keep learning, but the attention pattern has stopped being trainable. This is the slow failure I keep
hitting, and it bites hardest late in the run, when I most want attention to keep sharpening on the right
tokens rather than locking onto whatever it happened to saturate on first.

So the score scale is not a one-time constant; it is a moving target that the fixed `1/sqrt(d_k)` cannot
track, because it was calibrated for a `||q||, ||k||` that no longer hold once the weights have grown. The
question is how to make the logit scale *invariant* to the magnitude of q and k, so it depends only on their
*directions* — on the angle between query and key, which is what attention is supposed to be measuring — and
not on how big the projection matrices have grown.

The clean way to make a dot product depend only on direction is to normalize the vectors before taking it.
If I divide q and k each by their length, `q_hat = q / ||q||`, `k_hat = k / ||k||`, then
`q_hat^T k_hat = cos(angle)` lives in `[-1, 1]` no matter how `W_q`, `W_k` evolve. The logit is now a pure
cosine similarity per head, bounded, and completely insensitive to the magnitude drift — `c_q` and `c_k`
cancel exactly, because they factor straight out of the normalization. The saturation-creep failure is
removed at the root: the softmax input range can no longer blow up just because the weights grew.

But I have to check I have not over-corrected, and here the arithmetic says I have. A cosine similarity is
in `[-1, 1]`, so the *entire* logit vector, before masking, spans at most a range of 2. Push that through a
softmax and the sharpest contrast two positions can have is `exp(1)/exp(-1) = exp(2) = 7.39`. That means the
model cannot place more than `7.39/(7.39 + 1) ≈ 88%` of its mass on one key even against a single competitor
at the opposite extreme, and against 1023 competitors the achievable peak is far lower still. I have traded
saturation for the opposite pathology — a softmax that *cannot concentrate* even when it should. The network
needs the *range* of the logits to be a controllable quantity, decoupled from the *drift* of the q/k norms.
So after normalizing direction, I reintroduce a scale, but a *single deliberate* one, `g`, replacing the
fixed `1/sqrt(d_k)`: logit `= g · q_hat^T k_hat`. Now `g` sets how sharp attention is allowed to get — one
number per head (or per layer), rather than a side effect of the full `W_q`/`W_k` magnitudes — so sharpness
is learned on purpose instead of leaking in through weight growth. That is query-key normalization in its
cleanest form: L2-normalize q and k along the head dimension, then scale by a learned `g`.

That is what I would derive on a blank page. But the harness fixes the loop and hands me only the
`CausalSelfAttention` body, so I have to reconstruct what this edit surface actually lets me realize, and
there is a real gap between the clean derivation and the form that fits. Two constraints bind me. First, the
forward path goes through PyTorch's fused `scaled_dot_product_attention` for speed; that kernel applies its
own `1/sqrt(d_k)` internally and runs its own softmax, and I cannot slip a custom learned `g` *inside* the
fused kernel without abandoning the fused path and paying the memory of an explicit `T×T` map on every one
of the 24 layers. Second, the normalization I want on q and k is a per-vector rescale along the head
dimension, and the framework already exposes exactly that primitive: RMSNorm.

There is a fork inside that choice worth resolving, because the framework also has a mean-subtracting
LayerNorm and I should know why I am not reaching for it. LayerNorm normalizes as `(x - mean(x)) / std(x)`
— it removes the projection of `x` onto the all-ones direction before rescaling. RMSNorm skips the
centering: it is `x / rms(x)`, a pure rescale that leaves the direction of the raw vector untouched. Which
one do I want? I derived the target as "make the logit depend only on the *direction* of q and k," and the
direction I mean is the direction of the vector the projection actually produced. RMSNorm preserves that
direction exactly and only fixes its length; LayerNorm would additionally rotate q and k by projecting out
their mean component, so the cosine it measures is the angle between *centered* q and *centered* k — a
different similarity, and one with no clean interpretation in terms of the score I set out to stabilize.
Centering solves a problem I do not have (a shared additive offset across head coordinates) at the cost of
distorting the very angle I am trying to make the logit depend on. So RMSNorm is not just the convenient
primitive; it is the faithful one — the norm-preserving, direction-only operation my derivation asked for.

So let me line up what RMSNorm gives me against the L2-normalize-plus-`g` I derived, and compute the
realized logit by hand rather than assert that it works out. RMSNorm of a vector `x ∈ R^d` is `x / rms(x)`
with `rms(x) = sqrt(mean_i x_i^2) = ||x|| / sqrt(d)`. So RMSNorm is `sqrt(d) · x / ||x||` — it is L2
normalization times the *fixed* constant `sqrt(d)`. Apply it to both q and k along the head dimension
(`d = d_k = 64`), and the pre-scale dot product becomes
`(sqrt(d) q_hat)^T (sqrt(d) k_hat) = d · q_hat^T k_hat = 64 · cos(angle)`. Then the fused kernel divides by
`sqrt(d_k) = 8`, so the realized logit is `64 · cos(angle) / 8 = 8 · cos(angle)` — a cosine similarity
scaled by the *constant* `sqrt(d_k) = 8`. Now check whether ±8 gives back a usable sharpness, because that
is the whole worry from the over-correction above. The extreme contrast is `exp(8)/exp(-8) = exp(16) ≈
8.9 × 10^6` — plenty to concentrate mass when the cosine genuinely separates one key from the field, and
nominally the same ±8 range the original `1/sqrt(d_k)` produced at init. So the fixed `sqrt(d_k)` is quietly
playing the role my learned `g` would have played, pinned at 8 rather than tuned. The version that fits this
scaffold is therefore: RMSNorm q and k along the head dimension, and leave everything else — the fused
kernel, the causal mask, the `wpe` — exactly as is. Two lines added to the forward pass; no new parameters,
no learnable scale, no change to position.

Before I commit, one limit check I can do on paper, because it tells me whether I am perturbing the part of
training that is already well-tuned. What does this edit do *at initialization*? At init the projected q and
k entries are roughly unit-variance, so `||q_head|| ≈ sqrt(d_k) = 8` and likewise for k, and the vanilla
realized logit is `q^T k / 8 ≈ (||q|| ||k|| / 8) · cos(angle) ≈ (8 · 8 / 8) · cos = 8 · cos(angle)`. The
RMSNorm version forces `||q_head|| = sqrt(d_k) = 8` *exactly* and gives `8 · cos(angle)` by construction. So
the two coincide at init, up to the small fluctuation of `||q||, ||k||` around `sqrt(d_k)` — the edit is
very nearly a no-op at step zero, and it only begins to bite as the norms drift away from 8 over the run.
That is exactly the signature a stability fix should have: it does not touch the carefully-tuned starting
point, it just clamps the drift that the fixed scaling silently allows. I am reassured I am not paying an
init-time cost for a late-time correction.

I want to be honest with myself about what I am giving up relative to the clean derivation, because that is
the difference between the idea and the thing I can actually run. The learned per-head `g` is gone; the
sharpness ceiling is pinned at `sqrt(d_k) = 8` instead of being tuned by the optimizer. That is a real
omission — if some head would genuinely benefit from a sharper-than-±8 logit range, this version cannot
supply it, and I have no way to know from this rung alone whether any head wants that. But it keeps the
property I came for: the logit scale is now invariant to the *drift* of the q/k magnitudes, because RMSNorm
strips the magnitude out before the product. The saturation-creep failure is fixed; only the
deliberate-sharpening upside is left on the table. And keeping `use_pos_emb = True` is forced, not chosen:
this intervention says nothing about position, so the learned `wpe` table stays exactly as in the default. I
am changing *one thing* — the score's robustness to q/k norm drift — and nothing else, which is precisely
the clean-measurement property I wanted from the score-scale axis.

A couple of details decide whether the edit is faithful. RMSNorm must be applied *per head*, along the
`head_dim` axis, *after* the reshape to `(B, n_head, T, head_dim)`. If I normalized across the full
`n_embd = 1024` before splitting heads, I would be mixing all 16 heads' coordinates into one normalization
and destroying the per-head direction the cosine is supposed to measure; the whole point is that each head's
angle is computed in its own 64-dimensional subspace. So the order is: project, split into heads, RMSNorm q,
RMSNorm k, then attention. And I apply it to q and k only, never to v. The value path carries the content
that gets averaged into the output; normalizing v would distort the actual information the head reads out,
and the saturation problem lives entirely in the *logit*, which v is not part of. The dropout, the causal
masking, and the output projection all stay untouched. The compute overhead is nil — the two extra
normalizations touch `2 · B · n_head · T · head_dim` elements per layer against the
`B · n_head · T^2 · head_dim` of the attention product itself, a ratio of `2/T = 2/1024`, under a quarter of
a percent — which matters because the loop is frozen and I cannot afford a version that quietly changes the
training budget.

So at step 1 the edit is the minimal, position-agnostic one: leave the entire loop and the learned position
table alone, and insert two `F.rms_norm` calls on the per-head q and k right before the attention product
(the distilled module is in the answer). It is the floor I want to start the ladder from, for a reason I can
now state sharply. This intervention is a *stability* fix, not a *capacity* or *inductive-bias* fix. It
removes a failure mode — late-training logit-scale drift toward softmax saturation — without adding any new
information the model did not already have: position is still absolute-additive via `wpe`, exactly as in the
default. So I expect a *modest* improvement over an unmodified GPT-2 attention — cleaner late-training
gradients, a slightly lower and steadier validation loss, with the downstream metrics roughly tracking the
language-modeling gain rather than jumping on their own. What I do *not* expect it to touch is the thing the
background survey kept circling: position is still absolute and additive, so the relative structure of
language — that a dependency is about the *offset* `m - n`, not the absolute slots — is still left for the
network to reconstruct indirectly through the `wpe` cross terms. That is a *representational* limitation, and
RMSNorm-on-q-k does exactly nothing about it.

So my falsifiable expectation is sharp, and I will state it against the metrics this task actually reports.
On the primary objective, `val_loss`, this rung should land as the *weakest* of the attention interventions
I have in view — a small, real improvement over vanilla but beaten by anything that fixes how order enters,
because it only stabilizes the score and leaves the absolute-position handicap fully in place. A `val_loss`
somewhere in the low 2.3s is an in-domain perplexity of about ten (`exp(2.3) ≈ 10`), and I would expect
WikiText-2 and especially LAMBADA to sit well above that in-domain number, because LAMBADA's last-word
prediction leans on exactly the
long-range relative dependency this rung does not help. If a later rung that touches position does *not* beat
this floor, then my whole diagnosis is wrong and stability was the real bottleneck all along — but I do not
believe that, because the drift argument is a stability story and the position handicap is a representational
one, and the two are independent. The diagnosis already points at the next move: leave the score where I
have now stabilized it, and fix *position* — make the logit depend on the relative offset `m - n`, not the
absolute index.
