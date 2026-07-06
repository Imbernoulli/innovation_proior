The QK-Norm run came back almost exactly where I predicted it would, and the number that matters is the
one it *didn't* move. Validation loss landed at 2.2885, with WikiText-2 perplexity 43.65, LAMBADA 69.99,
and downstream ARC-Easy 55.64, HellaSwag 33.41, PIQA 63.17, WinoGrande 51.30. Before I read that as
success or failure, let me actually convert it into a picture of what the model can and cannot do, because
the raw table hides the diagnosis. The validation cross-entropy 2.2885 is an in-domain perplexity of
`exp(2.2885) = 9.86` — on FineWeb, the model is choosing the next token as if from about ten equally-likely
options. But WikiText-2 sits at 43.65 and LAMBADA at 69.99, seven times the in-domain figure. Some of that
gap is domain shift (WikiText and LAMBADA are not FineWeb), but LAMBADA being the *worst* of the three is
the tell: LAMBADA is deliberately built so the final word is only predictable from a long-range dependency
to earlier in the passage, and a perplexity of 70 there says the model is barely resolving those long-range
links — it predicts local continuations fine and falls apart when the needed evidence is far away. The
downstream numbers point the same direction once I put them against their chance baselines. ARC-Easy and
HellaSwag are four-way (25% chance), PIQA and WinoGrande two-way (50% chance). So ARC-Easy 55.64 and PIQA
63.17 are comfortably above chance — factual and physical-commonsense questions the model can mostly do —
while HellaSwag 33.41 is only eight points over chance and WinoGrande 51.30 is *at* chance. WinoGrande is a
pronoun-resolution task: "the trophy doesn't fit in the suitcase because it is too big" — resolving "it"
requires binding a word to a referent several tokens back, which is a purely *relational* dependency. The
model scoring at chance there is the same failure LAMBADA reports, seen through a different lens: whatever is
about the *relation between positions* rather than the content at a position, this model is not doing.

That is a clean, stable run — the score-stability fix did its job, the late-training gradients stayed clean,
nothing diverged — but it is also the *floor*, and it is the floor for the precise reason I argued before I
ran it: RMSNorm on q and k stabilizes the logit *scale* and touches nothing about *position*. The model is
still being fed order through the learned absolute `wpe` table, so every dependency in the text — a verb
agreeing with a subject three tokens back, a closing quote matching an opening one forty tokens back, the
pronoun in WinoGrande — has to be reconstructed by the network out of *absolute* slot signals. That is the
representational handicap RMSNorm was never going to fix, and the 2.2885 is what "stability fixed, position
still absolute" costs. The diagnosis points straight at the seam I deliberately left alone: fix how *order*
enters.

Let me be precise about what is wrong with the position signal, because I want the next move to be the
*right* relative scheme, not just any change. Self-attention is order-blind by construction: with q, k, v
linear in the token embeddings, the whole computation is permutation-equivariant — shuffle the tokens and
the outputs shuffle identically. "The dog bit the man" and "the man bit the dog" produce the same bag of
representations. So order must be injected by hand, and the only quantity that decides which token attends
to which is the logit `q_m^T k_n`; everything downstream is a consequence of those logits. The default
injects position *additively and absolutely*: `x_i <- x_i + wpe[i]` before the stack, so
`q_m = W_q(x_m + p_m)`, `k_n = W_k(x_n + p_n)`. Expand the logit and it is four terms:
`x_m^T W_q^T W_k x_n` (pure content, fine) plus three terms each carrying *absolute* `p_m` or `p_n`. The
logit therefore depends on *where m and n sit in the buffer*, not on the offset `m - n` — and the offset is
exactly what language relations are about. The model can in principle untangle relative offset from those
absolute signals — the cross term `p_m^T W_q^T W_k p_n` does carry both `m` and `n` and could in principle
encode `m - n` — but I am forcing it to learn indirectly, through the `wpe` table's cross-products,
something I could hand it directly and exactly. That indirection is the bottleneck the 2.2885 is paying for,
and QK-Norm did nothing about it because RMSNorm strips magnitude, not absolute-position-ness.

So I want the logit to depend on `x_m`, `x_n`, and `m - n` only. There are two ways to get there and they
are genuinely different. The first is to patch the additive expansion — replace the absolute `p_n` with a
learned relative table indexed by `m - n`, or add a scalar bias bucketed by distance; the whole
relative-attention family does exactly this, and every version ends up with a learned relative table or bias
living *inside* the pairwise logit, which means new parameters, a clip or bucket for distances beyond some
range, and a table the optimizer has to fill in. And the parameter cost is not trivial: a per-offset learned
vector in `head_dim` over a 1024-token context is up to `(2·1024 - 1) · 64 ≈ 131k` numbers before any
sharing, and even the bucketed scalar-bias variant is a few hundred parameters per head per layer plus the
bucket boundaries as new hyperparameters — all of it living on the frozen budget, all of it something the
optimizer must learn from scratch and that could interact with the drift I just stabilized. The second is to *demand* the relative property as a
constraint and solve for the injection that satisfies it, hoping the solution is parameter-free. Let me try
the second, because if it closes it is strictly cleaner. Write `q_m = f_q(x_m, m)`, `k_n = f_k(x_n, n)`, and
require `<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n)` for some `g` that sees position only through the
difference, with the boundary `f(x, 0) = W x` so it reduces to ordinary attention at position 0.

Solve this in the simplest nontrivial dimension, d = 2, by identifying R^2 with the complex plane and using
`<a, b> = Re[a b*]`. Write `f` in polar form, magnitude times phase, and match the two sides. The magnitude
equation, evaluated at offset 0 against the boundary, forces the magnitude to be position-independent — I
take the stable, norm-preserving branch, because I do not want position to amplify one side and shrink the
other. The phase equation forces the phase to be *arithmetic* in position: the same extra angle `m·theta`
added to the query and key on top of each vector's own angle. The solution is a rotation:
`f_q(x_m, m) = (W_q x_m) e^{i m theta}`, `f_k(x_n, n) = (W_k x_n) e^{i n theta}`. Let me verify it actually
delivers the demand rather than assert it. In matrix form, `f_q(x_m, m) = R(m theta) W_q x_m` with the 2×2
rotation `R(α) = [[cos α, -sin α], [sin α, cos α]]`, and the inner product is
`<R(m theta) a, R(n theta) b> = a^T R(m theta)^T R(n theta) b`. Rotations satisfy `R(α)^T = R(-α)` and
`R(-α) R(β) = R(β - α)`, so `R(m theta)^T R(n theta) = R((n - m) theta)`, and the logit becomes
`(W_q x_m)^T R((n - m) theta) (W_k x_n)` — the absolute indices `m` and `n` appear *only* through the single
rotation `R((n - m) theta)`. Exactly the demand, and I did not bolt anything on; the relative property fell
out of the constraint, with no table and no learned parameter.

Let me put a number on it to be sure I have not fooled myself with the algebra. Take the smallest case,
`d = 2`, `theta = 1`, and let the raw query and key both be the unit vector `(1, 0)`. Put the query at
position `m = 5` and the key at `n = 2`, offset `m - n = 3`. Then `f_q = R(5)·(1,0) = (cos 5, sin 5)` and
`f_k = R(2)·(1,0) = (cos 2, sin 2)`, and their dot product is
`cos 5 · cos 2 + sin 5 · sin 2 = cos(5 - 2) = cos 3 ≈ -0.99`. Now slide *both* positions up by 95 — query at
`m = 100`, key at `n = 97`, same offset 3 — and the dot product is `cos(100 - 97) = cos 3 ≈ -0.99`,
identical. The logit did not move when I shifted both absolute positions by the same amount; it tracks only
the offset. That is the invariance I demanded, confirmed on actual numbers, and it is exactly what the
absolute-additive `wpe` could never guarantee.

Lift to the real head dimension by chopping it into `d/2` independent 2-planes and rotating each at its own
frequency. The inner product is a sum over planes, each plane is relative-only by the 2D argument, and a sum
of relative-only-per-plane is relative-only — linearity does the gluing. Stack the rotations into a
block-diagonal `R_m`, the i-th 2×2 block a rotation by `m·theta_i`; block-diagonal rotations still compose
by adding angles, so `R_m^T R_n = R_{n-m}` and `q_m^T k_n = x_m^T W_q^T R_{n-m} W_k x_n` — the offset sits in
a single rotation matrix between the content projections, no learned table, no clip, no bias bucket. What
frequencies? Reuse the sinusoidal geometric spectrum, `theta_i = 10000^{-2(i-1)/d}`. Let me put actual
numbers on that spectrum for this head, because it decides what the scheme can resolve. With
`d = head_dim = 64`, the exponent is `2(i-1)/64 = (i-1)/32`. Plane 1 has `theta_1 = 10000^0 = 1`, a period
of `2π ≈ 6.3` tokens — it spins through a full turn every six positions, the fast plane that resolves local
offsets. Plane 32, the last, has `theta_32 = 10000^{-62/64} ≈ 1.33 × 10^{-4}`, a period of about 47,000
tokens — over the whole 1024-token context it rotates only `1024 · 1.33 × 10^{-4} ≈ 0.02` of a turn, so it
is essentially a slow ramp carrying coarse, near-absolute position. And the crossover, the plane whose
period equals the context length, is at `theta = 2π/1024`, which solving `10000^{-(i-1)/32} = 2π/1024` puts
at `i ≈ 18.7`: planes 1 through ~18 complete at least one full rotation *within* a 1024-token window (so
they encode genuine within-context offset), while planes ~19 through 32 do not close a full turn over the
context (so they carry position more like a monotone coordinate). That spread — fast planes for local
structure, slow planes for global position — is the geometric spectrum's whole point, and it gives the
long-range decay I want as a *prior*: as `|m - n|` grows, the per-plane phases `(m-n)·theta_i` spread across
frequencies and the summed positional contribution loses coherence, so far-apart tokens interact through a
less coherent positional signal, all else equal. Let me make "less coherent" a little more concrete rather
than leave it as a slogan. A plane contributes a usable, non-oscillating positional signal at offset `Δ`
only while `theta_i · Δ` is below roughly one radian — past that the cosine has turned far enough that the
plane's contribution is averaging over its own oscillation. At the maximum in-context offset `Δ = 1024`, the
planes that stay coherent are those with `theta_i < 1/1024`, i.e. period longer than ~6400 tokens, which
from the spectrum is `i ≳ 25` — the slowest seven or eight of the 32 planes still carry a clean relative
signal at the far end of the context, while the two dozen faster planes have smeared. So the decay is *graded*, not a cliff:
close tokens see all 32 planes coherently, distant tokens see a shrinking slow subset, and the crossover is
smooth across the whole context. That is the right shape for a language prior — local dependencies get full
positional resolution, long-range ones get a coarser but non-zero signal. And each `R` is orthogonal, so
applying it can never blow up or collapse the representation as it propagates through 24 layers — the same
norm-preservation I leaned on when the magnitude dropped out of the 2D solution.

Now make this concrete in the task's edit surface, because the harness fixes the loop and I only get the
`CausalSelfAttention` body, and the form that fits here is *not* the general machinery — it is one specific
layout, and two things are forced. First, position is no longer additive, so I must turn *off* the learned
`wpe`: set `self.use_pos_emb = False` in `__init__`. The loop gates the absolute table on exactly this flag
— `getattr(self.transformer.h[0].attn, 'use_pos_emb', True)` in `GPT.forward`, keyed on layer 0's attention
— so flipping it to False makes the loop skip the `wpe` add entirely. This is the one place a rung is
allowed to replace position without editing anything outside the attention class, and it is *mandatory*
here: leaving `wpe` on while also rotating q and k would stack two position schemes, an absolute-additive
one and a relative-rotational one, and the model would still be paying for the absolute handicap I am trying
to remove. Second, I do not build the sparse block-diagonal matrix; I realize the rotation elementwise. I
precompute `inv_freq = 1/(10000^{(arange(0, head_dim, 2)/head_dim))}` as a buffer, and per forward I form
`freqs = outer(arange(T), inv_freq)`, then `cos` and `sin`. The scaffold uses the *split-half* layout: split
each per-head vector into its first and second halves `x1 = x[..., :d]`, `x2 = x[..., d:]` with
`d = head_dim/2`, and rotate as `y1 = x1·cos - x2·sin`, `y2 = x1·sin + x2·cos`, then concatenate `[y1, y2]`.

That split-half choice deserves a check, because there is another obvious layout — the interleaved
`(2i, 2i+1)` pairing — and if I mismatch them I will silently break the relative property. Split-half pairs
coordinate `i` with coordinate `i + d` as the two legs of one 2-plane; interleaved pairs `2i` with `2i+1`.
Both partition the 64 coordinates into 32 planes and rotate each plane by its matched frequency; the two are
related by a fixed permutation of the head dimension, nothing more. Since `W_q` and `W_k` are *learned*, the
model can absorb any fixed coordinate permutation into those projection matrices at no cost, so the two
layouts are equivalent in expressivity — the network trained under one can represent exactly what it could
under the other. What is *not* free is consistency: the rotation `R((n-m)theta)` only telescopes if q and k
are rotated with the *same* pairing and the *same* frequency assignment, because the identity
`R_m^T R_n = R_{n-m}` is per-plane and only holds when the two sides share the plane structure. If I rotated
q split-half and k interleaved, the dot product would sum mismatched planes and the offset would no longer
factor out — the whole derivation collapses. So the rule is: pick one layout, apply it identically to q and
k. The harness's `_apply_rope` is the split-half one, so that is what I use, on q and k only, never on v:
v carries the content averaged into the output, and position belongs in the *logit*, not the values.

Here is the move that makes this step 2 and not a fresh start: I do not throw away the QK-Norm I just
validated. The 2.2885 run proved the score-stability fix is real and free — no parameters, no instability, a
clean run — and it is *orthogonal* to position: RMSNorm strips q/k magnitude, RoPE rotates q/k direction. So
I stack them. Now, the order of the two operations looks like it should matter, so let me actually check
whether it does before I pick one. RMSNorm is a pure rescale, `N(x) = sqrt(d)·x/||x||`; RoPE is a rotation
`R`, which is orthogonal and so preserves norm, `||R x|| = ||x||`. Compose them one way:
`N(R x) = sqrt(d)·(R x)/||R x|| = sqrt(d)·(R x)/||x|| = R(sqrt(d)·x/||x||) = R(N(x))`, where the middle step
uses `||R x|| = ||x||` and the last uses that `R` is linear so it pulls the scalar out. So `N(R x) = R(N x)`
exactly — the two operations *commute*, and the realized q and k are identical whichever order I write. The
order is cosmetic, not load-bearing; I had half-expected it to matter and it does not, precisely because
RoPE is norm-preserving and RMSNorm is a pure rescale. I will write RMSNorm-then-RoPE,
`q = _apply_rope(F.rms_norm(q, (q.size(-1),)), T)`, because it reads as the natural pipeline — normalize the
direction, then place it in position-space — but nothing downstream depends on the choice. Everything else —
the fused SDPA path, the causal masking, the output projection — stays exactly as in step 1. So this rung is
precisely "step 1 plus relative position": two `F.rms_norm` calls and two `_apply_rope` calls on the
per-head q and k, `use_pos_emb` flipped to False, and the `wpe` table dropped from the position path. The
full scaffold module is in the answer.

I should be honest about the degrees of freedom the scaffold does *not* expose, because they bear on what
this rung can and cannot buy. RoPE here is the *frozen* sinusoidal schedule at base 10000 — I am not
learning the frequencies (they barely move from this initialization anyway, so there is no reason to spend
parameters on them) and I am not tuning the base for the 1024-token context. That last one is a real choice I
should own: the spectrum I just worked out spends its slowest planes on periods of tens of thousands of
tokens, far past the 1024 I ever see, so a smaller base would pack more of the frequency ladder into the
window I actually use and arguably give finer relative resolution across the mid-range offsets. I decline it
anyway. Lowering the base is an untested hyperparameter search on a frozen training budget I cannot re-tune,
base 10000 is the well-validated default that transfers across scales, and — the discipline I care about
most on this ladder — retuning it here would fold a second, confounded change into a rung whose one job is to
measure what relative position buys. So I keep the standard base and let the *only* new content of this rung
be the position fix itself. And the QK-Norm half is still
the parameter-free RMSNorm version from step 1, not the learned-`g` form — so the realized logit is
`sqrt(d_k)·cos(angle)` modulated by the relative rotation, with the sharpness ceiling pinned at `sqrt(d_k)`
rather than tuned. Both are the same omissions I already accepted; the new content of this rung is purely
the relative-position fix layered on top.

Now the falsifiable expectations, stated against the 2.2885 floor. The whole thesis is that the bottleneck
the QK-Norm run was paying for is *absolute-additive position*, and RoPE removes it. So I expect this rung to
clear 2.2885 on validation loss by a real margin — not a stability margin but a representational one, which
in my experience at this scale is several hundredths of a nat rather than the thousandths a scaling tweak
buys. I expect the perplexities to follow, with LAMBADA the most sensitive: LAMBADA's last-word prediction
hinges on exactly the long-range dependency that a relative-offset encoding is built to serve, so the 69.99
should drop the most in relative terms, more than WikiText-2's 43.65, which leans more on local structure
that absolute position already handled passably. Downstream should move up with the LM gain — and if the
mechanism story is right, WinoGrande, the near-chance pronoun task, is where I would most hope to see
movement, because binding a pronoun to a referent is precisely a relative-offset dependency, though I hold
that one loosely since a score sitting at chance can move for noisy reasons. The risk I will be watching: the
two changes are stacked, so if the combined run does *not* beat the QK-Norm floor, the position story is
wrong and stability was the real bottleneck all along — but I do not believe that, because the four-term
expansion is an explicit, mechanical handicap and the constraint-solve provably removes it. The sharper
question this rung sets up for the *next* one is subtler: I am stacking RoPE *on top of* RMSNorm, and it is
entirely possible that once position is relative, the RMSNorm half contributes little or even slightly costs
me — that the fixed `sqrt(d_k)` sharpness ceiling it imposes is a worse trade once RoPE, itself
norm-preserving, is already damping the position-direction drift QK-Norm was guarding. If the combined run
lands good but not clearly better than plain RoPE would land, that is the tell that the QK-Norm half is
redundant once position is fixed, and the clean next experiment is to strip it back to plain RoPE and see
whether the cosine-similarity logit was quietly costing me the deliberate sharpening the fixed `sqrt(d_k)`
ceiling cannot provide.
