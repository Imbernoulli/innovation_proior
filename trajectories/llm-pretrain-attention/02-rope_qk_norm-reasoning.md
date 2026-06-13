The QK-Norm run came back almost exactly where I predicted it would, and the number that matters is the
one it *didn't* move. Validation loss landed at 2.2885, with WikiText-2 perplexity 43.65, LAMBADA 69.99,
and downstream ARC-Easy 55.64, HellaSwag 33.41, PIQA 63.17, WinoGrande 51.30. That is a clean, stable
run — the score-stability fix did its job, the late-training gradients stayed clean, nothing diverged.
But it is also the *floor*, and it is the floor for the precise reason I argued before I ran it: RMSNorm
on q and k stabilizes the logit *scale* and touches nothing about *position*. The model is still being
fed order through the learned absolute `wpe` table, so every dependency in the text — a verb agreeing
with a subject three tokens back, a closing quote matching an opening one forty tokens back — has to be
reconstructed by the network out of *absolute* slot signals. That is the representational handicap
RMSNorm was never going to fix, and the 2.2885 is what "stability fixed, position still absolute" costs.
The diagnosis points straight at the seam I deliberately left alone: fix how *order* enters.

Let me be precise about what is wrong with the position signal, because I want the next move to be the
*right* relative scheme, not just any change. Self-attention is order-blind by construction: with q, k, v
linear in the token embeddings, the whole computation is permutation-equivariant — shuffle the tokens and
the outputs shuffle identically. "The dog bit the man" and "the man bit the dog" produce the same bag of
representations. So order must be injected by hand, and the only quantity that decides which token
attends to which is the logit `q_m^T k_n`; everything downstream is a consequence of those logits. The
default injects position *additively and absolutely*: `x_i <- x_i + wpe[i]` before the stack, so
`q_m = W_q(x_m + p_m)`, `k_n = W_k(x_n + p_n)`. Expand the logit and it is four terms:
`x_m^T W_q^T W_k x_n` (pure content, fine) plus three terms each carrying *absolute* `p_m` or `p_n`. The
logit therefore depends on *where m and n sit in the buffer*, not on the offset `m - n` — and the offset
is exactly what language relations are about. The model can in principle untangle relative offset from
those absolute signals, but I am making it learn indirectly something I could just hand it directly. That
indirection is the bottleneck the 2.2885 is paying for, and QK-Norm did nothing about it because RMSNorm
strips magnitude, not absolute-position-ness.

So I want the logit to depend on `x_m`, `x_n`, and `m - n` only. The cleanest way to get there is not to
patch the additive expansion (replace `p_n` with a relative table, bucket a bias by distance — the whole
Shaw-2018 / Transformer-XL / T5 family does exactly that, and every one of them ends up with a learned
relative table or bias living *inside* the pairwise logit). It is to *demand* the relative property and
solve for the injection. Write `q_m = f_q(x_m, m)`, `k_n = f_k(x_n, n)`, and require
`<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n)` for some `g` that sees position only through the
difference, with the boundary `f(x, 0) = W x` so it reduces to ordinary attention at position 0. Solve
this in the simplest nontrivial dimension, d = 2, by identifying R^2 with the complex plane and using
`<a, b> = Re[a b*]`. Write `f` in polar form, magnitude times phase, and match the two sides: the
magnitude equation, evaluated at offset 0 against the boundary, forces the magnitude to be
position-independent (the stable, norm-preserving branch — I do not want position to amplify one side and
shrink the other), and the phase equation forces the phase to be *arithmetic* in position, the same extra
angle `m·theta` added to query and key on top of each vector's own angle. The solution is a rotation:
`f_q(x_m, m) = (W_q x_m) e^{i m theta}`, `f_k(x_n, n) = (W_k x_n) e^{i n theta}`, so
`<f_q, f_k> = Re[(W_q x_m)(W_k x_n)* e^{i(m-n)theta}]` — the absolute positions appear *only* through
`e^{i(m-n)theta}`. Exactly the demand, and I didn't add anything; I solved for it. Position is a rotation
by an angle proportional to the index.

Lift to the real head dimension by chopping it into d/2 independent 2-planes and rotating each at its own
frequency. The inner product is a sum over planes, each plane is relative-only by the 2D argument, and a
sum of relative-only-per-plane is relative-only — linearity does the gluing. Stack the rotations into a
block-diagonal `R_m`, the i-th 2×2 block a rotation by `m·theta_i`; rotations compose by adding angles,
so `R_m^T R_n = R_{n-m}` and `q_m^T k_n = x_m^T W_q^T R_{n-m} W_k x_n` — the offset sits in a single
rotation matrix between the content projections, no learned table, no clip, no bias bucket. What
frequencies? Reuse the sinusoidal geometric spectrum, `theta_i = 10000^{-2(i-1)/d}`: fast planes that
spin quickly and resolve local offsets, slow planes that barely move over the whole sequence and carry
coarse position. That choice is not arbitrary — it makes the scheme the relative-rotation version of the
sinusoidal encoding, and it gives the long-range decay I want as a *prior*: as `|m - n|` grows the
phases spread across frequencies, the partial sums lose coherence, and the averaged positional envelope
decays, so far-apart tokens contribute a less coherent positional signal, all else equal. And `R` is
orthogonal, so applying it can never blow up or collapse the representation as it propagates — the same
norm-preservation I leaned on when the magnitude dropped out of the 2D solution.

Now make this concrete in the task's edit surface, because the harness fixes the loop and I only get the
`CausalSelfAttention` body, and the form that fits here is *not* the general paper machinery — it is one
specific layout. Two things are forced. First, position is no longer additive, so I must turn *off* the
learned `wpe`: set `self.use_pos_emb = False` in `__init__`, and `GPT.forward` (which gates the `wpe` add
on exactly that flag) will skip the absolute table entirely. This is the one place a rung is allowed to
replace position without editing anything outside the attention class — and it is mandatory here, because
leaving `wpe` on while also rotating q/k would double up two position schemes. Second, I do not build the
sparse block-diagonal matrix; I realize the rotation elementwise. I precompute
`inv_freq = 1/(10000^{(arange(0, head_dim, 2)/head_dim))}` as a buffer, and per forward I form
`freqs = outer(arange(T), inv_freq)`, take `cos` and `sin`. The scaffold uses the *split-half* layout:
split each per-head vector into its first and second halves `x1 = x[..., :d]`, `x2 = x[..., d:]` with
`d = head_dim/2`, and rotate as `y1 = x1·cos - x2·sin`, `y2 = x1·sin + x2·cos`, then concatenate
`[y1, y2]`. This pairs coordinate `i` with coordinate `i + d` as the two legs of one 2-plane (the
LLaMA/NeoX convention), as opposed to the interleaved `(2i, 2i+1)` pairing — the two layouts are
equivalent up to a fixed permutation of the head dimension, but the code must use one consistently, and
the harness's `_apply_rope` is the split-half one. I apply it to q and k only, never to v: v carries the
content averaged into the output, and position belongs in the *logit*, not the values.

Here is the move that makes this step 2 and not a fresh start: I do not throw away the QK-Norm I just
validated. The 2.2885 run proved the score-stability fix is real and free — no parameters, no
instability, a clean run — and it is *orthogonal* to position: RMSNorm strips q/k magnitude, RoPE rotates
q/k direction, and they compose. So I stack them. The order matters and it is forced by what each
operation does. RMSNorm rescales each per-head vector to a fixed norm (`sqrt(d)·x/||x||`); RoPE rotates
it, and rotation is norm-preserving, so the norm RMSNorm imposes survives the rotation. If I rotated
*first* and normalized *second*, the normalization would still leave a clean direction, but I would be
re-deriving the norm after mixing the two planes — and more to the point, the reference realization (and
the cleanest reading of "normalize the score, then place it in position-space") is **RMSNorm then RoPE**:
`q = _apply_rope(F.rms_norm(q, (q.size(-1),)), T)`, same for k. Normalize the direction, then rotate it
by position. That is the literal edit. Everything else — the fused SDPA path, the causal masking, the
output projection — stays exactly as in step 1. So this rung is precisely "step 1 plus relative
position": two `F.rms_norm` calls and two `_apply_rope` calls on the per-head q and k, `use_pos_emb`
flipped to False, and the `wpe` table dropped from the position path. The full scaffold module is in the
answer.

I should be honest about the one degree of freedom the scaffold does *not* expose, because it bears on
what this rung can and cannot buy. RoPE here is the *frozen* sinusoidal schedule at base 10000 — I am not
learning the frequencies (they barely move from this initialization anyway, so there is no reason to
spend parameters on them) and I am not tuning the base for the 1024-token context. And the QK-Norm half
is still the parameter-free RMSNorm version from step 1, not the learned-`g` form — so the realized logit
is `sqrt(d_k)·cos(angle)` modulated by the relative rotation, with the sharpness ceiling pinned at
`sqrt(d_k)` rather than tuned. Both are the same omissions I already accepted; the new content of this
rung is purely the relative-position fix layered on top.

Now the falsifiable expectations, stated against the 2.2885 floor. The whole thesis is that the
bottleneck the QK-Norm run was paying for is *absolute-additive position*, and RoPE removes it. So I
expect this rung to clear 2.2885 on validation loss by a real margin — not a stability margin but a
representational one — and I expect the perplexities to follow, with LAMBADA the most sensitive: LAMBADA
is a last-word prediction task that hinges on a long-range dependency to the start of the passage, which
is exactly where relative-offset encoding should help most, so the 69.99 should drop meaningfully.
Downstream accuracies should move up with the LM gain, since better order modeling is a general
improvement, not a benchmark trick. The risk I will be watching: the two changes are stacked, so if the
combined run does *not* beat the QK-Norm floor, the position story is wrong and stability was the real
bottleneck all along — but I do not believe that, because the four-term expansion is an explicit,
mechanical handicap and RoPE provably removes it. The sharper question this rung sets up for the *next*
one is subtler: I am stacking RoPE *on top of* RMSNorm, and it is entirely possible that once position is
relative, the RMSNorm half contributes little or even slightly costs me — that RoPE alone, without the
parameter-free QK-Norm, is the stronger configuration. If the combined run lands good but not clearly
better than RoPE would alone, that is the tell that the QK-Norm half is redundant once position is fixed,
and the next rung is to strip it back to plain RoPE and see whether the cosine-similarity logit was
quietly costing me the deliberate-sharpening the fixed `sqrt(d_k)` ceiling cannot provide.
