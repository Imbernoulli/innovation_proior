The combined run confirmed the position thesis and, in the same breath, flagged the redundancy I said to
watch for. RoPE + QK-Norm landed validation loss 2.2589, against the QK-Norm floor of 2.2885 — a drop of
0.0296 in cross-entropy, which is exactly the representational margin I predicted relative position would
buy, far past anything a stability tweak alone produces. The perplexities moved with it: WikiText-2
43.65 → 43.44, and LAMBADA 69.99 → 67.20, the LAMBADA drop the largest as I expected, since last-word
prediction over a long passage is precisely where relative-offset encoding earns its keep. Downstream
followed: ARC-Easy 55.64 → 57.83, HellaSwag 33.41 → 34.24, PIQA 63.17 → 64.74. So the position fix is
real and it is general — not a single-metric artifact. But read the result against the *other* question I
left open at the end of step 2: I stacked RoPE on top of the parameter-free RMSNorm QK-Norm, and I said
that if the combined run came in good-but-not-clearly-dominant, the tell would be that the QK-Norm half is
redundant once position is relative. The 2.2589 is good. The question is whether the RMSNorm half is
*helping* or quietly *costing* me, and the way to answer it is the cleanest experiment on the ladder:
strip the QK-Norm back out and run RoPE alone.

Let me argue from mechanism why plain RoPE could be the stronger configuration, because I do not want to
run an ablation I can't predict the sign of. The QK-Norm half, in this scaffold, is the parameter-free
RMSNorm version: it forces each per-head q and k onto a fixed-norm sphere, so the realized logit becomes
`sqrt(d_k)·cos(angle)` — a cosine similarity scaled by the *constant* `sqrt(d_k) = 8`. That constant is a
hard ceiling on how sharp attention can get: the entire pre-mask logit vector spans at most ±8 around the
cosine range, so the softmax contrast between the best and worst position is bounded no matter how
confident the model should be. When position was *absolute and additive* (step 1), that ceiling was a
fair trade — I was buying robustness to q/k magnitude drift, and the drift was a genuine late-training
failure mode worth paying a sharpness ceiling to remove. But now position is *relative*, injected by
rotation, and the picture changes. RoPE itself is norm-preserving — the rotation `R_m` is orthogonal, so
it neither inflates nor collapses q and k — which means a chunk of the instability QK-Norm was guarding
against is already damped by the geometry of the position scheme. More importantly, plain RoPE keeps the
*magnitude* of q and k as a usable degree of freedom: the logit is `||q||·||k||·cos(angle)` modulated by
the relative rotation, and the model can grow `||q||,||k||` on heads that *should* attend sharply and
keep them small on heads that should stay diffuse. RMSNorm throws that away — it pins every head to the
same `sqrt(d_k)` sharpness ceiling and removes the model's ability to learn per-head, per-token
confidence through the q/k norms. So the hypothesis is concrete: once RoPE has fixed position, the
RMSNorm half is no longer buying enough stability to justify the sharpness it sacrifices, and removing it
should *recover* a small amount of loss by handing the q/k magnitudes back to the optimizer as a learnable
sharpness control. The 2.2589 → expect-slightly-lower is the falsifiable form.

So step 3 is plain RoPE: the same relative-position injection, with the QK-Norm stripped back out. Let me
re-establish the derivation cleanly, because this is the method I am landing and it has to stand on its
own, not as "step 2 minus a line." Attention is order-blind: with q, k, v linear in the token embeddings,
the computation is permutation-equivariant, so position must be injected by hand, and the only quantity
that decides which token attends to which is the logit `q_m^T k_n`. The default fed order in additively
and absolutely through `wpe`, and expanding `q_m^T k_n` with `q = W_q(x_m + p_m)`, `k = W_k(x_n + p_n)`
produced four terms, three carrying *absolute* `p_m`/`p_n` — so the logit depended on the buffer slot,
not the offset `m - n` that language relations actually turn on. That is the handicap the QK-Norm floor
was paying for and the one RoPE removes by construction.

The construction is a solve, not a patch. Demand that the encoded inner product depend on position only
through the difference: `<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n)`, with the boundary
`f(x, 0) = W x`. In two dimensions, identify R^2 with the complex plane and use `<a, b> = Re[a b*]`;
write `f` in polar form and match magnitude and phase. The magnitude equation, pinned by the boundary at
offset 0, forces the magnitude to be position-independent — the stable, norm-preserving branch, because I
do not want position to amplify one side and shrink the other — and the phase equation forces the phase
to be arithmetic in position, the same extra angle `m·theta` added to both query and key on top of each
vector's own angle. The solution is rotation: `f_q(x_m, m) = (W_q x_m) e^{i m theta}`,
`f_k(x_n, n) = (W_k x_n) e^{i n theta}`, so
`<f_q, f_k> = Re[(W_q x_m)(W_k x_n)* e^{i(m-n)theta}]` — absolute `m, n` appear *only* through
`e^{i(m-n)theta}`. Position is a rotation by an angle proportional to the index, and I didn't bolt
anything on; the relative property fell out of the demand.

Lift to the real head dimension by splitting it into `d/2` independent 2-planes and rotating each at its
own frequency. The inner product is a sum over planes; each plane is relative-only by the 2D argument;
the sum of relative-only-per-plane is relative-only — linearity glues it. Block-diagonal `R_m`, the i-th
2×2 block a rotation by `m·theta_i`; rotations compose by adding angles, so `R_m^T R_n = R_{n-m}` and
`q_m^T k_n = x_m^T W_q^T R_{n-m} W_k x_n` — the offset sits in a single rotation between the content
projections, no learned table, no clip, no distance bias. The frequencies reuse the sinusoidal geometric
spectrum `theta_i = 10000^{-2(i-1)/d}`: fast planes resolving local offsets, slow planes carrying coarse
position, which gives the long-range decay envelope as a free prior — as `|m - n|` grows the phases
spread across frequencies, the partial sums lose coherence, and the positional contribution decays, so
far-apart tokens interact less, all else equal. And because `R` is orthogonal it preserves norm, so it
can never blow up or collapse the representation as it propagates through 24 layers — which, note, is
exactly the property that makes the QK-Norm half partly redundant: RoPE already keeps q/k from drifting
in the position-dependent direction.

Now the literal scaffold edit, which is *simpler* than step 2 — I am removing the two `F.rms_norm` calls
and keeping everything else. Position is no longer additive, so `self.use_pos_emb = False` in `__init__`,
and `GPT.forward` skips the `wpe` add (it gates on exactly that flag — the one mechanism a rung has to
replace position without touching anything outside the attention class). I precompute
`inv_freq = 1/(10000 ** (arange(0, head_dim, 2)/head_dim))` as a buffer; per forward I form
`freqs = outer(arange(T), inv_freq)`, take `cos` and `sin`, and apply the *split-half* rotation —
`x1 = x[..., :d]`, `x2 = x[..., d:]` with `d = head_dim/2`, `y1 = x1·cos - x2·sin`,
`y2 = x1·sin + x2·cos`, concatenate `[y1, y2]` — to q and k only, never v. The split-half layout pairs
coordinate `i` with `i + d` as the two legs of one plane (the LLaMA/NeoX convention the harness's
`_apply_rope` uses), equivalent up to a fixed permutation to the interleaved `(2i, 2i+1)` pairing but the
code must be consistent. The fused SDPA path, the causal mask, the output projection are untouched. So
the difference from step 2 is exactly the QK-Norm removal: `q = self._apply_rope(q, T)` instead of
`q = self._apply_rope(F.rms_norm(q, ...), T)`. That is the whole edit, and it is deliberately the
*minimal* delta so the ablation is clean — whatever the loss does between 2.2589 and this run is
attributable to the RMSNorm removal and nothing else.

I will name the omissions again so the comparison is honest. RoPE here is the *frozen* sinusoidal
schedule at base 10000 — the frequencies are not learned (they barely move from this initialization
anyway) and the base is not tuned to the 1024-token context. Those are unchanged from step 2; the only
new content of this rung is the QK-Norm removal. So the experiment is as controlled as the scaffold
allows: one operation deleted, everything else held.

Falsifiable expectations, against the 2.2589 RoPE + QK-Norm run. If the QK-Norm half is redundant once
position is relative, plain RoPE should land *at or slightly below* 2.2589 on validation loss — I am
predicting a small win, on the order of a couple of thousandths, because handing the q/k magnitudes back
to the optimizer as a sharpness control should recover a little of what the `sqrt(d_k)` ceiling pinned.
On perplexity I expect the same direction: WikiText-2 and especially LAMBADA should hold or improve, since
better-calibrated per-head sharpness helps the confident long-range predictions LAMBADA tests. The
downstream story is the one I am least sure of, and it is where the ablation could split: removing
QK-Norm trades a stability guard for sharpness freedom, and it is entirely possible that plain RoPE wins
the language-modeling metrics (val_loss, perplexity) while RoPE + QK-Norm holds a slight edge on one or
two downstream tasks where the extra logit-scale robustness happens to help calibration — ARC-Easy and
PIQA, the multiple-choice tasks most sensitive to small probability differences, are where I would expect
the combined model to possibly cling on. If that split happens — plain RoPE best on val_loss/perplexity,
RoPE + QK-Norm best on a downstream task or two — then the honest reading is that the strongest *language
model* is plain RoPE, and that is the rung I would rank highest on the task's primary objective, while
noting the combined model is not strictly dominated. The risk that would *falsify* the whole hypothesis is
the opposite outcome: if plain RoPE comes in clearly *worse* than 2.2589, then RMSNorm was doing real
stabilization work even under relative position, the sharpness ceiling was cheap, and I should keep the
combined form. I do not expect that — RoPE's own norm-preservation is why I think the guard is now
largely redundant — but it is the clean way the experiment could tell me I am wrong.
