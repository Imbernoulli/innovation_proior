The RoPE + QK-Norm run came back where I hoped and, in the same breath, sharpened the one question I
deliberately left open. Validation loss landed at 2.2589 against the QK-Norm floor of 2.2885 — a drop of
0.0296 in cross-entropy. Let me turn that into something I can feel before I read it as success or failure:
the floor's token-level perplexity was `exp(2.2885) ≈ 9.860`, the combined run's is `exp(2.2589) ≈ 9.573`,
a ratio of `0.9709`, so about a 2.9% reduction in per-token perplexity. That is far past anything a
stability tweak alone buys — step 1's whole job was stability and it left position untouched — so this is
the representational margin I predicted relative position would open, arriving through exactly the channel I
argued it would. The perplexities moved with it, and the *pattern* of how they moved is the tell. WikiText-2
went 43.65 → 43.44, a drop of 0.21, which is 0.5% relative. LAMBADA went 69.99 → 67.20, a drop of 2.79,
which is 4.0% relative — eight times the relative move of WikiText-2. That asymmetry is the position story
writing itself into the metrics: LAMBADA is last-word prediction hinging on a dependency that reaches back
across the whole passage, precisely where a relative-offset encoding earns its keep, while WikiText-2
perplexity is dominated by short-range, locally-decidable continuations that absolute position already
handled passably. Downstream followed the language-modeling gain: ARC-Easy 55.64 → 57.83 (+2.19), HellaSwag
33.41 → 34.24 (+0.83), PIQA 63.17 → 64.74 (+1.57). WinoGrande alone did not track it — 51.30 → 50.67, down
0.63 — but it is sitting right on its 50-point two-way chance floor at both ends, so that wobble is within
single-seed noise and I read it as "no signal yet" rather than "position hurt coreference." So the position
fix is real and it is general, not a single-metric artifact.

But there is a thing this one number cannot tell me, and it is exactly the thing that decides the next rung.
The 2.2589 is the result of *two* changes stacked: RMSNorm on q and k from step 1, and RoPE layered on top.
I proved in step 1 that the RMSNorm half is real and free — a clean run, no parameters, the drift fix I
wanted — but I proved that in a world where position was absolute and additive. Now position is relative,
and I have no way, from one combined run, to attribute the 0.0296 between the two operations. Did RMSNorm
still contribute here, or did RoPE do all the work while the RMSNorm half rode along neutral, or — the
possibility I flagged at the end of step 2 — is the RMSNorm half now quietly *costing* me, its fixed-norm
sphere pinning a sharpness I would rather the optimizer control? A single combined run cannot separate those
three stories; they all produce a good number. The only way to read the answer off the metrics is the
cleanest experiment on the ladder: hold everything else fixed, delete the RMSNorm, run RoPE alone, and
attribute the entire difference to that one deletion.

Before I commit to the deletion, let me lay out what I could actually do from here, because "strip it out"
is only obviously right once I have looked at the alternatives and found them worse. Three moves are on the
table. I could keep the combined form as the accepted baseline and spend this rung on a *different* lever
entirely — head structure, or the shape of the distribution — treating the QK-Norm question as
settled-enough and moving on. I could delete the RMSNorm and run plain RoPE, the pure ablation. Or I could
*upgrade* the QK-Norm half: replace the parameter-free RMSNorm with the learned-`g` form I derived on the
blank page back in step 1 — L2-normalize q and k and multiply the logit by a per-head learned scale `g`, so
the sharpness ceiling stops being pinned at `sqrt(d_k)` and becomes a tuned quantity. That third option is
the tempting one, because it looks like it keeps the stability guard *and* removes the ceiling — the best of
both. Let me walk it and see why it is the wrong move for *this* rung rather than dismiss it. First, it does
not answer the question I actually have. Step 2 posed a specific question — is the guard redundant now that
position is relative? — and the way to answer "is this thing needed" is to remove it and look, not to
replace it with a more elaborate version of it. The learned-`g` form changes two things at once: it removes
the ceiling *and* it keeps stripping the q/k magnitude. If the run improved, I would not know whether the
gain came from the recovered sharpness or from residual stabilization, and I would be right back in the
confounded reading the combined run already handed me. Second, look at what it keeps and what plain RoPE
gives for free. Learned-`g` still throws away the magnitude of q and k and reintroduces sharpness through a
*separate* learned dial — one extra scalar per head, `16 · 24 = 384` numbers on the frozen budget, cheap in
parameters but still a new thing the optimizer has to find. Plain RoPE hands the *same* control back with no
new parameter at all: the magnitudes `||q||`, `||k||` that the projections already produce *are* the
sharpness dial, and the optimizer already knows how to move `W_q` and `W_k` to set them. So learned-`g`
spends parameters and an extra confound to recover a control that deleting RMSNorm gives me for nothing.
Third, the discipline of the ladder is one clean change per rung, and learned-`g` is two. So I set it aside —
not because it is wrong in general, but because plain RoPE is the strictly cleaner probe of the exact
question I am holding. The fourth thing I could fiddle, the RoPE base, I declined in step 2 for budget
discipline and decline again for the same reason: retuning it here would fold a confounded change into a
rung whose one job is to isolate the RMSNorm deletion.

So the whole bet rides on whether the RMSNorm ceiling is costing me sharpness the model wants, and I do not
want to run an ablation whose sign I cannot predict, so let me put a real number on the ceiling. The RMSNorm
form forces each per-head q and k onto a fixed-norm sphere, so the realized logit is `sqrt(d_k)·cos(angle)`
— a cosine similarity scaled by the *constant* `sqrt(d_k) = 8`. "Sharpness ceiling" is a phrase; I want the
peak attention mass it permits. Take a head that should attend almost entirely to one key. The best it can
do under the cap is drive that key's cosine to +1, giving a logit of 8. The competitors it sees — call it
~511 keys under the causal mask mid-sequence — sit at whatever cosine their content gives, which for
near-orthogonal directions in a 64-dimensional head space is close to zero. Let me pin that spread, because
the whole calculation leans on it: the dot product of two independent random unit vectors in `d` dimensions
has mean 0 and variance `1/d`, so here standard deviation `1/sqrt(64) = 0.125`; multiplied by the scale 8,
the competitor logits are `8·cos ~ N(0, 1)`. The mass the softmax can put on the target is
`exp(8) / (exp(8) + sum_competitors exp(8·cos_j))`. The expected value of `exp(8·cos)` for `8·cos ~ N(0,1)`
is `exp(1/2) = 1.649`, so the competitor sum is about `511 · 1.649 ≈ 843`, and with `exp(8) = 2981` the peak
mass is `2981 / (2981 + 843) ≈ 0.78`. Under the cosine cap, even a maximally confident head tops out around
78% of its mass on the single most relevant key — and this is not a late-training degradation, it is the
ceiling for *every* sharp head from step 0 to the last iteration.

Now let plain RoPE off the leash. Without RMSNorm the logit is `||q||·||k||·cos(angle) / sqrt(d_k)`; at init
`||q|| ≈ ||k|| ≈ sqrt(d_k) = 8`, so the coefficient is `8·8/8 = 8`, the same `8·cos` as the capped version —
but now the model can *grow* `||q||`, `||k||` on a head that should be sharp, and the coefficient grows as
`8·c²` if each norm grows by a factor `c`. Trace what modest growth buys, because the size of the recoverable
gain is exactly the size of this effect. To reach coefficient 16 each of `||q||`, `||k||` need grow by only
`c = sqrt(2) ≈ 1.41` — a 41% increase, well inside what weight decay 0.1 permits. Then the target logit is
16, the competitors are `16·cos ~ N(0, 4)` with std 2, `E[exp(16·cos)] = exp(2) = 7.39`, the competitor sum
is `511·7.39 ≈ 3776`, and against `exp(16) = 8.89×10^6` the peak mass is `1 − 3776/8.89×10^6 ≈ 0.9996`. Push
to coefficient 24 — each norm grown by `c = sqrt(3) ≈ 1.73`, a 73% increase — and the target logit is 24,
competitors `24·cos ~ N(0, 9)` std 3, `E[exp] = exp(4.5) = 90`, sum `511·90 ≈ 4.6×10^4`, and against
`exp(24) = 2.65×10^10` the peak is `1 − 1.7×10^-6 ≈ 0.999998`. So the capped head is frozen at 78% while a
modest, weight-decay-plausible magnitude growth carries the uncapped head to 99.96% and then essentially
100%. That is the sharpness the constant `sqrt(d_k)` is pinning, quantified — and note it is a *first-order*
limit: it applies to every sharp head throughout all of training, not just late.

I have to be honest that removing RMSNorm does not remove the problem it was invented to fix. Plain RoPE
hands the q/k magnitudes back to the optimizer, so the exact magnitude-drift failure mode from step 1 —
`W_q`, `W_k` growing over the run, the scaled logit standard deviation inflating, the softmax creeping
toward saturation late in training — is back in play. I am, on purpose, reintroducing the pathology I spent
step 1 removing. Why is that the right trade now, when it was not at step 1? Two reasons, and they are about
what changed between the rungs. First, the position scheme changed. RoPE applies an orthogonal rotation
`R_m` to q and k, and orthogonal maps preserve norm, so RoPE itself contributes *zero* drift — it cannot
inflate or collapse q and k as they propagate through the 24 layers. A chunk of the instability QK-Norm was
guarding against in step 1 was coupled to the free additive position table; that table is gone now, replaced
by a norm-preserving rotation, so the residual drift RMSNorm still had to fight is smaller than in the
step-1 setting where position was a free additive signal riding into the projections. Second, the two
effects live on different orders. The sharpness ceiling is *first-order*: it caps peak attention at 78% for
every sharp head, unconditionally, all through training. The drift is *second-order*: it only bites once the
weights have grown, and weight decay bounds how far they grow — the decay force `0.1·W_q` balances the loss
gradient at some finite equilibrium norm, so the drift is self-limiting in a way the ceiling is not. My bet
is concrete: the first-order sharpness I recover by removing the cap outweighs the second-order stability I
give up, so plain RoPE should land at or slightly below 2.2589.

One limit check I can do on paper before I trust the bet, because it tells me whether the ablation risks
disturbing the part of training that is already well-tuned. What does deleting RMSNorm do *at
initialization*? At init the projected q and k entries are roughly unit-variance, so `||q_head|| ≈ sqrt(d_k)
= 8` and likewise for k, and the plain-RoPE realized logit is `||q||·||k||·cos/8 ≈ (8·8/8)·cos = 8·cos`. The
RMSNorm version forces exactly `8·cos` by construction. So at step 0 the two configurations produce the
*same* realized logit, up to the small fluctuation of `||q||`, `||k||` around 8 — the deletion is very
nearly a no-op at initialization, and it only begins to matter as the norms drift away from 8 over the run.
That is the reassuring signature: plain RoPE cannot pay an init-time cost for a late-time freedom, exactly
the mirror of the argument that made me trust the RMSNorm *addition* was safe in step 1. Whatever the
ablation does, it does it by opening late-training headroom, not by disturbing the tuned starting point — so
the worst honest case is that it changes nothing, not that it breaks the run.

So step 3 is plain RoPE, and since this is the method I am landing it has to stand on its own, not as "step 2
minus a line." Attention is order-blind by construction: with q, k, v linear in the token embeddings the
whole computation is permutation-equivariant, so order must be injected by hand, and the only quantity that
decides which token attends to which is the logit `q_m^T k_n`. The default fed order in additively and
absolutely through `wpe`, and expanding `q_m^T k_n` with `q = W_q(x_m + p_m)`, `k = W_k(x_n + p_n)` produces
four terms, three of them carrying *absolute* `p_m` or `p_n` — so the logit depends on the buffer slot, not
the offset `m − n` that language relations actually turn on. That is the handicap the QK-Norm floor was
paying for, and RoPE removes it by construction. And it is a solve, not a patch. Demand that the encoded
inner product depend on position only through the difference, `<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m −
n)`, with the boundary `f(x, 0) = W x` so it reduces to ordinary attention at position 0. In two dimensions,
identify `R^2` with the complex plane and use `<a, b> = Re[a b*]`; write `f` in polar form and match
magnitude and phase. The magnitude equation, pinned by the boundary at offset 0, forces the magnitude
position-independent — the stable, norm-preserving branch, because I do not want position to amplify one side
and shrink the other — and the phase equation forces the phase arithmetic in position, the same extra angle
`m·theta` added to both query and key on top of each vector's own angle. The solution is a rotation:
`f_q(x_m, m) = (W_q x_m) e^{i m theta}`, `f_k(x_n, n) = (W_k x_n) e^{i n theta}`, so
`<f_q, f_k> = Re[(W_q x_m)(W_k x_n)* e^{i(m−n)theta}]` — absolute `m, n` appear *only* through
`e^{i(m−n)theta}`.

Let me put a number on it so I am not trusting the algebra alone. Take `d = 2`, `theta = 1`, and let the raw
query and key both be the unit vector `(1, 0)`. Put the query at `m = 5`, the key at `n = 2`, offset 3: then
`f_q = R(5)·(1,0) = (cos 5, sin 5)`, `f_k = R(2)·(1,0) = (cos 2, sin 2)`, and their dot product is
`cos 5 · cos 2 + sin 5 · sin 2 = cos(5 − 2) = cos 3 ≈ −0.99`. Now slide *both* positions up by 95 — query at
`m = 100`, key at `n = 97`, same offset 3 — and the dot product is `cos(100 − 97) = cos 3 ≈ −0.99`,
identical. The logit did not move when I shifted both absolute positions by the same amount; it tracks only
the offset, the invariance the absolute `wpe` could never guarantee. Lift to the real head dimension by
splitting it into `d/2` independent 2-planes and rotating each at its own frequency: the inner product is a
sum over planes, each relative-only by the 2D argument, and a sum of relative-only-per-plane is
relative-only, so linearity glues it. The block-diagonal `R_m` has i-th 2×2 block a rotation by `m·theta_i`;
rotations compose, so `R_m^T R_n = R_{n−m}` and `q_m^T k_n = x_m^T W_q^T R_{n−m} W_k x_n` — the offset sits
in a single rotation between the content projections, no learned table, no clip, no distance bias. The
frequencies reuse the geometric spectrum `theta_i = 10000^{−2(i−1)/d}`: fast planes resolve local offsets,
slow planes carry coarse position, giving a graded long-range decay as a free prior — as `|m − n|` grows the
per-plane phases spread across frequencies, the partial sums lose coherence, and the positional contribution
decays, so far-apart tokens interact through a less coherent positional signal, all else equal. And because
`R` is orthogonal it preserves norm, so it can never blow up or collapse the representation as it propagates
through 24 layers — which is exactly the property I leaned on above to argue that RoPE already damps the
position-coupled drift, making the QK-Norm guard partly redundant.

Now the literal scaffold edit, which is *simpler* than step 2 — I remove the two `F.rms_norm` calls and keep
everything else. Position is no longer additive, so `self.use_pos_emb = False` in `__init__`, and
`GPT.forward` skips the `wpe` add — it gates on exactly that flag,
`getattr(self.transformer.h[0].attn, 'use_pos_emb', True)`, the one mechanism a rung has to replace position
without touching anything outside the attention class. I precompute
`inv_freq = 1/(10000 ** (arange(0, head_dim, 2)/head_dim))` as a buffer; per forward I form
`freqs = outer(arange(T), inv_freq)`, take `cos` and `sin`, and apply the *split-half* rotation —
`x1 = x[..., :d]`, `x2 = x[..., d:]` with `d = head_dim/2`, `y1 = x1·cos − x2·sin`, `y2 = x1·sin + x2·cos`,
concatenate `[y1, y2]` — to q and k only, never v: v carries the content that gets averaged into the output,
and position belongs in the *logit*, not the values. The split-half layout pairs coordinate `i` with `i + d`
as the two legs of one plane, the LLaMA/NeoX convention the harness's `_apply_rope` uses, equivalent up to a
fixed permutation to the interleaved `(2i, 2i+1)` pairing and immaterial to the logit as long as q and k are
rotated identically. The fused SDPA path, the causal mask, and the output projection are untouched, so the
difference from step 2 is exactly the QK-Norm removal — `q = self._apply_rope(q, T)` instead of
`q = self._apply_rope(F.rms_norm(q, ...), T)`. That is deliberately the minimal delta so the ablation is
clean: whatever the loss does between 2.2589 and this run is attributable to the RMSNorm deletion and nothing
else. Both configurations share the identical RoPE — frozen sinusoidal schedule, base 10000, frequencies
unlearned, base untuned — so I am not confounding the ablation with any position change; the only moving part
is whether q and k are put onto the fixed-norm sphere before the rotation.

Now the falsifiable expectations, against the 2.2589 RoPE + QK-Norm run. If the QK-Norm half is redundant
once position is relative, plain RoPE should land *at or slightly below* 2.2589 on validation loss — I am
predicting a small win, on the order of a couple of thousandths, because handing the q/k magnitudes back to
the optimizer as a sharpness control recovers a little of what the `sqrt(d_k)` ceiling pinned. It should be a
*small* win and not a large one, and the sharpness arithmetic says exactly why: the ceiling only binds the
heads that want to be maximally sharp, and most attention is genuinely diffuse over several tokens, so the
recovered sharpness helps a minority of heads, which caps the size of the gain. On perplexity I expect the
same direction — WikiText-2 and especially LAMBADA holding or improving, since better-calibrated per-head
sharpness helps the confident long-range predictions LAMBADA tests, and LAMBADA has been the most sensitive
perplexity to every real improvement so far. The downstream story is the one I am least sure of, and it is
where the ablation could split. Removing QK-Norm trades a uniform logit-scale robustness for sharpness
freedom, and it is entirely possible that plain RoPE wins the language-modeling metrics (val_loss,
perplexity) while RoPE + QK-Norm holds a slight edge on one or two downstream tasks where the extra scale
robustness happens to help calibration. If I had to guess which, it would be ARC-Easy and PIQA — the
multiple-choice tasks scored by comparing option likelihoods, where a small, uniform logit-scale robustness
can nudge the ranking of near-tied completions — while the sharper plain model would be likelier to take
HellaSwag and WinoGrande, where committing hard to the right continuation matters more than calibrated ties.
If that split happens — plain RoPE best on val_loss/perplexity, RoPE + QK-Norm best on a downstream task or
two — then the honest reading is that the strongest *language model* is plain RoPE, the rung I would rank
highest on the task's primary objective (held-out val_loss), while noting the combined form is not strictly
dominated. I do not expect the opposite outcome — plain RoPE clearly worse — because RoPE's own
norm-preservation is why I think the guard is now largely redundant; but that is the clean way the experiment
could tell me I am wrong, and if it does, the ladder should keep the combined form and look elsewhere for the
next lever.
