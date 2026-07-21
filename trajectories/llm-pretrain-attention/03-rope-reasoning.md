The RoPE + QK-Norm run came back where I hoped and sharpened the one question I deliberately left open.
Validation loss landed at 2.2589 against the QK-Norm floor of 2.2885 — a drop of 0.0296. In perplexity that
is `exp(2.2885) ≈ 9.860 → exp(2.2589) ≈ 9.573`, about a 2.9% reduction per token — far past anything a
stability tweak buys, so this is the representational margin I predicted relative position would open,
arriving through exactly the channel I argued. The *pattern* of the perplexities is the tell: WikiText-2
went 43.65 → 43.44 (0.5% relative), LAMBADA 69.99 → 67.20 (4.0% relative) — eight times the relative move.
That asymmetry is the position story writing itself into the metrics, LAMBADA's long-range last-word
dependency being exactly where a relative encoding earns its keep while WikiText-2 is dominated by
short-range continuations. Downstream followed the LM gain: ARC-Easy 55.64 → 57.83, HellaSwag 33.41 → 34.24,
PIQA 63.17 → 64.74. WinoGrande alone did not track it (51.30 → 50.67), but it sits right on its 50-point
two-way chance floor at both ends, so I read that wobble as single-seed noise, not "position hurt
coreference." The position fix is real and general, not a single-metric artifact.

But one number cannot tell me the thing that decides the next step. The 2.2589 is *two* changes stacked —
RMSNorm on q and k, and RoPE on top. I proved in step 1 that the RMSNorm half is real and free, but in a
world where position was absolute and additive. Now position is relative, and I have no way, from one
combined run, to attribute the 0.0296 between the two operations. Did RMSNorm still contribute here, did
RoPE do all the work while RMSNorm rode along neutral, or — the possibility I flagged — is the RMSNorm half
now quietly *costing* me, its fixed-norm sphere pinning a sharpness I would rather the optimizer control? A
single combined run cannot separate those three stories; they all produce a good number. The only clean way
to read the answer off the metrics is to hold everything else fixed, delete the RMSNorm, run plain RoPE, and
attribute the entire difference to that one deletion.

Before committing to the deletion, the alternatives, because "strip it out" is only obviously right once I
have looked at the others and found them worse. I could keep the combined form as the accepted baseline and
spend this step on a *different* lever — head structure or the distribution shape — treating the QK-Norm
question as settled enough. I could delete the RMSNorm for the pure ablation. Or I could *upgrade* the
QK-Norm half to the learned-`g` form I derived on the blank page back in step 1 — L2-normalize q and k and
multiply the logit by a per-head learned `g`, so the sharpness ceiling stops being pinned at `sqrt(d_k)`.
That upgrade is tempting because it looks like it keeps the guard *and* removes the ceiling, but it is wrong
for *this* step. First, it does not answer the question I have: the way to learn whether a thing is needed
is to remove it and look, not replace it with a more elaborate version. Learned-`g` changes two things at
once — removes the ceiling *and* keeps stripping the magnitude — so an improvement would leave me back in
the confounded reading the combined run already handed me. Second, it spends parameters to recover a control
plain RoPE gives free: learned-`g` throws away the q/k magnitude and reintroduces sharpness through a
separate learned dial (one scalar per head, `16·24 = 384` numbers on the frozen budget), whereas plain RoPE
hands the same control back with no new parameter — the magnitudes `||q||, ||k||` the projections already
produce *are* the sharpness dial, and the optimizer already knows how to move `W_q, W_k` to set them. So I
set it aside; plain RoPE is the strictly cleaner probe of the exact question I am holding. The RoPE base I
decline again for the same budget discipline as step 2.

The whole bet rides on whether the RMSNorm ceiling is costing me sharpness the model wants, and I will not
run an ablation whose sign I cannot predict, so put a number on the ceiling. The RMSNorm form forces each
per-head q, k onto a fixed-norm sphere, realizing logit `sqrt(d_k)·cos(angle) = 8·cos`. Take a head that
should attend almost entirely to one key: the best it can do under the cap is drive the target's cosine to
+1, logit 8. Its competitors — call it ~511 keys under the causal mask mid-sequence — sit at whatever cosine
their content gives, which for near-orthogonal directions in 64 dimensions has mean 0 and variance `1/d`, so
standard deviation `1/sqrt(64) = 0.125`, and `8·cos ~ N(0, 1)`. Peak mass is
`exp(8)/(exp(8) + sum exp(8·cos_j))`; with `E[exp(8·cos)] = exp(1/2) = 1.649` the competitor sum is
`511·1.649 ≈ 843`, and against `exp(8) = 2981` the peak is `2981/(2981 + 843) ≈ 0.78`. Under the cosine cap a
maximally confident head tops out around 78% of its mass on the single most relevant key — not a
late-training degradation but the ceiling for every sharp head from step 0 to the last iteration.

Now plain RoPE off the leash. Without RMSNorm the logit is `||q||·||k||·cos/sqrt(d_k)`; at init
`||q|| ≈ ||k|| ≈ 8`, so the coefficient is `8·8/8 = 8`, the same `8·cos` as the capped version — but now the
model can *grow* `||q||, ||k||` on a head that should be sharp, and the coefficient grows as `8·c²`. To reach
coefficient 16 each norm need grow by only `c = sqrt(2) ≈ 1.41`, well inside what weight decay 0.1 permits:
then competitors are `16·cos ~ N(0, 4)`, `E[exp] = exp(2) = 7.39`, sum `511·7.39 ≈ 3776`, and against
`exp(16) = 8.89×10^6` the peak is `≈ 0.9996`. So the capped head is frozen at 78% while a modest,
decay-plausible magnitude growth carries the uncapped head to 99.96% — and this is a *first-order* limit,
binding every sharp head throughout all of training.

I have to be honest that removing RMSNorm reintroduces the exact drift it fixed: `W_q, W_k` growing over the
run, the scaled logit σ inflating, the softmax creeping toward saturation late. Why is that the right trade
now when it was not at step 1? Two reasons, both about what changed between the steps. First, position
changed: RoPE applies an orthogonal rotation, contributing *zero* drift, and it replaced the free additive
`wpe` table that a chunk of step-1's instability was coupled to — so the residual drift RMSNorm still fights
is smaller than in the step-1 setting. Second, the two effects live on different orders: the sharpness
ceiling is first-order (caps peak at 78% for every sharp head, unconditionally, all through training), the
drift is second-order (only bites once weights grow, and weight decay bounds how far, balancing at a finite
equilibrium norm — self-limiting in a way the ceiling is not). My bet is concrete: the first-order sharpness
I recover outweighs the second-order stability I give up, so plain RoPE should land at or slightly below
2.2589. And the deletion is nearly a no-op at init — with `||q|| ≈ ||k|| ≈ 8` both configurations realize
`8·cos` at step 0 — so plain RoPE opens late-training headroom without disturbing the tuned start; the worst
honest case is that it changes nothing, not that it breaks the run.

So step 3 is plain RoPE: the same relative-by-construction rotation I derived and validated last step — the
constraint-solve that makes the logit depend on `x_m, x_n, m-n` only, lifted across `d/2` geometric-frequency
2-planes, orthogonal and norm-preserving — with the QK-Norm removed. The literal edit is *simpler* than step
2: drop the two `F.rms_norm` calls and keep everything else. `self.use_pos_emb = False` so `GPT.forward`
skips the `wpe` add (it gates on exactly that flag); precompute
`inv_freq = 1/(10000^{(arange(0, head_dim, 2)/head_dim)})`; per forward build `cos, sin` from
`outer(arange(T), inv_freq)` and apply the split-half rotation to q and k only, never v. The fused SDPA
path, the causal mask, and the output projection are untouched, so the difference from step 2 is exactly the
QK-Norm removal — `q = self._apply_rope(q, T)` instead of `self._apply_rope(F.rms_norm(q, ...), T)`. That
minimal delta is the point: whatever the loss does between 2.2589 and this run is attributable to the
RMSNorm deletion, both configurations sharing the identical RoPE (frozen schedule, base 10000, frequencies
and base unlearned), so nothing confounds the ablation but whether q and k are put on the fixed-norm sphere
before rotating.

Now the expectations against the 2.2589 run. If the QK-Norm half is redundant once position is relative,
plain RoPE should land *at or slightly below* 2.2589 — a small win, on the order of thousandths, because
handing the q/k magnitudes back to the optimizer as a sharpness control recovers a little of what the
`sqrt(d_k)` ceiling pinned. It should be small and not large, and the sharpness arithmetic says why: the
ceiling only binds the heads that want to be maximally sharp, and most attention is genuinely diffuse over
several tokens, so the recovered sharpness helps a minority of heads. On perplexity I expect the same
direction, LAMBADA the most sensitive as ever. Downstream is where I am least sure, and where the ablation
could split: removing QK-Norm trades a uniform logit-scale robustness for sharpness freedom, so it is
entirely possible that plain RoPE wins the language-modeling metrics while the combined form clings to a
slight edge on one or two of the multiple-choice downstream tasks — ARC-Easy or PIQA, scored by comparing
option likelihoods, where a small uniform scale robustness can nudge the ranking of near-tied completions.
If that split happens, the honest reading is that the strongest *language model* is plain RoPE — the
configuration I would rank highest on the primary objective, held-out val_loss — with the combined form not
strictly dominated. I do not expect plain RoPE clearly worse, because RoPE's own norm-preservation is why I
think the guard is now largely redundant; but that is the clean way the experiment could tell me I am wrong,
and if it does, the ladder keeps the combined form and looks elsewhere for the next lever.
