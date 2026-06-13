Plain RoPE is the strongest rung so far, and the way it won tells me where the remaining loss is hiding.
Stripping the QK-Norm half edged the combined model out exactly as I predicted: validation loss 2.2570
against RoPE + QK-Norm's 2.2589, WikiText-2 43.17 vs 43.44, LAMBADA 65.81 vs 67.20 — plain RoPE best on
every language-modeling metric, confirming that once position is relative, pinning q/k to the
`sqrt(d_k)·cos` ceiling cost more sharpness than it bought in stability. The split I flagged did show up
on downstream: RoPE + QK-Norm kept a slight edge on ARC-Easy (57.83 vs 57.32) and PIQA (64.74 vs 64.42),
the multiple-choice tasks, while plain RoPE took HellaSwag (34.48 vs 34.24) and WinoGrande (51.70 vs
50.67). So plain RoPE is the strongest *language model* and the right rung to build on, with the combined
form not strictly dominated. But step back and look at what *every* rung so far has in common, because
that is the seam none of them touched. QK-Norm changed the logit *scale*. RoPE changed how *position*
enters the logit. Neither touched the *shape of the attention distribution itself* — all three rungs end
in the same operation: a single softmax over the context, then a weighted average of values. And that
single softmax has a structural defect that no amount of better position or better scaling can fix.

Here is the defect, stated precisely. The attention weights are `a_{m,n} = exp(s_{m,n}) / sum_{n'}
exp(s_{m,n'})`, and every `exp(·)` is strictly positive. So no finite logit can drive a weight to exactly
zero — the softmax can make an irrelevant token's weight small, but never zero, and it has no mechanism to
take mass *back off* a position. Over a 1024-token context, even if every irrelevant token gets a weight
of order `1/T`, the *aggregate* of those weights is most of the mass. The output `o_m = sum_n a_{m,n} v_n`
is therefore part signal — the few tokens that matter — and part a low-level average of the irrelevant
remainder. RoPE made the model place its *peak* attention in the right relative location; it did nothing
about the noise floor smeared under that peak, because the floor is a property of the softmax's positivity,
not of where the peak is. This is the attention-noise problem, and it is exactly the kind of structural
limitation that a position fix or a scale fix cannot reach. The 2.2570 is, in part, the cost of averaging
junk into every token's representation.

So the question is whether I can change the operator so it can *subtract* mass from irrelevant positions —
because subtraction is the one thing a single positive softmax cannot do. Where does "cancel a common
floor by subtracting two correlated signals" work? A differential amplifier rejects the voltage common to
its two inputs and amplifies only their difference, cancelling common-mode noise; noise-cancelling
headphones subtract an estimate of the ambient floor. The shared principle: the noise is *common* to two
channels, the signal *differs* between them, so the difference keeps the signal and kills the noise. I want
attention built the same way.

The construction follows directly. Form *two* softmax maps over the same context from two different
query/key projections — `A1 = softmax(Q1 K1^T / sqrt(d))`, `A2 = softmax(Q2 K2^T / sqrt(d))` — and take
the differential `A1 - lambda·A2` with a learnable scalar `lambda`, then average the values with that
signed map: `(softmax(Q1 K1^T/sqrt(d)) - lambda·softmax(Q2 K2^T/sqrt(d))) V`. Check it does what I claim.
Both maps see the same content, so over the irrelevant tail their floor patterns are *correlated* — that
is the common-mode part, and `A1 - lambda·A2` drives it toward zero. On the relevant tokens, the model has
an incentive to make the first map spike where the second does not, so the difference is large there. The
resulting weights no longer sum to one and are no longer all positive — they are *signed*, which is exactly
the new power I wanted: the operator can now push an irrelevant value's contribution to zero or below
instead of being stuck with the positive floor. The trivial solution `lambda = 0` is not the optimum,
because cancelling the floor genuinely lowers loss — a cleaner average over the relevant values predicts
the next token better — so the optimizer has a real reason to use the subtraction. This is the
representational lever that QK-Norm and RoPE both lacked, because both ended in one positive softmax.

Now the engineering that makes it a fair, stable drop-in, because a naive version would double the cost
and destabilize training. Three pieces. First, `lambda`: I do not learn it as a free scalar (badly
conditioned, can drift the subtraction into a wild magnitude). I re-parameterize it as
`lambda = exp(lambda_q1·lambda_k1) - exp(lambda_q2·lambda_k2) + lambda_init` with four learnable
`head_dim` vectors initialized `N(0, 0.1)`, so at init the two exponentials are ~1, roughly cancel, and
`lambda ≈ lambda_init` with well-scaled signed gradients. Second, the budget. I have doubled the queries
and keys, so to match a vanilla head's parameters and FLOPs I *halve the head dimension*: each logical
head uses `head_dim = n_embd / n_head / 2`, with `2·n_head` query/key sub-heads of that dimension and
`n_head` value heads of dimension `2·head_dim`. Total q/k width `2·n_head·(head_dim) = n_embd`, total v
width `n_head·(2·head_dim) = n_embd` — exactly the vanilla widths, so the fused `c_attn` projection is
unchanged and the compute is matched; the doubling is absorbed by the halving. Third, scale. The
subtraction makes heads heterogeneous and shrinks the operator gain roughly by `(1 - lambda)`, so I
per-head normalize each head's `2·head_dim` output (a per-head RMSNorm, the GroupNorm-across-heads
discipline) and then rescale by the *fixed* constant `(1 - lambda_init)` — fixed, not the learned
`lambda`, so the gain compensation is a stable constant and the normalization carries the rest. That fixed
compensation is what lets the frozen GPT-2 Medium optimizer, learning rate, and schedule transfer
unchanged, which matters here because the loop is fixed and I cannot retune.

Now the part that is specific to *this* edit surface, and where the harness forces two real compromises I
have to name. The editable region is only `CausalSelfAttention(config)` — and `config` does not carry the
*layer index*. The depth-scaled `lambda_init = 0.8 - 0.6·exp(-0.3·(l-1))` needs `l`, which I do not have:
every one of the 24 blocks constructs its attention from the same `config` with no `l`. So I cannot
realize the depth schedule; I set a single fixed `lambda_init = 0.8` (the schedule's deep-layer asymptote
and the operating point the reparameterization is centered on). This is a genuine omission relative to the
full method — the early layers will cancel harder than the schedule would prescribe — and it is forced by
the edit surface, not a choice. The second compromise: the fused SDPA path returns only the final averaged
output, not the two softmax maps I need to subtract. So I cannot use Flash here; I take the manual
masked-softmax path the scaffold already provides as the non-flash fallback, compute the `2·n_head`
attention maps explicitly, reshape to `(B, n_head, 2, T, T)`, subtract, and matmul with v. That costs the
memory of the explicit `T×T` maps — the same cost the manual fallback always had — and it is unavoidable
because the differential subtraction lives *between* the softmax and the value-average, exactly where the
fused kernel hides its internals. Position stays as the strongest baseline's: RoPE on the doubled q/k
sub-heads (`use_pos_emb = False`, the split-half rotation from step 3 applied to the `head_dim`-wide
sub-heads), so this rung is "the strongest position scheme, with the single softmax replaced by a
differential one." The full scaffold module is in the answer.

Let me state the bar this has to clear, against the real numbers, with no invented results. The rung to
beat is plain RoPE: validation loss 2.2570, WikiText-2 43.17, LAMBADA 65.81, ARC-Easy 57.32, HellaSwag
34.48, PIQA 64.42, WinoGrande 51.70. The differential operator is parameter- and FLOP-matched, so any
gain is purely from the changed attention *shape*, not from added capacity — which is the honest way to
claim it. My falsifiable expectation: because the noise floor is a real fraction of the attention mass on
a 1024-token context and the subtraction removes it, validation loss and both perplexities should fall
below the RoPE numbers — LAMBADA most of all, since its long-passage last-word prediction is exactly the
retrieval-under-noise setting the cancellation targets, so the 65.81 is the number I would watch first.
Downstream should move with the LM gain, with the in-context, retrieval-flavored tasks (ARC-Easy,
HellaSwag) the likeliest to lift, since cleaner attention to the relevant context is what those reward.
The risks I would actually validate before believing the win: (1) the fixed `lambda_init = 0.8` may
over-cancel in the early layers — if I could pass depth I would expect a further gain, so a plain-RoPE-beating
but modest result might be the depth-schedule omission showing, not a failure of the method; (2) the
signed map can, at init, produce a higher-variance output than a single softmax, so I would watch the
first few hundred iterations for an instability the `(1 - lambda_init)` compensation and the per-head norm
are supposed to prevent. If validation loss does *not* drop below 2.2570, the honest reading is that at
355M params and 7.1B tokens the attention-noise floor is not yet the binding constraint — the
differential mechanism's reported advantage grows with scale and context length, and this is a small-model,
1024-token regime — in which case the strongest rung remains plain RoPE and the differential operator is a
bet that pays off at larger scale than this task runs.
