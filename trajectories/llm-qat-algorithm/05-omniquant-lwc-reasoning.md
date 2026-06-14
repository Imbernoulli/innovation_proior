LSQ closed the ladder's biggest gap and, in doing so, exposed the one seam left to press. At INT4 it
landed at 12.17 (degradation −1.04) — essentially STE's 11.70, the learnable-scale machinery neither
helping nor hurting much when 16 codes are already fine, as I expected. At INT3, 13.49 (degradation
0.29), a shade better than STE's 13.75 and within a third of a point of full precision. And at INT2 it
delivered exactly the payoff I predicted: from STE's 72.55 down to 19.50, a degradation of 6.29. That
is the first method to be genuinely usable at two bits, beating the `finetune_then_ptq` control's 94034
by orders of magnitude and proving the contribution is real QAT signal — learning the per-group step
size let the optimizer place the *levels* as well as the weights, clipping outliers and rounding the
bulk finely, which is precisely what the fixed max-abs scale could not do. So LSQ is the strongest rung,
and the task's actual requirement — low perplexity uniformly across INT4/INT3/INT2 — is met for the
first time.

But I left a question open at the close of the LSQ rung, and the INT2 number sharpens it. LSQ learns the
scale `s` as a *free, unbounded* parameter, optimized by AdamW with one tensor-wide gradient
normalizer `1/sqrt(N_elem·qmax)`. At INT2 the loss surface in `s` is brutal: with four codes, a small
change in `s` reassigns large blocks of weights across codes, so the step-size gradient is huge and
noisy near transitions. An unbounded `s` driven by that gradient can drift — overshoot to a scale that
clips too aggressively, or undershoot to one that wastes codes — and there is nothing in the
parameterization to keep it sane. The 6.29 degradation at INT2 is small, but I suspect a meaningful
slice of it is LSQ's scale wandering during the 500 steps rather than settling at the loss-optimal grid.
The diagnosis is specific: LSQ's *degree of freedom* is right (the grid should be learned) but its
*parameterization* is too loose (the scale is free to roam anywhere in (0, ∞)). The move is to keep the
learnable grid but **bound** it — parameterize the scale so that what is learned is a clipping *ratio*
constrained to a sensible range, not an unconstrained step size.

So here is the move that does exactly this, and that fits this edit surface without breaking the
algorithm: learnable weight clipping. Go back to the max-abs scale as the *anchor* — the no-clip cover
STE used — and learn a multiplicative *clipping factor* that shrinks it, where the factor is forced into
(0, 1) by a sigmoid. Concretely, for each group take the raw extremes
`xmax = max(w_g)` and `xmin = min(w_g)`, introduce two learnable per-group parameters
`upbound_factor` and `lowbound_factor`, gate them through a sigmoid, and clip the extremes:
`xmax ← sigmoid(upbound_factor)·xmax`, `xmin ← sigmoid(lowbound_factor)·xmin`. For symmetric signed
weight quantization the scale is then `s = max(|xmax|, |xmin|) / qmax`, clamped to a safe range
`[1e-5, 1e4]`, and the fake-quant is the straight-through `round(clip(w/s, qmin, qmax))·s`. The
gradient reaches the bound factors through the differentiable `sigmoid · extreme / qmax` scale; the
round is STE as before. I set `init_value = 4.0` for the
factors (so `sigmoid(4) ≈ 0.982` — the grid starts essentially at the max-abs cover, identical to STE,
and *learns to clip inward* from there), `CLIPMIN = 1e-5`, and per-group factors with shape
`(out_features, n_groups, 1)`.

Let me be precise about why this should beat LSQ at INT2, because the two are close cousins and the
difference is the whole bet. Both learn a per-group grid; the difference is the *coordinate* in which
they learn it. LSQ learns `s ∈ (0, ∞)` directly. LWC learns `γ = sigmoid(factor) ∈ (0, 1)` and sets
`s = γ · max|w_g| / qmax`. Three consequences fall out. First, the search space is *bounded*: γ cannot
exceed 1, so the learned scale can never blow past the max-abs cover — it can only clip inward, which is
exactly the useful direction at low bits (clip the outliers, round the bulk finely). LSQ's scale has no
such ceiling and can wander above the cover, wasting codes. Second, the parameterization is *anchored*:
because the scale is always `γ · max|w_g|`, it rides the current weight magnitude, so as the weights
move during training the grid moves with them automatically and the factor only has to learn the
relative clip, a far gentler quantity to optimize than an absolute scale. Third, the sigmoid *bounds the
gradient*: `dγ/dfactor = γ(1−γ)` saturates as γ → 0 or 1, which damps the huge, noisy step-size
gradients that destabilize LSQ at four codes — the parameterization itself provides the stability that
LSQ has no mechanism for. The starting point at `sigmoid(4) ≈ 0.982` means LWC begins essentially where
STE sits (near the max-abs cover) and improves monotonically by clipping inward, so it should never be
*worse* than STE, and it inherits LSQ's "learn the grid" power with a tighter, gradient-damped
parameterization — the cure for exactly the drift I suspect in LSQ's INT2 number.

One more thing to be careful about, because it is where the original form of this clip and this harness
part ways and I must not import machinery the edit surface cannot run. The clip's natural home is a
*per-block local reconstruction*: quantize one transformer block, minimize the MSE between its output
and the full-precision block's output on a small calibration set, no gradient through the rest of the
model — a PTQ-grade budget. This task's harness does not expose a per-block reconstruction loop; it
exposes the *global* QAT loop — full-model forward, cross-entropy on WikiText-2, AdamW over all
`requires_grad` parameters for 500 steps. So I do not get that cheap local objective; I get the same
end-to-end finetune every other rung uses, and the learnable factors are trained by the global task
loss alongside the weights. That is actually a *stronger* training signal than a local
reconstruction — the factors are optimized against the very metric I am scored on — so the LWC
parameterization should, if anything, do better here than its budget-constrained origin, not worse. The
one piece of LWC I therefore drop is the per-block scheduling; the parameterization itself — sigmoid-
gated per-group clip factors, init 4.0 — is exactly the canonical one, and it is the part that fixes the
INT2 rounding error. I keep the weights trainable too (the harness trains them anyway), so this is LWC's
clip layered on top of STE's weight-QAT, both driven by the global loss.

Now I land it in the task's edit surface, following the same wiring decisions LSQ forced, because the
harness's fixed post-training RTN would clobber a learned grid just as it would have clobbered LSQ's
scales. So `quantize_dequantize_weight` is again a deliberate no-op `weight.clone()`, and the real
quantization lives in the wrapper's `forward`, branching on `self.training`: during training the
differentiable LWC fake-quant with the sigmoid-gated factors; at eval a genuine no-grad
`round(clip(w/s))·s` on the *learned* clipped grid (recomputing the per-group extremes and scale from
the final weights and the final factors), so evaluation sees the LWC grid and not the max-abs grid. The
factors `upbound_factor` and `lowbound_factor` are registered as `nn.Parameter`s of shape
`(out_features, n_groups, 1)` initialized to 4.0, so the harness's optimizer-over-all-`requires_grad`-
params trains them alongside the weights with no extra plumbing. Activations stay full precision
(weight-only), and the LM head is restored to plain Linear — both unchanged from LSQ. I keep the same
500-step schedule (`lr=2e-5`, `warmup=50`, cosine to 10%, grad-accum 4, grad-clip 1.0) so the
comparison to LSQ holds everything constant except the parameterization of the learned grid. One
detail I have to get right so the clip is faithful: I compute raw `xmax`/`xmin` as the per-group
*signed* max and min (not max-abs), gate each by its own sigmoid factor, and *then*
take `max(|xmax|, |xmin|)` for the symmetric scale — so I keep both factors and the signed-extreme
computation rather than collapsing to a single max-abs factor, which would be a different (and less
faithful) method. The full scaffold module is in the answer.

The causal chain in one breath: LSQ's measured INT2 (19.50, degradation 6.29) proved that *learning the
grid* is the right degree of freedom, but its *free unbounded scale* is too loose a parameterization and
likely drifts under the huge four-code step-size gradient → keep the learnable grid but bound it: learn a
sigmoid-gated clipping factor γ ∈ (0,1) on the max-abs cover, `s = γ·max|w_g|/qmax`, init `sigmoid(4)≈
0.982` so it starts at STE and clips inward, with `γ(1−γ)` damping the gradient → wire it into the task
exactly as LSQ (no-op real-QDQ, eval-branch QDQ in the wrapper, per-group `nn.Parameter`s) so the only
change from LSQ is the coordinate the grid is learned in.

This finale carries no feedback, so the bar it must clear is the strongest baseline's measured numbers,
stated as falsifiable expectations. At INT4 LSQ landed 12.17 and STE 11.70; since LWC starts at the
max-abs cover (≈ STE) and only clips inward, I expect it to match or modestly beat both — somewhere
around 11.7–12.2, certainly not worse than STE. At INT3 LSQ reached 13.49; I expect LWC to match or
slightly beat it, low-to-mid teens, within a point of full precision. INT2 is where the claim lives: LSQ
reached 19.50 (degradation 6.29), and if my diagnosis is right — that a bounded, gradient-damped,
weight-anchored clip is more stable than a free scale at four codes — LWC should clear LSQ at INT2,
landing *below* 19.50 (I would expect mid-to-high teens, degradation under 6). The falsifiable failure
mode is explicit: if LWC lands *at or above* 19.50 at INT2 with no INT4/INT3 regression, then the
bounded parameterization bought nothing over LSQ's free scale and the drift I hypothesized was not the
binding constraint — in which case LSQ remains the right answer and the grid-placement gap at two bits
must lie somewhere this clip also does not reach (e.g. the block-wise reconstruction objective the clip
is naturally paired with, which this single-linear edit surface does not expose). The minimal, honest
claim is that LWC
should not regress INT4/INT3 and should not be worse than LSQ at INT2; the optimistic claim is a clear
INT2 improvement from the bounded, stable clip.
