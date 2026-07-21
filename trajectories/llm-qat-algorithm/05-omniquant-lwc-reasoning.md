LSQ closed the ladder's biggest gap and, in doing so, exposed the one seam left to press. At INT4 it
landed at 12.17 (degradation −1.04), at INT3 13.49 (degradation 0.29), and at INT2 19.50 (degradation
6.29) — the first method genuinely usable at two bits, beating the control's 94034 by orders of magnitude
and proving the contribution is real QAT signal. Read the three against STE for *where* learning the grid
helped. At INT2, STE's 72.55 fell to 19.50 — in bits, `2.46 → 0.56`, a recovery of `1.90`, exactly the
payoff from letting the optimizer place the levels as well as the weights. At INT3, 13.75 to 13.49, a
slender `0.26` PPL — learning the 8-code grid trims a little. But at INT4 something worth staring at:
STE's 11.70 *rose* to LSQ's 12.17, about `0.056` bits per token *worse* at four codes, where STE already
sat below full precision.

That INT4 regression is small but it is not noise, and it is the most informative number in the table. At
INT4 the grid is fine enough that the max-abs scale is already very nearly optimal — a method that
*learns* the scale there should at worst learn to leave it alone. Instead LSQ's free scale drifted off the
near-optimal cover and cost accuracy. So the unbounded parameterization is not merely failing to help
where help is unneeded, it is an active liability, wandering when it should sit still. And if it wanders
at INT4 where the loss surface in `s` is gentle, it wanders more at INT2 where the surface is brutal: with
four codes a small change in `s` reassigns large blocks of weights across codes, so the step-size gradient
is huge and noisy near transitions, and an unbounded `s` — even damped by the tensor-wide
`1/sqrt(N_elem·qmax)` normalizer — can overshoot to a scale that clips too aggressively or undershoot to
one that wastes codes, with nothing in the parameterization to keep it sane. The `0.56` bits LSQ still
leaves at INT2 is small, but the INT4 regression makes me suspect a real slice of it is the scale
wandering during the 500 steps rather than settling at the loss-optimal grid. The diagnosis is specific:
LSQ's *degree of freedom* is right — the grid should be learned — but its *parameterization* is too loose,
free to roam anywhere in `(0, ∞)`, above the max-abs cover as easily as below it. The move is to keep the
learnable grid but **bound** it: parameterize the scale so that what is learned is a clipping *ratio*
constrained to a sensible range.

"Constrain `s`" admits more than one shape, but the alternatives fall short. A hard clamp of LSQ's free
scale to `[s_lo, s_hi]` gives no gradient as it approaches the wall, does nothing to damp the huge
four-code step-size gradient in the interior, and still learns an *absolute* scale that does not ride the
weight magnitude. Lowering LSQ's learning rate only slows the drift without bounding it, and it is blunt —
at INT2 the scale genuinely needs to travel a long way inward from its init, so a lower rate would starve
exactly the motion I want while only partly taming the drift I do not. What I need is a coordinate whose
*reachable set* is bounded and whose *gradient* is naturally quiet where the run is fragile and lively
where the clip is being decided. Go back to the max-abs scale as the *anchor* — the no-clip cover STE
used — and learn a multiplicative clipping factor that shrinks it, forced into `(0, 1)` with a sigmoid.

Concretely, for each group take the raw extremes `xmax = max(w_g)` and `xmin = min(w_g)`, introduce two
learnable per-group parameters `upbound_factor` and `lowbound_factor`, gate them through a sigmoid, and
clip: `xmax ← sigmoid(upbound_factor)·xmax`, `xmin ← sigmoid(lowbound_factor)·xmin`. For symmetric signed
quantization the scale is `s = max(|xmax|, |xmin|) / qmax`, clamped to `[1e-5, 1e4]`, and the fake-quant
is the straight-through `round(clip(w/s, qmin, qmax))·s`. The gradient reaches the factors through the
differentiable `sigmoid · extreme / qmax` scale; the round is STE as before. I set `init_value = 4.0` and
`CLIPMIN = 1e-5`, with per-group factors of shape `(out, n_groups, 1)`.

Both LSQ and this learn a per-group grid; the difference is the *coordinate*. LSQ learns `s ∈ (0, ∞)`
directly; LWC learns `γ = sigmoid(factor) ∈ (0, 1)` and sets `s = γ · max|w_g| / qmax`. Three consequences
fall out, each checkable. First, the search space is *bounded*: `γ ≤ 1` means `s ≤ max|w_g|/qmax`, exactly
STE's max-abs scale, so LWC can only clip inward, never coarsen past the cover — precisely the useful
direction at low bits, and precisely the direction LSQ violated when it drifted above the cover and
regressed INT4. A parameterization that structurally cannot make LSQ's INT4 mistake is already a reason to
expect no regression there. Second, it is *anchored*: because `s = γ · max|w_g|`, it rides the current
weight magnitude, so as the weights move during training the grid moves with them and the factor only
learns the *relative* clip. This matters more than it looks, because the weights are not frozen during the
500 steps — STE's weight-QAT runs underneath, reorganizing them, so `max|w_g|` drifts step to step. LSQ's
absolute `s` has to track that drift with its own gradient, spending effort just to stay in place; if the
weights grow, an `s` that was well-tuned is suddenly too small and clips too hard until it catches up.
LWC's factor is immune — when `max|w_g|` grows, `s = γ·max|w_g|` grows in lockstep for free, and "how far
inside the cover should I clip" is a stationary target even while the weights move. So the anchoring
decouples the grid-learning from the weight-learning that shares the loop, exactly the interference that
could make LSQ's scale wander at INT2 where weight updates are largest. Third, the sigmoid *bounds the
gradient*: `dγ/dfactor = γ(1−γ)` saturates as `γ → 0` or `1`, damping the huge noisy step-size gradients
that destabilize LSQ at four codes.

The reachable set tells me what LWC can and cannot become. As `factor → +∞`, `γ → 1` and `s →
max(|xmax|,|xmin|)/qmax` — *exactly STE's fixed scale*, so STE is the `γ = 1` corner of LWC's search
space: LWC contains STE as a boundary case and opens the interior `γ < 1` toward finer clipped grids. As
`factor → −∞`, `γ → 0`, `s → 0`, a degenerate corner held off by the `CLIPMIN = 1e-5` floor. This is the
property I most want from a finale: because the same global loss trains the factors and STE is inside the
search space, the trained LWC can always fall back to the STE corner if clipping does not help, so on the
*training* objective it can only match or beat STE — it strictly generalizes the rung that scored 11.70 /
13.75 / 72.55. Whether that survives to *test* perplexity is the empirical question INT2 will answer, but
I am not risking a regression against STE by construction, only betting the interior beats the corner.
The start confirms it: `sigmoid(4) = 1/(1 + e^{-4}) = 0.982`, so at init `s = 0.982 · max|w_g| / qmax`,
within 2% of STE's cover — LWC begins essentially where STE sits and learns to clip inward. And the
gradient there is `γ(1−γ) = 0.982 · 0.018 = 0.0177` against the peak `0.25` at `γ = 0.5`, a fourteen-fold
damping right at initialization, so the factor barely moves in the first steps and the run cannot lurch
off the cover; as a group's factor descends toward `γ = 0.5` the gradient grows to its `0.25` maximum —
most responsive in the middle clip regime where useful clipping happens, quiet at both ends.

Make the stability quantitative by comparing the two coordinates at the same loss gradient. LSQ moves the
absolute scale, its sensitivity `∂L/∂s` the summed step-size residual, large near a four-code transition
(the interior term reaches `≈ 0.5` per stranded weight, summed over the group). LWC moves the factor, and
by the chain rule `∂L/∂factor = ∂L/∂s · (max|w_g|/qmax) · γ(1−γ)`. So for the *same* loss pressure `∂L/∂s`,
LWC's factor update carries an extra `γ(1−γ)` gate — `0.018` at the start, rising only to `0.25` at
mid-clip. LWC applies at most a quarter, and at initialization a fifty-sixth, of the drive LSQ applies at
the same transition. The huge noisy four-code gradients that pushed LSQ off the grid at INT4 and made it
wander at INT2 are structurally throttled by the coordinate itself, most where the run is most fragile —
the opposite of LSQ, whose gradient was largest and noisiest precisely where it most needed to be calm.

I should be honest about what this does *not* recover. The final scale is still symmetric, `s =
max(|xmax|,|xmin|)/qmax`, so the extra negative code the signed-symmetric grid wastes — a full quarter of
the codebook at INT2, from the rung-1 accounting — stays unused. The two factors let the clip be
asymmetric in how far each extreme is pulled in before the symmetric scale is taken, which matters when a
group's tails differ, but they do not turn the grid asymmetric and so reclaim no code. LWC's entire gain
is the clip — the same rounding-versus-clipping trade LSQ found, now reached from the STE cover by a
damped, weight-anchored factor that cannot overshoot past the cover on the way. At INT2 a group with an
outlier at `0.9` and bulk near `±0.2` starts at `s ≈ 0.88`, essentially STE's cover, so the bulk is
crushed as before; as the factor learns `γ ≈ 0.3` the anchored scale becomes `s ≈ 0.27`, the bulk weight
`0.2` lands on a code instead of vanishing, and the outlier is clipped. The bet is that this controlled
descent lands closer to the loss-optimal clip, and more reliably, than LSQ's free scale drifting from its
own init.

One place the original form of this clip and this harness part ways, and I must not import machinery the
edit surface cannot run. The clip's natural home is a *per-block local reconstruction*: quantize one
transformer block, minimize the MSE to the full-precision block's output on a small calibration set, no
gradient through the rest — a PTQ-grade budget. This harness does not expose a per-block loop; it exposes
the *global* QAT loop — full-model forward, cross-entropy on WikiText-2, AdamW over all `requires_grad`
parameters for 500 steps. So I drop the per-block scheduling and train the factors by the global task
loss alongside the weights. That is actually a *stronger* signal than local reconstruction — the factors
are optimized against the very metric I am scored on, cross-entropy on the eval domain, not a proxy MSE —
so the parameterization should if anything do better here than its budget-constrained origin. The
parameterization itself — sigmoid-gated per-group clip factors, init 4.0 — is exactly the canonical one,
and it is the part that fixes the INT2 grid-placement error.

I land it in the edit surface following the same wiring LSQ forced, because the fixed post-training RTN
would clobber a learned grid. So `quantize_dequantize_weight` is again a no-op `weight.clone()`, and the
real quantization lives in the wrapper's `forward`, branching on `self.training`: differentiable LWC
fake-quant with the sigmoid-gated factors during training; a genuine no-grad `round(clip(w/s))·s` on the
learned clipped grid at eval, recomputing the per-group extremes and scale from the final weights and
factors, so evaluation sees the LWC grid. The factors are `nn.Parameter`s of shape `(out, n_groups, 1)`
initialized to 4.0, so the harness's optimizer trains them with no extra plumbing. Activations stay full
precision and the LM head is restored to plain Linear, both unchanged from LSQ, and the 500-step schedule
is held so the comparison to LSQ isolates only the parameterization. One detail for faithfulness: I
compute raw `xmax`/`xmin` as the per-group *signed* max and min (not max-abs), gate each by its own
sigmoid factor, and *then* take `max(|xmax|, |xmin|)` for the symmetric scale — keeping both factors and
the signed-extreme computation rather than collapsing to a single max-abs factor, which would be a
different and less faithful method. The shapes close: `w` reshapes to `(out, n_groups, group_size)`, `amax`
and `amin` over `dim=−1` give `(out, n_groups, 1)`, the gated product and `max(|·|,|·|)` stay `(out,
n_groups, 1)`, and `w/s` broadcasts that scalar over the 128 columns of its group — one independent clip
per group. The full scaffold module is in the answer.

This finale carries no feedback, so its bar is the strongest baseline's measured numbers, as falsifiable
expectations. At INT4 LSQ landed 12.17 and STE 11.70; since LWC starts at the cover (≈ STE) with a
fourteen-fold-damped gradient and can only clip inward, I expect it to match or modestly beat both, around
11.7–12.2, and specifically *not* to reproduce LSQ's INT4 regression, because the coordinate that caused
it — a scale free to drift above the cover — no longer exists. At INT3 LSQ reached 13.49; I expect LWC to
match or slightly beat it, low-to-mid teens within a point of full precision. INT2 is where the claim
lives: LSQ reached 19.50 (`0.56` bits/token), and if a bounded, gradient-damped, weight-anchored clip is
more stable than a free scale at four codes, LWC should clear LSQ at INT2, landing below 19.50 —
degradation under 6, recovering a further slice of that `0.56`-bit gap. The falsifiable failure mode is
explicit: if LWC lands at or above 19.50 at INT2 with no INT4/INT3 regression, the bounded
parameterization bought nothing over LSQ's free scale and the drift I hypothesized was not the binding
constraint — in which case LSQ remains the answer and the residual two-bit gap lies somewhere this clip
also does not reach, for instance the block-wise reconstruction objective this single-linear edit surface
does not expose. The minimal honest claim is that LWC should not regress INT4/INT3 and should not be worse
than LSQ at INT2; the optimistic claim is a clear INT2 improvement from the bounded, stable clip — the last
turn of a screw whose ledger is the INT2 column, five digits then two then teens, every drop bought by
moving first the weights, then the levels, then the coordinate the levels are learned in.
