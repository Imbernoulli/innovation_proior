LSQ closed the ladder's biggest gap and, in doing so, exposed the one seam left to press. At INT4 it
landed at 12.17 (degradation −1.04), at INT3 13.49 (degradation 0.29), and at INT2 19.50 (degradation
6.29) — the first method genuinely usable at two bits, beating the `finetune_then_ptq` control's 94034 by
orders of magnitude and proving the contribution is real QAT signal. Let me read the three numbers
against STE the way that tells me *where* learning the grid helped and where it did not, because that
pattern is what points to the finale. At INT2, STE's 72.55 fell to LSQ's 19.50 — in bits per token,
`2.46 → 0.56`, a recovery of `1.90` bits, exactly the payoff I predicted from letting the optimizer place
the levels as well as the weights. At INT3, STE's 13.75 to LSQ's 13.49, a slender `0.26` PPL better —
learning the 8-code grid trims a little. But at INT4 something worth staring at happened: STE's 11.70
*rose* to LSQ's 12.17. LSQ was `0.47` PPL — about `0.056` bits per token — *worse* than STE at four codes,
where STE already sat below full precision.

That INT4 regression is small, but it is not noise, and it is the most informative number in the table.
At INT4 the grid is fine enough that the max-abs scale is already very nearly optimal — STE's fixed cover
had almost nothing to improve on. A method that *learns* the scale there should, at worst, learn to leave
it alone. Instead LSQ's free scale drifted off the near-optimal cover and cost accuracy. So the free,
unbounded parameterization is not merely failing to help where help is unneeded — it is an active
liability, wandering when it should sit still. And if the free scale wanders at INT4 where the loss
surface in `s` is gentle, it is surely wandering more at INT2 where the surface is brutal: with four
codes, a small change in `s` reassigns large blocks of weights across codes, so the step-size gradient is
huge and noisy near transitions, and an unbounded `s` driven by that gradient — even one damped by the
tensor-wide `1/sqrt(N_elem·qmax)` normalizer — can overshoot to a scale that clips too aggressively or
undershoot to one that wastes codes, with nothing in the parameterization to keep it sane. The `0.56` bits
per token LSQ still leaves at INT2 is small, but the INT4 regression makes me suspect a real slice of it is
LSQ's scale wandering during the 500 steps rather than settling at the loss-optimal grid. The diagnosis is
specific: LSQ's *degree of freedom* is right — the grid should be learned — but its *parameterization* is
too loose — the scale is free to roam anywhere in `(0, ∞)`, above the max-abs cover as easily as below it.
The move is to keep the learnable grid but **bound** it: parameterize the scale so that what is learned is
a clipping *ratio* constrained to a sensible range, not an unconstrained step size.

Before I commit to a particular bound, let me name the alternatives, because "constrain `s`" admits more
than one shape. I could simply clamp LSQ's free scale to a box `[s_lo, s_hi]` — but a hard clamp gives no
gradient information as it approaches the wall (the gradient just vanishes at the boundary), it does
nothing to damp the huge four-code step-size gradient in the interior, and it still learns an *absolute*
scale that does not ride the weight magnitude as the weights move. I could keep LSQ and lower its learning
rate — but that only slows the drift, it does not bound it, and it is a blunt instrument that cannot tell
useful motion from wandering: at INT2 the scale genuinely needs to travel a long way inward from its init
to reach the loss-optimal clip, so a lower learning rate would starve exactly the motion I want while only
partly taming the drift I do not. What I actually need is a coordinate whose *reachable set* is bounded and
whose *gradient* is naturally quiet where the run is fragile and lively where the clip is being decided —
a shaping the learning rate, a single scalar multiplying every step equally, cannot provide. Or I could go back to the max-abs scale as the *anchor* — the no-clip cover STE used — and learn a
multiplicative *clipping factor* that shrinks it, forcing the factor into `(0, 1)` with a sigmoid. This
last one is the one that fixes all three defects at once, so let me build it and check each.

Concretely, for each group take the raw extremes `xmax = max(w_g)` and `xmin = min(w_g)`, introduce two
learnable per-group parameters `upbound_factor` and `lowbound_factor`, gate them through a sigmoid, and
clip the extremes: `xmax ← sigmoid(upbound_factor)·xmax`, `xmin ← sigmoid(lowbound_factor)·xmin`. For
symmetric signed weight quantization the scale is then `s = max(|xmax|, |xmin|) / qmax`, clamped to a safe
range `[1e-5, 1e4]`, and the fake-quant is the straight-through `round(clip(w/s, qmin, qmax))·s`. The
gradient reaches the bound factors through the differentiable `sigmoid · extreme / qmax` scale; the round
is STE as before. I set `init_value = 4.0` for the factors and `CLIPMIN = 1e-5`, with per-group factors of
shape `(out_features, n_groups, 1)`.

Let me be precise about why this should beat LSQ, because the two are close cousins and the difference is
the whole bet. Both learn a per-group grid; the difference is the *coordinate* in which they learn it. LSQ
learns `s ∈ (0, ∞)` directly. LWC learns `γ = sigmoid(factor) ∈ (0, 1)` and sets `s = γ · max|w_g| / qmax`.
Three consequences fall out, and I can check each with a number. First, the search space is *bounded*: `γ`
cannot exceed 1, so `s = γ·max|w_g|/qmax ≤ max|w_g|/qmax`, which is exactly STE's max-abs scale. LWC's
scale can therefore *only clip inward, never coarsen past the cover* — precisely the useful direction at
low bits, and precisely the direction LSQ violated when it drifted above the cover and regressed INT4. A
parameterization that structurally cannot make the INT4 mistake LSQ made is already a reason to expect no
INT4 regression. Second, the parameterization is *anchored*: because the scale is always `γ · max|w_g|`,
it rides the current weight magnitude, so as the weights move during training the grid moves with them
automatically and the factor only has to learn the *relative* clip — a far gentler quantity to optimize
than an absolute scale that must chase a moving weight distribution. This matters more than it first looks,
because the weights are *not* frozen during these 500 steps — STE's weight-QAT is running underneath,
reorganizing them to round cleanly, so `max|w_g|` drifts step to step. LSQ's absolute `s` has to track that
drift with its own gradient, spending optimization effort just to stay in place; if the weights grow, an
`s` that was well-tuned is suddenly too small and clips too hard until it catches up. LWC's factor is
immune to this: when `max|w_g|` grows, `s = γ·max|w_g|` grows in lockstep for free, and the factor's job —
"how far inside the current cover should I clip" — is a stationary target even while the weights move. So
the anchoring does not just make the factor gentler to optimize; it decouples the grid-learning from the
weight-learning that shares the same loop, which is exactly the kind of interference that could make LSQ's
scale wander at INT2 where the weight updates are largest. Third, the sigmoid *bounds the
gradient*: `dγ/dfactor = γ(1−γ)`, which saturates as `γ → 0` or `1`, damping exactly the huge, noisy
step-size gradients that destabilize LSQ at four codes — the parameterization itself supplies the
stability LSQ had no mechanism for.

Let me check the reachable set of this parameterization against the two limits, because they tell me what
LWC can and cannot become. As `factor → +∞`, `γ = sigmoid(factor) → 1`, and `s → max(|xmax|, |xmin|)/qmax`
— the max-abs cover, which is *exactly STE's fixed scale*. So STE is the `γ = 1` corner of LWC's search
space: LWC does not replace STE, it contains it as a boundary case and opens the interior `γ < 1` toward
finer, clipped grids. As `factor → −∞`, `γ → 0` and `s → 0`, which would collapse every weight — a
degenerate corner, held off by the `CLIPMIN = 1e-5` floor on the scale. So the reachable scales run from a
tiny floor up to the max-abs cover, with STE pinned at the top. This is the property I most want from a
finale: because the same global loss trains the factors and STE is inside the search space, the trained
LWC can always fall back to the STE corner if clipping does not help, so on the *training* objective it
can only match or beat STE — it strictly generalizes the rung that scored 11.70 / 13.75 / 72.55. Whether
that dominance survives to *test* perplexity is the empirical question the INT2 number will answer, but I
am not risking a regression against STE by construction, only betting that the interior is better than the
corner.

Now let me verify the starting point, because a finale that starts worse than an earlier rung is a
non-starter, and the init is what guarantees it does not. `sigmoid(4) = 1/(1 + e^{-4}) = 1/(1 + 0.0183) =
0.9820`. So at initialization `s = 0.982 · max|w_g| / qmax`, within 2% of STE's max-abs cover — LWC begins
essentially *where STE sits*, at the fixed grid that already scored 11.70 / 13.75 / 72.55, and learns to
clip inward from there. It cannot start worse than STE by construction. And the gradient at that start is
`γ(1−γ) = 0.982 · 0.018 = 0.0177`, against the peak of `0.25` at `γ = 0.5` — a fourteen-fold damping right
at the initialization, so the factor barely moves in the first steps and the run cannot lurch away from
the STE cover. As training proceeds and a group's factor descends toward `γ = 0.5`, the gradient grows to
its `0.25` maximum — the parameterization is *most* responsive in the middle clip regime, exactly where
useful clipping happens, and quiet at both ends where it should hold still.

Let me make the stability quantitative by comparing the two coordinates' updates at the same loss
gradient, because "the sigmoid damps it" should be a number. LSQ moves the absolute scale, and its
sensitivity `∂L/∂s` is the summed step-size residual, which near a four-code transition is large — recall
the interior term reaches magnitude `≈ 0.5` per stranded weight and the group sums many of them. LWC moves
the factor, and by the chain rule `∂L/∂factor = ∂L/∂s · ∂s/∂factor = ∂L/∂s · (max|w_g|/qmax) · γ(1−γ)`. So
for the *same* loss pressure `∂L/∂s`, LWC's update on the factor carries an extra `γ(1−γ)` gate relative to
LSQ's raw update on the scale: `0.018` at the `γ = 0.982` start, rising only to `0.25` at mid-clip. In
other words, LWC applies at most a quarter, and at initialization a fifty-sixth, of the drive that LSQ
applies at the same transition — the huge, noisy four-code gradients that pushed LSQ off the near-optimal
grid at INT4 and made it wander at INT2 are structurally throttled by the coordinate itself, most where
the run is most fragile. That is the opposite of LSQ, whose gradient was largest and noisiest precisely at
the four-code transitions where it most needed to be calm.

I should be honest about what this parameterization does *not* recover, so I do not oversell it. The final
scale is still symmetric — `s = max(|xmax|, |xmin|)/qmax` — so the extra negative code that the
signed-symmetric grid wastes (the `−2^{B−1}` rail, a full quarter of the codebook at INT2, from the
rung-1 accounting) stays unused. The two separate factors let the clip be *asymmetric in how far each
extreme is pulled in* before the symmetric scale is taken, which matters when a group's positive and
negative tails differ, but they do not turn the grid asymmetric and so do not reclaim that code. LWC's
entire gain is the clip — trading a little clipping error on the outliers for finer rounding on the bulk,
now in a bounded and stable coordinate — not a wider codebook. Concretely at INT2, a group with an outlier
at `0.9` and bulk near `±0.2` starts at `s ≈ 0.982 · 0.9 ≈ 0.88`, essentially STE's cover, so the bulk is
crushed exactly as before; but as the factor learns `γ ≈ 0.3`, the anchored scale becomes `s ≈ 0.27`, the
bulk weight `0.2` now lands on a code (`0.2/0.27 ≈ 0.74 → 1 → 0.27`) instead of vanishing, and the outlier
is clipped — the same rounding-versus-clipping trade LSQ found, but reached from the STE cover by a damped,
weight-anchored factor that cannot overshoot past the cover on the way. The bet is that this controlled
descent lands closer to the loss-optimal clip, and lands there more reliably, than LSQ's free scale
starting from its own init and drifting. So LWC inherits LSQ's "learn the grid" power with a tighter, gradient-damped, weight-anchored
coordinate — the cure for exactly the drift the INT4 regression exposed.

One more thing to be careful about, because it is where the original form of this clip and this harness
part ways and I must not import machinery the edit surface cannot run. The clip's natural home is a
*per-block local reconstruction*: quantize one transformer block, minimize the MSE between its output and
the full-precision block's output on a small calibration set, no gradient through the rest of the model —
a PTQ-grade budget. This task's harness does not expose a per-block reconstruction loop; it exposes the
*global* QAT loop — full-model forward, cross-entropy on WikiText-2, AdamW over all `requires_grad`
parameters for 500 steps. So I do not get that cheap local objective; I get the same end-to-end finetune
every other rung uses, and the learnable factors are trained by the global task loss alongside the
weights. That is actually a *stronger* training signal than a local reconstruction — the factors are
optimized against the very metric I am scored on, cross-entropy on the eval domain, rather than a proxy
MSE — so the LWC parameterization should, if anything, do better here than its budget-constrained origin,
not worse. The one piece I therefore drop is the per-block scheduling; the parameterization itself —
sigmoid-gated per-group clip factors, init 4.0 — is exactly the canonical one, and it is the part that
fixes the INT2 grid-placement error. I keep the weights trainable too (the harness trains them anyway), so
this is LWC's clip layered on top of STE's weight-QAT, both driven by the global loss.

Now I land it in the task's edit surface, following the same wiring decisions LSQ forced, because the
harness's fixed post-training RTN would clobber a learned grid just as it would have clobbered LSQ's
scales. So `quantize_dequantize_weight` is again a deliberate no-op `weight.clone()`, and the real
quantization lives in the wrapper's `forward`, branching on `self.training`: during training the
differentiable LWC fake-quant with the sigmoid-gated factors; at eval a genuine no-grad `round(clip(w/s))·s`
on the *learned* clipped grid (recomputing the per-group extremes and scale from the final weights and the
final factors), so evaluation sees the LWC grid and not the max-abs grid. The factors `upbound_factor` and
`lowbound_factor` are registered as `nn.Parameter`s of shape `(out_features, n_groups, 1)` initialized to
4.0, so the harness's optimizer-over-all-`requires_grad`-params trains them alongside the weights with no
extra plumbing. Activations stay full precision (weight-only), and the LM head is restored to plain Linear
— both unchanged from LSQ. I keep the same 500-step schedule (`lr=2e-5`, `warmup=50`, cosine to 10%,
grad-accum 4, grad-clip 1.0) so the comparison to LSQ holds everything constant except the parameterization
of the learned grid. One detail I have to get right so the clip is faithful: I compute raw `xmax`/`xmin`
as the per-group *signed* max and min (not max-abs), gate each by its own sigmoid factor, and *then* take
`max(|xmax|, |xmin|)` for the symmetric scale — so I keep both factors and the signed-extreme computation
rather than collapsing to a single max-abs factor, which would be a different and less faithful method. Let
me check the two factors close on the right shape: `w` reshapes to `(out, n_groups, group_size)`, `amax`
and `amin` over `dim=−1` give `(out, n_groups, 1)`, the factors are `(out, n_groups, 1)`, the sigmoid-gated
product and the `max(|·|,|·|)` stay `(out, n_groups, 1)`, and dividing `w/s` broadcasts that scalar over
the 128 columns of its group — consistent, one independent clip per group, exactly as intended. The full
scaffold module is in the answer.

The causal chain in one breath: LSQ's measured INT2 (19.50, degradation 6.29) proved that *learning the
grid* is the right degree of freedom, but its INT4 regression (11.70 → 12.17, `0.056` bits worse where the
grid should not have moved) proved its *free unbounded scale* is too loose a parameterization and drifts →
keep the learnable grid but bound it: learn a sigmoid-gated clipping factor `γ ∈ (0,1)` on the max-abs
cover, `s = γ·max|w_g|/qmax`, init `sigmoid(4) ≈ 0.982` so it starts at STE and clips inward, with
`γ(1−γ)` damping the gradient → wire it into the task exactly as LSQ (no-op real-QDQ, eval-branch QDQ in
the wrapper, per-group `nn.Parameter`s) so the only change from LSQ is the coordinate the grid is learned
in.

This finale carries no feedback, so the bar it must clear is the strongest baseline's measured numbers,
stated as falsifiable expectations. At INT4 LSQ landed 12.17 and STE 11.70; since LWC starts at the
max-abs cover (≈ STE) with a fourteen-fold-damped gradient and can only clip inward, I expect it to match
or modestly beat both — somewhere around 11.7–12.2, and specifically I expect it *not* to reproduce LSQ's
INT4 regression, because the coordinate that caused that regression (a scale free to drift above the
cover) no longer exists. At INT3 LSQ reached 13.49; I expect LWC to match or slightly beat it, low-to-mid
teens, within a point of full precision. INT2 is where the claim lives: LSQ reached 19.50 (degradation
6.29, `0.56` bits per token), and if my diagnosis is right — that a bounded, gradient-damped,
weight-anchored clip is more stable than a free scale at four codes — LWC should clear LSQ at INT2,
landing *below* 19.50 (I would expect mid-to-high teens, degradation under 6, recovering a further slice
of that `0.56`-bit gap). The falsifiable failure mode is explicit: if LWC lands *at or above* 19.50 at
INT2 with no INT4/INT3 regression, then the bounded parameterization bought nothing over LSQ's free scale
and the drift I hypothesized was not the binding constraint — in which case LSQ remains the right answer
and the grid-placement gap at two bits must lie somewhere this clip also does not reach (for instance the
block-wise reconstruction objective the clip is naturally paired with, which this single-linear edit
surface does not expose). The minimal, honest claim is that LWC should not regress INT4/INT3 and should
not be worse than LSQ at INT2; the optimistic claim is a clear INT2 improvement from the bounded, stable
clip. And I can state the reading of the whole ladder that this finale would complete, so the endpoint is
not just a number but a mechanism traced end to end: the format's blind floor detonated at INT2 because
three codes replace rather than perturb the weights; a grid-blind finetune could not touch it because it
never sees the grid; straight-through put the grid in the forward and hauled INT2 from worse-than-uniform
to alive but left it stranded on a fixed step; learning the step size placed the levels and cut the INT2
gap to under a bit, at the cost of a free scale that drifts; and bounding that learned scale into a
sigmoid-gated, weight-anchored clip is the last turn of the same screw — the same degree of freedom LSQ
found, in a coordinate that cannot overshoot the cover and whose gradient goes quiet where the run is
fragile. Each rung changed exactly one thing and the INT2 column is the ledger that recorded it: five
digits, five digits, two digits, then teens — every drop bought by moving the weights, then the levels,
then the coordinate the levels are learned in. The finale's bet is only the last of those, and it is the
smallest and best-hedged, because STE lives inside its search space and it starts there.
