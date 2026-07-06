The 2-bit grid finally broke the plateau, and reading the three baselines as a sequence is what tells me
where to push. Binary sat at val_loss 2.7352; ternary, adding only the zero, moved to 2.7213 — a drop of
0.0139, half a percent; then int2 fell to 2.4392, a drop of 0.2821, about ten percent. So the
ternary→int2 step was roughly twenty times larger than the binary→ternary step, and the two steps differ
in exactly one way: ternary added a *structural* state (the off-switch), while int2 added *graded
magnitude*. WikiText-2 tells the same story — 81.3, then 77.9, then 54.1, the collapse concentrated in
the last step — and LAMBADA barely moved until int2 (110.2, 109.8, 82.0). Most telling is downstream,
because that is where the earlier rungs were pinned at the floor: ARC-Easy went 47.0 → 46.7 → 53.8 and
HellaSwag 28.7 → 28.4 → 31.5, so the completion tasks were dead flat through the off-switch change and
only *moved* once the graded levels arrived. In perplexity terms the in-distribution `exp(val_loss)` went
from ~15.4 to ~11.5 across the plateau to int2. The resolution diagnosis was right: the binding
constraint on the earlier rungs was not the off-switch but the inability to express a graded weight
magnitude, and the five-level grid `{−1, −2/3, 0, +2/3, +1}` that the int2 fill actually realizes (its
`×1.5` scaling plus half-to-even rounding lands on five symbols, not the four its docstring names) fixed
exactly that.

Two rows I have not yet folded into that sequence complete it. PIQA went 58.9 → 59.7 → 62.9 — flat across the plateau (a 0.8-point wobble I have to treat as noise on one seed), then a clean +3.2 at int2, tracking the graded-magnitude gain exactly the way ARC and HellaSwag did; so three of the four downstream tasks moved together the moment resolution arrived, which is much stronger evidence the mechanism is real than any one metric alone. WinoGrande is the exception: 51.5 → 50.8 → 50.4, sitting at or just below its 50 floor through all three rungs, unmoved even by int2. That is worth holding onto — coreference resolution did not emerge from better weight fidelity at this scale, so whatever WinoGrande needs is not the axis I am on, and I should not expect the finale to rescue it. The wall-clocks — 22601, 22494, 23255 — are within about 3% of each other, int2's small bump the cost of its extra `×1.5` arithmetic; identical wall-clock across such different losses confirms all three runs were optimization-clean, none diverged, each simply converged to the optimum its grid allowed. And the in-distribution perplexity sharpens the target: `exp(2.7352) ≈ 15.4`, `exp(2.7213) ≈ 15.2`, `exp(2.4392) ≈ 11.5` — the plateau shaved 0.2 off per-token perplexity, int2 shaved 3.7, a ~25% cut. The gap I am now trying to close sits below that 11.5, and it is a smaller gap than the one int2 just closed, so I should expect a smaller absolute move and design for it.

But the win also tells me where the *next* limit is, and it is a subtler thing than "more levels." The
thing that moved the model was giving each weight more places to land; the thing I have *not* touched on
any rung is *where those places are*. On all three baselines the grid is anchored by one number per
tensor — the absmean `s = mean(|W|)` — and that number is chosen to minimize the squared reconstruction
error `‖W − s·grid‖²`, i.e. to make the discrete weights *close to the float weights*. But I do not care
whether the discrete weights are close to the float weights; I care whether they make the *loss* small.
Those are different objectives, and at two or three bits, where only a handful of levels exist per layer,
the gap between them is exactly the gap between int2 and a usable model.

Let me make the deficiency concrete, because it is the whole motivation. With so few levels, the single
scalar `s` decides everything about level placement: too small and I clip away the tails of the weight
distribution, throwing the large-magnitude weights into the top level and losing them; too large and I
waste my precious few levels on coarse spacing, so the dense core of the distribution near zero gets one
level where it wanted three. The absmean picks `s` to balance reconstruction across the whole tensor —
but reconstruction error weights every weight equally, while the loss does not: some weights matter
enormously and some not at all, and the right `s` for the loss is the one that places levels where the
loss-important weights are, not where the average weight is. There is no formula for that `s`. So I
should stop computing it from a statistic and *learn* it — make the per-layer step a trainable parameter,
optimized jointly with the weights against the actual next-token cross-entropy. That is the move: a
single extra scalar per `BitLinear`, the step size, trained by the same AdamW that trains the weights, so
the network tunes its own quantization grid to minimize the thing I measure.

Make the mismatch concrete with two weights in one layer, because the abstract "reconstruction is the wrong objective" only bites once I can see it choose wrong. Suppose weight A sits at exactly a bin center — the grid already represents it perfectly, its reconstruction error is zero — but it feeds a downstream feature the loss barely uses; and weight B sits halfway between two levels — its reconstruction error is near maximal — but it gates a feature the next-token loss leans on heavily. The absmean picks `s` to shrink the *total* squared error `‖W − s·grid‖²`, in which A and B count equally, so it will happily nudge the grid to trim A's already-tiny error while leaving B stranded between levels. The loss wants the opposite: move the grid so B lands cleanly on a level, even at the cost of unsettling A. A per-tensor statistic cannot tell A from B — that distinction lives only in `∂L/∂ŵ`, the backward signal, which absmean never sees. So the scale I actually want is the one that has *been shown the loss*, and the only way to give `s` that information is to route the loss gradient into it and let the optimizer place the grid. That is exactly what promoting `s` to a trained parameter does, and it is why no closed-form statistic — absmean, absmax, or any fixed percentile — can substitute: they are all functions of the weight magnitudes alone, blind to which weights the loss cares about.

The obstacle is immediate and it is the same wall every rung hit, but sharper. The quantizer is
`ŵ = round(clip(w/s, −Q_N, Q_P)) · s`. The round is flat almost everywhere, so the path from `s` through
`w/s` into the integer code has zero ordinary derivative — ordinary backprop sees the final multiply by
`s` but is blind to how `s` moves weights *toward or away from bin transitions*, which is the whole
reason `s` matters. The standard fix for the round is the STE I have used throughout: treat round as the
identity on the backward pass. But here I must be more careful than on the baselines, because they only
needed a gradient to the *weight*; now I need a correct gradient to the *step* too, and the easy thing —
cancel the round entirely — would zero out the interior step-gradient and learn nothing useful. So I
apply STE to the round node only, and differentiate the divide, the clip, and the multiply honestly.

Take the interior region, where `−Q_N < w/s < Q_P` so the clip is inactive: `ŵ = round(w/s)·s`.
Differentiate w.r.t. `s` by the product rule: `∂ŵ/∂s = [∂round(w/s)/∂s]·s + round(w/s)`. STE says treat
round as identity for the first term, so `round(w/s) ≈ w/s` there and `∂(w/s)/∂s = −w/s²`; times `s`
gives `−w/s`. The second term is `round(w/s)`. So in the interior `∂ŵ/∂s = round(w/s) − w/s`. Write
`z = w/s`, `n = round(z)`, `r = z − n` (the residual, between about `−½` and `½`): then
`∂ŵ/∂s = n − z = −r`, the negative signed residual between `z` and the level it rounds to. Stare at that,
because it is the payoff. The step-size gradient is *largest in magnitude exactly when `z` sits near a
bin transition* (`|r|` near ½) and *zero when `z` sits right on a level* (`r = 0`).

Let me trace that on a handful of concrete weights so I am sure the sign and magnitude come out right,
using a 3-bit signed grid (`Q_N = 4`, `Q_P = 3`) and a step `s = 0.02`. A weight `w = 0.040` gives
`z = 2.0`, `n = 2`, `r = 0`, so `∂ŵ/∂s = 0` — it sits exactly on a level and does not care where the step
moves. A weight `w = 0.038` gives `z = 1.9`, `n = 2`, `r = −0.1`, so `∂ŵ/∂s = +0.1` — a small pull. A
weight `w = 0.050` gives `z = 2.5`, which half-to-even rounds to `n = 2`, `r = 0.5`, so `∂ŵ/∂s = −0.5` —
the maximum interior magnitude, because it is balanced on a bin edge where a nudge of `s` flips its code.
And the clipped weights: `w = 0.071` gives `z = 3.55 > Q_P`, so `ŵ = Q_P·s` and `∂ŵ/∂s = Q_P = 3`, while
`w = −0.095` gives `z = −4.75 < −Q_N`, so `∂ŵ/∂s = −Q_N = −4`. So the step gradient is zero on levels,
`±½` at the interior bin edges, and jumps to the large fixed values `±Q_P / ∓Q_N` for the clipped tails —
the clipped weights, which want a bigger dynamic range, push `s` outward hardest, and the total
`∂L/∂s = Σ (∂L/∂ŵ)(∂ŵ/∂s)` is dominated by exactly the weights whose codes are most sensitive to `s`. The
fixed-absmean baselines had no version of this at all — their `s` was a constant of the weights, not a
parameter, so it could never respond to where the weights sit relative to the grid. And the data gradient
is the usual STE: `∂ŵ/∂w = 1` inside the range, `0` outside (clipped weights get no gradient, which is
correct — pushing a clipped latent weight further changes nothing in the forward pass, so it should feel
no pull).

Let me aggregate that per-weight residual over a tiny layer to see which direction `s` actually gets pushed, because `∂ŵ/∂s = −r` is only half the story — the loss gradient weights each term. Take four weights with residuals `r = (−0.4, +0.1, −0.45, 0.0)`, so their step-gradient factors `∂ŵ/∂s = −r = (+0.4, −0.1, +0.45, 0.0)`, and suppose the backward loss gradients are `∂L/∂ŵ = (+2, +1, −3, +1)` in some scaled units. Then `∂L/∂s = Σ (∂L/∂ŵ)(∂ŵ/∂s) = (2)(0.4) + (1)(−0.1) + (−3)(0.45) + (1)(0) = 0.8 − 0.1 − 1.35 + 0 = −0.65`. A negative `∂L/∂s` means gradient descent *increases* `s` — it widens the grid — and the term that dominates the sum is the third weight, balanced near a bin edge (`r = −0.45`) with a large loss gradient (`−3`), contributing `−1.35`. That is the behavior I wanted made literal: the weights sitting right on bin transitions, where a nudge to `s` flips a code and moves the loss most, vote hardest on where the step goes; the weight parked on a level (the fourth, `r = 0`) abstains entirely. A fixed absmean has no vote-counting step at all — it is a constant of the weight magnitudes, computed once per forward pass with no reference to `∂L/∂ŵ` — which is precisely the faculty I am adding, and it is why a single learned scalar can do something a whole family of reconstruction statistics cannot.

There is one more thing I have to get right or the learned step will destabilize training, and it is the
part the careless versions of this idea miss. I now have one scalar `s` per layer being optimized by the
same AdamW, with the same global learning rate, as the layer's hundreds of thousands of weights. Training
behaves well when, across parameters, the ratio of update magnitude to parameter magnitude is in the same
band — if `s` gets updates that are huge relative to its own size it overshoots and oscillates; if tiny,
it stalls. Let me check the ratio `R = (∇_s L / s) / (‖∇_w L‖ / ‖w‖)`. For the parameter sizes:
`‖w‖ ∝ √N_W` times a typical weight magnitude, and the typical weight magnitude is about `s·√Q_P` (with
`Q_P` levels the step shrinks like `1/√Q_P`), so `‖w‖/s ≈ √(N_W·Q_P)`. For the gradients: `∇_s L` is a
sum over all `N_W` weights of `(∂L/∂ŵ)·(∂ŵ/∂s)`, and treating the per-weight loss gradients as
uncorrelated zero-mean with the `∂ŵ/∂s` factor an order-one per-element constant, `E[(∇_s L)²] ≈
N_W·E[(∂L/∂ŵ)²]` — the *same order* as `E[‖∇_w L‖²]`. So numerator and denominator of `R` differ exactly
by the `‖w‖/s` factor: `R ≈ √(N_W·Q_P)`. That is not 1 — it grows with layer width and precision, so the
step would be over-driven by roughly that factor, worst in the widest layers. Put numbers on it: the MLP
`c_fc` has `N_W = 1024·4096 = 4.19M`, so `√(N_W·Q_P) = √(12.6M) ≈ 3547`; the attention `c_proj` has
`N_W = 1.05M`, so `√(N_W·Q_P) = √(3.1M) ≈ 1773`. Left unscaled the step's relative update would be one to
three *thousand* times the weights' — an instant blow-up. The fix is to cancel it directly: scale the
gradient flowing to `s` by `g = 1/√(N_W·Q_P)`, which is `≈ 2.8×10⁻⁴` in the widest layers and `≈ 5.6×10⁻⁴`
in the attention output — a *per-tensor* number, since it reads each tensor's own `N_W`. I inject this as
a transparent gradient multiplier with the same detach trick the STE uses — `gradscale(s, g) = (s −
g·s).detach() + g·s` is `s` in the forward pass (the step value is untouched) and multiplies the gradient
by `g` in the backward pass. So `s` trains in the same update/parameter band as the weights, and one
AdamW learning rate serves both.

I should be honest that the `‖w‖/s ≈ √(N_W·Q_P)` estimate rests on a heuristic — the claim that the RMS weight magnitude is about `s·√Q_P` — and check how far off it is here, because if it were wildly wrong the whole gradient scaling would be mis-set. At the `std=0.02` init, `s = 2·mean(|W|)/√Q_P`, so `s·√Q_P = 2·mean(|W|) = 2·0.798σ ≈ 1.596σ`, whereas the actual RMS weight magnitude is `σ`. So the heuristic overestimates the RMS by about `1.6×` at init, which means `R` is balanced to within a factor of ~1.6 of one, not exactly to one. That is completely acceptable, and worth being clear-eyed about: the entire purpose of the scaling is to kill the `√N_W` factor — 3547 in the widest MLP, 1774 in the attention output — that would otherwise over-drive the step by three orders of magnitude, and a residual `1.6×` on top of that is the difference between `s` learning slightly faster or slightly slower than the weights, not between stability and blow-up. AdamW's per-parameter second-moment normalization absorbs order-one discrepancies like this anyway. So I take `g = 1/√(N_W·Q_P)` as an order-of-magnitude balancer, not a precision instrument, and I expect it to land the step in the right band even though the heuristic behind it is loose — and I will not pretend the `1.6` cancels to `1`, because it does not.

Before I commit to a learned scalar I should walk the other ways to attack the fixed scale, because "learn it" is one move among several and I want the elimination on paper rather than assumed. One option is a finer *static* granularity: replace the per-tensor absmean with a per-output-row absmean, one scale per row of the weight matrix. That is cheap — it adds `out_features` scalars, a few thousand per layer, negligible against `N_W` — and it does tighten reconstruction where a tensor's rows differ in scale. But it is still a *reconstruction* statistic; it makes the discrete weights closer to the float weights, row by row, and I just argued at length that closeness to the float weights is not the objective. It would refine the wrong quantity more finely. A second option is to learn only a multiplier `θ` on the absmean — `s = θ·mean(|W|)` with `θ` trained — which is nearly equivalent to a free learned step but carries a needless coupling to the reconstruction statistic and no advantage over just learning `s` directly. A third is to also learn the *activation* step; I set that aside deliberately, because the activation path is the controlled-experiment spine held identical across every rung, and adding a second scratch-initialized learned scale to an already-unstable low-bit pretraining run is risk the clean weight-versus-weight comparison does not need. A fourth is simply to spend another bit at fixed absmean — a 3-bit grid with the old scale — but that re-tests "do more levels help," which int2 already answered, without touching the level *placement* that is the actual residual. The one move that attacks placement and only placement is a single free per-tensor step trained against the loss, which is the one I take.

Now adapt all of this to *this task's* substrate, because the canonical LSQ recipe assumes things the
harness does not provide and I must not import them blindly. Canonical LSQ *fine-tunes from a trained
full-precision model* with momentum SGD and cosine decay; here I am pretraining from scratch with AdamW
on the fixed 13,535-iteration schedule, so there is no FP teacher to initialize from — the latent weights
start from the scaffold's `std=0.02` init and the step must be initialized from them. The scale-aware
initializer `s = 2⟨|w|⟩/√Q_P` does exactly that from the initial weights, so I set the step parameter to
`2·mean(|W|)/√Q_P` at construction and let AdamW take it from there. Let me check that init places the
grid sensibly rather than trusting the formula. At `std=0.02`, `mean(|W|) = σ√(2/π) ≈ 0.01596`, so
`s_init = 2·0.01596/√3 ≈ 0.03192/1.732 ≈ 0.01843`. The grid then spans from `−Q_N·s = −4·0.01843 ≈
−0.0737` to `+Q_P·s = +3·0.01843 ≈ +0.0553`, i.e. `[−3.69σ, +2.76σ]` — it covers essentially the whole
Gaussian and clips only the ~0.3% beyond `+2.76σ` on the positive side, while a typical weight
(`|w| ≈ 0.016`, `z ≈ 0.87`) lands near the first level. So the `√Q_P` denominator correctly says "more
levels ⇒ finer initial step," and the eight levels are spread across the weight distribution from the
first step rather than bunched. Canonical LSQ keeps the first and last layers at 8 bits; the harness ties
`wte` to `lm_head` and quantizes every projection uniformly, so I keep the uniform treatment rather than
carve out exceptions the scaffold is not built to express. And I keep the activation path *identical* to
the three baselines — 8-bit per-tensor absmax, `Q_b = 127`, STE — for two reasons: it isolates the
contribution to the *weight* quantizer (the learned step), which is the honest comparison against int2;
and learning a second step for activations from scratch, on top of an unstable-by-nature low-bit
pretraining run, is a risk the controlled experiment does not need.

There is a self-correcting loop hidden in the clipped-tail gradient that reassures me the learned step will not run away from a rough scratch init, and it is worth tracing because it is what lets me tolerate an initializer I know is only approximate. Suppose `s` drifts too small: then a large fraction of weights push past `±Q_P` (or `−Q_N`) and clip, and every clipped weight contributes the *fixed, large* step-gradient `∂ŵ/∂s = +Q_P` or `−Q_N`, all pointing to widen the grid, each weighted by whatever loss gradient that saturated weight carries. So a too-small `s` generates strong outward pressure that *grows* with how much mass it is clipping — a restoring force that switches on exactly when clipping becomes severe. Suppose instead `s` drifts too large: almost nothing clips, the sum is dominated by interior residuals `∂ŵ/∂s = −r` bounded by `±½`, which largely cancel across the many weights straddling levels, so the net pull on `s` is weak — while the coarse-grid penalty (weights stranded far from their levels) instead shows up in the *data-gradient* path to the weights. The two regimes bracket `s` from opposite sides: severe clipping shoves it out hard, mild coarseness lets it settle. That negative feedback is why LSQ tolerates the rough `2·mean(|W|)/√Q_P` initializer — the step does not have to be right, only close enough to sit inside the basin, and the clipped-tail term drags it toward the loss-preferred range from wherever the scratch weights put it.

The bit-width I take is three: signed range `Q_N = 2^{b−1} = 4`, `Q_P = 2^{b−1} − 1 = 3`, an eight-level
grid `{−4,…,+3}`, `log₂8 = 3` bits. Three bits keeps me firmly in the few-bit native-low-bit regime this
task is about — at ~132 MB it is still five times under `bfloat16` — and it deliberately gives the learned
step *more* levels to place than int2's effective five, so the finale tests both halves of the thesis at
once: more levels, and levels placed by the loss rather than by a reconstruction formula. I could instead
have stayed at the *same* level count as int2 and changed only the placement, which would be the cleaner
one-variable experiment; but int2 already showed that more levels help, so pairing the learned step with
a genuine three-bit grid is the stronger finale, and if it beats int2 I will still know the learned
placement contributed because a fixed-absmean 3-bit grid is a thing I could reason about separately. No
SubLN inside the layer, same as every rung: the block's pre-projection `LayerNorm` holds the variance.

So the finale fill adds three things to the scaffold and changes one: a learnable `weight_scale`
parameter per `BitLinear` initialized to `2·mean(|W|)/√Q_P`; the two helper ops `roundpass` (STE round)
and `gradscale` (the `1/√(N_W·Q_P)` step-gradient multiplier); and a `weight_quant` that clips `w/s` to
`[−Q_N, Q_P]`, STE-rounds, and rescales by the gradient-scaled `s`. The activation path and the latent-
weight machinery are unchanged from the baselines (the full module is in the answer).

Two substrate checks before I trust this runs clean in the frozen harness, because the finale is the first rung that adds a *learned* parameter and I must not break the contract. First, `torch.compile`: the fill adds a learnable `Parameter` and two helpers built from the same detached-difference idiom as the STE, and nothing else — no RNG, no Python-level branch on tensor values, no `@torch.compiler.disable`. So the forward stays a pure function of its inputs and the graph traces exactly as the baselines did; `gradscale` and `roundpass` are just arithmetic with a `detach`, which inductor lowers without a graph break. Second, DDP across the two GPUs: `weight_scale` is a registered `Parameter`, so its gradient is all-reduced across ranks like every other parameter, and for that averaging to be consistent both ranks must compute the *same* gradient-scaling factor `g`. They do — `g = 1/√(numel·Q_P)` reads only `weight.numel()` and `Q_P`, both properties of the module's fixed shape and hyperparameters, identical on every rank and every step, never data-dependent. So the step parameter trains identically to how it would on one GPU, and the 2-GPU setup introduces no divergence between ranks. The learned scale slots into the fixed loop as just one more parameter the optimizer and the all-reduce already know how to handle, and the `min=1e-5` clamp at init keeps the very first `w/s` divide finite even if a layer's weights initialized unusually small.

Here is the bar this has to clear, against int2's measured numbers, and what I would validate. The thing
int2 left on the table is loss-aware level placement, and three bits gives more room to place; so I
expect validation loss *below* int2's 2.4392 — into the low-2.4s or high-2.3s, a smaller absolute step
than int2's leap off the plateau but a real one, because the mechanism says loss-aware level placement has
headroom that int2's fixed absmean step left on the table. I expect WikiText-2 below int2's 54.1 and LAMBADA below 82.0, and downstream ARC-Easy
and HellaSwag at or above int2's 53.8 / 31.5 — because graded, loss-placed magnitudes are exactly what
the completion tasks reward, and those were the metrics that finally moved at int2. The way I would be
wrong is if the learned step does not train cleanly under AdamW from a scratch init — if the `√(N_W·Q_P)`
gradient scaling is even slightly off, the step either oscillates and the grid thrashes, or stalls and I
have a fixed-step quantizer with one wasted parameter landing right back at int2. I would validate by
watching the per-layer `weight_scale` trajectories: if they move smoothly away from their absmean-flavored
init (that `~0.0184`) and settle at values the loss prefers, the learned placement is doing real work; if
they sit pinned at init, the gradient scaling needs the per-layer `N_W` it already uses but perhaps a
different AdamW treatment. The falsifiable claim is simply: a *learned* step at three bits beats the
*fixed* absmean step at int2's effective resolution — val_loss strictly under 2.4392.
