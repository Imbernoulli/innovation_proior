The squared-error run confirmed the magnitude diagnosis and, in doing so, exposed the next crack. Where
cosine had collapsed to 0.4664 at the large width, MSE lifted the large model all the way back to 0.6409
— the magnitude information the probe needed is now being trained, and the monotone-with-width decline
is gone. Base climbed to its best, 0.6506, and across sizes MSE averages 0.6311 against cosine's 0.5251,
a clean win that lands exactly where I predicted: put magnitude back and the big model recovers. So the
question "direction or magnitude" is settled — match both. But look at the *small* width: 0.6019, a
point or two *below* base and large, and the lowest of the three MSE numbers. That inversion is the
tell. The small model has the least capacity and the noisiest features, so its autoregressive roll-out
produces the largest and most erratic latent residuals; if a handful of those large residuals are
dominating the gradient, the loss most able to be destabilized by them is exactly the one with the
smallest, most fragile model — and that is the small width sagging. The Gaussian-default bet I made
explicitly when I chose squared error — "no heavy tails" — is the thing now worth interrogating, and the
small-model number is the first hint it does not hold.

Let me make the mechanism precise rather than hand-wave "outliers." What drives the parameter update is
not the loss `ρ(r)` but its derivative with respect to the residual, `ψ(r) = ρ'(r)` — the per-coordinate
pull. For squared error `ψ(r) = 2r`: unbounded. A residual of size 10 pulls ten times as hard as a
residual of size 1, a residual of size 100 a hundred times as hard, forever, no ceiling. In an
autoregressive roll-out the errors compound step to step; early in training the predictor is bad; some
frames are genuinely hard; and the encoder *target itself* is a moving, slightly noisy thing because it
is being learned at the same time. So at any moment a chunk of the residual coordinates are small and a
few are large, and with an unbounded `ψ` those few large ones produce gradients that swamp the batch.
The only lever to survive that is to drop the learning rate until the worst residual's gradient is
tolerable — but the schedule is fixed at `lr=1e-3` for all three sizes, and even if it were not, lowering
it would barely move the bulk of small residuals. The small model, with the largest roll-out residuals,
is the one this bites first. That is the squared-error failure mode in one sentence: its influence on
the estimate is unbounded, so it has no defense against the large residuals the small model produces in
abundance.

The opposite extreme fixes the tail and breaks the middle. Absolute error, `ρ(r) = |r|`, has
`ψ(r) = sign(r)`: the pull is `±1` no matter how large `r` is — bounded, so no single coordinate runs
away with the batch. But `ψ` is `±1` *all the way down to zero* — even a coordinate that is already
nearly right gets a full-magnitude, sign-only gradient — and at `r = 0` the loss has a kink, not
differentiable at the very point I want the predictor to settle into. There is no quadratic basin near
the optimum; the fit rattles around the minimum at constant gradient magnitude instead of easing in,
which is the same kinked-minimum disease cosine had. And statistically, if most residuals really are
small and roughly Gaussian, `|r|` fits them inefficiently — it is the maximum-likelihood estimator for
Laplace noise, the median, not the mean, and on clean-ish data the mean is the better estimate. So
absolute error fixes the tail and breaks the middle. The two losses each give me one half: squared error
is quadratic, smooth, and efficient where `r` is small but has unbounded influence where `r` is large;
absolute error has bounded influence in the tail but is kinked and inefficient in the middle. I want
both: quadratic-and-efficient where the residual is small, bounded-influence where it is large.

Make the trade-off precise instead of guessing a threshold. Both losses are M-estimators — I estimate by
`argmin Σ ρ(rᵢ)`, optimality condition `Σ ψ(rᵢ) = 0` — and the asymptotic variance of such an estimator
is `E[ψ²]/(E[ψ'])²`. For a *given* residual density `f`, minimizing that ratio (Cauchy–Schwarz on the
weighted inner product `⟨ψ, f'/f⟩`) gives equality exactly when `ψ ∝ −f'/f`, the maximum-likelihood
score: the best `ρ` for density `f` is its MLE. Gaussian `f` gives `ψ = r` (squared error, the mean);
Laplace `f` gives `ψ = sign(r)` (absolute error, the median). But my residuals are *neither* a clean
Gaussian nor a clean Laplace — they are mostly Gaussian-ish (the bulk of coordinates where the predictor
is decent) with a contaminating minority of large ones (the compounding roll-out, the hard frames, the
noisy moving target, concentrated in the small model). So the real model is the contaminated density
`F = (1 − ε)Φ + εH`, where `Φ` is the nominal Gaussian, `ε` a small contamination fraction, and `H`
*anything* — I refuse to assume the shape of the gross residuals. The right loss is the one that is best
against the *worst* `F` in this class: a minimax game where I pick `ψ` to minimize asymptotic variance
and an adversary picks the contamination `H` to maximize it.

The adversary's worst case decides the whole answer. The minimized variance for fixed `F` is
`1/(Fisher information of f)`, so the adversary *minimizes Fisher information* over the contamination
class — picks the least-informative density. A Gaussian has `(log f)' = −r`, which grows without bound in
the tails: far out, the Gaussian is *very* informative about location, and that is exactly the
informativeness squared error exploits to let outliers dominate. The adversary kills it by flattening the
tails — splicing on something whose log-slope stops growing. The least-informative such density is
Gaussian in a central interval and **exponential in the tails** (`(log f)' = −k·sign(r)`, constant-slope,
the flattest non-trivial thing), with the crossover `k` pinned by `ε`. The MLE score for *that*
least-favorable density reads straight off the pieces: in the middle (`|r| ≤ k`) it is `ψ = r`, the
squared-error pull; in the tails (`|r| > k`) it is `ψ = k·sign(r)`, the absolute-error pull. Together,
`ψ(r) = clamp(r, −k, k)` — and I did not choose this shape, the minimax problem forced it. It is the
squared-error pull capped at `±k`: efficient where the data look Gaussian, saturating so no single large
residual pulls harder than `k`. That cap *is* the robustness.

Integrate the score to get the loss I implement, demanding value- and slope-continuity at the join so
there is no kink (a kink in `ρ` is a jump in `ψ`, reintroducing exactly the non-differentiability I am
escaping). For `|r| ≤ k`, `ρ = ½r²`; for `|r| > k`, `ρ = k|r| − ½k²`, where the `−½k²` is forced by
value-continuity at `r = k` (left `½k²`, right `k·k − ½k² = ½k²` — equal) and the slopes already match
(`k` on both sides). This is the Huber / Smooth-L1 loss: quadratic basin in the middle, linear
bounded-gradient tail outside, spliced `C¹`. Squared error is the `k → ∞` limit (the mean — my step-2
loss), absolute error the `k → 0` limit (the median). So this is not a third competitor; it is the
one-parameter family that *contains* both, and it lives at the contaminated interior where my residuals
actually sit. That the two previous extremes drop out as limits is the test that I found the right
generalization rather than another option.

For these JEPA latents I have no clean `ε` to plug in, so the crossover `k` becomes a hyperparameter for
"what residual magnitude starts counting as an outlier." The natural default is `k = 1`: the encoder
produces latent features of order one, and after the VC regularizer keeps their spread normalized, a
residual of order one is the natural scale at which to flip from "fit this carefully" to "cap its
influence." In the unit-saturating parameterization the loss is `½r²` for `|r| < 1` and `|r| − ½` for
`|r| ≥ 1`, with gradient `clamp(r, −1, 1)` — the squared-error pull, capped at unit magnitude. I keep the
*mean* reduction over `[B, C, T, H, W]` for the same reason MSE needed it: a summed loss scales with
channel count and resolution and would couple the effective learning rate to model width, the
across-width sensitivity I removed at step 2; the per-coordinate mean keeps the term scale-matched across
the three sizes, and with mean reduction the per-coordinate influence is capped by `1/N` — no outlier
residual, however large, exceeds that. This is exactly PyTorch's `F.smooth_l1_loss` at its default
`beta = 1`. The full scaffold module is in the answer.

What I expect against MSE's numbers, falsifiably. The decisive test is the *small* width, where MSE
sagged to 0.6019 — if that sag is the unbounded influence of large roll-out residuals (the heavy-tail my
Gaussian default ignored), then capping the gradient should help the small model most, lifting its AP
toward and ideally past the base/large range and closing the inverted size profile. Base and large,
where the residuals are tamer and squared error was already near-optimal in its quadratic middle, should
hold roughly level with MSE's 0.6506 / 0.6409 — Smooth-L1 is identical to MSE for all residuals below the
threshold, so where outliers are rare it changes little, and I would not expect a large jump there. The
sharpest single number is therefore the small model: a Smooth-L1 loss that lifts small above 0.6019 while
holding base/large confirms that MSE's residual tail was the remaining failure; if small does *not* move,
the heavy-tail story is wrong and the next thing to interrogate would be the *temporal* structure the
flat per-coordinate reduction ignores — weighting the roll-out steps by horizon rather than treating
every `T` slice alike. The chain in one breath: MSE fixed cosine's magnitude failure (large recovered to
0.6409) but its unbounded influence `ψ = 2r` left the small model — which produces the largest roll-out
residuals — sagging to 0.6019 → model the residuals as a contaminated Gaussian and pick the
minimax-robust score, which forces `clamp(r, −k, k)`: squared-error pull capped at `±k`, efficient in
the Gaussian middle and bounded in the tail → integrate to the Huber/Smooth-L1 loss with `k = 1`, MSE's
robust generalization that recovers MSE as `k → ∞` → mean-reduce over `[B,C,T,H,W]` to stay scale-matched
across widths → expecting the small model to recover above 0.6019 while base and large hold, with the
roll-out's temporal weighting as the next lever if the heavy-tail fix is not enough.
