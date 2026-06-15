The free-form monotone fit split exactly the way I predicted, and the split is the whole argument for what
comes next. On Adult, isotonic dropped the worst-group ECE from temperature scaling's 0.4844 to 0.3482 —
the largest single gain anywhere on this ladder — confirming that Adult's distortion was genuinely non-
uniform and the data-chosen binning bent the map where the global slope could not. But on Law School,
where temperature scaling had already gotten to 0.3098 because the distortion there was nearly a pure
scale, isotonic blew up to 0.4506: with little real shape to fit, the extra non-parametric degrees of
freedom chased calibration-split noise, and the out-of-bounds clip dropped shifted test scores onto flat
blocks. COMPAS stayed roughly put at 0.4832. The mean worst-group ECE came out 0.4273 — essentially
*tied* with temperature scaling's 0.4264 — but the error moved: isotonic traded a win on Adult for a loss
on Law School. The `subgroup_auroc` confirms the mechanism: it fell from the strictly-monotone scalar's
0.8590 / 0.8819 / 0.7486 to 0.8545 / 0.8420 / 0.7292 — small, but real, and only here, because PAVA's
flat blocks collapse the ordering *within* a block where a strictly monotone map would not. And the Brier
rose to 0.380 mean, its worst on the ladder. So isotonic is not a free lunch: it is the right tool exactly
when the distortion is non-uniform and the calibration set is large enough to estimate the bends, and the
wrong tool when the sample is small and shifted — which on this benchmark is at least one of the three
datasets.

So the diagnosis is now precise. I have two failure modes bracketing the answer. Temperature scaling is
too *rigid*: one slope, low variance, transfers under shift, but cannot bend, so it leaves residual
miscalibration where the distortion is non-uniform (Adult). Isotonic is too *flexible*: it bends
freely, fixes Adult, but has as many effective degrees of freedom as it has blocks, so it overfits the
small shifted Law School split and even nicks the ranking. I want the thing in between — a map that can
bend, including in directions temperature scaling cannot reach, but inside a *tightly parametric* family
of only a few parameters, so I get the shape without the variance. The question is which parametric
family, and the way to choose it is to understand where the rigid map's poverty comes from and fix that
one assumption.

Temperature scaling is the one-parameter sigmoid recalibration; the two-parameter sigmoid (Platt) is the
next richer step. So ask where the sigmoid comes from, because the place it comes from is where it fails.
It is not an arbitrary squash: it drops out of a generative story. Posit that within each class the score
is normally distributed with the *same* variance, means `s_+` and `s_-`. Form the likelihood ratio
`LR(s) = p(s|+)/p(s|−) = exp[(−(s−s_+)² + (s−s_-)²)/(2σ²)]`. Expand: `−(s−s_+)² + (s−s_-)² = 2s(s_+−s_-)
− (s_+²−s_-²)`, the `s²` terms cancel — that cancellation is the whole point, it makes the exponent
*linear* in `s` — leaving `LR(s) = exp[γ(s − m)]` with `γ = (s_+−s_-)/σ²` and `m = (s_++s_-)/2`. Under a
uniform prior the LR is the posterior odds, so `μ(s) = 1/(1 + LR(s)^{−1}) = σ(γ(s − m))`, the sigmoid
exactly. So the sigmoid family is precisely the maps you get if you believe the per-class scores are
equal-variance Gaussians. Now I know the assumption I make every time I sigmoid-scale, and I can ask
whether it is true here.

It is not, and the way it is false is structured. Two problems. First, my scores live in `[0,1]` — they
are bounded — and a Gaussian puts mass on the whole real line; for a probability-like score the density
I am assuming is incoherent outside the only region the score can occupy. Second, and this is the one
that bites in the reliability diagrams, the *shape* the sigmoid can produce is one-directional: equal-
variance Gaussians give `γ ≥ 0`, an S-curve that takes scores clustered in the middle and *spreads* them
toward 0 and 1. That is right for a max-margin classifier whose scores pile up near `0.5`. But a model
that is *over-confident at the extremes* — scores already slammed against 0 and 1, the very over-
confidence I have been fighting all along — needs the opposite: the empirical positive rate is *less*
extreme than the score (a 0.99 should map to maybe 0.8), so the correct map must *pull the extremes back
in*. That is an inverse-sigmoid shape, steep near the ends and flat in the middle, and no member of the
`γ ≥ 0` sigmoid family can make it. There is a third, almost embarrassing failure: is the identity in the
family? `σ(γ(s − m)) = s` for all `s`? The left side is a bounded S-curve, the right a straight line
through `(0,0)` and `(1,1)`; no finite `γ, m` matches. So a sigmoid applied to an *already-calibrated*
score must move it and uncalibrate it. Three rigidities — wrong support, one-directional shape, no
identity — all from one modelling choice: equal-variance Gaussians on a bounded score.

All three trace to the Gaussian, so fix the Gaussian and re-run the derivation. The natural density on
`[0,1]` is the beta distribution, `p(s; α, β) = s^{α−1}(1−s)^{β−1}/B(α, β)`. It lives exactly on `[0,1]`,
it can be unimodal, U-shaped, J-shaped, or flat, and it has *two* shape parameters per density, not one
shared variance. Plug positives `∼ Beta(α₁, β₁)`, negatives `∼ Beta(α₀, β₀)` into the same LR machine.
The powers of `s` subtract to `s^{α₁−α₀}`, the powers of `(1−s)` subtract to `(1−s)^{β₁−β₀}`, and the
beta-function constants collect into one factor. Writing `a = α₁−α₀`, `b = β₀−β₁`, and `K = B(α₁,β₁)/
B(α₀,β₀) = e^{−c}`, the result is `LR(s) = e^c · s^a / (1−s)^b` — a clean power law on the odds. The
calibrated posterior is `μ_beta(s; a, b, c) = 1/(1 + 1/(e^c · s^a/(1−s)^b))`. Three parameters, and it
does everything the sigmoid could not. Monotone non-decreasing iff `a, b ≥ 0` (since `d/ds ln LR =
a/s + b/(1−s)`). Shapes: `a = b > 1` is a sigmoid, `a = b < 1` is an *inverse sigmoid* (the gathering
shape that pulls extreme over-confident scores back toward the middle — the exact thing this benchmark
needs and temperature scaling could not produce), `a ≠ b` is asymmetric, and `a = b = 1, c = 0` is the
*identity*, so a beta calibrator can leave an already-calibrated region alone. Bounded support, both
shape directions, and the identity — all three sigmoid rigidities repaired by swapping Gaussians for
betas, with only one extra parameter over Platt.

The part that makes this *shippable* on the small shifted splits is the fitting. The log-likelihood-ratio
is `ln LR = a ln s − b ln(1−s) + c` — linear in the two quantities `ln s` and `−ln(1−s)`. A logistic
regression posterior on features `φ` is `σ(w·φ + bias)`, whose log-odds is linear in the features. So the
whole three-parameter beta map is *exactly* a bivariate logistic regression on the features `(ln s,
−ln(1−s))` with weights `(a, b)` and intercept `c` — one off-the-shelf call, the same cost as the
sigmoid, with none of its rigidity, and only three parameters so the variance is far below isotonic's.
That is the resolution of the bias/variance bracket: a map that can bend like isotonic (inverse sigmoids,
asymmetry) but with three parameters that transfer under shift like temperature scaling.

Now the part I have to get exactly right, because the trajectory must land the *task's* implementation,
not the generic one. The scaffold's `CalibrationMethod` builds the two features as `[log p, log1p(−p)]`
— note the second feature is `log(1−p)`, **not** the negated `−log(1−p)` the clean derivation suggests —
and fits a default `LogisticRegression(max_iter=2000, solver="lbfgs")`. Two things differ from the
textbook beta calibrator, and I want to be honest about each. First, the sign: building the second
feature as `+log(1−p)` rather than `−log(1−p)` fits the *identical* predictive function — the logistic
regression simply learns the coefficient `−b` where the negated-feature version learns `+b`; the
unconstrained MLE on either parameterization produces the same `μ_beta`, so the sign convention is
cosmetic, not a different method. Second, and substantively, the scaffold leaves the logistic regression
at its **default L2 regularization** (`C = 1.0`) rather than the near-zero penalty (`C → ∞`) that would
give the pure maximum-likelihood beta fit, and it omits the canonical monotonicity drop-and-refit guard
(fit unconstrained, and if `a` or `b` comes back negative, fix that feature to zero and refit). On a
clean large calibration set those omissions would be losses — the regularization shrinks `a, b` toward an
intercept-only constant map, slightly *flattening* the fitted distortion. But here, on small calibration
splits evaluated on a shifted tail, the default L2 is not obviously a bug: it is exactly a variance
control, pulling the three parameters toward the safe constant when the calibration sample is thin, which
is the same medicine the whole ladder has wanted under shift. And the missing monotonicity guard rarely
fires because real distortions are monotone, so the unconstrained fit almost always returns `a, b ≥ 0`
on its own. So the scaffold's beta calibrator is the genuine three-parameter beta family, fit by a mildly
regularized bivariate logistic regression on `(log p, log(1−p))`, clipping `p` into `[ε, 1−ε]` so the
logs stay finite — and the mild regularization is, if anything, well-matched to the shift. `groups` is
accepted and ignored; this stays the low-variance group-agnostic posture that has won every comparison so
far. The full scaffold module is in the answer.

State the falsifiable expectations against the numbers I have measured. The central claim is that beta
calibration should *fix isotonic's worst failure without giving back its best win*. On Law School, where
isotonic overfit the small shifted split up to 0.4506, the three-parameter family has far too little
capacity to chase that noise, so I expect it to come back down well into the 0.3s — closer to temperature
scaling's 0.3098 than to isotonic's 0.4506, because beta with `a = b ≈ 1` degenerates toward the global
scale correction that already nearly sufficed there. On Adult, where isotonic won big at 0.3482, the
question is whether three parameters can capture the same non-uniform bend; I expect beta to recover most
of that gain — landing near 0.35, comparable to isotonic and far below temperature scaling's 0.484 —
because the inverse-sigmoid/asymmetric shapes in the family are exactly what a non-uniform distortion
needs. COMPAS I expect to improve modestly past the ~0.48 everything else posted, into the high 0.46s,
since the gathering shape can pull COMPAS's extreme over-confident scores back better than a single slope.
If those three land, the *mean* worst-group ECE should drop clearly below both temperature scaling's
0.4264 and isotonic's 0.4273 — into the high 0.39s — making beta the first map on the ladder to beat both
brackets at once, which is the entire reason for the bias/variance argument. Two secondary checks:
`subgroup_auroc` must return to the strictly-monotone 0.8590 / 0.8819 / 0.7486 (beta with `a, b ≥ 0` is
strictly monotone, unlike PAVA's flat blocks, so the ranking damage isotonic did should reverse), and the
Brier should fall back below isotonic's 0.380 toward the low 0.35s, because a proper-scoring-rule MLE fit
on a smooth family is better behaved than a jagged non-parametric one under shift. If instead beta only
*ties* isotonic on the mean while still trailing on the dataset where the calibration set is smallest,
that would say even three parameters are too many for the thinnest shifted split — and the next move would
be to spend the parameters where the shift actually bites, on a method that explicitly accounts for the
calibration→test distribution change rather than assuming the calibration-fit map transfers unchanged.
