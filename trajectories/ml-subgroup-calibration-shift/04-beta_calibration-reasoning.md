The free-form monotone fit split exactly along the fault line I expected, and the split is the whole
argument for what comes next. On Adult, isotonic dropped the worst-group ECE from temperature scaling's
0.4844 to 0.3482 — the largest single gain from any map I have tried — confirming Adult's distortion was
genuinely non-uniform and the data-chosen binning bent the map where the global slope could not. On Law
School, where temperature scaling had already reached 0.3098 because the distortion there was nearly a
pure scale, isotonic blew up to 0.4506: with little real shape to fit, the extra non-parametric degrees
of freedom chased calibration-split noise, and the out-of-bounds clip dropped shifted test scores onto
flat blocks. COMPAS stayed put at 0.4832.

The way it cancels is the point. The per-dataset changes from scalar to isotonic are `−0.136` (Adult),
`−0.002` (COMPAS), `+0.141` (Law School): two enormous moves in opposite directions and one nothing. The
mean worst-group ECE came out 0.4273, essentially tied with temperature scaling's 0.4264 — but the mean
is a lie, because the average absolute move was `0.093`, thirty times the `0.0009` change in the mean.
The error did not shrink; it *relocated*, out of Adult and into Law School. Reading the triple
`(ΔECE, ΔBrier, ΔAUROC)` per dataset sharpens each verdict. Adult is `(−0.136, −0.110, −0.004)`: both
errors fall while ranking is nearly untouched — a *real shape found*. Law School is
`(+0.141, +0.102, −0.020)`: both errors rise together — *overfitting*, freedom spent on noise that does
not transfer. COMPAS is `(−0.002, +0.032, −0.040)`: worst-group ECE flat, Brier up, the largest ranking
loss — *pooling without payoff*, PAVA collapsing ordered examples into flat blocks on a dataset whose
uniform badness offered no shape to find; the `−0.040` AUROC drop is the biggest I have seen so far, exactly
where my earlier gap analysis flagged uniform badness and the most violations for PAVA to pool. So
isotonic is the right tool when the distortion is non-uniform and the calibration set is large enough to
estimate the bends, and the wrong tool when the sample is small and shifted.

The diagnosis is now precise: two failure modes bracketing the answer. Temperature scaling is too
*rigid* — one slope, low variance, transfers under shift, but cannot bend, so it leaves residual
miscalibration where the distortion is non-uniform (Adult). Isotonic is too *flexible* — it bends freely
and fixes Adult, but has as many effective degrees of freedom as it has blocks, so it overfits the small
shifted Law School split and nicks the ranking. I want the thing between: a map that can bend, including
in directions temperature scaling cannot reach, but inside a *tightly parametric* family of only a few
parameters. Which family, and the way to choose it, is to understand where the rigid map's poverty comes
from.

Temperature scaling is the one-parameter sigmoid recalibration; the two-parameter sigmoid (Platt) is the
next step. Where does the sigmoid come from? It drops out of a generative story: posit that within each
class the score is normally distributed with the same variance, means `s_+` and `s_-`. The likelihood
ratio's exponent, `−(s−s_+)² + (s−s_-)² = 2s(s_+−s_-) − (s_+²−s_-²)`, has its `s²` terms cancel — that
cancellation makes the exponent *linear* in `s` — leaving `LR(s) = exp[γ(s − m)]` with `γ = (s_+−s_-)/σ²`,
`m = (s_++s_-)/2`. Under a uniform prior the posterior is `μ(s) = σ(γ(s − m))`, the sigmoid exactly. So
the sigmoid family is the maps you get if you believe the per-class scores are equal-variance Gaussians
— and I can now ask whether that is true here.

It is not, in a structured way. First, my scores live in `[0,1]` and a Gaussian puts mass on the whole
real line, so the assumed density is incoherent outside the only region the score can occupy. Second,
and the one that bites: equal-variance Gaussians give `γ ≥ 0`, an S-curve that takes scores clustered
near `0.5` and *spreads* them toward 0 and 1 — right for a max-margin classifier, but a model
over-confident at the extremes needs the opposite. Its empirical positive rate is *less* extreme than
the score (a 0.99 should map to maybe 0.8), so the correct map must *pull the extremes back in* — an
inverse-sigmoid shape, steep near the ends and flat in the middle, which no `γ ≥ 0` sigmoid can make.
Third, the identity is not in the family: `σ(γ(s − m)) = s` cannot hold for a bounded S-curve against a
straight line, so a sigmoid applied to an already-calibrated score must move it and uncalibrate it.
Three rigidities — wrong support, one-directional shape, no identity — all from one modelling choice.

So fix the Gaussian. The natural density on `[0,1]` is the beta, `p(s; α, β) ∝ s^{α−1}(1−s)^{β−1}`: it
lives exactly on `[0,1]`, can be unimodal, U-shaped, J-shaped, or flat, and has *two* shape parameters
per density, not one shared variance. Plug positives `∼ Beta(α₁, β₁)`, negatives `∼ Beta(α₀, β₀)` into
the same LR machine: the powers of `s` subtract to `s^{α₁−α₀}`, the powers of `(1−s)` to
`(1−s)^{β₁−β₀}`, and with `a = α₁−α₀`, `b = β₀−β₁`, `c = ln[B(α₀,β₀)/B(α₁,β₁)]`, the result is
`LR(s) = e^c · s^a/(1−s)^b`, a power law on the odds, with calibrated posterior
`μ_beta(s; a, b, c) = 1/(1 + 1/(e^c · s^a/(1−s)^b))`. Three parameters, and `d/ds ln LR = a/s + b/(1−s)`
makes it monotone non-decreasing iff `a, b ≥ 0`. The shapes: `a = b > 1` is a sigmoid; `a = b < 1` is
the *inverse sigmoid* that pulls extreme over-confident scores back toward the middle, the exact thing
this benchmark needs and temperature scaling could not produce; `a ≠ b` is asymmetric; and
`a = b = 1, c = 0` is the identity. Bounded support, both shape directions, and the identity — all three
sigmoid rigidities repaired by swapping Gaussians for betas, with one extra parameter over Platt.

The nesting is worth making exact, because it forces a design decision. Temperature scaling
`σ(logit(s)/T) = s^{1/T}/(s^{1/T} + (1−s)^{1/T})` is `μ_beta` with `a = b = 1/T`, `c = 0` — the
symmetric-power slice through the origin; Platt `σ(a' logit s + b')` is `a = b = a'` with free `c`; beta
frees `a`, `b`, and `c`. So the two degrees of freedom beta adds over the temperature I already ran are
precisely the intercept `c` and the asymmetry `a − b`. The cheaper stop is Platt — two parameters, lower
variance — so why not save the third? Because the intercept is not the freedom Adult required: isotonic
beat the scalar on Adult by bending *asymmetrically*, a real non-uniform shape across the score range,
which means the low and high ends needed different corrections — the `a ≠ b` axis. Platt, locked at
`a = b`, can slide and steepen the S-curve with `c` but cannot make its two ends bend by different
amounts, so it would recover part of the Adult gain and stall, landing me back at the same
rigid-vs-flexible bracket one parameter poorer. The third parameter is the specific axis isotonic
exploited, bought back at three-parameter variance instead of many-block variance.

What makes this shippable on small shifted splits is the fitting. `ln LR = a ln s − b ln(1−s) + c` is
linear in `ln s` and `−ln(1−s)`, and a logistic-regression log-odds is linear in its features, so the
whole three-parameter beta map is *exactly* a bivariate logistic regression on `(ln s, −ln(1−s))` with
weights `(a, b)` and intercept `c` — one off-the-shelf call, the cost of a sigmoid, three parameters so
the variance is far below isotonic's. And because temperature scaling sits inside it at `a = b ≈ 1`,
`c ≈ 0`, on a dataset like Law School where the scalar already nearly sufficed the beta fit can
*fall back* to that near-scale correction rather than overfit — the family degenerates toward the thing
that worked instead of chasing the noise isotonic chased.

I will implement it as a plain bivariate logistic regression, building the two features as
`[log p, log1p(−p)]`. Two details differ from the textbook beta calibrator and I want to be honest about
each. The second feature is `+log(1−p)` rather than the `−log(1−p)` the clean derivation suggests, but
this fits the *identical* predictive function — the regression simply learns the coefficient `−b` where
the negated version learns `+b`, so the sign convention is cosmetic. And I could push toward the pure
maximum-likelihood beta fit — near-zero penalty (`C → ∞`) plus the canonical monotonicity drop-and-refit
guard (fit unconstrained, and if `a` or `b` comes back negative fix that feature to zero and refit) —
but the off-the-shelf `LogisticRegression(max_iter=2000, solver="lbfgs")` at its default L2 (`C = 1.0`)
is not obviously worse here. On a clean large calibration set the default penalty would be a loss,
shrinking `a, b` toward a constant map and flattening the fitted distortion; but on small splits
evaluated on a shifted tail it is exactly variance control, pulling the three parameters toward the safe
constant when the sample is thin — the same medicine every fit so far has wanted, and toward the same
lowest-capacity object my first group-temperature shrinkage pulled toward. The missing monotonicity guard rarely
fires because real distortions are monotone and the unconstrained fit almost always returns `a, b ≥ 0`
on its own. So the shippable calibrator is the genuine three-parameter beta family fit by a mildly
regularized bivariate logistic regression on `(log p, log(1−p))`, clipping `p` into `[ε, 1−ε]` so the
logs stay finite, `groups` accepted and ignored. The full module is in the answer.

The falsifiable expectation is that beta should *fix isotonic's worst failure without giving back its
best win*. On Law School, where isotonic overfit up to 0.4506, three parameters have too little capacity
to chase that noise and sit right next to the `a = b ≈ 1` scale correction that already got Law School to
0.3098, so I expect it back down near temperature scaling's level. On Adult, where isotonic won big at
0.3482, I expect beta to recover most of that gain — the inverse-sigmoid and asymmetric shapes are
exactly what a non-uniform distortion needs, and the asymmetry `a ≠ b` is the freedom isotonic used and
the scalar lacked. COMPAS should improve modestly, since the gathering shape can pull its extreme
over-confident scores back better than a single slope, though its uniform badness caps how far any
monotone map can go. If those land, the *mean* worst-group ECE should drop clearly below both
temperature scaling's 0.4264 and isotonic's 0.4273 — beta the first map to beat both brackets at once,
which is the entire reason for the bias/variance argument. Two mechanism checks: `subgroup_auroc` must
return to the strictly-monotone 0.8590 / 0.8819 / 0.7486 (beta with `a, b ≥ 0` is strictly monotone,
unlike PAVA's flat blocks), and if it does not my monotonicity claim has a bug; and Brier should fall
back below isotonic's 0.380, a smooth three-parameter MLE fit better behaved under shift than a jagged
non-parametric one. Should beta only tie isotonic on the mean while trailing where the calibration
set is smallest, that would say even three parameters are too many for the thinnest shifted split, and
the next move would be to spend the parameters where the shift actually bites — a method that explicitly
accounts for the calibration→test change rather than assuming the fit map transfers unchanged.
