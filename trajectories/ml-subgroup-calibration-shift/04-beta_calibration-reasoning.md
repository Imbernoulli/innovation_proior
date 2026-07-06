The free-form monotone fit split exactly the way I predicted, and the split is the whole argument for what
comes next. On Adult, isotonic dropped the worst-group ECE from temperature scaling's 0.4844 to 0.3482 —
the largest single gain anywhere on this ladder — confirming that Adult's distortion was genuinely non-
uniform and the data-chosen binning bent the map where the global slope could not. But on Law School,
where temperature scaling had already gotten to 0.3098 because the distortion there was nearly a pure
scale, isotonic blew up to 0.4506: with little real shape to fit, the extra non-parametric degrees of
freedom chased calibration-split noise, and the out-of-bounds clip dropped shifted test scores onto flat
blocks. COMPAS stayed roughly put at 0.4832.

Let me read the split quantitatively, because the way it cancels is the point. The per-dataset changes
from the scalar to isotonic are `−0.136` (Adult), `−0.002` (COMPAS), `+0.141` (Law School): two enormous
moves in opposite directions and one nothing. The mean worst-group ECE came out 0.4273, essentially
*tied* with temperature scaling's 0.4264 — but the mean is a lie here, because the average absolute move
was `0.093`, thirty times the `0.0009` change in the mean. The error did not shrink; it *relocated*, out
of Adult and into Law School. So isotonic is not a better map than the scalar on this benchmark, it is a
*differently-shaped-error* map: it fixes the dataset whose distortion is non-uniform and breaks the one
whose distortion was already a clean scale. Two more columns confirm the mechanism rather than just the
outcome. The `subgroup_auroc` fell from the strictly-monotone scalar's 0.8590 / 0.8819 / 0.7486 to
0.8545 / 0.8420 / 0.7292 — small, but real, and the biggest drop is COMPAS's `−0.040`. That is exactly
the dataset my earlier gap analysis flagged as *uniform badness*, where every cell is mis-scaled and the
ranking has the most violations for PAVA to pool; so COMPAS is where isotonic pooled hardest, tying the
most previously-ordered examples into flat blocks, and it paid for that pooling in AUROC without buying
any worst-group ECE back (0.485 → 0.483). Pooling that costs ranking and returns no calibration is the
signature of flexibility spent where there is no shape to find. And the Brier tracks the worst-group
split limb for limb: `−0.110` on Adult (isotonic genuinely helped), `+0.032` on COMPAS, `+0.102` on Law
School (isotonic genuinely hurt), mean up to 0.380, its worst on the ladder. So isotonic is not a free
lunch: it is the right tool exactly when the distortion is non-uniform and the calibration set is large
enough to estimate the bends, and the wrong tool when the sample is small and shifted — which on this
benchmark is at least one of the three datasets.

It is worth reading the three datasets as three distinct *signatures* in the triple `(ΔECE, ΔBrier,
ΔAUROC)` from the scalar to isotonic, because each signature is a different verdict on what freedom
bought. Adult is `(−0.136, −0.110, −0.004)`: worst-group ECE and Brier both fall while ranking is nearly
untouched — the fingerprint of a *real shape found*, a genuine non-uniform distortion the extra freedom
captured cleanly. Law School is `(+0.141, +0.102, −0.020)`: both errors rise together — the fingerprint
of *overfitting*, freedom spent fitting calibration-split noise that does not transfer. COMPAS is
`(−0.002, +0.032, −0.040)`: worst-group ECE flat, Brier up, and the largest ranking loss — the
fingerprint of *pooling without payoff*, PAVA collapsing ordered examples into flat blocks on a dataset
whose uniform badness offered no shape to find. Three signatures, three verdicts, and they tell me the
family I want must do different things on each: recover Adult's real shape, refuse Law School's noise,
and at least not damage COMPAS's ranking while nudging its scale.

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

I should verify those claimed shapes rather than assert them, and the cleanest way is to locate the
methods I already ran *inside* this family, because that both checks the algebra and shows me exactly what
freedom I am adding. Take temperature scaling: `σ(logit(s)/T) = 1/(1 + (s/(1−s))^{−1/T}) = s^{1/T}/
(s^{1/T} + (1−s)^{1/T})`. Now take `μ_beta` with `a = b = t` and `c = 0`: `1/(1 + (1−s)^t/s^t) = s^t/
(s^t + (1−s)^t)`. These are the *same function* with `t = 1/T`. So temperature scaling is exactly the
`a = b`, `c = 0` diagonal slice of beta, with the beta shape parameter equal to the inverse temperature.
That is a real check, so let me push numbers through it: `T = 2` should be `a = b = 0.5`. At `s = 0.99`,
`s^{0.5}/(s^{0.5} + (1−s)^{0.5}) = 0.995/(0.995 + 0.100) = 0.909`; at `s = 0.9`, `0.949/(0.949 + 0.316) =
0.750` — both matching the `T = 2` softening I traced two rungs ago to the digit. The `a = b < 1` case
is not merely *an* inverse sigmoid, it is the gathering map I have wanted all along, and it was hiding
inside temperature scaling as the `T > 1` softening. Contrast the `a = b > 1` sigmoid: `a = b = 2` at
`s = 0.9` gives `0.81/(0.81 + 0.01) = 0.988`, pushing `0.9` *out* to `0.988` — the spreading a max-margin
model needs and an over-confident one does not. And the identity check: `a = b = 1, c = 0` gives
`s/(s + (1−s)) = s`, the identity on the nose, so the family really can leave a calibrated region alone.
The verification also lays the nesting bare. Platt is `σ(a' logit s + b') = s^{a'}/(s^{a'} + e^{−b'}(1−s)^{a'})`,
which is `μ_beta` with `a = b = a'` and `c = b'` — the symmetric-power slice *with* a free intercept. So
the three methods nest cleanly: temperature scaling is one parameter (`a = b`, `c = 0`), Platt is two
(`a = b`, free `c`), and beta is three (free `a`, `b`, `c`). The two degrees of freedom beta adds over the
temperature I already ran are precisely the intercept `c` and the asymmetry `a − b`, and the asymmetry is
what lets the low and high ends of the score range be corrected by different powers — the one thing every
symmetric map on this ladder, temperature and Platt alike, structurally cannot do.

The nesting also forces a design decision I should make out loud rather than skip past. The cheaper stop
is Platt: two parameters, lower variance than beta's three, and it already restores the intercept `c`
that temperature lacked. Under a shift, where variance is the enemy, why not take the two-parameter map
and save a degree of freedom? Because the intercept is not the freedom the Adult win required. Isotonic
beat the scalar on Adult by bending the map *asymmetrically* — the triple-signature said a real,
non-uniform shape was found, and non-uniform across the score range means the low and high ends needed
different corrections, which is exactly the `a ≠ b` axis. Platt, locked at `a = b`, can slide the S-curve
sideways with `c` and steepen it, but it cannot make its two ends bend by different amounts; a Platt fit
would likely recover part of the Adult gain and stall, landing me back at this same rigid-vs-flexible
bracket one parameter poorer. So the third parameter is not decoration — it is the specific axis isotonic
exploited on Adult, bought back at three-parameter variance instead of many-block variance. That is why I
spend it, and why I do not stop at Platt.

The part that makes this *shippable* on the small shifted splits is the fitting. The log-likelihood-ratio
is `ln LR = a ln s − b ln(1−s) + c` — linear in the two quantities `ln s` and `−ln(1−s)`. A logistic
regression posterior on features `φ` is `σ(w·φ + bias)`, whose log-odds is linear in the features. So the
whole three-parameter beta map is *exactly* a bivariate logistic regression on the features `(ln s,
−ln(1−s))` with weights `(a, b)` and intercept `c` — one off-the-shelf call, the same cost as the
sigmoid, with none of its rigidity, and only three parameters so the variance is far below isotonic's.
That is the resolution of the bias/variance bracket: a map that can bend like isotonic (inverse sigmoids,
asymmetry) but with three parameters that transfer under shift like temperature scaling. And because
temperature scaling sits inside it at `a = b ≈ 1`, `c ≈ 0`, on a dataset like Law School where the scalar
already nearly sufficed, the beta fit can *fall back* to that same near-scale correction rather than
overfit — the family degenerates toward the thing that worked, instead of chasing the noise isotonic
chased. That is the structural reason to expect it to fix isotonic's Law School failure without a special
case.

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
is the same medicine the whole ladder has wanted under shift — and it pulls toward the constant map, the
lowest-capacity object, which is the same direction the shrinkage on the very first rung pulled the
per-group temperatures. And the missing monotonicity guard rarely fires because real distortions are
monotone, so the unconstrained fit almost always returns `a, b ≥ 0` on its own. So the scaffold's beta
calibrator is the genuine three-parameter beta family, fit by a mildly regularized bivariate logistic
regression on `(log p, log(1−p))`, clipping `p` into `[ε, 1−ε]` so the logs stay finite — and the mild
regularization is, if anything, well-matched to the shift. `groups` is accepted and ignored; this stays
the low-variance group-agnostic posture that has won every comparison so far. The full scaffold module is
in the answer.

State the falsifiable expectations against the numbers I have measured. The central claim is that beta
calibration should *fix isotonic's worst failure without giving back its best win*. On Law School, where
isotonic overfit the small shifted split up to 0.4506, the three-parameter family has far too little
capacity to chase that noise, and it sits right next to the `a = b ≈ 1` scale correction that already got
Law School to 0.3098; so I expect it to come back down well into the 0.3s, closer to temperature
scaling's 0.3098 than to isotonic's 0.4506. On Adult, where isotonic won big at 0.3482, the question is
whether three parameters can capture the same non-uniform bend; I expect beta to recover most of that
gain — landing near 0.35, comparable to isotonic and far below temperature scaling's 0.484 — because the
inverse-sigmoid and asymmetric shapes in the family are exactly what a non-uniform distortion needs, and
the asymmetry `a ≠ b` is the freedom isotonic used and the scalar lacked. COMPAS I expect to improve
modestly past the ~0.48 everything else posted, into the high 0.46s, since the gathering shape can pull
COMPAS's extreme over-confident scores back better than a single slope, though its uniform badness caps
how far any monotone map can go. If those three land, the *mean* worst-group ECE should drop clearly
below both temperature scaling's 0.4264 and isotonic's 0.4273 — into the high 0.39s — making beta the
first map on the ladder to beat both brackets at once, which is the entire reason for the bias/variance
argument. Two secondary checks that test the mechanism rather than the score. First, `subgroup_auroc`
must return to the strictly-monotone 0.8590 / 0.8819 / 0.7486: beta with `a, b ≥ 0` is strictly monotone,
unlike PAVA's flat blocks, so the ranking damage isotonic did — most of all its `−0.040` on COMPAS —
should reverse completely, and if it does not, my "beta is strictly monotone" claim has a bug. Second,
the Brier should fall back below isotonic's 0.380 toward the low 0.35s, because a proper-scoring-rule MLE
fit on a smooth three-parameter family is better behaved than a jagged non-parametric one under shift. If
instead beta only *ties* isotonic on the mean while still trailing on the dataset where the calibration
set is smallest, that would say even three parameters are too many for the thinnest shifted split — and
the next move would be to spend the parameters where the shift actually bites, on a method that
explicitly accounts for the calibration→test distribution change rather than assuming the calibration-fit
map transfers unchanged.
