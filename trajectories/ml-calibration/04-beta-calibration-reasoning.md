Temperature scaling won the ladder by making the *minimal* correction, and reading its numbers tells me
both why it won and exactly the one thing it still can't do. The wins are clean where the dominant error
was scale: RF on MNIST ECE fell back to 0.0101 mean (below isotonic's 0.0156 and far below Platt's 0.0254),
with NLL 0.1553 holding Platt's gain and the seed spread tight; SVM on Breast Cancer ECE dropped from
Platt's blown-up 0.0493 to 0.0305 and NLL improved to 0.0864 — one shared scalar fixed the overconfidence
that Platt's per-class sigmoid had overfit. Accuracy preserved by construction, the best win-count across
the twelve cells. But look where it *didn't* move: GBM on Madelon ECE sits at 0.0305, essentially Platt's
0.0288 and *worse* than isotonic's 0.0161 — temperature scaling left Madelon's calibration error almost
untouched. That is the gap I flagged when I built it: one shared `T` can *soften* a confidence vector but
it cannot move a *location* (it has no offset to shift where `P=½` sits) and it cannot *reshape* a curve
that bends the wrong way (it has no second degree of freedom to gather extreme scores back toward the
middle). Madelon's GBM is exactly such a column — its reliability curve needs both a location shift and a
shape that bends two ways, and a single scalar reaches neither. And the binary tasks generally are where
this bites, because there the harness gives me only the one positive-class probability, so there is no
joint softmax for the scalar to exploit — it's just a one-dimensional curve, and a one-parameter rescaling
of a one-dimensional bounded curve is a thin tool.

So the verdict across the three rungs is sharp. Isotonic: too flexible, great ECE, bad proper scores, high
variance. Platt: two-parameter sigmoid, good proper scores, but one rigid shape and per-column fitting that
regressed ECE. Temperature: one-parameter scale, recovered ECE where the error was scale, but left the
columns that need a *location* or a two-way *shape* correction untouched — Madelon being the proof. What I
want now is the family that keeps Platt's cheapness and data efficiency but is rich enough to bend in the
directions temperature and Platt both miss: a parametric map that contains the sigmoid (so it never loses
where Platt won), contains the *inverse* sigmoid (so it can gather extreme scores, which neither Platt nor
temperature can), can place its midpoint freely (the location offset temperature lacks), and — a quiet but
decisive property — contains the *identity*, so on a column that some earlier classifier already calibrated
it can learn to leave it alone instead of moving it the way Platt's sigmoid is forced to.

Let me derive that family rather than guess it, because the rigidity of the sigmoid traces to one specific
modelling choice and I want to fix exactly that choice. Platt's sigmoid is not an arbitrary squashing
function; it drops out of a generative story. Posit that within each class the score is normally
distributed with the *same* variance `σ²`, means `s₊` and `s₋`. Form the likelihood ratio
`LR(s) = p(s|+)/p(s|−)` and the `s²` terms cancel (that cancellation is the whole point — it's what makes
the exponent linear in `s`), leaving `LR(s) = exp[γ(s − m)]` with `γ = (s₊ − s₋)/σ²` and `m = (s₊ + s₋)/2`.
Under a uniform prior the LR is the posterior odds, so `μ(s) = 1/(1 + exp(−γ(s − m)))` — the sigmoid,
exactly, and the map runs both ways: any such sigmoid corresponds to *some* pair of equal-variance
Gaussians. So the sigmoid family is precisely "I believe the per-class scores are equal-variance Gaussians."
Now I can see why it's too rigid, in three concrete ways, all from that one assumption. First, my scores
live in `[0,1]` — they're bounded — and a Gaussian puts mass on the whole real line, so modelling a
bounded score as Gaussian is incoherent on its face. Second, the equal-variance Gaussians force `γ ≥ 0`, an
S-curve that only ever *spreads* scores toward 0 and 1; it can never *gather* them back, so on a classifier
that slams its scores to the extremes (forest votes, GBM additive logits, naive-Bayes double-counting) the
correct map is an *inverse* sigmoid — steep at the ends, flat in the middle — and no sigmoid member looks
like that. Third, the identity `μ(s) = s` is a straight line through `(0,0)` and `(1,1)`, and no bounded
S-curve equals it for any finite `γ, m`, so the identity is not in the family — Platt applied to an
already-calibrated column necessarily *un*calibrates it. All three trace to one place: equal-variance
Gaussians on a bounded score. Fix the assumption and re-run the derivation.

Where did the trouble enter? Infinite support and a single shared variance. My score lives on `[0,1]`. The
natural density on `[0,1]` is the **beta distribution**, `p(s; α, β) = s^{α−1}(1−s)^{β−1}/B(α, β)`, which
lives exactly on `[0,1]`, can be unimodal, U-shaped, J-shaped, or flat, and crucially has *two* shape
parameters per class, not one shared variance. Plug it into the same likelihood-ratio machine: positives
`~ Beta(α₁, β₁)`, negatives `~ Beta(α₀, β₀)`. The powers of `s` subtract, the powers of `(1−s)` subtract,
and the beta-function constants collect into one factor, giving `LR(s) = e^c · s^a/(1−s)^b` with
`a = α₁ − α₀`, `b = β₀ − β₁`, and `c` absorbing the constant — a clean power law on the odds. Turn it into
a calibrated posterior the same way, `μ(s) = 1/(1 + LR(s)^{−1})`, and the family is
`μ_beta(s; a, b, c) = 1/(1 + 1/(e^c · s^a/(1−s)^b))`. Three parameters: `a, b` shapes, `c` location.

Check it does the things the sigmoid and the scalar couldn't, because that's the entire reason I changed
the assumption. Monotonicity: `μ` increases in `s` iff `LR` does, and `d/ds ln LR = a/s + b/(1−s) ≥ 0` for
all `s ∈ (0,1)` exactly when `a, b ≥ 0` — the analogue of the sigmoid's `γ ≥ 0`. The shapes: the curves
are *not* translation-invariant in `s` (unlike the sigmoid, where moving `m` just slid the same S-curve),
because `s^a` and `(1−s)^b` aren't symmetric unless `a = b` — and that asymmetry is exactly the location
freedom temperature scaling lacked, with the midpoint `m` (where `μ = ½`, i.e. `LR(m) = 1`) at
`c = b ln(1−m) − a ln m`. The decisive test, the gathering shape neither prior rung could make: take
`a = b < 1`, then `LR(s) = e^c (s/(1−s))^a` and the exponent `< 1` *damps* the log-odds, pulling extreme
scores toward the middle — an inverse sigmoid. Sigmoid is `a = b > 1`. And the identity, the thing the
sigmoid family flatly lacked: set `a = b = 1, c = 0`, then `LR(s) = s/(1−s)` and `μ(s) = s` exactly. So
the one move of swapping Gaussians for betas repairs all three sigmoid failures at once — bounded support
is now correct, both shape directions are reachable, and the identity is a member — and it adds the location
offset temperature scaling didn't have. A concrete sanity check on the gathering case: naive-Bayes fed `k`
identical copies of a calibrated feature outputs `s = x^k/(x^k+(1−x)^k)`, and the exact map that recovers
`x` is `μ_beta` with `a = b = 1/k, c = 0` — double-counting is a power on the odds, and the beta family is
precisely powers on the odds, so the correction falls right out.

Now make it concrete in this task's edit surface, and here I want to land the *task's* implementation, not
the generic one. The map's log-likelihood-ratio is `ln LR = a ln s − b ln(1−s) + c`, linear in `ln s` and
`−ln(1−s)`, so in principle the whole family is a bivariate logistic regression on those two features —
that's the "easily implemented" result. But the task's surface gives me `numpy`, `scipy.optimize`, and
asks for a `BaseEstimator` filling `fit`/`predict_proba`, so the literal fill builds the two features
directly — `f1 = log(p/(1−p))` (the log-odds) and `f2 = log(1−p)` — note `a·log(p/(1−p)) + b·log(1−p) + c =
a·ln s − a·ln(1−s) + b·ln(1−s) + c`, the same linear-in-`(ln s, ln(1−s))` object up to a relabeling of the
two coefficients, so this featurization fits the identical three-parameter beta map. The objective is the
mean cross-entropy of `q = 1/(1 + exp(−(a f1 + b f2 + c)))` against the labels, minimized with
`optimize.minimize(method="L-BFGS-B")` from `x0 = [1, 0, 0]` — which starts at `a = 1, b = 0, c = 0`,
i.e. plain log-odds-identity, the natural neutral start. For multiclass I do the same per-class
one-against-all fit (a separate `(a_c, b_c, c_c)` per column on `1{label==c}` against that column's
probability) and renormalize the rows to sum to one, exactly as Platt and isotonic did, which keeps the
output a valid distribution for the harness's assertion. The full scaffold module is in the answer.

One thing the task's surface deliberately omits, and I should be explicit about it because it's a real
difference from the canonical method: the unconstrained `optimize.minimize` can in principle return `a < 0`
or `b < 0`, which would make the calibration map non-monotone over part of the range. The canonical
recipe guards this with a drop-and-refit step (if a coefficient comes out negative, fix it to zero and
refit the reduced model). This task's edit has no such guard — it fits all three coefficients
unconstrained and uses them as-is. In practice the guard rarely fires, because real distortions are
monotone and the data pulls both coefficients positive; on these four well-behaved classifiers I expect it
not to matter, but it is the one place the harness's beta calibrator is a slightly looser implementation
than the reference, and I'm noting it rather than pretending the guard is there. I also leave the fit
essentially unregularized (no explicit penalty in the scipy objective), which is correct — beta calibration
*is* the maximum-likelihood / log-loss fit, and strong shrinkage would pull `a, b` toward zero, collapsing
the map toward an intercept-only constant, the opposite of fitting the distortion.

This carries no feedback, so let me state the bar it has to clear and what I would validate, against
temperature scaling's real numbers. The clearest target is the column temperature scaling left untouched:
**GBM on Madelon**, where temperature's ECE was 0.0305 (essentially Platt's, and worse than isotonic's
0.0161). Beta calibration's whole reason to exist is that it can supply the location offset and the
two-way shape that a single scalar can't, so the falsifiable claim is that Madelon's ECE should fall below
temperature's 0.0305 — toward, and possibly below, isotonic's 0.0161 — *without* the proper-score bleed
isotonic paid for it, because beta is a smooth three-parameter map, not a coarse block fit. On the columns
temperature already calibrated well by scale — RF on MNIST (ECE 0.0101, NLL 0.1553) and SVM on Breast
Cancer (ECE 0.0305, NLL 0.0864) — the bar is that beta should *not regress*: because the sigmoid is a
strict sub-case (`a = b > 1`) and the identity is reachable (`a = b = 1, c = 0`), the richer family can
always fall back to what Platt/temperature found, so it should match or modestly beat them rather than
overfit three parameters where two sufficed. The real risk to watch is exactly that — three parameters per
column on the smaller binary calibration splits is more capacity than two, so on SVM (the smallest split)
beta could overfit and give back some of temperature's ECE; if it does, the diagnosis would be that the
binary columns want the *symmetric* `a = b` restriction (univariate logistic regression on the log-odds,
two effective parameters) rather than the full three. The success condition, then, is concrete: beat
temperature scaling on the Madelon ECE it couldn't move, hold its RF and SVM wins within noise, and improve
or hold NLL and Brier everywhere — which is what a family that *contains* every rung below it as a special
case should be able to do.
