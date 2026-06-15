The isotonic numbers tell me exactly what I bought and exactly what I paid, and the price is in the
proper scores. Read the split it landed. On the two binary tasks the ECE is excellent — SVM on Breast
Cancer 0.0188, GBM on Madelon 0.0161 — and on the multiclass tasks too, RF on MNIST 0.0156 mean, MLP on
Fashion-MNIST 0.0122. That is the monotone non-parametric map doing exactly what it was built to do:
matching confidence to accuracy bin by bin. But look at NLL. RF on MNIST sits at 0.2369 mean, and the
seeds wobble — 0.2233, 0.2718, 0.2155 — a spread of more than 0.05 in NLL across seeds, which is the
non-parametric variance I warned myself about showing up directly. Worse, GBM on Madelon's NLL is 0.582
on its one seed, much higher than the other three tasks, and that is the tell I was watching for: Madelon
is the dataset with the steep curve and the smaller calibration split, and there the coarse piecewise-constant
blocks, each estimated from few examples, hand out the same flat probability to test points that deserve
different ones — ECE looks fine because the *average* in each bin is right, but NLL pays for every example
where the block value is wrong. So the diagnosis is precise: isotonic spends too many degrees of freedom,
the blocks are estimated from too little data, and the proper scores bleed where the calibration set is
small. The fix isn't a different non-parametric scheme; it's to *spend fewer degrees of freedom* — go
parametric, buy back data efficiency, accept a little shape rigidity in exchange for stable proper scores.

So let me start from what a parametric calibrator has to be, and let the form be forced rather than
chosen. I have a frozen classifier handing me, for each calibration example, an uncalibrated probability
— a single positive-class number `p` for the binary tasks, a full vector for the multiclass ones. The
ranking is good (isotonic confirmed it: a monotone map sufficed), the magnitudes are wrong. I want a
*two-parameter* map from that score to a calibrated probability, fit by likelihood on the calibration
split, with so little capacity that the few-example variance that wrecked isotonic's Madelon NLL simply
can't happen. The textbook generative route — fit a class-conditional density to the positive scores and
another to the negatives, then Bayes' rule — is a place to *derive* the form from, not to implement,
because I do not want to estimate two whole distributions when I only care about the one posterior.

What do the class-conditional scores actually look like? For a margin classifier the cross-validated
score histograms are nowhere near Gaussian; they have kinks and decay roughly exponentially between the
class clusters. Hold onto the exponential part. Write the two class-conditionals as exponential tails,
`p(f|+) ∝ exp(γ₁ f)` and `p(f|−) ∝ exp(−γ₀ f)` with `γ₀, γ₁ > 0`, where `f` is the score in a space
where it ranges over the real line — I'll come back to *which* space. Put priors `π₁, π₀` into Bayes'
rule: `P(+|f) = π₁ e^{γ₁ f} / (π₁ e^{γ₁ f} + π₀ e^{−γ₀ f})`. Divide top and bottom by the numerator:
`P(+|f) = 1 / (1 + (π₀/π₁) e^{−(γ₀+γ₁) f}) = 1 / (1 + exp(A f + B))`, with `A = −(γ₀+γ₁) < 0` the slope
and `B = log(π₀/π₁)` the offset that absorbs the class-prior ratio. There it is — two exponential
class-conditionals give a *sigmoid* in `f`. It has a second reading I like even more: `1/(1+exp(Af+B))`
is exactly the model "`f` is, up to an affine transform, the log-odds of the positive class," so the
positive-class probability increases with `f` precisely when `A < 0`, which honors the monotone prior
isotonic just validated, and the output is automatically in `[0,1]`. Two free parameters fit to data —
`A` the slope (how fast confidence grows with the score), `B` the offset (where `P=½` actually sits) —
not one parameter read off a Gaussian variance.

Now the question this task's surface forces: the sigmoid is naturally a function of a real-line score,
but the harness hands me a *probability* `p ∈ [0,1]`, not a margin. If I feed `p` straight into
`1/(1+exp(A p + B))` I'm putting a bounded variable through a function designed for an unbounded one, and
the affine `A p + B` can't reach the log-odds the sigmoid wants. The clean fix is to map the probability
back to the space where the sigmoid's derivation lives: the **log-odds**, `f = log(p/(1−p))`. That is the
right `f` — it ranges over the whole real line, it's monotone in `p`, and it's exactly the quantity the
"score is proportional to log-odds" reading wants as input. So the calibrator I land on, in this task's
vocabulary, is: take the uncalibrated probability, transform to log-odds `f = log(p/(1−p))` (clipping `p`
into `(0,1)` so the log is finite), and fit `calibrated = 1/(1 + exp(A f + B))`. This is logistic
regression of the label on the single transformed feature `f`.

How do I fit `A, B`? The right objective for a probability model is its negative log-likelihood. Encode
the label as a target `t_i ∈ {0,1}` and minimize the cross-entropy
`−Σ_i [t_i log p_i + (1−t_i) log(1−p_i)]` with `p_i = 1/(1+exp(A f_i + B))`. Two parameters, smooth,
convex (the Hessian is a Gram matrix, positive definite unless every `f_i` coincides), so an off-the-shelf
quasi-Newton solver — L-BFGS-B from a flat start `[A,B] = [1.0, 0.0]` — finds the unique optimum. But
plain `{0,1}` targets have a failure mode I have to head off: if the calibration scores happen to be
linearly separable in `f`, maximum likelihood tries to drive every positive's `p_i` to exactly 1 and
every negative's to exactly 0, which the sigmoid can only do by becoming infinitely steep — `A → −∞`, no
finite optimum. I need regularization, but I refuse to add a hyperparameter to tune (that would reopen
the cross-validation problem isotonic suffered from). The hyperparameter-free fix is Platt's: instead of
asserting a positive example has target probability exactly 1, model it as positive with a small residual
chance of being negative out of sample. The rule of succession pins the numbers: with a uniform prior
over the true positive probability, seeing `N₊` positives gives a positive example the soft target
`t₊ = (N₊ + 1)/(N₊ + 2)`, and a negative example the soft target `t₋ = 1/(N₋ + 2)`. These are strictly
inside `(0,1)`, so pushing `A → −∞` would overshoot `t₊ < 1` and *increase* the loss — the optimum
becomes finite — and as `N₊, N₋ → ∞` they converge to `{0,1}`, recovering plain MLE in the large-data
limit. No knob, exactly the regularizer I want.

That handles one binary column completely: log-odds transform, Laplace-smoothed targets, minimize the
cross-entropy with L-BFGS-B, and at predict time return `1/(1+exp(A f + B))` clipped to `[0,1]`. Now the
multiclass case, where I again land *this task's* choice and not a generic one. I will not try to
calibrate the whole softmax jointly — same reason isotonic didn't — but unlike isotonic I have a
parametric per-class fit, so the one-against-all decomposition is even cheaper: for each class `c`, treat
the binary problem "is the true label `c`?" against the classifier's probability for class `c`, fit a
separate `(A_c, B_c)` sigmoid on the log-odds of that column with its own Laplace-smoothed targets, and
do this independently for all `C` classes. Independently fit, the per-class calibrated outputs won't sum
to one, so reconcile by renormalizing — each one-against-all output already estimates `P(c|x)` directly,
so dividing each by the row sum is the natural combiner, and it gives the harness the valid distribution
its row-sum assertion requires. The full scaffold module is in the answer.

There is one honest worry I should name before I run it, because it's the flip side of the data efficiency
I'm buying. The sigmoid is *one* shape. On a column whose true distortion really is an inverse-sigmoid —
a tree-style classifier that slams its probabilities to the extremes, so the correct map has to *pull*
them back toward the middle — the sigmoid (with `A < 0`, the only sign monotonicity allows) can only push
toward the extremes, never gather, so the best fit can be a *bad* fit, and on a column that was already
nearly calibrated the sigmoid can't represent the identity, so it would move probabilities it should have
left alone. That rigidity is the price of the two-parameter stability, and the multiclass tasks — RF and
MLP, whose per-class probability columns can be the extreme-pushed kind — are exactly where I'd expect it
to show. I'm accepting that for now because the proper-score bleed from isotonic's variance is the more
urgent problem; if the rigidity bites visibly, the next rung will have to be a parametric family rich
enough to bend both ways.

So here is what I expect against isotonic's measured numbers, falsifiably. On the proper scores where
isotonic bled — RF on MNIST NLL (0.2369 mean, seed-spread > 0.05) and especially GBM on Madelon NLL
(0.582) — the two-parameter fit should *cut* NLL, because there is no longer a coarse flat block handing
out the same wrong probability to many examples; the smooth sigmoid assigns a graded probability to every
score. I expect Madelon's NLL in particular to drop substantially below 0.582, and RF's NLL below 0.237
with a tighter seed spread, because that is the data-efficiency I went parametric to buy. The risk runs
the other way on ECE: where isotonic's data-chosen binning matched accuracy nearly perfectly, the rigid
sigmoid may match it *less* well, so I would not be surprised to see ECE *rise* on the tasks isotonic
calibrated best — the SVM column especially, if its log-odds curve isn't quite sigmoidal, could see ECE
go up even as Brier/NLL hold or improve. If that is the pattern — proper scores down, ECE up where the
true curve fights the single sigmoid shape — then the verdict is clear and points straight at the next
rung: I traded shape flexibility for data efficiency, and the residual ECE is the shape flexibility I
gave up, which a richer-but-still-parametric family would buy back without reopening the variance problem.
