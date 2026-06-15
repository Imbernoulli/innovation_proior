Platt did exactly what I went parametric to get, and it also did exactly the thing I was afraid of, and
both are legible in the numbers. The proper-score bleed I diagnosed on isotonic is gone: RF on MNIST NLL
fell from 0.2369 to 0.1627 mean and the seed spread tightened (0.1703, 0.1549, 0.1628 — under 0.016 wide,
where isotonic was over 0.05); GBM on Madelon NLL dropped from 0.582 to 0.451; Brier improved across the
board. Two parameters bought the data efficiency I wanted. But look at ECE, and look specifically where
isotonic had been strongest. SVM on Breast Cancer ECE *rose* from 0.0188 to 0.0493 — nearly tripled —
with a wild seed spread (0.0595, 0.0562, 0.0323). RF on MNIST ECE rose from 0.0156 to 0.0254. GBM on
Madelon ECE rose from 0.0161 to 0.0288. That is precisely the failure I predicted: the single sigmoid
shape, applied per-column, fits the *proper scores* well because it assigns a graded probability to every
point, but it cannot bend to the true reliability curve where that curve isn't sigmoidal, so the binned
ECE — which isotonic's data-chosen blocks matched almost perfectly — comes back worse. The SVM blowup is
the sharpest tell: Platt fits a separate sigmoid per class and on the binary SVM column the log-odds curve
clearly isn't a clean sigmoid, so the fit is confident in the wrong place and ECE pays.

So now I can see the whole tension. Isotonic: too flexible, great ECE, bad proper scores, high variance.
Platt: rigid two-parameter sigmoid per column, good proper scores, but the per-column independent fitting
plus the wrong shape spends its capacity badly — on the multiclass tasks it fits `C` separate sigmoids on
`C` separate log-odds columns and then renormalizes, which both multiplies the chances of a bad per-column
fit and throws away the joint structure of the softmax. The diagnosis points two ways at once: I want a
parametric map (keep the data efficiency, keep the proper scores) but I want it to *cost even less
capacity* and to *respect the joint distribution* rather than calibrate columns independently. The cleanest
hypothesis that does both: maybe the dominant miscalibration isn't a per-class shape problem at all but a
single shared *scale* problem — the classifier's confidence vector has the right *direction* (the ranking
is good, isotonic confirmed it; the argmax is fine) and the wrong *magnitude*. If that's the dominant
error, the right correction is the minimal one: rescale the whole confidence vector by a single shared
number, and nothing else.

Let me make that precise and check it has the properties I need before I commit. A classifier's output is
a softmax of some logits `z`. The minimal transform that fixes a uniform scale error, and *only* that, is
to divide every logit by one shared positive number `T`: calibrated `= softmax(z/T)`, one scalar `T > 0`
shared across all classes and all examples. Borrow the name from where this operation already lives in
statistical mechanics and distillation — call `T` the temperature. Now check the properties. First,
because `T` is the *same* for every class, dividing by a positive `T` is a monotone transform of the
logits that doesn't reorder them: `argmax_k z_k/T = argmax_k z_k`. So the predicted class never changes,
the accuracy is *exactly* preserved — not approximately, exactly. That's a property Platt's per-class
renormalized fit could not promise (different `(A_c, B_c)` per class can reorder the argmax after
renormalization). Second, the effect on the probabilities is precisely a softening: as `T → ∞` the scaled
logits flatten toward the uniform `1/C`; at `T = 1` I recover the original; as `T → 0⁺` mass collapses
onto the argmax. So sweeping `T` upward does exactly one thing — bleed confidence out of an overconfident
model toward uniform — which is the single knob the shared-scale hypothesis says I need. And with *one*
parameter there is even less capacity to overfit than Platt's two, which is the direction the variance
diagnosis told me to go.

There's a subtlety the task surface forces, and getting it right is the whole ballgame. Temperature
scaling is naturally defined on *logits*, but the harness hands me *probabilities* — a vector `p` whose
rows already sum to one for multiclass, a single positive-class `p` for binary. I have to recover logits
to divide. For multiclass, the softmax is invariant to an additive shift in the logits, so I can take
`z = log(p)` as a valid logit vector (any constant offset washes out in the next softmax); dividing
`log(p)` by `T` and re-softmaxing is exactly temperature scaling on the implied logits. For binary, the
harness gives only the positive-class probability `p`, so I reconstruct a two-class logit pair
`z = [log(1−p), log(p)]` and run the same scalar division and two-class softmax, returning the
positive-class component. That's the literal fill: build logits from the probabilities, divide by `T`,
softmax (numerically stable — subtract the row max before exponentiating), and for binary peel off the
positive column.

How do I pick `T`? I want honest probabilities, and the clean differentiable objective that rewards honest
probabilities is the negative log-likelihood — it's a strictly proper scoring rule, minimized in
expectation when the predicted distribution matches the truth, and it's the same proper score where Platt
already did well and isotonic bled. The binned ECE I also care about is non-differentiable, a bad thing to
optimize directly; NLL is its smooth cousin and the right lever. So: freeze everything, compute the
implied logits on the calibration split once, and minimize the calibration NLL over the single scalar `T`.
Since the argmax-preservation argument requires `T > 0`, I bound the optimization to `T ∈ [0.01, 20.0]`
and clamp `max(T, 0.01)` inside the objective, run L-BFGS-B from `x0 = 1.5` (I expect an overconfident
model, so I start above 1 to soften rather than sharpen). One parameter, a smooth one-dimensional problem,
a couple hundred iterations — done.

I want to know this is principled and not a lucky hack that happened to match my shared-scale story, because
if it falls out of a clean optimization principle I'll trust it across these four very different
classifiers. Calibration is about *not being more confident than the evidence warrants* — being as
high-entropy as possible while staying consistent with the data. Literally maximize the entropy of the
predicted distributions subject to one coupling constraint: the average logit assigned to the *true* class
must equal the average logit under the predicted distribution (a moment-matching condition that pins the
distributions to the data without dictating their shape), plus the per-example normalization constraints.
Maximum entropy under a moment constraint produces an exponential-family form: turning the Lagrangian crank,
each `q_i^{(k)} = exp(λ z_i^{(k)} + β_i − 1)`, and imposing `Σ_k q_i^{(k)} = 1` divides out the per-example
factor to leave `q_i^{(k)} = softmax(λ z_i)^{(k)}`. Identify `λ = 1/T` and that is temperature scaling
exactly. So the single scalar isn't a convenient small choice — it is the lone Lagrange multiplier of the
lone coupling constraint. And it makes the overconfidence story precise: an overconfident model has
low-entropy outputs, and raising `T` (lowering `λ`) raises the entropy back to what the calibration data
support. The full scaffold module is in the answer.

I should be honest about where this can lose to Platt, because the hypothesis it rests on is strong. If a
classifier's miscalibration *isn't* a shared scale error — if different classes (or the two extremes of a
binary score) are miscalibrated in *different* directions — then one global `T` can't fix it, and a method
with a per-class or shape degree of freedom can do better on that column. Concretely: on a binary task
whose reliability curve needs a *location* shift, not just a scale, temperature scaling has no `B`-like
offset to move the midpoint, so it could leave residual ECE that Platt's offset would have caught; and on
a column whose true curve is an inverse-sigmoid (extreme-pushed probabilities that need *gathering*), one
shared `T` can soften but cannot reshape. So I'd expect temperature scaling to win cleanly where the
dominant error is overconfidence-as-scale — the multiclass tasks especially, RF and MLP, where one
softening `T` over the whole softmax should beat Platt's `C` independent per-column sigmoids and respect
the joint distribution — and to be roughly a wash, or slightly behind on a metric or two, where the error
has a location or shape component a single scalar can't reach.

Here is what I expect against Platt's measured numbers, falsifiably. The ECE regressions Platt introduced
should *reverse* on the tasks where a single scale is the right correction: RF on MNIST ECE should fall
back well below Platt's 0.0254 (toward isotonic's 0.0156 or better), because one `T` over the whole softmax
is the natural fix for forest overconfidence; SVM on Breast Cancer ECE should drop substantially below
Platt's blown-up 0.0493, because the per-class sigmoid was overfitting that binary column and one scalar
won't. On the proper scores I expect temperature scaling to *hold* Platt's gains and likely improve NLL on
the tasks it calibrates by scale — RF NLL at or below 0.163, SVM NLL at or below 0.101 — because the
maxent argument says the scalar `T` is the NLL-optimal global correction. Where I am *not* confident is
GBM on Madelon and the joint-vs-marginal question on the binary tasks: if Madelon's error has a location
component, temperature scaling's lack of an offset could leave its ECE near or even slightly above Platt's,
and Brier on a task that Platt's per-column fit happened to suit could be a wash. If the pattern is what I
expect — ECE recovered where the error was scale, proper scores held or improved, accuracy exactly
preserved by construction — then temperature scaling is the strongest of the three: it gets the single
clearest win count across the twelve cells by making the *minimal* correction the data support. The one
gap it leaves open, by construction, is the column whose distortion is not a shared scale — a curve that
needs both a *shape* that bends two ways and a *location* offset, on bounded scores — and that gap is
exactly the opening a richer parametric family would have to fill.
