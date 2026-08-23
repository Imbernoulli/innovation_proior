An imputed cell is not a measurement; it is a guess, and different holes
support guesses of very different quality. A cell in a column tightly
coupled to its neighbours can be inferred with confidence, while a cell
in a weakly coupled column -- or in a row stripped of most of its context
-- admits little more than the column's typical value. This variant asks
for an imputer that knows the difference and acts on it: deviation from a
safe central anchor must be earned by evidence, and where evidence is
thin the method should prefer admitting ignorance to manufacturing
detail.

The core requirement is a monotone evidence-to-boldness relationship.
Quantify, per column or per cell, how much support the observed data
gives an inference -- fraction observed, strength of dependence on other
columns, how much genuine context the row retains -- and let the
completion interpolate between the conservative anchor and a model-based
estimate as that support grows. Weak coupling must never license bold
extrapolation. The scaffold computes a per-column evidence score from
observed fraction and dependence strength and exposes the interpolation,
but currently pins it at the anchor end: full conservatism, confidence
measured but never spent. Calibrating that middle ground is the
contribution.

What disciplines the design is that the benchmark's two readouts punish
different sins. Pointwise error punishes wild
guesses, so conservatism under weak evidence is rewarded there;
downstream quality punishes invented structure that misleads the learner,
which is precisely the risk of confident hallucination. Make the case
that a method calibrated to its own ignorance concedes almost nothing
where inference was genuinely possible, and dodges the losses that
overconfidence takes where it was not.
