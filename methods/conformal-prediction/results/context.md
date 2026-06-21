## Prediction Without a Model

A learning algorithm can produce a point prediction long before it can justify a reliable uncertainty statement. A support vector machine, nearest-neighbor rule, regression fit, or neural classifier may rank candidate answers well in practice, but its raw confidence is usually a modeling claim. If the likelihood, posterior, variance model, or asymptotic approximation is wrong, the stated uncertainty can become overconfident.

The desired object is instead a set-valued prediction rule. At significance level `eps`, the rule should output a set of labels or responses that misses the next true label with probability at most `eps`, using finite data and without assuming that the fitted model is correctly specified.

## What Symmetry Can Still Say

IID sampling is stronger than needed for some finite-sample conclusions. If the observed examples and the next example are exchangeable, then their order carries no information once the unordered collection is fixed. This is the same kind of symmetry that makes permutation tests and some classical prediction arguments exact without estimating an unknown parametric distribution.

The challenge is to turn that symmetry into a prediction set for a new object whose label is not yet known. The object features may be highly informative, but the proof must not depend on knowing the true data-generating law.

## Existing Routes and Their Cost

Classical confidence and prediction intervals can be exact in special models, such as normal linear settings, but those intervals inherit the assumptions of the model. Algorithmic-randomness ideas suggest testing whether a completed data sequence looks typical, but universal tests are not computable. Early transductive schemes attach confidence to particular support-vector predictions, yet they remain tied to a specific algorithmic construction.

A general solution has to keep the useful part of those routes: judge candidate completions by how compatible they are with the old examples, while avoiding both distributional modeling and algorithm-specific confidence formulas.

## The Reliability Gap

The unresolved problem is how to use a practical prediction heuristic without trusting its numerical confidence as a probability. The heuristic may be excellent, mediocre, or bad; a finite-sample guarantee should not require it to be a likelihood, posterior, variance estimate, or true conditional error. It should only need the heuristic to rank which candidate completions look more or less plausible relative to the observed data.

This separates two demands that are often conflated. The reliability statement has to come from the sampling structure, while the predictive heuristic should only affect how small or broad the returned set becomes.

## Success Criteria

The final rule should work for classification and regression, finite label spaces and continuous responses, and should remain valid under exchangeability. It should degrade by becoming broad or uninformative when the heuristic is poor, rather than by silently losing coverage.

It should also expose the computational tradeoff. The most direct version may test many candidate labels using completed samples, while a cheaper version should be able to train or choose a score first and reserve fresh data only for calibration.
