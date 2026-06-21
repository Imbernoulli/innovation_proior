## Prediction Without a Model

A learning algorithm can produce a point prediction long before it can justify a reliable uncertainty statement. A support vector machine, nearest-neighbor rule, regression fit, or neural classifier may rank candidate answers well in practice, and its raw confidence is typically a modeling claim that rests on a likelihood, posterior, variance model, or asymptotic approximation.

A different object of interest is a set-valued prediction rule. At significance level `eps`, such a rule outputs a set of labels or responses, and one would like that set to miss the next true label with probability at most `eps`, using finite data and without assuming that the fitted model is correctly specified.

## What Symmetry Can Say

IID sampling is stronger than needed for some finite-sample conclusions. If the observed examples and the next example are exchangeable, then their order carries no information once the unordered collection is fixed. This is the same kind of symmetry that makes permutation tests and some classical prediction arguments exact without estimating an unknown parametric distribution.

The setting is to turn that symmetry into a prediction set for a new object whose label is not yet known. The object features may be highly informative, and one wants statements that hold under exchangeability of the data-generating law.

## Existing Routes

Classical confidence and prediction intervals are exact in special models, such as normal linear settings, with intervals derived from the assumptions of the model. Algorithmic-randomness ideas test whether a completed data sequence looks typical, using universal tests. Early transductive schemes attach confidence to particular support-vector predictions, tied to a specific algorithmic construction.

These routes judge candidate completions by how compatible they are with the old examples, by way of distributional modeling or algorithm-specific confidence formulas.

## The Reliability Setting

The question is how to use a practical prediction heuristic. The heuristic may be excellent, mediocre, or bad, and one would like a finite-sample guarantee that treats it as a way to rank which candidate completions look more or less plausible relative to the observed data, rather than as a likelihood, posterior, variance estimate, or true conditional error.

This keeps two demands distinct: the reliability statement comes from the sampling structure, while the predictive heuristic affects how small or broad the returned set becomes.

## Scope

A rule of this kind would apply to classification and regression, finite label spaces and continuous responses, under exchangeability. There is also a computational dimension: a direct version may test many candidate labels using completed samples, while a cheaper version may train or choose a score first and reserve fresh data only for calibration.
