## Research Question

A learner sees independent training pairs from an unknown distribution and must choose a rule that will perform well on future pairs from the same source. The quantity that matters is expected loss on the distribution, but the learner cannot compute that expectation directly.

The immediate observable proxy is the average loss on the sample. The hard question is not whether this average estimates the expectation for one preselected rule. Classical probability already says that it does. The hard question is whether the rule chosen after looking at the sample can still be trusted.

The target problem is therefore: under what conditions can a learner search a whole class of candidate rules using only sample-average loss and still obtain a rule whose true risk is near the best risk available in that class?

## Background

For a fixed candidate rule `f`, the loss values on i.i.d. data are i.i.d. random variables. A law of large numbers and concentration inequalities can make

```text
R_emp(f) = (1/n) sum_i L(f(x_i), y_i)
```

close to

```text
R(f) = E[L(f(X), Y)].
```

But a learning procedure usually does not evaluate one fixed rule. It compares many candidates and keeps the one with the lowest observed loss. This creates a selection effect: the selected rule may look good on the sample precisely because the sample underestimates its true risk.

Finite candidate sets show the shape of a solution. If there are only `N` possible rules, a union bound can extend a fixed-rule concentration inequality to all of them at once, paying a complexity cost involving `log N`. Useful rule classes, however, are often infinite, so raw counting cannot be the final answer.

## Baselines

One baseline is to trust the training loss directly. This can work when the candidate class is tiny or heavily constrained, but it fails when the class can memorize the observed examples.

Another baseline is to analyze one predetermined statistical model. That gives useful estimator theory, but it dodges the central learning issue: the rule is selected from a class after the data are known.

A finite-class union bound is a stronger baseline. It correctly accounts for searching over multiple candidates, but it only applies when the effective number of candidates is finite and known.

A holdout set estimates future performance empirically after selection. It is practical, but it consumes data and does not by itself explain what structural property of the candidate class makes learning possible.

## Failure Modes

A misleading solution can prove only that one fixed rule has small sample-to-population error. That is not enough, because the eventual rule is the one favored by the realized sample.

Another misleading solution can report perfect fit on the training data without asking how many different label patterns the candidate class can realize. If the class can match arbitrary labels on arbitrary samples, low observed loss may be only memorization.

A useful principle must therefore account for both selection after sampling and the effective size of the class being searched. The class must be fixed before the sample is used, and the probability statement must cover the whole class rather than only the final selected rule.

## Evaluation Settings

The cleanest setting is binary classification with bounded loss. The sample is i.i.d.; the distribution is arbitrary and unknown; the learner chooses a rule from a fixed class; performance is measured by true error or expected loss.

Diagnostic candidate classes include finite rule lists, thresholds on the line, intervals, half-spaces in Euclidean space, and the class of all labelings of a sample. The finite and low-capacity examples should allow sample averages to track true risks across the class. The all-labelings example should expose memorization.

A proposed solution should be judged by whether it can give a high-probability guarantee that the largest class-wide gap between true and observed loss is small:

```text
P( class-wide risk gap > epsilon ) <= delta.
```

If such a guarantee holds for arbitrary distributions as sample size grows, then the sample-chosen rule can be certified rather than merely hoped to generalize.

