## Research Question

A learner sees independent training pairs from an unknown distribution and must choose a rule that will perform well on future pairs from the same source. The quantity that matters is expected loss on the distribution, but the learner cannot compute that expectation directly. The immediate observable proxy is the average loss on the sample.

For a single preselected rule, classical probability already says the sample average estimates the expectation. A learning procedure instead compares many candidates and keeps the one with the lowest observed loss, so the rule is chosen after the data are seen. The setting is therefore: a learner searches a class of candidate rules using only sample-average loss and selects from it. The question is when the selected rule's true risk can be related to the best risk available in that class.

## Background

For a fixed candidate rule `f`, the loss values on i.i.d. data are i.i.d. random variables. A law of large numbers and concentration inequalities make

```text
R_emp(f) = (1/n) sum_i L(f(x_i), y_i)
```

close to

```text
R(f) = E[L(f(X), Y)].
```

A learning procedure usually does not evaluate one fixed rule. It compares many candidates and keeps the one with the lowest observed loss, so the selected rule is the one favored by the realized sample.

For finite candidate sets, a union bound extends a fixed-rule concentration inequality to all candidates at once, paying a complexity cost involving `log N` for a class of `N` rules. Many useful rule classes are infinite.

## Baselines

One baseline is to trust the training loss directly. This is used when the candidate class is tiny or heavily constrained.

Another baseline is to analyze one predetermined statistical model, giving estimator theory for a fixed model rather than a class selected after the data.

A finite-class union bound accounts for searching over multiple candidates, applying when the effective number of candidates is finite and known.

A holdout set estimates future performance empirically after selection, using held-out data to score the selected rule.

## Failure Modes

A solution can establish only that one fixed rule has small sample-to-population error, while the eventual rule is the one favored by the realized sample.

A class that can match arbitrary labels on arbitrary samples can report perfect fit on the training data: low observed loss may then reflect memorization rather than the rule's behavior on the distribution.

## Evaluation Settings

The cleanest setting is binary classification with bounded loss. The sample is i.i.d.; the distribution is arbitrary and unknown; the learner chooses a rule from a fixed class; performance is measured by true error or expected loss.

Diagnostic candidate classes include finite rule lists, thresholds on the line, intervals, half-spaces in Euclidean space, and the class of all labelings of a sample. The finite and low-capacity examples are cases where sample averages can be compared against true risks across the class; the all-labelings example is the case where every sample can be fit exactly.

A proposed solution can be judged by whether it gives a high-probability statement about the largest class-wide gap between true and observed loss:

```text
P( class-wide risk gap > epsilon ) <= delta,
```

for arbitrary distributions as sample size grows.
