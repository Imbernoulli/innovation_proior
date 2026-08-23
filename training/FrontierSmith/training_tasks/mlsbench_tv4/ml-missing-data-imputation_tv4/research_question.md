Columns of a real table are not interchangeable real numbers. Some are
counts or graded categories with a handful of distinct values, others are
smooth physical measurements, and scales within one dataset can differ by
orders of magnitude. An imputer that treats every column as a generic
Gaussian-ish quantity produces completions that are wrong in kind: a 2.7
planted in a column whose observed values are only 1, 2, and 3, or a
smoothed compromise in a column that is effectively categorical. The
learner consuming the completed matrix then gets to invent split points
between real data and imputation artifacts.

This variant makes column heterogeneity the organizing problem. At fit
time, decide from observed values alone what each column is -- a small
discrete support, an integer grid, a continuous range -- and give each
kind its own completion rule and its own notion of central tendency. The
scaffold ships mode-on-support for discrete-like columns, mean for
continuous ones, and a projection hook that snaps any refined estimate
back onto a discrete column's observed support. The contribution is a
typing discipline strong enough that every imputed cell is plausible as a
value of its column: on-support where the column is discrete, in-range
and scale-respecting where it is continuous.

Boundaries: typing must be inferred, never keyed to a known dataset, and
a wrong type call should degrade gracefully rather than catastrophically.
How the result is judged does not move; the payoff from type fidelity
has to show up through the usual readouts or it does not count. The claim
to defend is that respecting column kind is not cosmetic -- identify where
type-blind filling loses, and show the mechanism that recovers it.
