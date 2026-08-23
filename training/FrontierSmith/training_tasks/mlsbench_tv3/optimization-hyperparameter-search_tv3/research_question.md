Before the first validation score returns, a tuner knows nothing about
the problem except the declaration of its search space: which
parameters exist, their types, their bounds, which axes are marked
log-scaled, and which choices a categorical offers. This variant asks
how much performance can be extracted from that declaration alone —
the entire subject is the opening, the trials proposed while history
is empty or nearly so.

The constraint that gives the question teeth: the opening must be a
function of space structure and the seed, and of nothing else. No
defaults memorised from other tuning problems, no recognising a
benchmark by its parameter names, no knowledge smuggled in from
outside the run. What structure licenses is real, though. A log-marked
axis announces that its plausible values span orders of magnitude and
that uniform coverage in raw units covers almost nothing; bounds
define a geometry in which stratifying, spreading and centering are
meaningful acts; a categorical with three choices demands that all
three be seen early. An opening that actually reads the declaration
should place its first trials very differently from one that samples
blind.

convergence_auc integrates incumbent quality over spent cost, so the
earliest segment of the curve — precisely the part the opening
controls — counts the same as any other segment; best_val_score
records whether the head start compounds once feedback arrives or
merely evaporates into what any search would have found.

To defend: a structured, declaration-derived opening lifts the early
curve on all three benchmarks, and the lift survives to the final
incumbent — against the null hypothesis that before data arrives,
uniform random is as good as anything.
