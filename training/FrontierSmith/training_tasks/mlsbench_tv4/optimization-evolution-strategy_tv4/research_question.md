One restriction defines this variant: the optimizer may compare fitness
values, but it may never do arithmetic with them. Ranks, medians taken
as order statistics, success counts from head-to-head comparisons — all
admissible. Fitness-proportional weights, magnitude thresholds, averages
of objective values steering any decision — all forbidden.

What that discipline buys is invariance: compose the objective with any
strictly increasing transformation and a compliant strategy's
trajectory is unchanged, seed for seed. Invariance is the classical
insurance policy of evolution strategies — it is what makes behaviour
portable to objectives whose numeric values are arbitrary, distorted or
untrustworthy — and it is a property of the code, checkable by reading
the editable section.

The scientific question is the price of the insurance on this suite.
The four settings report best_fitness and convergence_gen under a fixed
seed, and those numbers are indifferent to how fitness was consumed
internally; if comparison-only machinery — rank-weighted parent
sampling, success-driven step control in the spirit of the one-fifth
rule, elitism kept by comparison — reaches values competitive with
magnitude-using designs, the insurance was free. If it cannot, the gap
measures exactly what magnitude information was worth here.

Two rules keep the exercise honest. Sorting and order statistics are
the only permitted lens on fitness anywhere in the loop, including any
adaptation mechanism; and the discipline may not be laundered through
the harness — printing requires fitness values, but printing is not a
decision.

Deliverable, in the terms this environment scores: a strategy whose
every use of fitness is a comparison, defended by an audit naming each
place fitness is consumed, together with four best_fitness values and
convergence_gen numbers showing the restriction did not slow the
approach to the final value.
