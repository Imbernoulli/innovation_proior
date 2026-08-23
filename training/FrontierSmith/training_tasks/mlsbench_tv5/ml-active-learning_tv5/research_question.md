No single acquisition heuristic wins the whole curve. Greedy confidence-based
picking has documented dead zones where it trails uniform sampling for
rounds; committee methods shine mid-run and stall late; representativeness
schemes do the opposite. Yet the reported score integrates accuracy across
all twenty budgets on three quite different datasets, so a strategy with one
bad regime anywhere pays for it at every subsequent budget on that dataset.
The engineering target of this variant is regret, in the online-learning
sense: at no point on any of the three learning curves should the deployed
strategy sit meaningfully below the best member of a reference family of
heuristics — and it may never peek at test accuracy to find out which member
that is.

The natural architecture is a portfolio. Maintain several qualitatively
different acquisition experts — confidence-seeking, disagreement-seeking,
representativeness-seeking, uniform — and let each round's batch be a
weighted blend of their nominations. The scientific content is the weighting
signal: everything must be estimated from quantities visible inside the run,
such as how much the model's predictions on the pool actually shifted after
an expert's labels arrived, whether training loss on newly purchased points
was high or trivial, or how quickly each expert's nominations are becoming
redundant with the labeled set. Static blend ratios are the fallback, not the
answer; the interesting rule adapts them per dataset, per phase, from
evidence. Batch composition must remain sensible as weights drift — experts
nominating overlapping points should not double-charge the budget.

Scoring is exactly the harness's usual pair — endpoint accuracy plus the
integrated curve, multiplied across letter, spambase, and splice — and the
success criterion mirrors the regret framing: on each dataset separately,
the portfolio's curve should hug or exceed the strongest fixed heuristic for
that dataset, demonstrating insurance without a premium.
