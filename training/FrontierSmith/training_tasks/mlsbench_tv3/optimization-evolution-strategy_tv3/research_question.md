The suite runs the same separable multimodal landscape twice, at thirty
and at one hundred dimensions, and the distance between those two
best_fitness numbers is the most honest measurement this environment
makes: it exposes how much of a strategy's apparent skill was a
thirty-dimensional tuning accident. This variant is about making that
distance small.

Scaling discipline is the whole subject. Every quantity that shapes
variation must carry its dependence on dimension explicitly:
per-coordinate step lengths, expected genes changed per individual, the
strength of survivor pressure — none of them tuned as a constant at n=30
and silently wrong at n=100. Whatever structure the strategy maintains
about coordinate scales must cost O(n) per generation, so that the
algorithm the hundred-dimensional setting runs is the same algorithm the
thirty-dimensional settings ran, only wider. The ill-conditioned valley
and the flat-rim basin remain in the suite precisely so that
dimension-robustness cannot be bought by overfitting the separable
landscape.

Concretely, a submission should be built and argued in that order. State
the intended scaling law for each internal quantity before showing any
results. Then let the reported metrics check the law: rastrigin-100d
best_fitness may degrade from rastrigin-30d no faster than the statement
predicts, and convergence_gen at one hundred dimensions should stay a
bounded multiple of its thirty-dimensional counterpart rather than
exploding.

The claim to defend is dimension-invariance: same code, same constants,
a written scaling argument, and four reported numbers consistent with it.
