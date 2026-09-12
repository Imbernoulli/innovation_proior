"""Score spec for llm-hybrid-posttraining.

The task score is the geometric mean of final validation accuracies on the
three math benchmark splits used by the shared HPT training run.
"""

from mlsbench.scoring.dsl import *


term(
    "aime24",
    col("test_score_aime24").higher().id().bounded_power(bound=1.0),
)

term(
    "amc23",
    col("test_score_amc23").higher().id().bounded_power(bound=1.0),
)

term(
    "math_500",
    col("test_score_math_500").higher().id().bounded_power(bound=1.0),
)

setting("AIME24", weighted_mean(("aime24", 1.0)))
setting("AMC23", weighted_mean(("amc23", 1.0)))
setting("MATH-500", weighted_mean(("math_500", 1.0)))

task(gmean("AIME24", "AMC23", "MATH-500"))
