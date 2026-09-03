"""Score spec for llm-on-policy-distillation.

Primary metric = arithmetic mean of three math-reasoning benchmark accuracies
(GSM8K, MATH-500, AMC23) after on-policy distillation training
(Qwen2.5-0.5B student, Qwen2.5-Math-7B-Instruct teacher).

Each per-benchmark accuracy is bounded to 1.0 (no normalization vs leaderboard
needed since accuracy is already a [0, 1] quantity).
"""

from mlsbench.scoring.dsl import *

term(
    "gsm8k",
    col("gsm8k_accuracy").higher().id().bounded_power(bound=1.0),
)

term(
    "math500",
    col("math500_accuracy").higher().id().bounded_power(bound=1.0),
)

term(
    "amc",
    col("amc_accuracy").higher().id().bounded_power(bound=1.0),
)

setting("qwen2.5-0.5b", weighted_mean(
    ("gsm8k", 1.0),
    ("math500", 1.0),
    ("amc", 1.0),
))

task(gmean("qwen2.5-0.5b"))
