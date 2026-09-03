"""Score spec for libero-lifelong.

LIBERO paper (Liu et al. NeurIPS 2023) canonical lifelong metric is AUC —
area under per-task success curve, which captures both forward learning and
backward retention in one number. avg_final_success is a coarse fallback;
regularization methods (EWC) score ~0 on avg_final due to catastrophic
forgetting yet ~0.06 AUC in the paper.
"""
from mlsbench.scoring.dsl import *

term("auc",
    col("auc").higher().id()
    .bounded_power(bound=1.0))

setting("train", weighted_mean(("auc", 1.0)))

task(gmean("train"))
