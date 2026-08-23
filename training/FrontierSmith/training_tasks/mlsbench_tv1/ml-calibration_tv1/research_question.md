Post-hoc calibration has a cheap trick and an expensive truth. The cheap trick:
pool predictions into coarse bins, or shrink everything toward the base rate,
and the binned calibration error falls while the probabilities become less
informative — the log-likelihood and squared-error terms quietly pay for the
ECE improvement. The expensive truth: the three metrics scored here move
together only when a mapping genuinely repairs miscalibration instead of
blurring confidence, and each setting averages all three, so a method that
games ECE at NLL's expense buys nothing.

This variant makes that tension the objective. Build one calibration mapping —
a single class with fixed hyperparameters and no per-setting switches — that
must digest four very different probability geometries from the fixed
pipeline: a random forest whose vote-ratio outputs contain exact zeros and
ones (unbounded log-loss until floored), an overconfident neural-network
softmax, boosted-tree scores, and an SVM whose outputs have already passed
through one sigmoid. Every fit sees only the modest held-out calibration
split, so a mapping flexible enough to repair the forest's per-class
distortions must not overfit the smallest split, barely a hundred points on
breast cancer.

The success criterion is agreement under the worst case. Settings combine
multiplicatively, so your score is dominated by whichever classifier-dataset
pair the mapping serves worst, and inside that pair by whichever of the three
error measures you sacrificed. Aim for sharpness-preserving recalibration:
lower ECE while leaving Brier and NLL at least as good as the best simple
alternative would achieve, on all four settings simultaneously — not
confidence laundering. The scaffold's placeholder applies a small fixed
shrinkage toward the calibration-set base rates and records a reliability
summary at fit time; the strength, the shape, and the class-awareness of that
correction are yours to redesign.
