The held-out calibration split is not one resource but four very unequal
ones: about four thousand points for each image benchmark, roughly five
hundred for madelon, and barely over a hundred for breast cancer. A single
fixed method must be fitted independently on each. That thirty-five-fold
spread turns capacity into the central design variable — a mapping flexible
enough to repair a forest's warped reliability curve from four thousand
examples will happily memorise the sampling noise of a hundred-point split,
and the resulting jitter lands directly in the scored ECE, Brier and NLL of
the smallest setting.

This variant asks for calibration whose effective complexity is priced by the
evidence available. The contribution is the adaptation rule, not any single
mapping: how bin counts, smoothing strength, prior pseudo-counts, or
parametric-versus-nonparametric structure should scale with the size of the
split being fitted. A method may consult nothing about the setting except
what fit() receives — the probabilities, the labels, and their count — so the
complexity choice must be automatic, driven by n and by measured stability
rather than by hand-set per-dataset knobs.

Establish this on the unchanged four-setting protocol: at the small end, the
sample-adaptive method beats every fixed-capacity configuration of itself —
the variance it sheds exceeds the bias it accepts — while at four thousand
points it recovers essentially everything the flexible configuration would
have delivered. The scaffold is an equal-frequency binning calibrator whose
bin count follows a square-root rule and whose bin estimates shrink toward
the global accuracy under a fixed pseudo-count; both rules are crude
placeholders for the evidence-pricing this variant is actually about.
