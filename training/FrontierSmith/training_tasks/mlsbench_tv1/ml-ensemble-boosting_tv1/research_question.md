Two hundred rounds is a long time to keep shallow trees useful. Early in a
boosting run almost any update rule makes progress; the regime that separates
strategies is the late one, where the ensemble already fits the easy structure
and each new depth-3 tree is aimed, by construction, at whatever residual
signal remains — which on real tabular targets is increasingly noise and
outliers. Exponential reweighting concentrates the sample distribution onto a
few stubborn points; plain gradient fitting keeps chasing residuals it can no
longer reduce. Either way the tail of the run is wasted or actively hurts the
held-out metric.

This variant asks for a boosting strategy designed around that late-round
regime. The strategy controls four things inside the fixed loop — initial
weights, per-round pseudo-targets, the learner coefficient, and the weight
update — and the design question is how these should evolve as a function of
progress through the run and of the residual distribution itself: when should
a sample's influence be capped, when should step sizes shrink, when is a round
better spent consolidating than correcting. One instance of the class must
serve both problem families the pipeline evaluates — a near-separable binary
classification task and two regression tasks with noisy targets — so the
reweighting, damping, and scheduling machinery must be shared, not two
unrelated algorithms selected by task type (only the loss whose gradient you
follow may differ).

The scoreboard is unchanged: held-out accuracy on the classification dataset
and held-out RMSE on the two regression datasets, combined multiplicatively,
so the weakest dataset dominates. The scaffold ships a deliberately damped
residual-fitting placeholder with an influence hook that currently treats
every sample equally and a running residual-scale statistic nothing consumes
yet; the intended contribution is the mechanism that makes rounds 100 through
200 earn their keep instead of memorizing noise.
