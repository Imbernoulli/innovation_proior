# Changelog

## 2026-08-18 — svfix(epistemic)
The 2026-08-18 W3_ancestors_only pass (commit b2681588b) grounded the bias-toward-zero
decisive step in a real trial (placebo-controlled attitude-change field experiment, 501
respondents, 26 baseline covariates, placebo-arm R^2=0.77, direct ATE 0.22/SE 0.072/t=3.1)
instead of an invented synthetic escalation — but it wrote the narrator as having actually
*fit the single pooled model on that trial and observed the result* ("I fit my single
pooled model ... The estimates collapse toward zero ... trees very rarely split on it ...
The fear is confirmed"). A single-turn proposal has no results yet, real numbers or not.

Rewrote the passage in `results/reasoning.md` to keep: the trial's own pre-dating facts
(design, sample size, covariate count, placebo-arm R^2, direct randomized ATE — these are
known facts about the trial, not the method's output); the discriminating-experiment DESIGN
(fit the single pooled S-learner with the treatment flag as a feature on this trial's data,
then run the same tree-inspection diagnostic at 100,000-tree scale, tallying splits on the
flag); and the PREDICTION (if the fear is right, tau_hat should collapse toward zero and the
flag should rarely be split on, even though the direct estimate says the effect is real) plus
the reason this test is more trustworthy than a self-built simulation ("I don't get to pick
the numbers"). Removed the claimed fit, the claimed read-off of tau_hat, the "estimates
collapse" / "trees very rarely split" / "fear is confirmed" reported outcomes, and the
framing that treated the failure as already demonstrated on data.

The landing (single-model recipe biased toward zero when treatment signal is weak relative
to the covariates, motivating the "two ends of a knob" characterization) is now unjustified
by this specific passage without an actual observation — expected per the epistemic-fix rule;
this unit needs trajectory-track conversion to carry the real run and its result.
