# Changelog

## 2026-08-19 — svfix(W3_ancestors_only)
`reasoning.md` and `train_answer.md` misattributed the reason for adding the variational/KL
latent on top of the deterministic ED: both framed it as the fix for the mean-regression /
lost-variability complaint (from Rasp et al. 2018). The primary source (Behrens et al. 2022)
does not support that — its own text ties the KL term's demonstrated benefit specifically to
latent-space *interpretability* (sharper regime separation; the deterministic ED's latent is
"significantly harder to interpret," missing cumulus/deep-convective regimes the VED resolves),
and its own R² comparison shows the VED's reproduced variability is *lower*, not higher, than
the deterministic reference ANN — "stochastic convection parameterizations" is stated as future
work, not something this VED already delivers.

Fixed: `results/reasoning.md` — the decisive-step paragraph now derives the need for a
probabilistic/KL-regularized latent from a self-contained argument (the deterministic
bottleneck's code has no constraint on its geometry: any invertible relabeling of the linear
code leaves the reconstruction loss unchanged, so nothing pins samples into a shared, comparable
frame across inputs) and demotes the mean-regression motivation to an honestly-hedged secondary
hope, not a claimed resolution. `results/train_answer.md` — removed the flat claim "addressing
the lost-variability problem"; replaced with the hedged version plus the actually-demonstrated
payoff (latent geometry / regime separation).

No architecture, code, or numeric content changed — landing unchanged.
