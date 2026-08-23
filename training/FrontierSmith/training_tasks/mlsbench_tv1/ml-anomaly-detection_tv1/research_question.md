The four benchmarks share a format but not a regime: anomalies are 2.5% of
thyroid, about 10% of cardio, 7% of shuttle — and nearly a third of satellite.
Most unsupervised detectors carry a hidden assumption about which regime they
live in. Tail-probability and density methods presume rarity and degrade when a
third of the data is "anomalous"; boundary methods that concede a large outlier
fraction throw away resolution exactly where anomalies are needles in a
haystack. The multiplicative aggregation across datasets makes this the whole
game: a detector that is excellent in three regimes and collapses in the fourth
scores like a collapse.

This variant therefore asks for regime-robust scoring: one detector with one
fixed configuration, adapted to each dataset only through statistics it
computes from the unlabeled features at fit time — no per-dataset
hyperparameters and no contamination constant tuned to a known anomaly rate.
The failure mode it targets is the gap between ranking and decision: a score
distribution can order test points well (high AUROC) yet be so flat, or so
heavy-tailed near the operating point, that the F1 measured after thresholding
falls apart — and the two metrics carry equal weight here. The shape of the
score distribution matters, not just its ordering; the method should either
estimate from unlabeled data where its own threshold will land, or emit scores
whose separation survives both a 2.5% and a 31.6% contamination.

The scaffold provides a subsampled multi-view distance ensemble and an explicit
contamination-signal hook that currently returns a constant; the contribution
is what statistic fills that hook and how the scoring rule exploits it. Success
means the weakest of the four settings — measured by both metrics — is as
strong as you can make it, without surrendering the regimes that rarity
assumptions currently favor.
