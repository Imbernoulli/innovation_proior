Between thyroid's six features and satellite's thirty-six lies the quietest
failure mode in unsupervised detection: coordinates that carry no anomaly
signal at all. Standardisation gives every feature the same variance, so a
distance, density, or tail computation weights the informative and the
irrelevant alike; each useless coordinate adds noise to every pairwise
distance and every pooled tail statistic. An anomaly that deviates sharply in
three coordinates out of thirty-six can be washed out — its deviation
averaged against thirty-three dimensions of ordinary fluctuation — while
full-space distances concentrate and lose contrast.

This variant asks for scoring that finds and protects the informative
subspace without labels. Candidate mechanisms: many low-dimensional views
whose verdicts are pooled, so that signal-bearing subsets can outvote diluted
full-space evidence; per-feature relevance weights estimated from unlabeled
structure (dispersion shape, bimodality, dependence with other features) and
used to bias which coordinates the score listens to; or sparse projections
that hunt for directions of unusual mass. The catch that keeps the problem
honest: thyroid has no dimensions to spare, so machinery that pays off at
thirty-six features must degrade gracefully to six, within one fixed
configuration and with no per-dataset switches.

What to establish on the unchanged AUROC and F1 protocol: subspace- or
relevance-aware scoring beats its own full-space counterpart on the wide
datasets and matches it on the narrow ones. The scaffold pools rank verdicts
from random low-dimensional histogram views but still samples coordinates
uniformly at random; making the sampling — or the vote weighting —
relevance-driven is the intended contribution.
