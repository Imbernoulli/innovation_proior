## Research Question

How can maximum likelihood estimation be carried out when the data actually recorded are an incomplete, coarsened, grouped, censored, or latent-variable view of a fuller data structure whose likelihood would be easy to optimize if it had been observed?

The setting has an observed value `y`, a hypothetical complete value `x`, and a many-to-one observation rule that maps possible complete values into `y`. The observed likelihood is a marginal or summed likelihood over all complete values compatible with `y`, so the log likelihood of `y` places a logarithm outside an integral or sum over the missing part.

## Background

Classical likelihood theory supplies the language: likelihood, score, information, and sufficient statistics. Fisher's statistical-estimation program describes how coarsening a full sample relates to information and why likelihood equations are naturally expressed through scores.

Incomplete-data work also supplies pieces. Missing-information arguments factor complete-data likelihoods into observed and conditional pieces, then relate the observed score to the conditional expectation of the complete-data score. In mixture models, unobserved component labels lead to posterior class probabilities and weighted estimating equations. In genetics, variance components, missing normal observations, grouped observations, and mixture distributions, specialized substitution or iterative likelihood procedures have appeared.

## Baselines

The direct baseline is to maximize the observed likelihood numerically. This can require repeated integration, differentiation through marginal likelihoods, or solving nonlinear likelihood equations with missing-data terms.

Another baseline is hard substitution: fill in missing values or latent labels with a current best guess, then solve the complete-data problem.

A third baseline is a special-case estimating-equation iteration. Mixture and missing-information procedures use posterior weights for hidden categories or missing measurements, each derived for its original example.

## Evaluation Settings

The method should be judged in settings where the complete-data model has simple maximum-likelihood updates but the observed-data likelihood is difficult. Examples include grouped multinomial counts, missing entries in normal models, finite mixtures, variance components, factor analysis, hidden Markov models, and empirical Bayes or random-effects likelihoods.

The observed-data likelihood is the objective of interest. Relevant questions are how an iteration relates to that likelihood, what its fixed points correspond to under regularity conditions, and how it behaves relative to the global maximum.

## Code Framework

A concrete implementation setting is Gaussian mixture estimation. The observed data are feature vectors. The complete data would also include a component indicator for each vector. If those indicators were known, the maximum-likelihood estimates would be component counts, means, and covariances computed from assigned observations.

The modern software testbed is scikit-learn's mixture code: compute log component probabilities, normalize them into posterior label probabilities, update Gaussian weights and parameters by weighted sufficient statistics, track a lower-bound or observed-log-likelihood quantity, and stop when its change is small.
