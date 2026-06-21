## Research question

Minimax lower bounds ask what error is unavoidable before choosing any particular estimator. Given a model class `P in mathcal P`, a target functional `theta(P)`, a loss scale `rho`, and `n` observations, the object is

`inf_{hat theta} sup_{P in mathcal P} E_P[Phi(rho(hat theta, theta(P)))]`.

The question is: how can one prove such a universal lower bound without analyzing every possible decision rule directly?

## Statistical setup

The minimax risk is defined over a model class and a loss geometry. Proving a lower bound means ruling out the possibility that any measurable estimator achieves small risk uniformly over the class. The separation between different parameter values is measured in the loss geometry of the functional. Whether two probability distributions can be told apart is measured in the experiment geometry: total variation directly controls the best binary test; KL divergence relates to total variation through Pinsker-type inequalities; Hellinger distance is convenient because it behaves well under products and mixtures.

The typical settings include normal mean estimation, nonparametric smoothness lower bounds, density estimation, functional estimation, and privacy-constrained estimation.

## Baselines

- **Analyze a concrete estimator.** Studying a specific procedure can show how that procedure performs in the model class.

- **Local asymptotic calculations.** These reveal rates around a fixed parameter value, characterizing estimation difficulty in a neighborhood.

- **Bias-variance decompositions.** These are useful for upper bounds and for understanding an estimator's error structure. They depend on the specific estimator's form.

- **Large packing arguments such as Fano or Assouad.** These handle many-way ambiguity and often capture dimension dependence of minimax rates. Fano's inequality relates the minimax risk to the average KL divergence in a packing set. Assouad's lemma reduces to coordinate-wise testing and is especially useful for product-structured model classes.

## Evaluation settings

The difficulty of a minimax problem is jointly governed by two quantities: how separated the hardest instances are in the parameter space, and how indistinguishable the corresponding observations are in probability space. Typical examples span parametric and nonparametric settings:

- **Normal mean:** estimating `mu` from `X_1, ..., X_n ~ N(mu, sigma^2)` with squared loss.
- **Nonparametric regression:** estimating `f` from a Sobolev or Hölder smoothness class.
- **Density estimation:** total variation or L2 risk over a smoothness class.
- **Functional estimation:** estimating a scalar feature such as the entropy or a linear functional of an unknown density.

Success in proving a lower bound means finding distributions within the model class where the target values are well-separated while the observed data distributions remain statistically close.
