Every CATE estimator leans on nuisance functions — an outcome regression,
a propensity model, or both — and the three synthetic settings are built
to break different ones. The small nonlinear setting punishes parametric
outcome surfaces; the high-dimensional confounded setting makes the
propensity genuinely hard to learn; the economic setting's outcome scale
magnifies any first-stage sloppiness. A plug-in construction inherits the
full first-order error of whichever nuisance the current dataset happens
to break, which is why single-nuisance methods look brilliant on one
benchmark and embarrassing on the next.

What is demanded here is orthogonality as an engineering discipline, not
a citation. Construct the estimator so that nuisance errors enter the
effect estimate only through products of errors — doubly robust
pseudo-outcomes, residual-on-residual formulations, or an equivalent
device — with cross-fitting inside your fit routine so no nuisance is
evaluated on its own training rows. Then verify the property you are
claiming: deliberately degrade each nuisance in turn (coarsen the
propensity, misspecify the outcome model) in your own experiments and
show the effect estimate bends rather than breaks. A method that only
works when both first stages are lucky is precisely what this variant
exists to rule out.

Judgment is rendered by the existing harness: PEHE and ATE error on all
three datasets, five-fold cross-fitting, ten replications, one
configuration throughout. The signature of genuine orthogonality is the
absence of a disaster column — no setting where the numbers reveal that
a nuisance failure propagated at full strength — combined with
competitive accuracy where nuisances are learnable. Report which
nuisance stressed each dataset and how the construction absorbed it.
