## Research Problem

Finite data often have to choose their own model structure. A time series needs an autoregressive order, a symbolic sequence needs a Markov order, and a regression problem needs a degree or subset of predictors. Once the structure is fixed, estimating the real-valued coefficients is routine: maximize likelihood, minimize squared error, or solve the corresponding normal equations. The difficult part is the integer choice.

Raw fitted likelihood cannot make that choice. In nested families, the best fit at order `k + 1` can reproduce the best fit at order `k` and then adjust one more degree of freedom, so the maximized likelihood cannot decrease. If this score is used alone, the selected model is the largest model under consideration. The missing ingredient is a principled way to charge complexity on the same scale as fit.

## Available Ingredients

Information theory already gives a unit for measuring surprises. If a distribution assigns probability `P(x)` to an observation, an ideal code spends about `-log P(x)` units on that observation. Entropy is the expected cost under the source distribution, and prefix coding supplies the bridge between probability assignments and decodable code lengths.

Algorithmic information theory gives a sharper philosophical slogan: regular structure in a finite string is whatever lets the string be specified more compactly than by copying it verbatim. That slogan is attractive for inference, but the universal version is too broad for ordinary statistics. General program-size complexity is not computable in the form needed for model selection, and additive language constants can swamp finite samples.

## Existing Baselines

Maximum likelihood gives the best coefficients at a fixed order, but across orders it rewards extra degrees of freedom even when they fit noise. It solves estimation inside a model and leaves model choice open.

Akaike's information criterion corrects maximized likelihood by adding a parameter-count penalty derived from expected predictive bias. It is an important information-theoretic baseline, but its penalty does not grow with sample size, and its derivation is tied to expected divergence from a data-generating distribution.

Wallace and Boulton's message-length approach uses a two-part message for a hypothesis and the data given that hypothesis. Its hypothesis part is explicitly prior-based, so it belongs to a Bayesian message interpretation rather than a coding rule whose primary object is a decodable description of the observed data.

## Evaluation Setting

The relevant experiments are classical order-selection problems. In autoregression, a fitted residual variance falls as order increases. In Markov modeling, longer contexts fit transition counts more closely. In polynomial or subset regression, more predictors reduce in-sample error. A useful criterion must stop before the richest model simply memorizes sample noise.

## Open Requirements

The score must be computable from quantities already present in the fitting problem: the achieved fit, the size of the model class being used, and the finite amount of data available. It should choose both the integer structure and the fitted coefficients without adding a hand-tuned constant that changes the outcome by convention.
