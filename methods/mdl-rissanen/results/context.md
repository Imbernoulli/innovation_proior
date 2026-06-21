## Research Problem

Finite data often have to choose their own model structure. A time series needs an autoregressive order, a symbolic sequence needs a Markov order, and a regression problem needs a degree or subset of predictors. Once the structure is fixed, estimating the real-valued coefficients is routine: maximize likelihood, minimize squared error, or solve the corresponding normal equations. The difficult part is the integer choice, and the question is how to put model complexity and data fit on a common scale.

## Available Ingredients

Information theory already gives a unit for measuring surprises. If a distribution assigns probability `P(x)` to an observation, an ideal code spends about `-log P(x)` units on that observation. Entropy is the expected cost under the source distribution, and prefix coding supplies the bridge between probability assignments and decodable code lengths.

Algorithmic information theory provides a philosophical slogan: regular structure in a finite string is whatever lets the string be specified more compactly than by copying it verbatim. General program-size complexity relates regularity to compressibility.

## Existing Baselines

Maximum likelihood gives the best coefficients at a fixed order and is the standard approach to estimation inside a parametric model.

Akaike's information criterion corrects maximized likelihood by adding a parameter-count penalty derived from expected predictive bias. Its penalty is a constant multiple of the parameter count, independent of sample size.

Wallace and Boulton's message-length approach uses a two-part message for a hypothesis and the data given that hypothesis, with the hypothesis part encoded under an explicit prior.

## Evaluation Setting

The relevant experiments are classical order-selection problems. In autoregression, a fitted residual variance falls as order increases. In Markov modeling, longer contexts fit transition counts more closely. In polynomial or subset regression, more predictors reduce in-sample error. A useful criterion must select an order from among these nested families.
