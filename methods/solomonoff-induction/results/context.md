## Research question

Induction asks how an observer should generalize from a finite history to future observations. Ordinary Bayesian inference gives a clean answer only after the observer has chosen a model class and a prior over that class. That choice is powerful but also constraining: a conclusion can depend less on the evidence than on which hypotheses were admitted before the calculation began.

The question is whether the model class can be made universal rather than hand selected. Instead of choosing a family such as polynomials, Markov chains, linear regressions, or a finite expert set, one can consider every computable way the observed sequence might have been generated. The only fixed background object would then be a universal machine, chosen before seeing the data.

The core question is therefore: can induction be defined as Bayesian prediction over all computable explanations, with simpler explanations receiving more prior mass?

## Background

A finite binary string `x` can be explained by any program `p` that makes a universal Turing machine output a sequence beginning with `x`. Short programs correspond to compact rules; long programs can encode special cases, lookup tables, or arbitrary continuations. Solomonoff's move is not to pick the shortest program alone, but to sum over all programs that explain the data.

For a prefix universal machine `U`, the Solomonoff universal semimeasure is commonly written

`M_U(x) = sum_{p: U(p) starts with x} 2^{-|p|}`.

The factor `2^{-|p|}` makes program length into prior probability. This is the formal Occam term: halving the code space for every extra bit of description length. The summation over all compatible programs is the complementary principle of multiple explanations: no computable hypothesis is thrown away just because it is not the simplest.

Prediction is then Bayesian conditioning on the observed prefix. For a candidate next symbol or continuation `y`, the predictive weight is proportional to `M_U(xy)` relative to `M_U(x)`. The posterior is not over a hand-built statistical family; it is over the computable processes that could have generated the observations.

## Baselines

- **Classical Bayesian modeling.** Choose a model class `H`, assign a prior `P(h)`, update by Bayes' rule.

- **Maximum likelihood or empirical risk minimization.** Fit the best member of a prespecified family. Complexity control can be added separately as a regularizer.

- **Minimum description length.** Prefer compact descriptions of data and model. It captures the compression perspective and is commonly used as a model-selection criterion.

- **Shortest consistent program.** Pick the minimum-length program that outputs the observed data. This gives a single best explanation rather than a mixture.

- **Expert advice over a finite or countable list.** Mix predictions from a named set of experts, assigning prior weights to each expert in the list.

## Evaluation settings

The clean setting is sequence prediction over finite strings, usually binary strings, with the assumption that the environment is computable or at least representable by a lower semicomputable semimeasure. The predictor sees a prefix `x` and assigns probabilities to future continuations.

Success is theoretical rather than computational. The universal semimeasure is not computable in general, because enumerating all programs that may eventually output a prefix runs into the halting problem. Its role is as an ideal limit or gold standard: a precise definition of what unbiased induction would do if it could range over every computable explanation.

Machine dependence is controlled but not eliminated. Different universal machines change prior weights by constant factors, so asymptotic predictions are robust up to the usual invariance behavior of algorithmic information theory. The reference machine still has to be fixed before the data, or the language choice can smuggle in a post hoc bias.
