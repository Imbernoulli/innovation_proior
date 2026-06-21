# Solomonoff Induction

Solomonoff induction is an ideal Bayesian theory of sequence prediction over all computable explanations. Fix a universal prefix Turing machine `U` before seeing the data. For a finite observed string `x`, define the universal semimeasure

`M_U(x) = sum_{p: U(p) outputs a string starting with x} 2^{-|p|}`.

Each program `p` is a hypothesis about how the data are generated. The weight `2^{-|p|}` gives exponentially more prior mass to shorter programs, so program length becomes a formal Occam penalty. The summation over all programs means the method does not keep only the simplest explanation; it averages every computable explanation that is compatible with the observations.

Prediction is Bayesian conditioning on this universal prior. After observing prefix `x`, the probability of a continuation `y` is represented by the relative mass of programs that output `xy`:

`P(y | x) ~= M_U(xy) / M_U(x)`.

The exact normalization details depend on the semimeasure formulation, but the conceptual structure is stable: posterior prediction is obtained by conditioning a length-weighted mixture of computable hypotheses.

The unique insight is the move from hand-selected model classes to all computable explanations. Standard Bayesian inference requires the user to choose a hypothesis class first: linear models, Markov models, grammars, neural nets, expert lists, and so on. Anything outside that class receives zero prior probability. Solomonoff induction replaces that modeling choice with a universal computational class. If an explanation can be implemented as a program, it is already in the prior.

This unifies three ideas:

1. **Occam's razor:** shorter explanations get larger prior mass through `2^{-|p|}`.
2. **Bayesian prediction:** future observations are predicted by posterior averaging over hypotheses.
3. **Computability theory:** hypotheses are programs for a universal machine, making the hypothesis space the class of computable explanations.

The result is not a practical exact algorithm. `M_U` is incomputable in general because computing it would require solving halting-style questions about arbitrary programs. Its value is theoretical: it gives a precise gold-standard account of universal induction, showing what it would mean to predict without manually restricting the model class except by computability and a fixed reference machine.

In short, Solomonoff induction says: do not choose one narrow family of explanations; place a universal prior over every computable explanation, weight each by description length, and use Bayes' rule to predict.
