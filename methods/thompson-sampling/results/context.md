## The Acting Stream

A sequence of individuals arrives one at a time, and each must be assigned immediately to one of two treatments. Each assignment produces only a binary observation: the critical event occurs or it does not. The experiment cannot be separated into a clean learning phase followed by a clean acting phase, because the very act that produces data also exposes an individual to a possibly inferior treatment.

The scientific objective is therefore not just to estimate two unknown chances accurately. It is to reduce the number of future individuals lost to the worse treatment while the evidence is still thin. The hard case is small samples, slow accumulation of observations, or valuable individuals, where waiting for statistical comfort is itself costly.

## Evidence On Two Unknown Chances

For treatment `i`, let the unknown event probability be `p_i`. The available evidence consists only of counts: `r_i` occurrences and `s_i` failures, with `n_i = r_i + s_i`. The two treatments are sampled independently, and the question of interest is comparative: which unknown probability is larger?

A natural state of ignorance treats each `p_i` as equally likely to fall in any two equal subintervals of `(0, 1)`. With Bernoulli sampling, this gives the familiar inverse-probability update: after `r` occurrences and `s` failures, the density over the unknown probability is proportional to `p^r (1-p)^s`, normalized by the beta integral.

## Available Probability Machinery

Bayes's theorem already supplies a way to turn observations into a distribution over an unknown Bernoulli probability. The older inverse-probability tradition, summarized by Todhunter, makes clear that this is not a new kind of posterior calculation.

Pearson's incomplete B-function work supplies another needed ingredient: the tail area of a beta-shaped density can be expressed as a finite sum of binomial terms. Muller provides related notation and computational methods for incomplete beta integrals. The mathematical apparatus can therefore evaluate probabilities about one unknown chance exceeding a fixed threshold, and it can do so in a form suitable for tables.

## Unsatisfactory Disciplines

One crude discipline is fixed alternation: assign half the individuals to each treatment until enough evidence accumulates. It protects against total commitment, but it keeps spending half of the stream on the inferior treatment even after the evidence has shifted strongly.

The opposite discipline is immediate final choice: estimate which treatment is better and give all future individuals that treatment. It exploits the current evidence, but it converts a small-sample comparison into a permanent commitment. If the current apparent winner is wrong, every later individual pays the gap.

Large-sample approximations and yes-or-no significance tests also miss the operational target. They can help describe evidence, but they do not by themselves say how the next individual should be allocated before large samples exist.

## The Empty Slot

What is missing is a decision principle that uses the graded uncertainty itself. The principle has to hedge when the evidence is genuinely balanced, lean harder when the evidence is persuasive, and let the leaning change automatically as observations arrive.

It also has to be computable from small counts. A useful rule cannot depend on asymptotic comfort or on arbitrary stopping thresholds, because the next assignment is required now and the cost of a wrong assignment is part of the experiment.

## Code framework

```python
import numpy as np

class ThompsonSampler:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select_arm(self):
        samples = np.random.beta(
            self.successes, self.failures
        )
        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
```
