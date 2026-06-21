## Confounded Treatment

We want the causal effect of a treatment, exposure, price, schooling level, or other regressor `D` on an outcome `Y`. The regressor `D` is not assigned as if by a clean experiment. People select it, institutions target it, markets determine it jointly with outcomes, or measurement error distorts it.

In potential-outcome terms, the observed comparison between units with different `D` values mixes causal response with baseline differences. In regression terms, the structural disturbance still contains omitted ability, demand, severity, preference, or institutional information, so `D` moves with the disturbance.

## Simultaneous Movement

Many target settings are simultaneous systems rather than one-way interventions. Price and quantity emerge together from demand and supply. Schooling and earnings reflect choices, laws, family background, and labor-market returns. Medical treatment and health outcomes reflect disease severity as well as care.

Multiple forces move the regressor and the outcome at the same time. The target is to say which movement in `D` is informative about the equation of interest and which movement reflects selection or simultaneity.

## Partial Assignment

Some empirical settings contain assignment, eligibility, timing, or encouragement variation that changes access or incentives without forcing every unit to comply. A draft number, school-entry rule, policy cutoff, physician tendency, or randomized encouragement can change probabilities rather than deterministically setting treatment.

The effect of being assigned, eligible, or encouraged is distinct from the effect of receiving the treatment. Some units would receive treatment regardless, some would never receive it, and some change behavior only because the assignment state changes.

## Existing Baselines

Naive differences in means are transparent and report the raw contrast between units at different `D` values. Ordinary least squares with controls adjusts for observed and modeled confounders. Matching and propensity-score adjustment estimate the effect under selection on observables.

Randomized assignment or intention-to-treat comparisons estimate the effect of the assignment itself. Fully structural simultaneous-equation models target deeper parameters using functional form and exclusion restrictions defended from substantive knowledge.

## Research Question

Given a regressor `D` that moves together with the disturbance in the outcome equation `Y = alpha + tau D + U`, the question is how to estimate the causal slope `tau` from observational data, and how to characterize which population the estimate describes when treatment effects differ across units and only some units respond to an available source of assignment or incentive variation.

## Code framework

```python
import numpy as np


def estimate_effect(X, Y, ...):
    """
    Estimate the causal coefficient of endogenous regressors X on outcome Y.
    X, Y are assumed to be NumPy arrays.
    """
    raise NotImplementedError


def scalar_ratio(Y, D, ...):
    """Return a scalar causal-slope estimate for one treatment."""
    raise NotImplementedError
```
