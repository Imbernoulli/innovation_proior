## Confounded Treatment

We want the causal effect of a treatment, exposure, price, schooling level, or other regressor `D` on an outcome `Y`. The central difficulty is that `D` is not assigned as if by a clean experiment. People select it, institutions target it, markets determine it jointly with outcomes, or measurement error distorts it.

In potential-outcome terms, the observed comparison between units with different `D` values mixes causal response with baseline differences. In regression terms, the structural disturbance still contains omitted ability, demand, severity, preference, or institutional information, so `D` moves with the disturbance rather than independently of it.

## Simultaneous Movement

Many target settings are simultaneous systems rather than one-way interventions. Price and quantity emerge together from demand and supply. Schooling and earnings reflect choices, laws, family background, and labor-market returns. Medical treatment and health outcomes reflect disease severity as well as care.

Because multiple forces move the regressor and the outcome at the same time, an observed slope can point in the wrong direction or have the wrong magnitude. A usable design must say which movement in `D` is informative about the target equation and which movement is contaminated by selection or simultaneity.

## Partial Assignment

Some empirical settings contain assignment, eligibility, timing, or encouragement variation that changes access or incentives without forcing every unit to comply. A draft number, school-entry rule, policy cutoff, physician tendency, or randomized encouragement can change probabilities rather than deterministically setting treatment.

This creates a second problem beyond confounding. The effect of being assigned, eligible, or encouraged is not automatically the effect of receiving the treatment. Some units would receive treatment regardless, some would never receive it, and some change behavior only because the assignment state changes.

## Existing Baselines

Naive differences in means are transparent, but they absorb all selection into the estimated effect. Ordinary least squares with controls removes only observed and correctly modeled confounding. Matching and propensity-score adjustment share the same selection-on-observables requirement.

Randomized assignment or intention-to-treat comparisons can be credible for the assignment itself, but they do not by themselves recover the effect of the treatment when compliance is incomplete. Fully structural simultaneous-equation models can target deeper parameters, but their credibility depends on functional form and exclusion restrictions that must be defended from substantive knowledge.

## Design Requirements

A stronger design needs a way to separate usable variation in `D` from contaminated variation in `D`. It must make a concrete argument that the usable variation is strong enough to matter, detached from unobserved determinants of `Y`, and connected to `Y` only through `D`.

The design also has to state the population whose effect is being learned. When treatment effects differ and only some units respond to an assignment or incentive, the answer may be local to those responsive units. Any final estimator therefore needs both an algebraic recipe and a design audit: strength of the induced movement, plausibility of no direct path to the outcome, and clarity about who is moved.

## Code framework

```python
import numpy as np


def two_stage_least_squares(X, Y, Z):
    """
    Compute the 2SLS coefficient for endogenous regressors X
    using instruments Z and outcome Y.
    X, Y, Z are assumed to be NumPy arrays.
    """
    raise NotImplementedError(
        "Project X onto the instrument space Z, then regress Y on the fitted X."
    )


def wald_iv(Y, D, Z):
    """Return the Wald ratio Cov(Z, Y) / Cov(Z, D) for a scalar instrument."""
    raise NotImplementedError("Compute reduced-form over first-stage covariance ratio.")
```
