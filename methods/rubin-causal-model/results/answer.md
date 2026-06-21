# Potential-Outcome Assignment Model

## Artifact Type

There is no executable code artifact to mirror. Faithfulness here means preserving the source-level statistical scaffold: potential outcomes, observed-outcome switching, finite-study estimands, and the assignment mechanism.

## Treatment-Specific Outcomes

For trial `j`, define two mutually exclusive treatment conditions, experimental `E` and control `C`, with a specified initiation time and outcome-measurement time.

Define the two treatment-specific outcomes:

`y_j(E)` = outcome for trial `j` if assigned `E`.

`y_j(C)` = outcome for trial `j` if assigned `C`.

The unit-level causal effect is

`tau_j = y_j(E) - y_j(C)`.

Only one member of the pair is observed. With `W_j=1` for `E` and `W_j=0` for `C`,

`Y_j^obs = W_j y_j(E) + (1-W_j) y_j(C)`.

So the two observed cases are:

`W_j=1 -> Y_j^obs = y_j(E)` and `y_j(C)` is missing.

`W_j=0 -> Y_j^obs = y_j(C)` and `y_j(E)` is missing.

## Causal Estimands

For `M` trials, the finite-study average causal effect is

`T_M = (1/M) sum_{j=1}^M [y_j(E) - y_j(C)]`.

For a superpopulation or target population, the analogous estimand is

`tau = E[y(E) - y(C)]`.

These estimands are defined before modeling and before estimator choice.

## Randomized Identification In A Balanced Finite Study

For `2N` trials with exactly `N` assigned to `E`, let `S_E` and `S_C` be the treatment and control index sets. The observed difference is

`y_d = (1/N) sum_{j in S_E} y_j(E) - (1/N) sum_{j in S_C} y_j(C)`.

Under balanced random assignment, every trial appears in `S_E` in half of the allocations and in `S_C` in half. Therefore

`E_R[y_d] = (1/(2N)) sum_{j=1}^{2N} [y_j(E) - y_j(C)] = T_{2N}`.

Randomization also defines the randomization set for significance statements: under a hypothesized complete set of trial-level effects, fill in the missing treatment-specific outcomes, compute `y_d` for every allocation allowed by the design, and compare the observed `y_d` with that design distribution.

## Nonrandomized Studies

Without random assignment,

`E[Y^obs | W=1] - E[Y^obs | W=0] = E[y(E) | W=1] - E[y(C) | W=0]`,

which need not equal

`E[y(E) - y(C)]`.

A nonrandomized comparison is causal only if the assignment mechanism can be defended. That means the investigator has controlled the important prior variables, usually through design, matching, or a prespecified adjustment, so that treatment assignment within the controlled comparison is plausibly as-if random. If an important prior variable systematically differs between treatment groups, the causal interpretation requires a defensible adjustment model or should be withheld.

## Stability Requirement

The two-outcome notation assumes a stable treatment description:

1. `E` and `C` are well-defined treatment conditions, not bundles of hidden versions.
2. A unit's outcome under one treatment does not depend on other units' assignments.

If interference or treatment-version variation matters, replace `y_j(E)` and `y_j(C)` with outcomes indexed by the full assignment and version structure, then define the causal contrast on that expanded table.

## Estimator Separation

The procedure is:

1. Define the unit, treatment conditions, timing, and outcome.
2. Define treatment-specific outcomes and the causal estimand.
3. State the assignment mechanism: randomized, randomized within blocks, or nonrandomized with a defended as-if-random control argument.
4. Use the assignment mechanism to justify which missing treatment-specific means are recoverable from observed outcomes.
5. Only then choose an estimator such as a difference in means, blocked difference, matched comparison, regression adjustment, or weighting rule.
