I consider units that each receive one of two treatment conditions, an experimental condition `E` or a control condition `C`, and then record an outcome `Y`. The simplest comparison computes the average outcome among units that received `E`, subtracts the average outcome among units that received `C`, and presents the difference as a causal effect. I distrust that calculation immediately, because it compares different units. If the units assigned to `E` were already different from those assigned to `C`, the contrast mixes the treatment with the selection that produced the groups, and the arithmetic alone cannot separate the two. A before-after comparison on the same unit fails from a different direction: holding the unit fixed but moving in time allows maturation, trends, and intervening events to change the outcome, so an earlier measurement is not the untreated outcome at the later time. Perfect matching would pair units so closely that either member would respond identically under either treatment, making the pair contrast compelling, but exact equality is rarely defensible outside tightly controlled physical settings. Regression adjustment can be useful, yet only after it is attached to a prior causal argument; otherwise it will adjust for post-treatment variables, irrelevant variables, or variables selected after seeing the answer. Even randomized assignment, which protects against systematic assignment bias and yields exact reference distributions, leaves the unit-level alternative outcome unobserved. What these approaches share is a failure to state the causal contrast before data analysis begins, to identify which part of that contrast goes missing once treatment is assigned, and to make the assignment argument explicit. The estimand, the assignment story, and the estimator become blurred, and causal claims leak out of statistical machinery that was never asked to define them.

I therefore propose the Potential-Outcome Assignment Model, also called the Rubin causal model. It is a statistical scaffold rather than a single estimator. It defines causal effects as comparisons between treatment-specific outcomes attached to the same trial, treats the unobserved member of each pair as a genuine missing value, and forces the assignment mechanism to be stated as a separate, defensible object before any estimator is chosen. The first move is to fix a trial, meaning a unit, a treatment-initiation time, and an outcome-measurement time, and to attach to that single trial two treatment-specific outcomes rather than one. If the trial receives `E`, its outcome is `y_j(E)`; if the same trial had instead received `C`, its outcome is `y_j(C)`. The unit-level causal effect is `tau_j = y_j(E) - y_j(C)`, experimental minus control, so an `E` that raises the outcome relative to `C` gives a positive effect and an `E` that lowers it gives a negative effect. This definition immediately creates the central difficulty, and that difficulty is a feature rather than a flaw: a trial receives only one treatment at the initiation time, so only one of `y_j(E)` and `y_j(C)` is observed and the other is missing. Repeating the unit later does not recover the missing value, because a later exposure is a different trial subject to carryover, aging, learning, and changing conditions. The missing value is not a regression residual that disappears with better notation; it is the counterfactual half of the contrast. Writing `W_j = 1` for assignment to `E` and `W_j = 0` for assignment to `C`, the observed outcome is the switch `Y_j^obs = W_j y_j(E) + (1 - W_j) y_j(C)`, so `W_j = 1` reveals `y_j(E)` and hides `y_j(C)`, and `W_j = 0` does the reverse. Causal inference is thereby reframed as a missing-data problem on a table of potential outcomes.

Because the unit-level effect is never fully observed, the target must be an average defined before any estimator is picked. For `M` trials the finite-study average causal effect is `T_M = (1/M) sum_{j=1}^M [y_j(E) - y_j(C)]`, the average of within-trial contrasts rather than the contrast of whichever outcomes happen to be revealed. For a target population the analogue is `tau = E[y(E) - y(C)]`. These estimands are defined before modeling and before estimator choice. The framework then asks when an observable quantity recovers one of these estimands. The cleanest case is balanced randomization. Take `2N` trials with exactly `N` assigned to `E`, let `S_E` and `S_C` be the treatment and control index sets, and form the observed difference `y_d = (1/N) sum_{j in S_E} y_j(E) - (1/N) sum_{j in S_C} y_j(C)`. In any realized allocation this is not algebraically equal to `T_{2N}`, because it uses `y_j(E)` only for trials in `S_E` and `y_j(C)` only for trials in `S_C`, whereas `T_{2N}` wants both values for every trial. Randomization bridges the gap in expectation. Under a balanced random mechanism every balanced allocation is equally likely, so a fixed trial `j` lands in `S_E` in half the allocations, contributing `+y_j(E)/N`, and in `S_C` in the other half, contributing `-y_j(C)/N`. Averaging over the randomization set, trial `j` contributes `(1/2) y_j(E)/N - (1/2) y_j(C)/N = [y_j(E) - y_j(C)]/(2N)`, and summing over all `2N` trials gives `E_R[y_d] = (1/(2N)) sum_{j=1}^{2N} [y_j(E) - y_j(C)] = T_{2N}`. The normalization is easy to get wrong: each group mean carries a `1/N`, but each trial occupies each role in only half the balanced allocations, so the randomization expectation carries `1/(2N)` on each unit-level contrast, and that is exactly why the finite-study average is recovered in expectation over the design.

I am careful not to overstate what randomization buys. It makes the observed difference unbiased over the design and licenses exact randomization statements, but it does not make the realized `y_d` equal to `T_{2N}`, and it says nothing about whether the study trials represent future units. Generalization beyond the study needs a separate sampling or representativeness argument. Matching supplies a different bridge. If two trials are matched so tightly that both would respond essentially the same under `E` and essentially the same under `C`, the observed pair difference is close to the pair's causal effect even without randomization. Since exact matching is almost never certain, randomizing within matched pairs is valuable: it preserves the matched comparison while making any residual systematic bias random inside the pair, and it correctly shrinks the relevant randomization set rather than pretending every balanced allocation was possible.

The same lens exposes what breaks in a nonrandomized study. There the observed contrast is `E[Y^obs | W=1] - E[Y^obs | W=0] = E[y(E) | W=1] - E[y(C) | W=0]`, which need not equal the desired `E[y(E) - y(C)]`. The conditioning is the problem: the first term comes from units selected into `E` and the second from units selected into `C`, and if treatment status depends on the potential outcomes or on prior variables that drive them, the two revealed means do not reconstruct the two average potential-outcome means. A controlled observational study can still be analyzed as if assignment were random, but only by making that an explicit, substantive claim about the assignment mechanism. The investigator must argue that important prior variables have been controlled, by design, matching, or a prespecified adjustment, so that within the controlled comparison a unit was effectively just as likely to receive `E` as `C`. If an important prior variable differs systematically between groups, the causal interpretation requires a defensible adjustment model or should be withheld.

Writing only `y_j(E)` and `y_j(C)` also assumes a stable treatment description: `E` and `C` are well-defined conditions rather than bundles of hidden versions, and a unit's outcome under its treatment does not depend on which treatments other units receive. If there is interference across units or hidden treatment-version variation, the two-value table is too small, and the outcome must be indexed by the full assignment and version structure before the causal contrast is defined on that expanded table. With stability in hand, the discipline is to keep three jobs separate and in order: estimand, then assignment mechanism, then estimator. A difference in means, a blocked difference, a matched comparison, a regression coefficient, or a weight is causal only after the estimand has been defined and the assignment mechanism has justified which missing treatment-specific means are recoverable.

The code below implements a small finite-population simulation of the core randomization identity. It fixes potential outcomes for each trial, repeatedly draws balanced treatment allocations, computes the observed treated-minus-control difference, and checks that the average of those observed differences converges to the true finite-study average treatment effect.

```python
import itertools
import numpy as np


def simulate_rubin_causal_model(n_per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    m = 2 * n_per_group
    # Fix potential outcomes for every trial
    y_E = rng.normal(5.0, 1.0, size=m)
    y_C = rng.normal(4.0, 1.0, size=m)
    true_ate = float(np.mean(y_E - y_C))

    # Enumerate all balanced allocations if feasible, else Monte Carlo
    indices = np.arange(m)
    if m <= 12:
        allocations = list(itertools.combinations(indices, n_per_group))
    else:
        allocations = [
            tuple(rng.choice(indices, n_per_group, replace=False))
            for _ in range(5000)
        ]

    observed_diffs = []
    for treated in allocations:
        treated = set(treated)
        control = [j for j in indices if j not in treated]
        yd = float(np.mean(y_E[list(treated)]) - np.mean(y_C[control]))
        observed_diffs.append(yd)

    mean_observed = float(np.mean(observed_diffs))
    return true_ate, mean_observed, len(observed_diffs)


true_ate, mean_observed, n_allocations = simulate_rubin_causal_model()
print(f"True finite-study ATE: {true_ate:.6f}")
print(f"Mean observed diff over randomization set: {mean_observed:.6f}")
print(f"Number of allocations: {n_allocations}")
```

Running this script shows that the average of the observed treated-minus-control differences across the randomization set equals, up to Monte Carlo or enumeration error, the finite-population average of `y_j(E) - y_j(C)`. The exercise makes the normalization concrete: each trial contributes its contrast weighted by `1/(2N)` once the averaging over allocations is performed, which is why the simple difference in group means is unbiased under balanced randomization even though the realized difference is not identical to the estimand.
