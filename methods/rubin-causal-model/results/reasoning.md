I start with the ordinary comparison and immediately distrust it. I can compute the average outcome among units that receive `E` and subtract the average outcome among units that receive `C`, but that calculation is not yet a causal effect. It compares different units. If the units assigned to `E` are already different from the units assigned to `C`, the contrast carries both treatment and selection.

So I first need the object that the observed comparison is trying to estimate. I fix a trial: a unit, a treatment-initiation time, and an outcome-measurement time. For that same trial there are two treatment-specific outcome values. If the trial receives `E`, the outcome value is `y_j(E)`. If the same trial receives `C` instead, the outcome value is `y_j(C)`. The causal effect for trial `j` is therefore

`y_j(E) - y_j(C)`.

The sign is deliberate: it is experimental treatment minus control treatment. If `E` raises the outcome relative to `C`, the effect is positive; if it lowers the outcome, the effect is negative.

This definition immediately creates the wall. I cannot observe both `y_j(E)` and `y_j(C)` for the same trial, because the trial receives only one treatment at the initiation time. Repeating the unit later is not the same trial: carryover, time trends, aging, learning, or changing conditions can alter the response. The unobserved treatment-specific value is not a nuisance term that disappears with notation. It is the missing counterfactual member of the pair.

For a collection of `M` trials, the clean average target is

`T_M = (1/M) sum_{j=1}^M [y_j(E) - y_j(C)]`.

That is the average of within-trial contrasts, not the contrast of whichever outcomes happen to be revealed. In the simple balanced study with `2N` trials, let `S_E` be the `N` trials assigned to `E` and `S_C` the `N` trials assigned to `C`. The observed treated-minus-control difference is

`y_d = (1/N) sum_{j in S_E} y_j(E) - (1/N) sum_{j in S_C} y_j(C)`.

This is computable, but it is not algebraically equal to `T_{2N}` in a realized allocation. It uses `y_j(E)` only for trials in `S_E` and `y_j(C)` only for trials in `S_C`; `T_{2N}` uses both treatment-specific values for every trial.

Now randomization gives the first real bridge. Suppose the design assigns exactly `N` of the `2N` trials to `E` by a balanced random mechanism, so every balanced allocation in the randomization set is equally likely. For a fixed trial `j`, half the allocations put it in `S_E`, where its contribution to `y_d` is `+y_j(E)/N`, and half put it in `S_C`, where its contribution is `-y_j(C)/N`. Averaging over the randomization set, trial `j` contributes

`(1/2) y_j(E)/N - (1/2) y_j(C)/N = [y_j(E) - y_j(C)]/(2N)`.

Summing over all `2N` trials gives

`E_R[y_d] = (1/(2N)) sum_{j=1}^{2N} [y_j(E) - y_j(C)] = T_{2N}`.

The constant is easy to get wrong. The observed difference has `1/N` in each group mean, but each trial appears in each role in only half the balanced allocations, so the randomization expectation has `1/(2N)` times each unit-level contrast. That is why the finite-study average is recovered exactly in expectation over the randomization set.

Randomization also gives a reference distribution. If I hypothesize a complete set of trial-level effects, then the missing treatment-specific value for each trial is filled in under that hypothesis. I can compute the treated-minus-control difference for every allocation in the randomization set, locate the observed `y_d` in that distribution, and ask how extreme it is. Under the sharp no-effect hypothesis, all missing values are especially easy to fill in because `y_j(E)=y_j(C)` for every trial. More general constant-effect hypotheses can also be checked by shifting the missing values by the hypothesized effect.

But I should not overstate randomization. It makes the observed difference unbiased over the design and gives exact randomization statements. It does not make the realized `y_d` equal to `T_{2N}`, and it does not prove that the study units represent future units. Generalizing from the study to a target population needs another argument, such as random sampling or a defensible claim that the study trials are representative for the causal effects of interest.

Matching gives a different kind of bridge. If two trials are matched so closely that both would have essentially the same outcome under `E` and essentially the same outcome under `C`, then the observed pair difference is close to the pair's causal effect even without randomization. Exact matching would make randomization unnecessary for estimation of that pair contrast. In practice exact matching is almost never certain. Randomization within matched pairs is therefore useful because it preserves the matched comparison while making any remaining systematic source of bias random within the pair.

Now I can see what goes wrong in a nonrandomized study. The observed difference is still

`(1/N) sum_{j in S_E} y_j(E) - (1/N) sum_{j in S_C} y_j(C)`,

but the sets `S_E` and `S_C` may have been formed by prior variables that also affect `Y`. In expectation notation, the raw observed contrast has the form

`E[Y^obs | W=1] - E[Y^obs | W=0] = E[Y(E) | W=1] - E[Y(C) | W=0]`.

The average causal contrast I want is

`E[Y(E) - Y(C)]`.

The conditioning is the problem. The first term comes from units selected into `E`, and the second comes from units selected into `C`. If treatment status depends on the unobserved treatment-specific outcomes, or on prior variables that determine those outcomes, the two revealed means do not reconstruct the two average potential-outcome means.

A carefully controlled observational study can still be useful, but only by making the missing assignment argument explicit. If the investigator has controlled the important prior variables, perhaps through matching or a prespecified adjustment, then it may be credible that within the controlled comparison a unit was effectively just as likely to receive `E` as `C`. That is a substantive claim about the assignment mechanism, not a fact read off the outcome regression. If an important prior variable is found to differ systematically between groups, I either adjust using a model I can defend before seeing the desired answer, or I stop trusting the causal interpretation.

The compact notation also needs a stability condition. Writing only `y_j(E)` and `y_j(C)` assumes that each treatment condition is well defined and that unit `j`'s outcome under its treatment does not depend on which treatments other units receive. If there are hidden versions of `E` or `C`, or interference across units, then the two-value notation is too small. I have to index the outcome by the whole assignment, and possibly by treatment versions. The simple two-outcome table is valid only after the treatment conditions and interference structure have been made stable enough for the comparison at hand.

The full structure is now clear. Define causal effects as comparisons between treatment-specific outcomes for the same trial. Accept that one member of each pair is missing. Define the finite-study average before choosing an estimator. Use randomization to make the observed difference unbiased over the randomization set and to get exact reference distributions. Use matching and prior-variable control to make nonrandomized comparisons credible only when the assignment story can be defended. Keep the estimator last, because a regression coefficient or matched difference is causal only after the estimand and assignment mechanism have already done their work.
