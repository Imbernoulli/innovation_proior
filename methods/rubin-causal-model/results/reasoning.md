I start with the ordinary comparison and immediately distrust it. I can compute the average outcome among units that receive `E` and subtract the average outcome among units that receive `C`, but that calculation is not yet a causal effect. It compares different units. If the units assigned to `E` are already different from the units assigned to `C`, the contrast carries both treatment and selection.

So I first need the object that the observed comparison is trying to estimate. I fix a trial: a unit, a treatment-initiation time, and an outcome-measurement time. For that same trial there are two treatment-specific outcome values. If the trial receives `E`, the outcome value is `y_j(E)`. If the same trial receives `C` instead, the outcome value is `y_j(C)`. The causal effect for trial `j` is therefore

`y_j(E) - y_j(C)`.

The sign is deliberate: it is experimental treatment minus control treatment. If `E` raises the outcome relative to `C`, the effect is positive; if it lowers the outcome, the effect is negative.

As soon as I write this down I notice that I can never actually evaluate it. To compute `y_j(E) - y_j(C)` I would need both treatment-specific values for the same trial, but the trial receives only one treatment at the initiation time, so one of the two is never realized. My first instinct is to recover the missing value by repeating the unit under the other treatment later. That does not give me the same trial, though: carryover, time trends, aging, learning, or changing conditions can alter the response, so the second measurement is `y_j'(C)` for a different trial `j'`, not the `y_j(C)` I needed. The unobserved treatment-specific value is not a nuisance term that disappears with notation. It is a genuinely missing counterfactual, and any usable causal statement will have to come from filling it in under some assumption rather than from observing it.

For a collection of `M` trials, the clean average target is

`T_M = (1/M) sum_{j=1}^M [y_j(E) - y_j(C)]`.

That is the average of within-trial contrasts, not the contrast of whichever outcomes happen to be revealed. In the simple balanced study with `2N` trials, let `S_E` be the `N` trials assigned to `E` and `S_C` the `N` trials assigned to `C`. The observed treated-minus-control difference is

`y_d = (1/N) sum_{j in S_E} y_j(E) - (1/N) sum_{j in S_C} y_j(C)`.

This is computable, but it is not algebraically equal to `T_{2N}` in a realized allocation. It uses `y_j(E)` only for trials in `S_E` and `y_j(C)` only for trials in `S_C`; `T_{2N}` uses both treatment-specific values for every trial.

Now randomization gives the first real bridge. Suppose the design assigns exactly `N` of the `2N` trials to `E` by a balanced random mechanism, so every balanced allocation in the randomization set is equally likely. For a fixed trial `j`, half the allocations put it in `S_E`, where its contribution to `y_d` is `+y_j(E)/N`, and half put it in `S_C`, where its contribution is `-y_j(C)/N`. Averaging over the randomization set, trial `j` contributes

`(1/2) y_j(E)/N - (1/2) y_j(C)/N = [y_j(E) - y_j(C)]/(2N)`.

Summing over all `2N` trials gives

`E_R[y_d] = (1/(2N)) sum_{j=1}^{2N} [y_j(E) - y_j(C)] = T_{2N}`.

The constant is the part I least trust, so I want to check it on a case small enough to enumerate by hand. Take `2N=4` trials with the potential outcomes `y(E) = (10, 7, 4, 9)` and `y(C) = (6, 5, 1, 2)`. The unit-level contrasts are `(4, 2, 3, 7)`, so `T_4 = (4+2+3+7)/4 = 4`. With `N=2`, the balanced allocations are the `C(4,2)=6` ways to choose the two `E` trials. Writing each as `(S_E -> S_C)` and computing `y_d = (mean of y(E) over S_E) - (mean of y(C) over S_C)`:

`{1,2}->{3,4}`: `(10+7)/2 - (1+2)/2 = 8.5 - 1.5 = 7`.
`{1,3}->{2,4}`: `(10+4)/2 - (5+2)/2 = 7 - 3.5 = 3.5`.
`{1,4}->{2,3}`: `(10+9)/2 - (5+1)/2 = 9.5 - 3 = 6.5`.
`{2,3}->{1,4}`: `(7+4)/2 - (6+2)/2 = 5.5 - 4 = 1.5`.
`{2,4}->{1,3}`: `(7+9)/2 - (6+1)/2 = 8 - 3.5 = 4.5`.
`{3,4}->{1,2}`: `(4+9)/2 - (6+5)/2 = 6.5 - 5.5 = 1`.

The average of `7, 3.5, 6.5, 1.5, 4.5, 1` is `24/6 = 4`, which is exactly `T_4`. So the identity holds here, and it pins down the constant: if I had instead written the expectation as `(1/N) sum [y_j(E)-y_j(C)] = 16/2 = 8`, I would have been off by a factor of two. The reason is now concrete. Each trial appears in `S_E` in exactly three of the six allocations and in `S_C` in the other three, so its averaged contribution is `(1/2) y_j(E)/N - (1/2) y_j(C)/N = [y_j(E)-y_j(C)]/(2N)`. For trial 1 that is `(10-6)/4 = 1`; summing the four trial contributions `1 + 0.5 + 0.75 + 1.75 = 4` reproduces the answer a third way. The `1/(2N)` is right because each trial spends half its allocations in each role, and the `2N` in the denominator is what recovers the finite-study average in expectation.

Randomization also gives a reference distribution. If I hypothesize a complete set of trial-level effects, then the missing treatment-specific value for each trial is filled in under that hypothesis. I can compute the treated-minus-control difference for every allocation in the randomization set, locate the observed `y_d` in that distribution, and ask how extreme it is. Under the sharp no-effect hypothesis the filling-in is trivial: `y_j(E)=y_j(C)` for every trial, so the missing value is just a copy of the observed one and the whole `y(E)`/`y(C)` table is reconstructed from the data alone. I should be careful, though, not to expect this to make the reference distribution collapse to a point. With the sharp null applied to, say, `y = (6, 5, 1, 2)`, the six allocations give `y_d` values `4, 0, 1, -1, 0, -4` — a real spread, because `y_d` still averages different trials' outcomes in the two arms even when each trial's own pair is equal. The null fixes the table, not the allocation, so the reference distribution is exactly this set of recomputed `y_d` values and the test asks where the observed one sits in it. More general constant-effect hypotheses are handled the same way, shifting each missing value by the hypothesized effect before recomputing.

But I should not overstate randomization. It makes the observed difference unbiased over the design and gives exact randomization statements. It does not make the realized `y_d` equal to `T_{2N}`, and it does not prove that the study units represent future units. Generalizing from the study to a target population needs another argument, such as random sampling or a defensible claim that the study trials are representative for the causal effects of interest.

Matching gives a different kind of bridge. If two trials are matched so closely that both would have essentially the same outcome under `E` and essentially the same outcome under `C`, then the observed pair difference is close to the pair's causal effect even without randomization. Exact matching would make randomization unnecessary for estimation of that pair contrast. In practice exact matching is almost never certain. Randomization within matched pairs is therefore useful because it preserves the matched comparison while making any remaining systematic source of bias random within the pair.

Now I can see what goes wrong in a nonrandomized study. The observed difference is still

`(1/N) sum_{j in S_E} y_j(E) - (1/N) sum_{j in S_C} y_j(C)`,

but the sets `S_E` and `S_C` may have been formed by prior variables that also affect `Y`. In expectation notation, the raw observed contrast has the form

`E[Y^obs | W=1] - E[Y^obs | W=0] = E[Y(E) | W=1] - E[Y(C) | W=0]`.

The average causal contrast I want is

`E[Y(E) - Y(C)]`.

The conditioning is the problem. The first term comes from units selected into `E`, and the second comes from units selected into `C`. If treatment status depends on the unobserved treatment-specific outcomes, or on prior variables that determine those outcomes, the two revealed means do not reconstruct the two average potential-outcome means.

It helps to see how badly this can go even when the treatment effect is a clean constant. Suppose every unit has a baseline `b` and outcomes `y(C)=b`, `y(E)=b+2`, so the true effect is `+2` for everyone and the average causal effect `E[y(E)-y(C)]` is exactly `2`. Now let assignment track the baseline: the two high-baseline units `b=10, 9` select into `E` and the two low-baseline units `b=2, 1` end up in `C`. The naive contrast is `(12+11)/2 - (2+1)/2 = 11.5 - 1.5 = 10`. The truth is `2`, so the observed difference overstates it by `8`. None of that gap is treatment; it is the baseline difference `(10+9)/2 - (2+1)/2 = 8` leaking through the conditioning, exactly the term that `E[y(E)|W=1]-E[y(C)|W=0]` carries and `E[y(E)-y(C)]` does not. The structure here is the same as the randomized case — same potential outcomes, same observed-difference formula — but the allocation is no longer balanced across the randomization set, and that is the entire difference between a number I can interpret causally and one I cannot.

A carefully controlled observational study can still be useful, but only by making the missing assignment argument explicit. If the investigator has controlled the important prior variables, perhaps through matching or a prespecified adjustment, then it may be credible that within the controlled comparison a unit was effectively just as likely to receive `E` as `C`. That is a substantive claim about the assignment mechanism, not a fact read off the outcome regression. If an important prior variable is found to differ systematically between groups, I either adjust using a model I can defend before seeing the desired answer, or I stop trusting the causal interpretation.

The compact notation also needs a stability condition. Writing only `y_j(E)` and `y_j(C)` assumes that each treatment condition is well defined and that unit `j`'s outcome under its treatment does not depend on which treatments other units receive. If there are hidden versions of `E` or `C`, or interference across units, then the two-value notation is too small. I have to index the outcome by the whole assignment, and possibly by treatment versions. The simple two-outcome table is valid only after the treatment conditions and interference structure have been made stable enough for the comparison at hand.

Stepping back, every piece above has pushed in the same direction, so I can read off the order the analysis has to follow. The causal quantity had to be defined as a comparison between treatment-specific outcomes for the same trial, because that is the only contrast that holds the unit fixed; and once defined that way, one member of each pair is unavoidably missing. The finite-study average `T_M` had to be written down before any estimator, since the small example showed the observed difference equals it only in expectation and only under the right allocation weighting. Randomization earned its place by the computation: it makes the observed difference unbiased over the randomization set and supplies an exact reference distribution, but it does not make the realized `y_d` equal `T_{2N}`. Matching and prior-variable control are what stand in for randomization when it is absent, and the confounded example showed they are load-bearing — without a defensible assignment story the observed difference can be off by more than the effect itself. That forces the estimator to come last: a regression coefficient or matched difference is causal only after the estimand and the assignment mechanism have already done their work, never on its own.
