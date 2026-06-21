## Research question

A study observes units that receive one of two treatment conditions, call them `E` and `C`, and then records an outcome `Y`. The scientific question is not just whether the observed `E` group has a different average outcome from the observed `C` group. The question is what it would mean for `E` rather than `C` to cause a different outcome for the same kind of unit at a specified time.

The hard part is that the study design exposes each unit to one treatment condition, not both. A useful causal framework must therefore say what the causal contrast is before the data analysis begins, what part of that contrast is missing after treatment assignment, and what kind of assignment or control argument makes an observed comparison informative.

## Background

Randomized experiments already separate two issues that are often blurred. First, there is a target contrast about treatments. Second, there is an assignment process that decides which treatment each unit receives. The assignment process can make a treated-control comparison unbiased or diagnostically useful, but it does not by itself define the causal quantity.

Agricultural experimentation supplied an earlier template. A field can be imagined as a collection of plots, each with a possible yield under each variety. Only one variety is actually planted on a plot, so only one yield is observed, but the comparison concerns a table of possible plot-variety yields. That template is useful because it treats the unobserved entries as part of the design problem rather than as regression residuals.

Observational studies create the same conceptual need with less protection. Treatment labels may reflect prior variables, choices, institutional rules, or investigator selection. If those prior differences also affect the outcome, a treated-control contrast mixes treatment with selection. Matching, blocking, or adjustment can help only after the target contrast and the assignment story are explicit.

## Baselines

- **Raw observed difference.** Compare the mean observed outcome among units assigned `E` with the mean observed outcome among units assigned `C`. This is easy to compute, but by itself it does not say whether treatment or preexisting differences produced the contrast.

- **Before-after comparison.** Compare the same unit before and after treatment. This keeps the unit fixed, but time, maturation, and other events can change the outcome, so the earlier measurement is not automatically the untreated outcome at the later time.

- **Perfect matching.** Pair units so closely that either member would respond the same way under either treatment. This would make the pair contrast compelling, but exact equality is rarely defensible outside tightly controlled physical settings.

- **Regression adjustment.** Fit a model using treatment labels and prior variables. This can be a useful estimator, but the model has to be attached to a prior causal argument; otherwise it can adjust for irrelevant variables, post-treatment variables, or variables selected after seeing the result.

- **Randomized assignment.** Assign treatments by a known random device. This protects against systematic assignment bias over the randomization set and gives exact reference distributions, but it still leaves the unit-level alternative treatment outcome unobserved.

## Evaluation Settings

A satisfactory framework must handle a fixed finite study with `2N` trials, `N` assigned to `E` and `N` assigned to `C`, where the immediate target is the average treatment effect for those trials. It must get the sign and normalization of the treated-minus-control contrast right, and it must explain why balanced randomization makes the observed average difference unbiased over the randomization set.

It must also handle blocked or matched designs. If assignment is randomized within matched pairs, the relevant randomization set is smaller than in a completely randomized design, and the analysis should respect that design instead of pretending every balanced allocation was possible.

For nonrandomized studies, it must state the extra assumption rather than hide it. A carefully controlled observational study can be analyzed as if treatment assignment were random only when the investigator can defend that no important prior variable systematically distinguishes the treatment groups, or can specify a credible adjustment for such variables.

Finally, it must distinguish the finite-study estimand from generalization to other units or times. Random assignment protects the comparison inside the study; generalizing beyond the study requires a separate sampling or representativeness argument.

## Code Framework

The artifact is a statistical scaffold, not a software implementation. The available inputs are unit identities, treatment labels, observed outcomes, possible pre-treatment variables, and a description of how treatment labels were assigned or selected.

The framework must provide:

1. A treatment definition with mutually exclusive conditions and a specified initiation time.
2. A way to name the two treatment-specific outcome values attached to each trial.
3. A rule saying which one of those values becomes observed.
4. A finite-study and, when needed, population-level causal estimand.
5. An assignment-mechanism statement: randomized, randomized within blocks, or nonrandomized but defended by matching and prior-variable control.
6. A rule that keeps estimand, assignment assumption, and estimator separate.
