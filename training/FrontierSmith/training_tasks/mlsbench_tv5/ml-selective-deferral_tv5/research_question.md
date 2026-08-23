Three unrelated decision problems — census income, recidivism, and
law-school outcomes — receive the identical deferral policy, and the final
grade is effectively set by whichever run goes worst. Nothing may be tuned
per problem: no knob whose value differs by dataset, no fallback selected by
inspecting which dataset is in play. How much abstention quality survives
that discipline is precisely what this variant measures.

The design pressure points toward distribution-free machinery. Cutoffs
placed by rank rather than by value, scores that need no dataset-specific
scaling, and fitting procedures insensitive to calibration-set size and
class balance are all favoured, because each removes a way the policy could
silently specialise to one problem and crater on another. Sensitivity is
itself measurable on calibration data — resample the fit and watch how far
the decision boundary moves — and such stability evidence belongs inside the
method, not only in the write-up.

Every reported column matters simultaneously here: accepted-set error,
weakest-subgroup error, burden spread, ranking quality, and realised
coverage each take a value on every dataset, and this variant cares about
the worst of each across the three problems more than about any
single-dataset triumph. An approach that wins two datasets by a wide margin
while degrading the third has answered the wrong question.

Defend the robustness directly: report the cross-dataset spread of each
metric, identify which dataset binds and why, and trace the stability of the
fitted rule to a specific design choice that a tuned-per-dataset alternative
would lack.
