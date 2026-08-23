This variant imposes a structural restriction and studies what it costs:
the deployed mapping must be one shared transform applied identically to
every example. No parameter, lookup, or branch in the prediction path may
key on subgroup membership; whatever group awareness exists must be spent
at fit time and compiled into that single rule. Group ids remain available
for analysis and for the write-up, but the transform itself is group-blind.

The restriction is not arbitrary. Per-group corrections carry real
liabilities — they need the group id at prediction time, they invite
overfitting on thin groups, and several deployment regimes bar them
outright. The open question is how much of the subgroup-reliability
objective survives without them: whether a well-chosen pooled transform,
fitted on all calibration pairs at once, can hold the worst subgroup's
calibration error and the between-group spread close to what group-aware
machinery achieves on the shifted tail.

Within the constraint the design space stays rich: the family of the
global map (scaling, affine-in-logit, monotone nonparametric), the loss it
is fitted under — including fitting choices that stop the pooled fit from
being captured wholesale by the largest group — and safeguards for score
regions into which the shift moves mass. Only the per-group escape hatch
is forbidden. Sharpness, read off the pooled Brier column, must survive
whichever pooled correction is chosen.

The defence is a constraint-cost accounting: state where the group-blind
map matches group-aware alternatives on the reported columns and where it
genuinely cannot, and explain which structural property of the chosen
global family — not which per-group parameter — held the weakest group's
reliability.
