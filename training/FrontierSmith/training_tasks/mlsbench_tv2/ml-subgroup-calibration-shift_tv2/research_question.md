A single number governs this variant: the calibration error of whichever
subgroup fares worst once the shifted test tail is scored. Not the average,
not the spread — the maximum. Everything upstream — model, shifted splits,
metric battery — stays frozen, so the probability mapping is the only
instrument available, and its success is judged minimax.

Working minimax changes the research posture. The first task is diagnostic:
on the calibration split, locate which group's reliability is worst and
decompose why — a prevalence mismatch, a score distribution the base model
places differently for that group, or plain sample poverty. The second task
is surgical: spend modelling capacity where the maximum sits, rather than
smoothing every group a little. A repair that mildly improves six groups
while the worst one stays put has, by this variant's accounting, achieved
nothing at all.

The remaining columns act as guardrails. Overall Brier must not be
sacrificed wholesale to polish one group; the between-group gap should
shrink as a by-product of lifting the floor; discrimination inside each
group merely needs to stay intact. These are constraints, not objectives — the variant
deliberately refuses the balanced-scorecard framing.

One stability wrinkle deserves thought: the group that is worst on
calibration data need not remain worst on the shifted tail. A method that
hard-wires its repair to the calibration-time arg-max should argue why that
identification transfers, or hedge against getting it wrong.

Defend the outcome by naming the worst group before and after, quantifying
the lift of the reliability floor, and itemising what — if anything — the
guardrail columns paid for it.
