Treat the acceptance score itself as the deliverable. Downstream, that same
number will be consumed as a triage signal — cases sorted by it, review
effort spent from the bottom of the ordering upward — so the property being
purchased is ordinal: examples the base model gets wrong must sit below
examples it gets right, as uniformly as possible across the score range. The
reported AUROC of the score against correctness is the primary column of
this variant; the accept/defer decision is deliberately boring, a single cut
through that ordering at the budgeted coverage.

The pipeline is fixed end to end. What remains open is how the score is
constructed from calibration evidence: raw softmax confidence, margins,
entropies, learned combinations of these, or any compact model of "will the
classifier be right here" that the interface admits. The scientific content
is which construction ranks correctness best on held-out data — and why the
ranking transfers, given that calibration and test differ in composition.

Risk metrics are expected to improve as a corollary: a better ordering makes
any budget cut cleaner, lowering accepted error overall and usually for the
weakest subgroup too. The variant explicitly rejects the converse route —
tuning thresholds or per-group carve-outs to move the risk columns while the
underlying ordering stays mediocre. Coverage should land on target, and the
burden spread across subgroups is monitored without being the objective.

The claim to defend is an ordering claim: demonstrate the ranking-quality
gain over the plain confidence baseline, localise where in the score range
the ordering improved, and show that the selective-risk gains line up with —
rather than outrun — the movement in AUROC.
