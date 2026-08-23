Most calibration failures can be repaired without ever changing what the
classifier ranks above what. This variant imposes that as a hard design rule:
the mapping must be order-preserving — a strictly increasing transform of the
scores, applied so that it can never swap two predictions, never change an
argmax, and therefore never touch accuracy. Inside that constraint set live
temperature scaling, positive-slope Platt maps, monotone splines and their
compositions; outside it live histogram binning, unconstrained isotonic fits
that collapse score regions into ties, and per-class corrections that let
classes trade places after renormalisation.

The research question is how much of the measured miscalibration this
restricted family can remove. Far from being an aesthetic preference, the
restriction guards generalisation: mappings free to reorder can chase
bin-level noise in the calibration split, manufacturing ECE gains that
dissolve into Brier and NLL losses on the test split, whereas the monotone
family simply cannot express those pathologies. The interesting engineering sits at the family's flexible end:
strictly increasing maps with enough curvature to fix a forest's vote-ratio
distortions — far more warped than one temperature can express — while
remaining well-conditioned on a hundred-point calibration split.

Two comparisons to defend on the unchanged three metrics across the four
settings. Against the identity: the monotone fit removes most of each
setting's measurable calibration error. Against an unconstrained alternative
of your choice fitted on the same splits: the monotone family gives up little
or nothing on ECE and repays the restriction with equal or better NLL and
Brier. The scaffold fits the family's crudest member — one temperature on
log-probabilities by calibration-set likelihood — and is the floor to build
monotonically upward from.
