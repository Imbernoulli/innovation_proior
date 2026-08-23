The cheap way to make every subgroup look calibrated is to hedge: pull
probabilities toward a base rate until confidence means little, and the
calibration-error columns improve while the Brier score records exactly what
was surrendered. The expensive failure is the opposite: a mapping fit sharply
to the region where calibration data lives, which the shifted test tail then
falsifies — worst of all for the smallest subgroup. This variant asks for a
calibrator that refuses both outs: reliability that is not purchased with
sharpness, sustained on data the calibration set does not represent.

The evaluation is deliberately hostile to per-group curve fitting. Test
examples are drawn from the tail of a domain score, so the score distribution
your mapping sees when fitting differs from the one on which it is judged;
subgroup sizes are badly uneven; and the reported quantities — worst-subgroup
calibration error, the widest between-subgroup error gap, and overall Brier —
are combined per dataset and then multiplied across the three datasets, so the
weakest dataset bounds the outcome. A method that overfits one small group's
calibration slice, or that buys flatness with sharpness, loses on the very
terms it optimized.

The design question is therefore evidential: how much correction can each
subgroup's data actually support? That means shrinking noisy per-group
corrections toward a global mapping in proportion to sample support, keeping
every output a legitimate probability, and anticipating that at test time the
mapping will be queried where its training evidence is thin. Subgroup ids are
available at both fit and prediction time and may be exploited or ignored —
but a group-blind mapping must then argue how it protects a worst subgroup it
never identifies.

Defend the result mechanistically: show which reported term a naive
alternative (global-only, or fully per-group) would have degraded on the
shifted tail, and what in your method held that term.
