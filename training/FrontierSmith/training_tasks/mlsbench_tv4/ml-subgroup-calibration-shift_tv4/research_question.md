The evaluation of this variant asks a mapping fitted in one region of score
space to answer for its behaviour in another. Because the held-out region is
selected from the extreme of a domain variable, the probabilities needing
correction at deployment sit where calibration coverage is thin or absent.
The variant therefore optimises a property most calibration work never states
explicitly: graceful degradation as the queried region moves away from the
fitted one.

Capacity control is the central trade. High-capacity monotone fits can be
locally superb on calibration data yet arbitrary two bins beyond it; rigid
one-parameter families extrapolate predictably but correct less where data
is dense. The design question is where on that spectrum — or through what
explicit safeguard, such as flattening corrections outside the observed
score range or blending toward identity in unsupported regions — the unseen
tail is best protected. Internal rehearsal of the shift, for instance
holding out the calibration tail to preview extrapolation behaviour, is
fair game and encouraged.

All reported columns are computed on the shifted tail, so they already
measure transfer: worst-subgroup calibration error and the between-group
spread reveal which populations the extrapolation fails first, while the
Brier score exposes corrections that turned meaningless outside their
fitted support. The subgroup AUROC column serves as a health check, nothing
more.

Defend the method as an extrapolation argument: characterise how its output
behaves as inputs leave the calibration support, show that failure is
gradual rather than cliff-shaped, and contrast with a flexible fit that
wins on calibration data and loses on the tail.
