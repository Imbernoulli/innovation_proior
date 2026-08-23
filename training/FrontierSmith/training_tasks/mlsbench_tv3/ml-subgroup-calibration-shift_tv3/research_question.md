Between-group reliability gaps are rarely caused by the big groups. The
subgroups that set the gap column are the thin ones — a few dozen
calibration examples, sometimes fewer — where any curve fitted to the group
alone is mostly noise and any global hand-me-down mapping is mostly bias.
This variant makes that tension its whole subject: every subgroup,
regardless of headcount, is entitled to probabilities of equal
trustworthiness, and the widest between-subgroup error difference is the
number under attack.

The intended machinery is evidence-proportional borrowing. Per-group
behaviour should be estimated exactly as far as the group's own sample
supports and no further, with the remainder inherited from pooled structure
— the classical partial-pooling posture, applied to calibration. What
distinguishes a good design is the weighting law: how correction strength
grows with group size, what plays the role of the prior, and whether the
pooled anchor itself deserves trust after the covariate shift the test
split bakes in.

Guardrails: worst-group calibration error should fall along with the gap —
a gap closed by degrading the large groups is a false economy — the pooled
Brier score is watched so probability quality is not silently spent, and
outputs must remain proper probabilities everywhere, including for group
ids never observed at fit time.

The defence this variant requires is a small-sample argument: show, group
by headcount, where a per-group fit would have overfitted and where a
global map would have underfitted, and demonstrate that the proposed
weighting law threads between the two — with the smallest groups ending up
no less reliable than the largest.
