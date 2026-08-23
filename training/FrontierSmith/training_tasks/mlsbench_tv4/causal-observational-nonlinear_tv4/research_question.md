Averaged leaderboards forgive specialists; this variant does not. The three
generators were chosen to be mutually hostile — Gaussian-process functions
with exponential noise on a scale-free graph, mixed function families with
Gaussian noise, and a 150-sample Laplace regime — so any algorithm carrying a
hidden assumption about smoothness, tail weight, or sample abundance will
excel somewhere and crater somewhere else. Here the design objective is the
floor, not the average: build the discovery procedure whose worst per-setting
result is as high as possible, and treat the spread between its best and
worst settings as a defect to be engineered away.

That objective dictates the toolbox. Statistics enter only if their null
behavior is stable across noise families — ranks, signs, quantiles, medians
and median absolute deviations — and heavy-tailed observations must be tamed
before any moment is trusted. Model capacity should be bounded a priori: a
degree cap and conservative decision rules protect the low-sample setting
from hallucinated structure, while the same fixed rules must still find
enough edges at n = 2000 to be competitive. Exactly one configuration is
allowed. If a constant would be tuned differently for Gaussian versus Laplace
noise, or for 150 versus 2000 rows, that constant must instead be a
continuous function of quantities the algorithm can measure itself.

The scoreboard stays as it is — F1 on directed edges, precision, recall, SHD,
reported per setting — but success is read off differently: narrow the gap
between the strongest and weakest of the three columns while keeping the
weakest column clearly above the linear-method baselines. A five-point gain
on the friendly setting bought with a collapse under Gaussian noise is, for
this variant, a regression.
