## Incomplete Follow-Up Records

Each observation begins at a comparable origin and then ends at an observed age. The endpoint has two possible meanings. In one case the event of interest is actually seen. In the other case observation stops first, so the only known fact is that the event had not occurred before that stopping time.

The data are therefore pairs `(time_i, event_i)`, where `time_i` is either an event time or a last observed event-free time. The inferential target is the survival function

```text
S(t) = P(T > t)
```

for the unobserved event time `T`.

## Denominator Risk

The hard part is not sorting times. It is deciding who belongs in each denominator. A subject whose event is observed before a time point is no longer at risk. A subject whose follow-up stopped before that time point is also no longer informative about that time point. But before follow-up stops, that same subject is real evidence of survival.

Any estimator must therefore use partial information without turning censoring into failure and without pretending that censored subjects remain observed forever. It also needs an explicit assumption that stopping observation is not itself carrying hidden information about the event time.

## Existing Baselines

A complete-data empirical survival curve works when every event time is observed. It fails under censoring because the original sample size stops being the right denominator.

Grouped life-table calculations estimate survival interval by interval. They are natural for coarse follow-up records, but the grouping choice can hide the exact event times and can mix deaths and withdrawals inside broad bins.

Parametric survival models impose a distributional shape. They can be efficient when the shape is credible, but the desired object here is a distribution-free step curve determined by the observed follow-up table.

## Required Behavior

The curve should start at one and never increase. It should change only when observed events occur. Censoring marks may appear on a plot, but censoring alone should not create a downward step.

Tied times require grouped bookkeeping. Records with the same observed age must be handled as one group, because an arbitrary order inside a tie would change denominators without adding information. With no censoring, the procedure should reduce to the ordinary empirical survivor fraction.

## Code Surface

The implementation receives one-dimensional arrays of observed durations and event indicators. It must validate the shape contract, sort observed ages, group equal ages, and maintain enough event-table counts to distinguish observed events from stopped follow-up.

```python
import numpy as np

def estimate_survival(durations, events):
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    # TODO: validate one-dimensional matching arrays.
    # TODO: build grouped counts over the observed ages.
    # TODO: turn the grouped counts into the survival table.
    raise NotImplementedError
```
