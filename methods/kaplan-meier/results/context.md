## Incomplete Follow-Up Records

Each observation begins at a comparable origin and then ends at an observed age. The endpoint has two possible meanings. In one case the event of interest is actually seen. In the other case observation stops first, so the only known fact is that the event had not occurred before that stopping time.

The data are therefore pairs `(time_i, event_i)`, where `time_i` is either an event time or a last observed event-free time. The inferential target is the survival function

```text
S(t) = P(T > t)
```

for the unobserved event time `T`.

## Denominator Risk

The hard part is not sorting times. It is deciding who belongs in each denominator. A subject whose event is observed before a time point is no longer at risk. A subject whose follow-up stopped before that time point is also no longer informative about that time point. Before follow-up stops, that same subject is real evidence of survival.

The standard assumption is that stopping observation carries no hidden information about the event time — censoring is uninformative.

## Existing Baselines

A complete-data empirical survival curve works when every event time is observed. It uses the full sample size as the denominator at each time.

Grouped life-table calculations estimate survival interval by interval. They are natural for coarse follow-up records and accumulate counts of deaths and withdrawals within each broad bin.

Parametric survival models impose a distributional shape. They can be efficient when the shape is credible.

## Research Question

How can a nonparametric survival curve be estimated from right-censored follow-up data?

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
