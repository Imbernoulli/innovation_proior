# Kaplan-Meier Estimator

Given right-censored observations `(time_i, event_i)`, where `event_i=True` means the event was observed and `False` means right-censored, let `u_1 < ... < u_m` be the distinct observed event-or-censoring times. At time `u_j`, define:

- `n_j`: number of subjects at risk immediately before `u_j`
- `d_j`: number of observed events at `u_j`
- `c_j`: number of censored observations at `u_j`

The survival estimate is

```text
S_hat(t) = product_{j: u_j <= t and d_j > 0} (1 - d_j / n_j)
```

Censoring does not create a drop. A subject censored at time `c` contributes to risk sets before and at `c`, then leaves later risk sets. If events and censoring share a recorded time, all tied records are in the risk set immediately before that time; the events determine the drop, and both events and censored records are removed before the next time. If the greatest observed time is censoring rather than an event, the nonparametric curve is not determined beyond that time.

```python
import numpy as np

def kaplan_meier(durations, events):
    """Return unique observed times and Kaplan-Meier survival estimates.

    durations: one-dimensional observed event-or-censoring times.
    events: boolean array; True means event observed, False means right-censored.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)

    if durations.shape != events.shape:
        raise ValueError("durations and events must have the same shape")
    if durations.ndim != 1:
        raise ValueError("durations and events must be one-dimensional")
    if durations.size == 0:
        raise ValueError("durations and events must not be empty")

    order = np.argsort(durations, kind="mergesort")
    durations = durations[order]
    events = events[order]

    n_at_risk = len(durations)
    survival = 1.0
    out_times = []
    out_survival = []

    start = 0
    while start < len(durations):
        time = durations[start]
        stop = start
        while stop < len(durations) and durations[stop] == time:
            stop += 1

        d_j = int(events[start:stop].sum())
        removed = stop - start

        if d_j:
            survival *= 1.0 - d_j / n_at_risk

        out_times.append(time)
        out_survival.append(survival)

        n_at_risk -= removed
        start = stop

    return np.asarray(out_times), np.asarray(out_survival)
```

This returns every unique observed time, matching event-table implementations: censor-only times repeat the previous survival value. A compressed mathematical curve can omit those flat rows without changing `S_hat(t)`.

With no censoring and distinct event times, the product telescopes to the complete-data empirical survival curve: after `r` failures in a sample of size `m`, `S_hat = (m - r) / m`.
