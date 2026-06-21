We are given follow-up records that all begin at a comparable origin and end at some observed age, where each endpoint carries one of two meanings: either the event of interest was actually seen, or observation simply stopped first and all we know is that the event had not yet occurred by that stopping time. The data are pairs $(\text{time}_i, \text{event}_i)$ in which a true event time and a last event-free time look identical except for the indicator, and the target is the survival function $S(t) = P(T > t)$ for the unobserved event time $T$. The difficulty is not in sorting the times; it is in deciding who belongs in each denominator. A subject whose event has already been seen before a time point is no longer at risk, and a subject whose follow-up stopped before that point is no longer informative about it either — yet before its follow-up stopped, that very subject was genuine evidence of survival. The naive options each break on exactly this. The complete-data empirical survivor fraction works only when every event time is observed; under censoring the original sample size is no longer the right denominator. Grouped life tables estimate survival interval by interval, which suits coarse records but hides exact event times and mixes deaths with withdrawals inside broad bins. Parametric survival models impose a distributional shape that can be efficient when credible but is not what we want here — we want a distribution-free step curve determined purely by the observed follow-up table. So any acceptable estimator must use partial information without turning censoring into failure and without pretending censored subjects remain under observation forever, and it must rest on the explicit assumption that the act of stopping observation carries no hidden information about the event time.

I propose the Kaplan-Meier estimator. The central move is to abandon the search for one global denominator at time $t$ and make the denominator local. Just before each observed event time I can list exactly the subjects still under observation and not yet failed — the risk set — and a censored subject simply belongs to the earlier risk sets and then drops out of the later ones, never inventing follow-up I did not observe. This is only valid when censoring is not selectively removing subjects on account of their hidden event times; otherwise the surviving risk set would stop being representative for the next local question. The shape of the construction comes from the old life-table identity that survival over several intervals is a product of conditional survivals: survive the first interval, then survive the second given you reached it, and so on, so I never have to estimate the whole distribution at once — only each local chance of not failing among the subjects for whom that local question is actually observed. I then refine the intervals until the observed event times themselves define the only possible jumps. Letting $u_1 < \dots < u_m$ be the distinct observed times, I write $n_j$ for the number at risk immediately before $u_j$ and $d_j$ for the number of observed events at $u_j$. Among the current risk set the empirical conditional failure fraction is $d_j/n_j$, so the empirical conditional survival factor is $1 - d_j/n_j$, and multiplying these factors through time gives

$$\hat S(t) = \prod_{j:\,u_j \le t,\ d_j > 0} \left(1 - \frac{d_j}{n_j}\right).$$

Each component earns its place. The product structure is the conditional-survival chaining; the factor $1 - d_j/n_j$ is the local survivor fraction estimated only on the people genuinely observed across $u_j$; and the restriction $d_j > 0$ encodes the design choice that censoring alone must never create a downward step. A censored-only time contributes a factor of one, so it can be dropped from a compressed mathematical jump table, though a software event table may retain it carrying the previous survival value so that later denominators stay auditable. This same local bookkeeping is also what resolves ties cleanly: when several records share an observed time there is no meaningful order to impose inside that time, so I count all of them in the risk set just before the time, let the observed events among them determine the drop, and then remove the entire group — events and censored records alike — before moving on, which is strictly better than arbitrarily ordering tied records, since any such order would shift denominators without adding information. The load-bearing sanity check is the no-censoring limit: with distinct complete event times in a sample of size $m$, the factors are $(m-1)/m$, then $(m-2)/(m-1)$, and so on, and after $r$ events the product telescopes to $(m-r)/m$, exactly the usual empirical survival fraction. That confirms this is not a different estimator in the complete-data case but the complete-data curve with its denominator corrected for incomplete follow-up. One boundary genuinely remains: if the greatest observed lifetime is only a censoring time, the nonparametric curve is not determined beyond that last observation; I can report the value reached there but should not invent tail behavior without further assumptions.

The implementation is just that bookkeeping made explicit — sort the observed times, group equal values, hold $n_{\text{at risk}}$ before each group, multiply survival by $1 - d_j/n_{\text{at risk}}$, and subtract the whole group from later risk sets; when $d_j$ is zero the factor is one, so keeping that time in the output yields a flat row rather than a jump.

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
