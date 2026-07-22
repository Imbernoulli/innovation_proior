# Field-Camera Trap Network: Adaptive Reservoir Budgeting Across Unknown, Skewed Species Strata

A wildlife-monitoring array streams photo detections from camera traps scattered
across a reserve. Every detection is tagged with a coarse **species class**
(a "stratum") and carries a numeric **reading** (e.g. a scene-exposure statistic
used downstream to compute a class-weighted population mean, such as mean
brightness deviation used for activity-time inference). Storage is limited: only a
fixed total number of full-resolution frames — the **reservoir budget `K`** — can
be archived from the entire deployment; everything else is discarded. You must
decide, **per species class**, how many of the `K` reservoir slots to devote to it.

Both the true relative frequency of each class and the true variability of its
readings are **unknown in advance and skewed**: most classes are common with mild,
fairly uniform readings, but occasionally one class is **rare and highly erratic**
(a skittish nocturnal animal seen only a handful of times, whose few readings swing
wildly). You only get to see a **burn-in prefix** of the stream — the first batch
of detections, in arrival order, tagged by class — before you must commit to a
final per-class reservoir allocation for the rest of the deployment.

## The objective (what "good" means)

Downstream, the archived reservoir for class `s` (`n_s` slots) is used to estimate
that class's mean reading, and a class-frequency-weighted combination of these
per-class means estimates the whole-network mean. The (textbook) asymptotic
variance of that stratified estimator is

```
Var(alloc) = sum_s  w_s^2 * sigma_s^2 / max(n_s, 0.5)
```

where `w_s` is class `s`'s true population share and `sigma_s` is its true reading
standard deviation — **both hidden from you**; you only ever see the noisy burn-in.
Your job is to choose integer-ish slot counts `n_s` (`sum_s n_s <= K`) that
**minimize** `Var(alloc)`.

This has no easy optimum: splitting `K` equally ignores the skew; allocating purely
by how often you've *seen* a class so far trusts noisy burn-in counts at face
value — a class that showed up only a few times can still have enormous true
variance, so starving it is costly. A sound strategy must (1) estimate each
class's population share in a way that survives zero/near-zero counts, (2)
estimate (or hedge) its variability, staying cautious about barely-observed
classes, and (3) reallocate budget toward classes whose *(estimated share) ×
(estimated variability)* is largest — not toward whichever class is most common.

## Public instance (stdin JSON)

```json
{
  "S": 6,                                   // number of species classes (0..S-1)
  "K": 90,                                  // total reservoir budget
  "burnin": [                               // detections observed so far, in order
    {"s": 3, "v": 1.284},                   // s = class id, v = reading
    {"s": 0, "v": -0.63},
    ...
  ]
}
```

Not every class is guaranteed to appear in `burnin` — a class with zero
occurrences is still a valid class you may (and generally should) allocate to.

## Answer (stdout JSON)

```json
{"alloc": [n_0, n_1, ..., n_{S-1}]}   // reservoir slots per class, n_s >= 0, sum <= K
```

Values may be given as integers or floats; wasting budget (`sum_s n_s < K`) is
never advantageous.

## Scoring

The evaluator computes a **baseline** `b` = `Var(alloc)` of the equal-share
allocation `n_s = K/S` for every class. For a feasible answer with objective `obj`:

```
r = min(1, 0.1 * b / obj)
```

so equal-share scores exactly `0.1`, and an allocation that drives the true
stratified variance `k` times below equal-share scores `min(1, 0.1k)`. Any answer
that is malformed, has the wrong length, has a negative or non-finite entry, or has
`sum_s n_s > K` is **infeasible and scores 0** on that instance. The reported
`Ratio` is the mean of `r` over 10 deterministic, seeded instances (including
larger, tighter held-out ones), each built by drawing a burn-in stream from fixed
but hidden true per-class frequencies and variances.

Your program reads one public instance JSON from stdin and writes one answer JSON
to stdout. It runs in an **isolated subprocess** and only ever sees the public
instance — never the true `w_s`/`sigma_s` used for scoring.
