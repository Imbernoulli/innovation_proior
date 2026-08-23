The two reported metrics are not equally demanding about the same part of the
score axis. AUROC integrates ranking quality over every threshold, but the F1
measured at the contamination threshold is decided by a thin slice: the test
points your scores place above roughly the anomaly rate. On thyroid that
slice is about forty points out of fifteen hundred — a handful of
confident-looking normals floating into it costs precision and recall
simultaneously, and no amount of good ordering in the bulk buys it back.

This variant makes the top of the ranking the object of design. Think of the
threshold as an alert budget: the detector may nominate only a small fraction
of the test set, and every nomination wasted on a borderline normal is a miss
it cannot recover. A scoring rule tuned for global separation often spends
that budget carelessly — one spectacular coordinate, a duplicated
near-anomaly, or a single view's idiosyncratic heavy tail can flood the top
slice. The design question is what evidence a point must present before it
outranks the budget line, and how scores should be shaped so the slice just
above the operating threshold stays both pure and full.

Mechanisms on the table: consensus requirements across dissimilar views
before a point may rank high, rank-domain fusion that stops any single view's
tail from dominating the top, de-duplication so one anomaly cluster does not
exhaust the budget, and explicit estimation of where the operating threshold
will fall. Defend, on the unchanged protocol, that top-slice discipline lifts
F1 across all four datasets while leaving AUROC — the bulk ordering — intact
or better. The scaffold fuses an isolation view with a distance view by rank
consensus; it demands agreement but does not yet reason about the budget,
which is the gap left open for you.
