On a sparse graph, the candidate pool works against you: Win95pts offers
2850 variable pairs of which only 112 are true edges, so even a
well-calibrated test making one percent false-positive calls floods the
report with more wrong adjacencies than there are right ones to find.
Every one of those wrong edges is billed twice — once through adjacency
precision and once through SHD — while a missed true edge is billed once
through recall. The asymmetry is the premise of this variant: the report
should read like an audited ledger in which no entry appears without
having survived scrutiny.

Build an admission-controlled discovery procedure. An edge enters the
output only after passing explicit vetting: a demanding marginal bar
followed by conditional interrogation against its plausible common causes
(does the association survive when the strongest shared neighbours are
conditioned on?), with thresholds justified by a false-discovery argument
rather than convenience. Recall is then grown outward from that vetted
core — a weaker candidate may be admitted when the already-trusted
structure raises its plausibility — but never by loosening the bar
wholesale. Orientation follows the same law: an arrowhead is a stronger
claim than an adjacency, so directions appear only where the vetted
skeleton makes them defensible, and arrow precision must not become the
leak in an otherwise tight report.

The case to make with the reported numbers: adjacency precision at or
near the top of what these five datasets permit, on the small networks
and — the hard part — on Hailfinder and Win95pts where the candidate
pool is largest; SHD low because false positives were refused up front;
adjacency recall as high as the vetting discipline allows, presented
honestly as the price of the precision guarantee rather than hidden.
The same admission rule must govern all five networks.
