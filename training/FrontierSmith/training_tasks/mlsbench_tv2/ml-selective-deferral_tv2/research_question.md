An operations team has provisioned human review capacity for exactly one case
in five, and everything upstream of the accept-or-defer choice arrives
frozen. What is commissioned is a deferral mechanism whose realised
acceptance rate, measured on the held-out split, lands on that 80% figure
with near-zero slack — and which, at that realised budget, hands over the
cleanest accepted pool it can.

Coverage transfer is where the difficulty lives. A cutoff placed at a
naive empirical quantile of calibration scores drifts when scores are heavily
tied, when the calibration sample is modest, or when the test composition
wobbles; each drift either floods the reviewers (over-deferral) or leaves paid
capacity idle while unvetted errors ship (under-deferral). Both directions
count as failures. The tracked actual-coverage figure is the audit, and
deviation from target is the first thing checked, not a stylistic footnote.

Subject to landing on budget, the accepted set should minimise error —
overall and for the weakest subgroup — and the cutting score must stay
informative as a correctness ranking, because a degenerate score can hit any
coverage by accident. The spread of deferral burden across groups is watched
as a sanity term. The same discipline is exercised on three separate tabular
problems, so coverage fidelity has to hold everywhere rather than on one
lucky split.

The claim to defend: quantify the calibration-to-test coverage error of the
proposed cutoff placement, explain the finite-sample or tie-handling
mechanism that keeps it small, and show that the selective error achieved
sits at or near the best available at that realised coverage — that is,
precision was not bought by quietly shrinking or inflating the accepted set.
