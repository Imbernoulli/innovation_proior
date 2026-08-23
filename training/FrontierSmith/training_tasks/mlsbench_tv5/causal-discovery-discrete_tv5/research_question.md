Discrete conditional-independence testing has a cardinality problem.
A contingency test between two variables with many states, or conditioned
on a many-state neighbour, burns degrees of freedom exponentially: cells
outnumber samples, the test loses power, and the edge quietly vanishes
from the output. The bnlearn suite makes this concrete — Hailfinder and
Win95pts mix binary flags with variables of up to eleven states — and the
victims are systematic, not random: exactly the edges touching
high-cardinality variables go missing. The result is an output whose
adjacency recall (and, downstream, arrow recall) is capped by test
starvation rather than by genuine ambiguity in the data.

What this variant asks for is recall recovery without densification.
Attack the starvation mechanism directly: degrees-of-freedom-aware
statistics that put a 2x2 table and an 11x8 table on a comparable
footing, principled category pooling or coarsening that spends samples
where the association signal lives, and targeted rescue passes for
variables whose detected neighbourhood is implausibly bare given their
role elsewhere in the graph. The forbidden move is buying recall with a
lowered global bar — that trade shreds adjacency precision and inflates
SHD, and the harness will price it accordingly. Precision on the small,
well-powered networks (Cancer, Child) is the canary: if it slips, the
recall gains are counterfeit.

The defensible end state, from one fixed procedure on all five datasets:
adjacency recall materially higher on the cardinality-heavy networks
than a naive fixed-threshold tester achieves, arrow recall lifted with
it, adjacency precision held near its previous level everywhere, and
SHD improved on the large networks because found-and-right edges now
outnumber the false alarms the rescue machinery admits.
