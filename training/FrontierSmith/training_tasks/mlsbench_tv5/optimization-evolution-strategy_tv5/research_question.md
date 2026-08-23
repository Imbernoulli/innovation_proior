The deliverable here is a calendar rather than an operator. The
variation machinery is deliberately ordinary and held
fixed; the design surface is the policy layer above it: the test that
declares a population trajectory exhausted, the sizing rule that
decides how large the next attempt deserves to be, and the choice of
what, if anything, is carried across the boundary between attempts.

The suite makes that decision genuinely contested. Trap-dense
landscapes reward abandonment, because a population camped on a local
optimum turns every further evaluation into nothing; the curved
unimodal valley punishes it, because one long uninterrupted trajectory
is exactly what that landscape wants. A viable trigger therefore
cannot be a timer. It has to infer from the trajectory itself —
improvement drought, collapsed spread, whatever the run can measure
about its own progress — that the marginal value of continuing has
dropped below the expected value of beginning again. Sizing is the
other half of the same decision: successive epochs need not be equal,
and the growth or shrinkage of later populations is part of the
policy, with every epoch, re-seeding included, paid from the one
evaluation pool the harness grants.

Ground rules: one policy with one set of constants serves all four
settings, with no per-function switches; the run reports its archived
best, and the per-generation record must track that archive so a
restart can never erase progress from history.

The claim to defend, in the reported numbers: governed by this
calendar, the ordinary operators beat their own single-trajectory run,
with the margin concentrated on the trap-dense settings and nothing
surrendered in the valley — and the defence must point at the epochs
in the trace and say which restart earned the difference.
