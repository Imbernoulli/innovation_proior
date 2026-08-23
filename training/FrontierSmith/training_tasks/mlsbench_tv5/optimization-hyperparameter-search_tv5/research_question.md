Most search strategies keep uniform randomness on retainer: some
fraction of proposals drawn blind, insurance against a misled model.
This variant cancels the policy. After a declared seed batch of
minimal size, every configuration proposed for the rest of the run
must be the output of an explicit model fitted to the run's own
history — a distribution, a surrogate, a structured rule, but never a
uniform fallback, never an epsilon of blind sampling, never a shrug.

Removing the crutch turns exploration into a modelling obligation.
The stray blind draw that usually rescues a collapsing search is
unavailable, so the proposal distribution itself must be engineered
against collapse: dispersion floors, forced spread across categorical
choices, widths tied to the remaining budget — whatever keeps the
model generating genuine hypotheses instead of photocopies of the
incumbent. Over-exploitation is the announced failure mode, and the
design is judged by how it is prevented from within the model.

One further requirement: nothing may be specialised per benchmark — a
single modelling policy must cope with three to six dimensions,
log-marked ranges, integers and categorical choices alike — and the
seed batch is part of the declared design, not a pool to quietly
re-open whenever the model feels thin.

Arbitration comes from the scoreboard: a collapsed proposer shows
itself twice, once as an incumbent that stalls early and caps
best_val_score, and again as a convergence_auc curve that goes flat
after its first rise. The claim to defend is
that a fully model-committed proposer sustains discovery across the
whole budget — that on these benchmarks a well-built model needs no
random insurance, and the run's sequence of proposals is itself the
proof that the exploration came from the model.
