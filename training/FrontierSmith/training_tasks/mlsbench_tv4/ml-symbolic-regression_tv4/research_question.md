An expression tree conflates two search problems with nothing in common: a
discrete, combinatorial choice of operators and wiring, and a continuous
estimation problem over the numeric leaves. Vanilla GP attacks both with the
same blunt instruments, so a structurally correct candidate is routinely
discarded because its constants happened to be wrong on the day it was
evaluated. This variant makes the division of labor explicit: variation
operators are responsible for proposing skeletons, and a separate,
deliberately cheap coefficient-refinement routine is responsible for letting
each promising skeleton testify at its best before selection judges it.

Design questions the split raises, which the study should answer: how many
refinement probes per candidate are worth their evaluation cost when the
driver's generation and population budgets are immovable; whether refinement
should touch every individual or only the current elite; how to keep tuned
constants from becoming a memorization channel (a skeleton with enough free
coefficients can shadow-fit the sample exactly the way a bloated tree does);
and how crossover ought to trade numeric material between lineages without
destroying the structures that carry it. The harness is untouched — the same
driver schedule, the same held-out R2 — so every probe spent polishing
coefficients is a probe not spent exploring structure, and the accounting
must be honest.

The claim to defend: with an explicit structure/coefficient split, the
search reaches skeletons that raw joint evolution misses at the same total
budget, and the refined constants generalize — the gap between training fit
and held-out fit does not widen as refinement is made stronger. Report where
the budget split landed and why that allocation, rather than a bigger or
smaller one, was the right trade.
