Every fitness call in this harness costs the same whether it teaches the
optimizer something or merely confirms what it already knew, and the run is
allotted a fixed pool of such calls, set by the population size and the
generation count it is launched with. This variant takes that pool as the
scarce resource and asks for the largest fitness gain per evaluation
actually spent, on all four benchmark settings at once.

best_fitness records where the run ends, while convergence_gen records
the first generation whose best already lies
within one percent of that endpoint, so a strategy that drifts for hundreds
of generations before its final small improvement is exposed even when its
endpoint looks strong. The pair has to be won jointly: plateauing early at
a mediocre value trades one metric for the other and defends nothing.

The design must honour a short list of obligations. An explicit ledger
has to account for every evaluation, and a redesigned loop — steady-state, unequal offspring
counts, whatever — may never overdraw what the harness grants. Nothing may
be spent re-scoring genomes that variation left untouched; a duplicate
genome is a known result, not a fresh purchase. And variation intensity
must be scheduled against the remaining allowance, so that broad, costly
exploration happens while the pool is full and cheap refinement dominates
as it drains — front-loading improvement by construction rather than by
accident.

What must be defended at the end is a spend-rate argument, read off the
run's own generation trace: which fraction of the allowance bought which
fraction of the final fitness gain, and why the same schedule holds from
thirty dimensions to one hundred.
