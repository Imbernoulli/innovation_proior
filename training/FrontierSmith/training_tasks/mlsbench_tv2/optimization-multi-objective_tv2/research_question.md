Nothing in the fixed loop is charged except truth: an offspring whose fitness has been
invalidated triggers one call to the black-box objective, and everything else — selection,
sorting, survival, bookkeeping — is free. The schedule in the problem spec (population size
times generation count) is therefore a ceiling on objective calls, not an entitlement, and a
strategy chooses its own bill by choosing which clones it actually perturbs. This variant makes
that bill the object of study: build a metered algorithm that decides, offspring by offspring,
whether new information is worth buying, and let the untouched clones ride through the loop
carrying the valid objective vectors they inherited.

Scoring is exactly the standard readout — dominated volume, distance to the withheld reference
front, and evenness of the final non-dominated set, per problem — and no evaluation count is
ever reported. The economics must therefore be defended inside those three numbers: calls spent
where the front is still moving buy convergence, calls spent re-measuring a settled region buy
nothing, and a strategy that meters well should match or beat an unmetered twin on the final
readout while provably invoking the objective far less often. Keep an internal ledger of paid
evaluations — that count, laid beside the closing front, is the entire exhibit.

Rules of the game: the meter may consult only what the runner legitimately holds — genotypes and
already-paid-for objective values — because problem identity is withheld and per-instance tuning
is impossible. Free-riding clones must stay honest: their stored objectives are their true ones,
and inheriting a guessed fitness for a perturbed genotype is out of bounds. A single mechanism
has to cover the 2-objective and 3-objective cases without modification, and it must tolerate
being wrong — a meter that misreads progress may slow the search, but it must never corrupt it.

The deliverable is an account in three parts: where the evaluations went, why the meter opened
and closed when it did, and per-problem final numbers showing that frugality was not paid for
in front quality.
