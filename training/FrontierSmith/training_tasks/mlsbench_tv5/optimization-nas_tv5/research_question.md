Permit the search a single act of chance: one random cell to stand on. From there, every
oracle call must evaluate an architecture exactly one edge-edit away from the current position
— a walk through the benchmark, not a survey of it. Random restarts, global draws, and
parallel portfolios are all off the table by construction, so the entire algorithmic content
of a run lives in three small decisions: which neighbour to try next, when a result justifies
moving the position, and what to do once the immediate neighbourhood is spent.

Cell benchmarks of this kind are known to be locally structured —
single-edit neighbours correlate strongly in accuracy — so a disciplined trajectory is not an
obviously handicapped strategy; whether it can match unrestricted search at this budget on the
actual scoreboard is precisely the open question. The walk's pathologies are equally real:
twenty-four neighbours around any position, thirty lookups in the whole purse, plateaus wide
enough to swallow half of it. Move ordering (which edges to probe first, informed by edits
that paid off earlier in the walk), tolerance for sideways drift, and a re-centring rule for
exhausted neighbourhoods — one that continues the trajectory rather than teleporting — are
where a good design separates from uniform choices.

Accounting is the usual: the cell nominated when the walk halts is tested once on the held-out
split, per dataset, per seed, and the best cell ever probed comes free with the trail — no
extra lookup is needed to remember it. The claim on offer: locality plus discipline recovers
nearly everything unrestricted sampling gets from its freedom, on all three datasets, with the
walk's own move log as the exhibit showing every step obeyed the single-trajectory contract.
