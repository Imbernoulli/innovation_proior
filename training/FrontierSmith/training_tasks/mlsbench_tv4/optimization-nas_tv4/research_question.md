A cell in this search space is six labelled edges, and that structure is legible without
spending anything: which operation sits on which edge, how convolutions chain into paths,
which motifs the already-evaluated cells share. This variant puts such free signal in charge.
The oracle's thirty answers are demoted from search driver to audit: broad, cheap candidate
pools are ranked by structural estimates computed from prior observations, and the single pool
member the estimate likes best is the only one that costs a query. Search quality then rises
or falls with the quality of the free signal, which is the point.

The environment constrains what the signal may be. A parameter check compares any learned
model against the reference implementations and rejects heavyweight predictors, so the proxy
must stay in the regime of light arithmetic — per-edge statistics of paid observations, path
composition features, linear fits at most — rather than expressive networks. Within that
regime the design space is real: per-(edge, operation) value tables versus path-level tables,
smoothing toward a global mean where evidence is thin, interaction terms between edges, and
the size and composition of the free pool screened each round.

Both classic failure modes of proxy-led search sit on the scoreboard. Trust the estimate too
much and the run tunnels into whatever the early evidence flattered, never spending a query
where the tables are ignorant; trust it too little and the machinery decays to unguided
sampling with extra steps. What decides between designs is unchanged: the test accuracy of the
finally nominated cell on each of the three datasets over the five seeds. The claim worth
defending is that thirty audited guesses, each pre-screened by an honest cheap model of the
space, land higher than thirty guesses steered by the oracle alone.
