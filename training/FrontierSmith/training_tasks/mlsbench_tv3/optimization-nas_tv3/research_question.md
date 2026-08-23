Treat the moment of stopping as adversarial. The evaluation loop can end at any epoch — the
budget guard aborts a step in flight the instant the allotment runs dry — and whatever
architecture the optimizer names at that instant is the one sent to the untouched test split.
A search that spends its early budget building apparatus and cashes in near epoch thirty is
betting that the end of the run arrives on schedule — a bet this variant refuses to place. The
working requirement: after every single paid query, the nominated cell must already be a
defensible answer.

That requirement reshapes the design rather than merely constraining it. The incumbent must be
refreshed inside the step, the moment the accuracy comes back, and the final read must be free
of computation — at stop time there may be no budget left to run anything. Improvement has to
arrive as a steady drip rather than a terminal payoff, which argues for interleaving fresh
coverage with refinement of the current leader at a fine grain, and against any long
uninterrupted phase whose value materialises only at its end. Plans are allowed; plans whose
worth is hostage to their own completion are the specific thing being disallowed.

Judgment is still the standard readout — final test accuracy on each dataset, five seeds
apiece — so an anytime design must not buy robustness-to-interruption with a weaker finish.
The interesting claim is that it does not have to: at this budget, a policy that is always
ready tends also to finish well, because readiness forces early exploitation of whatever has
been learned. Argue the claim with the incumbent trajectory recorded across the whole run — a
curve that climbs early and never waits — placed next to final numbers that concede nothing to
end-loaded designs.
