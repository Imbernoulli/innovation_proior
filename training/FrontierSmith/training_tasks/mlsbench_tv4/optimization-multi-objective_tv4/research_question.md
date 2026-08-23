Adaptive machinery is the fashionable answer in evolutionary multi-objective search —
schedules, feedback loops, self-tuning operators — and every adaptive knob is also a place
where a strategy can quietly fit itself to one landscape at the expense of another. This
variant runs the opposite experiment: a strategy whose every control quantity is a literal
constant, chosen once before the first individual exists, identical for every instance in the
suite and for both objective counts. If the mechanism is structurally right, one static recipe
should hold up across convex, disconnected, spherical, and deceptive geometries at once; if it
only survives through runtime adjustment, this variant is built to expose that dependence.

The contract is strict. Nothing may be recomputed from the population and fed back into any
operator: no generation-indexed schedules, no stagnation detectors, no probabilities that
drift. The per-generation callback stays inert, and no attribute is written after
construction. What remains designable is structure — which operators exist, in what fixed
proportions they fire, how survival trades rank against density — and structure is exactly
what gets evaluated. A fixed portfolio of variation intensities, for example, is legal
(constant shares, constant distribution indices) precisely because it is decided in advance;
the same portfolio with a share that responds to progress is not.

The three per-problem numbers on the scoreboard are computed exactly as for every other
variant, so the static recipe is judged head-to-head against adaptive rivals and against the
leaderboard anchors alike, with nowhere to hide a weak instance inside an average. The claim
to defend is uncomfortable and therefore interesting: that at this suite's scale a well-built
constant configuration concedes little or nothing to adaptivity, and that whatever gap remains
is the honest price of running blind — stated problem by problem, metric by metric, not
smoothed away.
