The pipeline will run two hundred rounds, but pretend the run could be
interrupted after twenty or fifty and forced to hand over whatever
ensemble exists at that instant. This variant asks for a boosting strategy
optimized under that reading: held-out quality should be front-loaded,
with a large share of final performance in place within a small fraction
of the budget, and the remaining rounds refining rather than rescuing.

Front-loading inverts the usual schedule. Uniform tiny steps spread
learning evenly across the whole run -- exactly wrong here. Take the
largest, most informative steps first (bigger effective coefficients,
targets that expose the dominant structure early), then decelerate
deliberately, moving late rounds into a consolidation role where they may
polish predictions but can no longer destabilize them. A useful mental
test: truncate the learner sequence at an arbitrary prefix -- is the
truncated ensemble close to the best achievable with that many trees?

Two further constraints. Aggression must not buy early speed with
final-quality debt: what is scored is still the complete run, so the
deceleration schedule has to land at a competitive endpoint. And the
frugality extends to compute: a strategy justified by budget discipline
should add only trivial per-round overhead of its own, so heavyweight
bookkeeping is out of bounds. Success here means prefix dominance -- at
essentially every truncation point the paced run should match or beat a
constant-step run of the same length, without giving up the endpoint.
