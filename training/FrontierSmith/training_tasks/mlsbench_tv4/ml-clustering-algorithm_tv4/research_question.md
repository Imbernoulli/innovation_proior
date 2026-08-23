Run a stock partitional clusterer twice with two seeds and you often get
two different stories about the same data. For a method whose output feeds
downstream decisions, that variance is a defect independent of accuracy.
This variant elevates run-to-run determinism to the primary discipline: the
partition should be a function of the data, with the seed reduced to a
tie-breaking formality.

Optimize for reproducibility first. The same input must yield the same
labelling — up to renaming of cluster ids — whatever value arrives in
random_state, and predict must be a deterministic read-out of the fitted
model rather than a fresh stochastic fit. Within that discipline, push all
three reported scores as high as they will go: a stable but wrong partition
earns nothing, and the interesting designs are those where stabilization
machinery (consensus over restarts, medoid partitions, co-association
voting, deterministic initialization) doubles as a denoiser that lifts the
scored numbers.

Constraints defining the variant:
- Internal randomness must be quenched by construction — averaged out,
  voted out, or replaced by deterministic rules — not hidden by hardcoding
  one lucky seed and hoping.
- The stabilization mechanism must be algorithmic in form, identical for
  every input it meets.
- Fit-then-predict on the same matrix must reproduce self.labels_ exactly.

The final claim is mechanism-level: an argument for why your partition is
the one any seed would find, supported by the standard reported metrics on
all three input geometries.
