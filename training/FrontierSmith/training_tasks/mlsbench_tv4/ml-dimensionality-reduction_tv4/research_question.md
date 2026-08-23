The reference embedding methods spend lavishly: full k-NN graph
construction over every sample, hundreds of gradient epochs over every
point. This variant asks what fraction of their quality survives when the
arithmetic itself is rationed. Build an embedding whose entire
fit_transform costs a small, self-imposed slice of the allowed time — the
working target is well under a minute per dataset on CPU — and whose
machinery stays within simple, auditable algebra: subsampling, landmark
projections, matrix-vector products, short fixed iteration counts. No full
pairwise distance matrix over all samples, no long stochastic optimization.

The scored quantities are what they always were, so frugality is not an
excuse but a constraint to design against. The interesting question becomes
which structural investments buy the most metric per second: landmark
methods answer for many points with few; power iterations extract dominant
directions without a decomposition; sparse or implicit operators avoid
materializing anything they only need to multiply by.

Rules of the game:
- Declare the compute discipline in the code — an explicit budget guard —
  and honor it on the 50-dimensional input and the 784-dimensional ones
  alike.
- Costs must scale gently: near-linear dependence on sample count, never
  quadratic memory over the full input.
- The frugal recipe is fixed once; no dataset-specific budget switches.

The argument to make at the end is quality per unit compute: state what the
map achieves on the reported metrics, what it spends to get there, and
which cheap structural trick did the heavy lifting.
