An embedding method that ships a different picture of the same data on
Monday than on Tuesday is hard to defend, whatever its average scores.
Stock neighbor-embedding optimizers are noisy by construction — random
initializations, negative sampling, order-dependent updates — and their
scored numbers wobble from run to run. This variant makes reproducibility
the headline property: the map must be a deterministic function of the
input matrix, with random_state accepted for interface compatibility but
rendered irrelevant by design.

Concretely, two calls on the same X must produce embeddings whose reported
metrics agree to numerical precision, the strong version being bitwise
identical output. Sources of run-to-run variance — initialization, sampling,
asynchronous accumulation, even eigenvector sign flips — are to be
eliminated by construction: closed-form stages, canonical orderings and
sign conventions, fixed deterministic schedules.

Within that discipline, quality still decides the score on the three fixed
datasets. Settling for a weak linear map is the easy exit; the research
content is a nonlinear-quality embedding that is nonetheless exactly
repeatable, demonstrating that the stochastic machinery of the reference
methods is replaceable rather than essential.

Variant terms:
- No stage of the pipeline may consume the seed in a way that changes the
  output; any iterative component needs a deterministic initialization and
  a fixed update order.
- Tie-breaking rules must be explicit and data-independent in form.
- Data-driven behavior is permitted only via deterministic functions of
  the input matrix.

The claim to close with: your map is the unique answer your algorithm gives
for this data, and that uniqueness cost little on the reported metrics.
