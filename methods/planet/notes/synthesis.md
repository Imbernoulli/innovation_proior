# PlaNet Synthesis

This note is retained as a compact orientation file. The strict rebuild evidence
and detailed corrections are in `notes/source_matrix.md` and
`notes/discovery_synthesis.md`.

## Core Reconstruction

PlaNet learns a latent dynamics model from images and uses online planning
instead of a learned policy/value network. The problem is a visual POMDP:
history is needed to infer velocity and hidden physical state.

The model is an RSSM:

- deterministic belief: $h_t=f(h_{t-1},s_{t-1},a_{t-1})$;
- stochastic latent: $s_t\sim p(s_t\mid h_t)$;
- heads: $p(o_t\mid h_t,s_t)$ and $p(r_t\mid h_t,s_t)$;
- filtering posterior: $q(s_t\mid h_t,o_t)$;
- decoder/reward features: `[sample, belief]`.

The ELBO maximizes reconstruction/reward log-likelihood and minimizes
`KL(q || p)`. With unit-variance Gaussian heads, negative log-likelihood is half
squared error plus constants. Latent overshooting trains multi-step predictive
priors by comparing open-loop predictions to informed posteriors in latent
space. The data-processing relation used to connect multi-step and one-step
predictive distributions is a conjecture in expectation, not a pointwise proof.

## Code Grounding

Canonical code was cloned to `code/google-research-planet/` and checked at:

- `planet/models/rssm.py`
- `planet/control/planning.py`
- `planet/training/utility.py`
- `planet/scripts/configs.py`
- `planet/tools/preprocess.py`
- `planet/networks/conv_ha.py`

Important code-faithfulness points:

- `RSSM._posterior()` calls transition first, then combines `prior["belief"]`
  with the current image embedding.
- `features_from_state()` returns `concat([sample, belief])`.
- Free nats are `max(0, KL - free_nats)`, not `max(KL, free_nats)`.
- CEM samples actions, clips to `[-1, 1]`, selects top returns, and refits
  standard deviation using variance from `tf.nn.moments`; this differs from the
  mean-absolute-deviation formula printed in the primary algorithm.
- Released config defaults use reward loss scale 10, divergence scale 1,
  overshooting scale 0, 5-bit image preprocessing with uniform dequantization,
  state size 30, model size 200, and planner `H=12`, `J=1000`, `K=100`,
  `I=10`.

## Deliverable Fixes

`results/context.md` is now pre-method only and has exactly five `##` sections.
`results/reasoning.md` has no markdown headers and keeps the derivation in
first-person present tense. `results/answer.md` separates primary-source math
from canonical-code behavior, especially free nats, CEM refit, reward scaling,
and overshooting default.
