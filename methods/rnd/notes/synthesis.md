# RND Synthesis

This file is retained as a compatibility pointer for older method directories. The strict audit
evidence and reconstruction notes are now in:

- `notes/source_matrix.md`
- `notes/discovery_synthesis.md`
- `refs/primary/`
- `refs/ancestors/`
- `refs/explainers/`
- `refs/self_accounts/`
- `code/openai-random-network-distillation/`

Key strict-audit corrections relative to the previous draft:

- The canonical code uses feature-wise MSE mean for the intrinsic reward and predictor loss, not a
  summed squared norm.
- The code trains two value heads with separate value losses, then combines weighted advantages
  \(A=1\cdot A_I+2\cdot A_E\).
- Intrinsic GAE ignores dones; extrinsic GAE uses dones.
- Intrinsic rewards are normalized by the running standard deviation of discounted intrinsic
  returns.
- Predictor/target observation normalization is one-frame whitening plus clip to `[-5,5]`, separate
  from the policy's four-frame `x/255` input.
- For 128 environments, predictor updates use a 25 percent Bernoulli keep mask.
