# VICReg Synthesis

This file is retained as a concise compatibility note. The strict source-grounded reconstruction for
this audit is in `notes/discovery_synthesis.md`; source provenance is in `notes/source_matrix.md`.

Key corrected points:

- The paper-form loss is
  `lambda * s_paper + mu * (v(x) + v(y)) + nu * (c(x) + c(y))`.
- The canonical PyTorch code uses `F.mse_loss`, which averages over all `B*d` entries, and computes
  `std_loss = mean(relu(1 - std_x))/2 + mean(relu(1 - std_y))/2`.
- The standard-deviation hinge amplifies small nonzero deviations compared with a variance hinge.
  With `epsilon=1e-4`, an exactly constant column still has zero embedding-gradient, so the precise
  claim is positive loss plus near-collapse gradient amplification, not a guaranteed nonzero gradient
  at exact equality.
- The official augmentation code is asymmetric BYOL-style blur/solarization: first view blur 1.0 and
  solarization 0.0; second view blur 0.1 and solarization 0.2.
