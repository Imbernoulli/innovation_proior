# Sources — focal-frequency-loss

Track D_single_source. Before this pass, `notes/`, `refs/`, and `src/` did not exist at all for
this method; the trace had nothing checkable behind it. Fetched the primary and the official code
repo below; no independent self-account / blog / interview material was found (see "searched, not
found" at the bottom).

## type=primary

- **path**: `methods/focal-frequency-loss/refs/primary/ffl_2012.12821.pdf` (+ `.txt`, pdftotext -layout)
- **source**: arXiv 2012.12821v3, "Focal Frequency Loss for Image Reconstruction and Synthesis" (ICCV 2021), fetched from https://arxiv.org/pdf/2012.12821
- **what it supplies**: ground truth for the `w(u,v) = |F_r(u,v) - F_f(u,v)|^alpha` weight matrix
  (Eq. 9) and the full FFL definition (Eq. 10), and the paper's own statement of the detach, Section
  3.3, immediately before Eq. 10:
  > "We further normalize the matrix values into the range [0, 1], where the weight 1 corresponds to
  > the currently most lost frequency, and the easy frequencies are down-weighted. The gradient
  > through the spectrum weight matrix is locked, so it only serves as the weight for each frequency."
- **verified**: the weight formula, the Hadamard-product loss (Eq. 10), and the detach are exactly as
  reasoning.md/answer.md describe. The paper states the detach as a design fact but does not derive
  the `(alpha+2)/2` gradient-inflation consequence of *not* detaching — that derivation is not in the
  primary; I re-derived and numerically cross-checked it myself (see changelog.md) against both a 1D
  scalar proxy and the actual 2-real-parameter complex form `(a_f, b_f)`; both give exactly 1.5 at
  alpha=1 via autograd.

## type=official-code

- **path**: `methods/focal-frequency-loss/refs/primary/focal_frequency_loss.py`
- **source**: official repo github.com/EndlessSora/focal-frequency-loss, file
  `focal_frequency_loss/focal_frequency_loss.py`, fetched from
  https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/focal_frequency_loss/focal_frequency_loss.py
- **what it supplies**: the actual detach mechanism, confirming the paper's "locked" gradient is
  implemented as a literal `.detach()` on the weight tensor, not just described:
  ```python
  matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
  matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
  weight_matrix = matrix_tmp.clone().detach()
  ```
  (lines 80-82). No code comment explains *why* beyond what's in the paper (no comment on the gradient
  ratio); the derivation of the constant-factor consequence in reasoning.md is this trace's own
  addition, not sourced from either the paper or the code.
- **verified**: `tensor2freq` (patch cropping -> `torch.fft.fft2(..., norm='ortho')` -> stacked
  real/imag) and `loss_formulation` (weight = `sqrt(dist^2)^alpha`, per-image or batch max
  normalization, NaN-zero, clamp to [0,1], detach, Hadamard product, mean) in reasoning.md's code
  block match this file's `tensor2freq`/`loss_formulation` structurally line-for-line (renamed
  locals aside); reasoning.md's landing code was already an accurate re-derivation of the real
  implementation.

## type=self_account

- **path**: `methods/focal-frequency-loss/refs/self_accounts/README.md`
- **source**: same repo, root `README.md`, fetched from
  https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/README.md
- **what it supplies**: confirms the official repo, links (project page / poster / slides), and one
  hyperparameter-tuning note: "a larger alpha indicates that the model is more focused." Does not
  contain any additional rationale for the detach beyond the paper/code above — checked and ruled
  out as a richer source.

## Searched, not found

- `SELF_ACCOUNT_SOURCES.md` at repo root: grepped for "focal frequency loss", "Jiang", "Liming
  Jiang" — no hits.
- Web search for author blog posts / interviews / OpenReview discussion specific to this paper's
  design choices: nothing beyond the paper, its poster/slides (not fetched — same content as the
  paper, no additional rationale expected), and the official code repo above.
