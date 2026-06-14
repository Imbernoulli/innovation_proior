# Focal Frequency Loss (FFL), distilled

Focal Frequency Loss (Jiang, Dai, Wu & Loy, ICCV 2021; arXiv:2012.12821) is a frequency-domain training
loss for image reconstruction and synthesis. It measures the distance between the 2D spectra of the
generated and target images and **adaptively concentrates on the frequency components that are currently
hard to synthesize**, down-weighting the easy ones. It is a drop-in *complement* to spatial losses
(`L1`/`L2`, perceptual, adversarial), shown to improve VAE, pix2pix, and SPADE in perceptual quality and
quantitative metrics.

## Problem it solves

Neural image generators trained under spatial losses exhibit a *spectral bias*: they fit low-frequency
content fast and lose high-frequency detail, because per-pixel error pays for the bulk pixel energy
(low frequencies) and barely penalizes the small-energy high-frequency residual that carries perceptual
detail. Spatial perceptual/adversarial terms patch this indirectly but have no explicit per-frequency
notion of what is being lost. A plain (unweighted) spectral distance fails too: equal weighting lets the
many already-matched easy frequencies dominate the gradient, so the few hard frequencies that account
for the gap get almost no attention.

## Key idea

Define the loss on the 2D DFT spectrum and **gate each frequency component by how mismatched it
currently is**. (1) Take the orthonormal 2D FFT of both images, splitting each spectrum into real/imag.
(2) Per component, the frequency distance is the squared Euclidean distance between complex values,
`|F_r(u,v) − F_f(u,v)|²`. (3) The spectrum weight matrix is `w(u,v) = |F_r(u,v) − F_f(u,v)|^α`,
normalized to `[0,1]` and **detached** (stop-gradient) so it is a fixed importance spotlight, not a
variable the optimizer can game. (4) The loss is the weighted average `FFL = mean(w · freq_distance)`.
Because `w` is computed fresh each step from the current error, it forms a self-generated curriculum:
focus on a hard frequency, and once it is matched its weight decays and the loss moves to the next.

## Final form

For real spectrum `F_r` and fake spectrum `F_f` over `M·N` components (per channel, per image):

```
FFL = (1 / MN) Σ_u Σ_v  w(u,v) · |F_r(u,v) − F_f(u,v)|²,
      w(u,v) = clamp( normalize( |F_r(u,v) − F_f(u,v)|^α ), 0, 1 ),   w detached.
```

`α = 0` recovers the unweighted spectral distance; `α = 1` (default) makes the weight scale linearly
with the current error magnitude, concentrating gradient on the hardest components.

## Why each choice

- **Frequency domain, not pixels** — spatial loss has a frequency bias that ignores high-frequency
  detail; the spectrum makes that structure explicit and measurable.
- **Squared complex distance** — exact image match ⇔ exact spectrum match, so spectral distance is a
  legitimate reconstruction objective; orthonormal FFT keeps its scale comparable to a spatial loss.
- **`|F_r − F_f|^α` weight** — "hard" = currently mismatched; weighting by the current error focuses
  gradient on the bands that account for the gap and is dynamic (decays as they are matched).
- **Detached weight** — if gradients flowed through `w`, the optimizer could game the importance map; as
  a constant gate the gradient is exactly `w · ∇(distance²)`, the clean reweighted spectral gradient.
- **Normalize to `[0,1]` (per-image default; optional log/batch)** — keeps the loss scale stable as
  absolute errors shrink and prevents a single outlier frequency from pinning the spotlight.
- **Complement, not replacement** — FFL constrains the spectrum but not absolute pixel values, so it
  rides on top of a spatial reconstruction term: `total = spatial + loss_weight · FFL`.

## Defaults

`loss_weight = 1.0`, `alpha = 1.0`, `patch_factor = 1` (single global FFT; `>1` = patch-wise spectra for
larger images), `ave_spectrum = False`, `log_matrix = False`, `batch_matrix = False`.

## Working code

```python
import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1,
                 ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        pf = self.patch_factor
        _, _, h, w = x.shape
        assert h % pf == 0 and w % pf == 0, "patch_factor must divide H and W"
        ph, pw = h // pf, w // pf
        patches = [x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                   for i in range(pf) for j in range(pf)]
        y = torch.stack(patches, 1)
        freq = torch.fft.fft2(y, norm='ortho')                # orthonormal 2D FFT
        return torch.stack([freq.real, freq.imag], -1)

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            tmp = (recon_freq - real_freq) ** 2
            tmp = torch.sqrt(tmp[..., 0] + tmp[..., 1]) ** self.alpha   # |F_r - F_f|^alpha
            if self.log_matrix:
                tmp = torch.log(tmp + 1.0)
            if self.batch_matrix:
                tmp = tmp / tmp.max()
            else:
                tmp = tmp / tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            tmp[torch.isnan(tmp)] = 0.0
            weight_matrix = torch.clamp(tmp, min=0.0, max=1.0).clone().detach()  # spotlight, detached

        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]             # |F_r - F_f|^2
        return torch.mean(weight_matrix * freq_distance)

    def forward(self, pred, target, matrix=None):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
```
