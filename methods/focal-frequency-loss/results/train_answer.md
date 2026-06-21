The problem is that image generators trained with spatial losses keep losing fine detail. A per-pixel L1 or L2 loss treats pixels as independent and, under reconstruction uncertainty, is minimized by averaging plausible high-frequency completions into a blur. Perceptual losses and adversarial losses improve sharpness indirectly, but none of them measures error where the gap actually lives: in the frequency spectrum. They have no explicit notion of which frequency bands are missing, so the structural high-frequency residual survives every spatial fix.

A plain frequency-domain distance is not enough either. If I simply take the squared Euclidean distance between the 2D FFT of the generated image and the target, every spectral component gets equal weight. Most components are already matched, especially the easy low-frequency ones, so their sheer number dominates the gradient. The few hard, mostly high-frequency components that actually account for the perceptual gap receive almost no attention. Equal weighting reproduces the same bias I am trying to fix.

The method I propose is Focal Frequency Loss (FFL). It is a frequency-domain reconstruction loss that adaptively concentrates on the spectral components the model currently gets wrong, while down-weighting the components that are already matched. It is designed as a drop-in complement to existing spatial losses, not a replacement, and it closes the frequency gap those losses leave behind.

The idea is to measure reconstruction error on the orthonormal 2D FFT spectrum. For each frequency component, I compute the squared Euclidean distance between the complex values of the predicted and target spectra, which is exactly the natural spectral distance. The crucial step is gating each component by a weight matrix that is large where the current frequency error is large. I set the weight to the error magnitude raised to a tunable power alpha, normalize it into the range [0, 1], and detach it from the gradient. With alpha equal to 1 the weight scales linearly with the error, so hard frequencies get a strong spotlight and easy frequencies are nearly ignored. Because the weight is recomputed every step from the current error, it forms a self-generated curriculum: focus on a hard frequency, and once it is matched its weight decays so the loss moves to the next hard one.

The weight must be detached. If gradients flowed through it, the optimizer could game the importance map and the focusing effect would be tangled with the optimization dynamics. With the weight frozen as a constant gate, the gradient is simply the ordinary frequency-distance gradient reweighted by the spotlight. I also normalize the weight matrix to keep the loss scale stable as absolute errors shrink during training, and I clamp it to [0, 1] and zero out NaNs so the map stays clean. For larger images, the loss can optionally split the image into a grid of patches and apply the same weighted spectral distance per patch, but a single global FFT is the default.

In code, I take the 2D FFT of the prediction and target, split real and imaginary parts, compute the per-component squared distance, build the detached weight matrix from the error magnitude, and return the weighted mean. The loss is used as total = spatial_loss + loss_weight * FFL.

```python
import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """Focal frequency loss: a spectral-distance loss whose frequency components
    are gated by how mismatched they currently are. Use as a complement to a
    spatial reconstruction loss: total = spatial + loss_weight * FFL."""

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
        patches = [x[:, :, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
                   for i in range(pf) for j in range(pf)]
        y = torch.stack(patches, 1)
        freq = torch.fft.fft2(y, norm='ortho')
        return torch.stack([freq.real, freq.imag], -1)

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            tmp = (recon_freq - real_freq) ** 2
            tmp = torch.sqrt(tmp[..., 0] + tmp[..., 1]) ** self.alpha
            if self.log_matrix:
                tmp = torch.log(tmp + 1.0)
            if self.batch_matrix:
                tmp = tmp / tmp.max()
            else:
                tmp = tmp / tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            tmp[torch.isnan(tmp)] = 0.0
            tmp = torch.clamp(tmp, min=0.0, max=1.0)
            weight_matrix = tmp.clone().detach()

        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        return torch.mean(weight_matrix * freq_distance)

    def forward(self, pred, target, matrix=None):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
```
