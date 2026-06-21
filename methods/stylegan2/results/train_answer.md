I propose the canonical method name **StyleGAN2**. It denotes a redesigned style-based generator that removes the characteristic artifacts of its predecessor while preserving, and even improving, the controllability that made style-based generation attractive.

The starting point is StyleGAN, which already produced compelling high-resolution images, but two families of artifacts kept betraying the outputs. The first is a stereotyped water-droplet blob that appears in essentially every intermediate feature map from 64×64 upward; the second is a location-locked or phase artifact, where details such as teeth or eyes stick to preferred pixel positions and jump discretely rather than gliding smoothly. A successful redesign has to eliminate both without sacrificing style mixing, and it should do so without making training appreciably slower.

The blob is the key diagnostic. It is too consistent to be noise, and the discriminator ought to be able to penalize it if it were merely a defect. That the blob persists, and that removing it sometimes severely corrupts the image, suggests the generator relies on it. The mechanism that creates it is adaptive instance normalization, AdaIN. AdaIN normalizes each feature map independently by its own mean and standard deviation, computed from the actual contents of that map, and then re-applies a style-driven scale and bias. The normalization is what makes the style scale dominate each map and thereby enables style mixing. But because the divisor is data-dependent, the generator can game it: by planting a single dominant spike in a feature map, it makes the measured standard deviation essentially the size of that spike, so dividing by it rescales the rest of the map by a chosen factor. The spike becomes a private channel for relative feature magnitudes, and that spike is the blob.

Removing normalization entirely does make the blob vanish, but it also removes the per-sample scale control that style mixing requires; later layers receive activations at wildly inconsistent magnitudes and cannot be controlled by feeding different styles to different layers. The right move is therefore to keep the effect of normalization while removing its data-dependence. Instead of measuring the standard deviation of each actual output map, I predict the standard deviation that the output map would have under a unit-variance-input assumption, and I do so analytically from the weights and the style.

The derivation is straightforward. Modulation multiplies input feature map i by the style scalar s_i. I can fold that scaling into the convolution weights: w'_{ijk} = s_i · w_{ijk}, where i is an input map, j is an output map, and k indexes the kernel footprint. If the inputs are independent with unit variance, the variance of output map j is the sum of squared weights feeding it, so the predicted standard deviation is σ_j = sqrt( Σ_{i,k} (w'_{ijk})² ). Demodulation simply divides by that predicted value, w''_{ijk} = w'_{ijk} / sqrt( Σ_{i,k} (w'_{ijk})² + ε ). The whole style block, modulation, convolution, and normalization, collapses into a single convolution with per-sample adjusted weights. Because the divisor is a deterministic function of the style and the weights, there is no measured content statistic for the generator to game, so the blob loses its purpose and disappears. At the same time, the style scale s is still fully present in w', so scale-specific control and style mixing remain intact.

A few details keep the system calibrated. The to-RGB output layers should be modulated by the style but not demodulated, because they produce an image rather than features feeding another layer. Bias and noise are moved outside the style block so their effect does not depend on the current style magnitude. The activation function is scaled so that leaky ReLU preserves unit variance, which keeps the demodulation assumptions valid as the signal flows deeper. Because the effective weights are different for every sample, I run them through grouped convolution, reshaping the batch dimension into groups so that the convolution sees one sample per group; the reshape is a view and adds no copies.

The second major addition is path-length regularization. Standard metrics such as FID and precision/recall have a known blind spot: two generators can score identically while one looks clearly better to humans. The difference tends to track perceptual path length, which measures how much the generated image changes under a small latent step. Generators with lower PPL generally look better. The intuition is that without a smoothness pressure, the adversarial objective can improve average quality by squeezing bad images into tiny regions of latent space and stretching good regions; those squeezed regions create violent changes in the latent-to-image map, which degrades overall quality and makes inversion unreliable.

I therefore regularize the mapping so that a fixed-size step in latent space produces a fixed-magnitude image change in every direction. Let g be the generator and J_w its Jacobian at w. For a random image-direction y drawn from a standard normal, the vector-Jacobian product J_w^T y is obtained by back-propagating through the scalar g(w)·y. The regularizer is the squared deviation of the length of J_w^T y from a target a, averaged over w and y. The target a is not fixed by hand; it is an exponential moving average of the observed lengths, so the optimizer equalizes the spectrum around whatever scale the network already has rather than fighting an arbitrary scale from initialization. In high dimensions this prior is minimized when the Jacobian is orthogonal up to a global scale, meaning all singular values are equal and the map is a local isometry. Straight latent interpolations then follow geodesics on the image manifold, and inversion by optimizing a single latent code becomes far more reliable.

Both the discriminator's R1 gradient penalty and the new path-length term change slowly, so I evaluate them lazily rather than every iteration. The path-length term runs once every eight generator steps, and R1 once every sixteen discriminator steps. To keep the optimizer state consistent, I adjust the hyperparameters with c = k/(k+1): the learning rate becomes c·λ, the Adam momenta become β_1^c and β_2^c, and the regularizer is multiplied by k so its accumulated gradient magnitude matches what it would have been if applied every step. The path-length term can also be computed on a fraction of the minibatch to save memory, since it is only a regularizer.

The remaining architectural change is to abandon progressive growing, which is the source of the phase artifacts. Progressive growing is good at enforcing a coarse-to-fine schedule, but because each resolution momentarily serves as the output resolution during fade-in, it is pushed to emit maximal-frequency detail at that stage. That leaves the intermediate layers with excessive high frequencies and breaks shift invariance. I keep the coarse-to-fine behavior by using a skip generator: every resolution block has its own to-RGB layer, and the final image is the sum of upsampled per-resolution RGB contributions. The discriminator is made residual, with residual merges scaled by 1/√2 to cancel the variance doubling of adding two paths. A sweep confirms that skip connections in the generator improve perceptual path length, while a residual discriminator improves FID, so the final design is skip generator plus residual discriminator with no progressive growing. Because the skip generator exposes how much each resolution actually contributes, I noticed that the highest resolutions were under-used; doubling the number of feature maps from 64×64 up to 1024×4 fixes that capacity shortfall and improves both FID and recall.

Putting these pieces together gives the StyleGAN2 generator: weight demodulation removes the blob while keeping style control; path-length regularization smooths the latent-to-image map and improves perceived quality and invertibility; lazy regularization keeps the computation affordable; and the skip generator with residual discriminator removes phase artifacts without progressive growing. The result is a method that fixes both artifact families and retains the signature capability of controlling images by feeding different styles to different layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# A tiny, self-contained illustration of the two core StyleGAN2 ideas:
# (1) weight demodulation, and (2) path-length regularization via VJPs.

class TinyDemodulatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        # Shared convolution weight w[i, o, k, k]; PyTorch uses (out, in, k, k).
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        # Affine style transform: latent -> per-input-map scale.
        self.style = nn.Linear(16, in_ch)

    def forward(self, x, latent):
        # x: (B, in_ch, H, W); latent: (B, 16)
        B = x.shape[0]
        # Style scale s_i for each input map; init around 1 by adding 1.
        s = self.style(latent) + 1.0                 # (B, in_ch)
        # Modulate: w'_ijk = s_i * w_ijk
        w = self.weight.unsqueeze(0)                 # (1, out, in, k, k)
        s = s.reshape(B, 1, -1, 1, 1)                # (B, 1, in, 1, 1)
        w_prime = w * s                              # (B, out, in, k, k)
        # Demodulate: divide each output map by predicted std under unit-variance input.
        sigma = torch.sqrt(torch.sum(w_prime ** 2, dim=[2, 3, 4], keepdim=True) + 1e-8)
        w_dmod = w_prime / sigma                     # (B, out, in, k, k)
        # Run as grouped convolution: one group per sample.
        x_ = x.reshape(1, B * x.shape[1], x.shape[2], x.shape[3])
        w_ = w_dmod.reshape(B * w_dmod.shape[1], w_dmod.shape[2],
                            w_dmod.shape[3], w_dmod.shape[4])
        out = F.conv2d(x_, w_, padding='same', groups=B)
        return out.reshape(B, -1, out.shape[2], out.shape[3])

class TinyGenerator(nn.Module):
    def __init__(self, latent=16, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(latent, hidden * 4 * 4)
        self.conv1 = TinyDemodulatedConv(hidden, hidden, kernel=3)
        self.conv2 = TinyDemodulatedConv(hidden, 3, kernel=3)

    def forward(self, z):
        x = F.relu(self.fc1(z)).view(z.shape[0], -1, 4, 4)
        x = F.relu(self.conv1(x, z))
        x = self.conv2(x, z)
        return x

# 1. Demonstrate weight demodulation on a single forward pass.
g = TinyGenerator()
z = torch.randn(2, 16, requires_grad=True)
img = g(z)
print("Generated shape:", img.shape)

# 2. Demonstrate path-length regularization: J^T y via one backward pass.
y = torch.randn_like(img) / np.sqrt(np.prod(img.shape[2:]))
dot = torch.sum(img * y)
grad_z = torch.autograd.grad(dot, z, create_graph=True)[0]
length = torch.sqrt(torch.mean(torch.sum(grad_z ** 2, dim=1)))
# Dynamic target a as an EMA of observed lengths; here we just print the current value.
print("Path-length estimate:", length.item())

# 3. A tiny loss that nudges the mapping toward constant path length.
target = 0.5
pl_loss = (length - target) ** 2
pl_loss.backward()
print("Path-length regularizer:", pl_loss.item())
```
