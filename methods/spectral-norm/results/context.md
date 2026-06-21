# Context

## Research question

Generative adversarial networks train a generator `G` only through the signal produced by a discriminator `D`. In the standard game,

    min_G max_D V(G,D),
    V(G,D) = E_{x~q_data}[log D(x)] + E_{x'~p_G}[log(1 - D(x'))],

the discriminator is not just an evaluator; it is the generator's teacher. The question is how to restrict the discriminator function class so that `D` cannot become arbitrarily sharp, while preserving enough capacity to use many features.

## Background

**Optimal discriminator and derivative pressure.** For a fixed generator in the original GAN objective,

    D*_G(x) = q_data(x) / (q_data(x) + p_G(x)) = sigmoid(f*(x)),
    f*(x) = log q_data(x) - log p_G(x).

The derivative of the log-density-ratio form is

    grad_x f*(x) = grad_x q_data(x) / q_data(x) - grad_x p_G(x) / p_G(x),

which can be unbounded or ill-defined near regions where one density is small. This points to the discriminator's input sensitivity, not merely its classification accuracy, as the quantity to control.

**Lipschitz continuity.** A function is `K`-Lipschitz when

    ||f(x) - f(x')|| <= K ||x - x'||

for all inputs under the Euclidean norm. Searching over a `K`-Lipschitz discriminator family is a common way to keep the discriminator from changing too abruptly:

    argmax_{||f||_Lip <= K} V(G,D).

For differentiable maps on the relevant domain, the local stretching is governed by the spectral norm of the Jacobian:

    ||g||_Lip = sup_h sigma(grad g(h)),

where `sigma(A)` is the largest singular value,

    sigma(A) = max_{h != 0} ||A h||_2 / ||h||_2.

For a linear layer `g(h) = W h`, the Jacobian is `W`, so its Lipschitz constant is `sigma(W)`. Lipschitz constants are submultiplicative under composition, and common discriminator activations such as ReLU and leaky ReLU with slope at most one are 1-Lipschitz. Thus a feed-forward discriminator's overall bound can be controlled through the stretching factors of its linear and convolutional layers.

**Power iteration.** The largest singular value and its leading singular vectors can be estimated without a full SVD. Starting with a nonzero vector `u`, repeated updates

    v <- W^T u / ||W^T u||_2,
    u <- W v / ||W v||_2

converge to the leading right and left singular vectors when the top singular value is isolated and the initialization has a nonzero top-component. Then `u^T W v` estimates `sigma(W)`. Each iteration costs two matrix-vector multiplies.

## Baselines

**Weight clipping in Wasserstein GANs.** WGAN uses the Kantorovich-Rubinstein dual form of Wasserstein-1 distance, which requires a 1-Lipschitz critic. Its simple enforcement mechanism clips every weight entry into a fixed box after each update, `w <- clip(w, -c, c)`.

**Gradient penalty.** WGAN-GP replaces clipping with a soft penalty,

    lambda E_{x_hat}[(||grad_{x_hat} D(x_hat)||_2 - 1)^2],
    x_hat = epsilon x + (1 - epsilon) x_tilde,

usually with `lambda = 10`. This directly regularizes input gradients at sampled interpolation points.

**Weight normalization and Frobenius normalization.** Weight normalization reparametrizes each row/vector as `w = g v / ||v||_2`. For a pure row-normalized comparison with the learned scales removed, each row has unit norm, so the sum of squared singular values is fixed:

    sum_t sigma_t(W)^2 = tr(W W^T) = d_out.

Frobenius normalization fixes the same type of budget to one.

**Orthonormal regularization.** Adding `||W^T W - I||_F^2` pushes all singular values toward one.

**Spectral norm regularization.** A prior supervised-learning regularizer penalizes large `sigma(W)` terms in the loss, estimated by power iteration.

## Evaluation settings

The natural test bed is image generation with convolutional discriminators and generators: CIFAR-10 at `32 x 32`, STL-10 at `48 x 48`, and ImageNet/ILSVRC2012 downsampled to `128 x 128` for conditional generation. Architectures include a standard CNN/DCGAN-style stack and a ResNet-style discriminator/generator.

The relevant comparisons are weight clipping, gradient penalty on WGAN or the standard GAN loss, batch normalization, layer normalization, weight normalization with learned multipliers removed for the Lipschitz comparison, and orthonormal regularization. Optimizers are Adam under several choices of learning rate, momentum parameters, and number of discriminator updates per generator update, including aggressive high-learning-rate settings. Quality metrics include inception score, FID, and diversity/mode-collapse diagnostics such as intra-class MS-SSIM.

## Code framework

```python
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim_z=128, ch=512):
        super().__init__()
        # Standard GAN generator: dense projection, upsampling/deconv blocks,
        # batch normalization in G, and a final image-valued output.
        ...

    def forward(self, z):
        ...


def constrain_weight(module, name="weight"):
    # Open slot: attach a cheap, per-forward transform to each discriminator
    # weight so the layer's input-output stretching is bounded without removing
    # the layer's ability to use many singular directions.
    raise NotImplementedError


class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            constrain_weight(nn.Conv2d(3, ch, 3, 1, 1)),
            nn.LeakyReLU(0.1),
            constrain_weight(nn.Conv2d(ch, ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            constrain_weight(nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            constrain_weight(nn.Conv2d(ch * 4, ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )
        self.fc = constrain_weight(nn.Linear(ch * 8 * 4 * 4, 1))

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def train_step(D, G, opt_d, opt_g, x_real, z, n_dis=1):
    for _ in range(n_dis):
        opt_d.zero_grad()
        d_loss(D, G, x_real, z).backward()
        opt_d.step()
    opt_g.zero_grad()
    g_loss(D, G, z).backward()
    opt_g.step()
```
