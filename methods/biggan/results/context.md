# Context: high-fidelity class-conditional image synthesis with adversarial nets

## Research question

Generative adversarial networks can produce sharp images, but on a large, diverse, many-class dataset like ImageNet the samples remain far from real photographs in both *fidelity* (does each image look like a convincing instance of its class?) and *variety* (does the model cover the full diversity within and across classes?). The best class-conditional ImageNet generators of the time reach an Inception Score around 52 against roughly 233 for real data — a wide gap. Two facts make closing this gap hard.

First, unlike supervised learning, adversarial training does not obviously benefit from brute-force scaling. It is a two-player game whose only fixed point is a Nash equilibrium, not the minimum of a single loss; the dynamics are notoriously sensitive to architecture, optimizer, and capacity. It is unknown whether simply enlarging the networks and the batch will help or will destabilize training.

Second, when a generator *can* produce high-fidelity samples, it tends to do so only in a narrow region of its output space, and pushing for fidelity tends to cost variety. There is no clean, controllable knob to navigate that trade-off, and no characterization of *why* large-scale adversarial training becomes unstable.

The goal: produce diverse, high-fidelity, class-conditional samples at 128–512 px on ImageNet; find whatever architectural, regularization, and sampling changes make adversarial training survive at scale; and understand the instabilities that appear there.

## Background

**The adversarial game.** A generator G maps a latent z drawn from a fixed prior p(z) to images; a discriminator D scores images as real or generated. The original objective is the two-player minimax

  min_G max_D  E_{x∼q_data}[log D(x)] + E_{z∼p(z)}[log(1 − D(G(z)))].

The prior p(z) is conventionally N(0, I) or U[−1, 1], chosen by habit rather than derivation. G and D are convolutional networks in the DCGAN lineage (Radford et al. 2016). Without auxiliary stabilization this game is brittle and requires careful tuning to train at all.

**Stabilizing the game.** A large body of work attacks instability. One line changes the objective (Wasserstein GAN, least-squares GAN, the margin-based *hinge* objective of Lim & Ye 2017 and Tran et al. 2017). Another constrains D so it provides bounded, informative gradients everywhere. *Spectral Normalization* (Miyato et al. 2018) does this by dividing each weight matrix by its largest singular value σ₀(W), estimated cheaply with one step of power iteration reusing a persisted left singular vector; this bounds the Lipschitz constant of D and adaptively damps its top singular direction. Self-Attention GAN (Zhang et al. 2018) found that applying spectral normalization in G as well improves stability, allowing fewer D steps per iteration. Odena et al. (2018) observed that GAN performance correlates with the *conditioning* of G — the singular values of G's Jacobian — suggesting that the spectra of the weights are worth monitoring.

**Modeling global structure.** Stacks of small convolutions model local texture well but struggle with long-range spatial dependencies. The non-local / self-attention block (Wang et al. 2018) computes a softmax over pairwise feature similarities so that each location can attend to all others; SA-GAN inserts this block into both G and D.

**Conditioning on class.** In conditional GANs (Mirza & Osindero 2014) the class label can enter in several ways. AC-GAN (Odena et al. 2017) concatenates a one-hot class vector to z and adds an auxiliary classifier term to D's loss — but rewarding easily-classified samples tends to suppress variety. A cleaner route injects the class through normalization: conditional BatchNorm (Dumoulin et al. 2017; de Vries et al. 2017) and the more general FiLM layer (Perez et al. 2018) replace BatchNorm's single learned gain and bias with *per-conditioning* affine parameters γ(c), β(c) produced from a conditioning input, so that BN(h)·γ(c) + β(c) modulates features by class. On the discriminator side, the *projection* approach (Miyato & Koyama 2018) starts from the optimal-discriminator identity f(x,y) = log q(y|x)/p(y|x) + log q(x)/p(x); assuming the class posteriors are log-linear in a shared feature φ(x), the conditional term collapses to an inner product between a learned per-class embedding and φ(x), giving f(x,y) = yᵀVφ(x) + ψ(φ(x)) — class information added as evidence via a dot product rather than concatenated.

**Conditioning and smoothness of G.** Orthogonal initialization (Saxe et al. 2014) preserves signal norms through depth (dynamical isometry), easing training of deep nets. Orthogonal *regularization* (Brock et al. 2017) softly pushes weight matrices toward orthonormality with a penalty β‖WᵀW − I‖²_F, keeping the learned map well-conditioned and close to norm-preserving.

**Diagnostic phenomena known about adversarial training.** It is established that the discriminator must remain near-optimal with respect to the current generator to supply useful gradients; if it falls behind, training degrades. The margin-based hinge objective gives exactly zero gradient once an example is confidently on the correct side of the margin, while the original log objective's gradient rapidly vanishes in the same regime. Averaging the generator's weights over training (an exponential moving average, as used by Karras et al. 2018 and Yazici et al. 2018) yields smoother, higher-quality samples at evaluation time. Progressive growing (Karras et al. 2018) trains high-resolution single-class GANs by adding resolution stages over time.

## Baselines

- **DCGAN (Radford et al. 2016).** The convolutional G/D backbone: strided/transposed convolutions, BatchNorm, ReLU/LeakyReLU, weights initialized from N(0, 0.02 I). It made GAN training reproducible at low resolution but does not reach high-fidelity many-class synthesis and offers no class-conditioning mechanism beyond naive concatenation.

- **AC-GAN (Odena et al. 2017).** Conditions G by concatenating a one-hot class vector to z and adds an auxiliary-classifier loss to D. Gap: the auxiliary classifier rewards samples that are *easy to classify*, biasing the generator toward low-variety, prototypical images; concatenation is a weak channel for class information.

- **SN-GAN (Miyato et al. 2018).** A ResNet-style GAN with spectral normalization on D's weights and a hinge objective. Spectral normalization bounds D's Lipschitz constant — for the discriminator output f, normalizing each layer keeps ‖∇D‖ controlled, giving G a usable gradient everywhere. Gap: demonstrated at modest scale; class-conditional ImageNet quality is still far below real data, and behavior at large batch/large model is unexplored.

- **SA-GAN (Zhang et al. 2018).** The strongest prior class-conditional ImageNet model and the natural starting point. It is SN-GAN plus: the non-local self-attention block in G and D (for global structure), spectral normalization in *both* networks, the hinge objective, class-conditional BatchNorm in G, a projection discriminator, and a two-time-scale update rule (different learning rates for G and D). It reaches IS ≈ 52 / FID ≈ 18.6 on ImageNet at 128×128. Gaps it leaves open: a large absolute fidelity/variety gap to real data; no controllable fidelity/variety trade-off; small batch and modest model size; no account of what happens, or why training breaks, at scale.

- **ProGAN (Karras et al. 2018).** Trains high-resolution GANs by progressively growing resolution; excellent on constrained single-class datasets. Gap: a multi-scale training schedule rather than a single model trained directly, and demonstrated mainly in the single-class regime rather than diverse many-class ImageNet.

The hinge objective these baselines use is, for the discriminator,
  L_D = E_{x∼q_data}[max(0, 1 − D(x))] + E_{z}[max(0, 1 + D(G(z)))],
and for the generator L_G = −E_z[D(G(z))]. It pushes real scores above +1 and fake scores below −1, and contributes zero gradient once an example is on the correct side of its margin.

## Evaluation settings

- **Datasets.** ImageNet ILSVRC 2012, at 128×128, 256×256, and 512×512, preprocessed by cropping along the long edge and area-resampling to the target resolution; class-conditional generation over the 1000 classes. CIFAR-10 as a small-scale check. A much larger proprietary web-image corpus is available as a stress test of whether design choices transfer.

- **Metrics.** *Inception Score* (Salimans et al. 2016), IS = exp(E_x KL(p(y|x) ‖ p(y))) using a pretrained Inception classifier, rewarding confident per-image class posteriors (fidelity/"objectness") and a diverse marginal over classes, but *not* penalizing missing intra-class variety. *Fréchet Inception Distance* (Heusel et al. 2017), FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^{1/2}) on Inception pool features, which penalizes both poor fidelity and dropped variety. Both are known to be imperfect but are the standard yardsticks for comparison. For reference, real ImageNet data has IS ≈ 233 at 128×128.

- **Protocol.** Large-scale distributed training across many accelerator cores; evaluation by sampling many images and computing IS and FID against the dataset statistics. Generator weights may be averaged over training for evaluation.

## Code framework

The primitives below already exist: a spectrally-normalized convolution/linear layer, a self-attention block, a class-conditional normalization layer that produces affine parameters from a conditioning vector, a projection-style conditional discriminator output, the hinge losses, an EMA helper, and a latent/label sampler. The open slots are the scalable generator/discriminator wiring, the weight-conditioning penalty, the evaluation-time latent sampler, and the training step that ties them together.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- existing primitives -------------------------------------------------

class SNConv2d(nn.Conv2d):   # weight divided by top singular value (power iteration)
    ...
class SNLinear(nn.Linear):
    ...
class SNEmbedding(nn.Embedding):
    ...

class SelfAttention(nn.Module):   # non-local block: softmax over pairwise feature similarities
    def forward(self, x): ...

class ConditionalBN(nn.Module):
    """BatchNorm whose affine gain/bias are produced from a conditioning vector."""
    def __init__(self, num_features, cond_dim, which_linear):
        super().__init__()
        self.gain = which_linear(cond_dim, num_features)
        self.bias = which_linear(cond_dim, num_features)
        self.bn = nn.BatchNorm2d(num_features, affine=False)
    def forward(self, x, cond):
        gain = (1 + self.gain(cond)).view(x.size(0), -1, 1, 1)
        bias = self.bias(cond).view(x.size(0), -1, 1, 1)
        return self.bn(x) * gain + bias

def loss_hinge_dis(d_fake, d_real):
    return F.relu(1. - d_real).mean(), F.relu(1. + d_fake).mean()
def loss_hinge_gen(d_fake):
    return -d_fake.mean()

class EMA:                    # exponential moving average of G's weights for sampling
    def update(self): ...

def sample_z_y(batch, dim_z, n_classes, device):
    z = torch.randn(batch, dim_z, device=device)
    y = torch.randint(0, n_classes, (batch,), device=device)
    return z, y

# --- empty slots the method will fill ------------------------------------

class GBlock(nn.Module):
    """Upsampling residual block with conditional normalization. TODO."""
    def __init__(self, in_ch, out_ch, cond_dim, which_conv, which_bn):
        super().__init__()
        pass   # TODO: conditional-BN -> act -> upsample -> conv -> conditional-BN -> act -> conv (+ shortcut)
    def forward(self, x, cond):
        pass

class DBlock(nn.Module):
    """Downsampling residual block for the discriminator. TODO."""
    def __init__(self, in_ch, out_ch, which_conv, preactivation=True, downsample=True):
        super().__init__()
        pass
    def forward(self, x):
        pass

class Generator(nn.Module):
    """Class-conditional generator. TODO: how to inject class AND latent at every resolution,
       how wide/deep, how to keep the learned map smooth enough for controlled sampling."""
    def __init__(self, dim_z, n_classes, ch):
        super().__init__()
        pass
    def forward(self, z, y):
        pass

class Discriminator(nn.Module):
    """Class-conditional discriminator. TODO: residual down-blocks + attention, projection output."""
    def __init__(self, n_classes, ch):
        super().__init__()
        pass
    def forward(self, x, y):
        pass

def ortho(model, strength, blacklist=()):
    """TODO: a weight-conditioning penalty."""
    pass

def sample_latents_for_eval(batch, dim_z, knob):
    """TODO: an evaluation-time latent sampler."""
    pass

def train_step(G, D, GD, real_x, real_y, z, y, ema, cfg):
    # TODO: n D-steps (hinge L_D) then 1 G-step (hinge L_G); apply ortho; EMA update
    pass
```
