# Research question

We want to learn a probability distribution over high-dimensional data — concretely,
learn to generate natural images. The classical recipe is to posit a parametric density
family `(P_theta)` and maximize the log-likelihood of the data, which asymptotically
minimizes `KL(P_r || P_theta)`. But real image data lives on (or very near) a
low-dimensional manifold inside pixel space, and a model that maps a low-dimensional noise
variable through a network produces a distribution that also concentrates on a
low-dimensional set. Two such sets, in general position, intersect on a set of measure
zero — so the model density and the true density effectively do not overlap, and the KL
(and the likelihood) is undefined or infinite. The usual patch is to convolve the model
with Gaussian noise so a density exists everywhere; the required noise is large enough
(σ ≈ 0.1 per pixel on `[0,1]` images) to visibly blur the samples, and practitioners
quietly drop it when displaying results — a sign the patch is wrong for the problem.

The implicit-sample approach — fix a prior `z ~ p(z)`, push it through `g_theta` to get
`P_theta`, vary `theta` to bring `P_theta` close to `P_r` — sidesteps densities entirely
and can represent manifold-supported distributions. The open question is how to measure
"close": which distance `rho(P_theta, P_r)` should be the training loss? The choice is not
cosmetic. A distance defines which sequences of distributions converge, and the
training loss `theta -> rho(P_theta, P_r)` can only be descended by gradient methods if it is
continuous in `theta` — yet, as the diagnostics below record, the distances in current use go
flat or infinite exactly in the non-overlapping-support regime that the manifold geometry forces.
The problem is to find a training loss that survives that regime.

# Background

**Learning by an implicit sampler.** Rather than a density, define `g_theta: Z -> X`
(a neural net) and let `P_theta` be the law of `g_theta(Z)`, `Z ~ p(z)`. VAEs (Kingma &
Welling 2013) and GANs (Goodfellow et al. 2014) are the two well-known instances; VAEs
optimize an approximate likelihood and so inherit the noise-term problem, while GANs allow
a freely-chosen objective.

**Candidate distances between distributions on a compact metric space `X`.**
- Total Variation: `δ(P_r,P_g) = sup_A |P_r(A) − P_g(A)|`.
- Kullback–Leibler: `KL(P_r||P_g) = ∫ log(P_r/P_g) P_r dμ`, asymmetric, `+∞` when
  `P_g=0` but `P_r>0`.
- Jensen–Shannon: `JS(P_r,P_g) = ½KL(P_r||P_m) + ½KL(P_g||P_m)`, `P_m=(P_r+P_g)/2`,
  symmetric and always finite.
- Earth-Mover / Wasserstein-1: `W(P_r,P_g) = inf_{γ∈Π(P_r,P_g)} E_{(x,y)~γ}[||x−y||]`,
  where `Π` is the set of couplings with marginals `P_r,P_g`. `γ` is a transport plan;
  `W` is the minimal cost of moving mass to turn one distribution into the other.

A textbook fact from optimal transport (Villani 2009): `W` metrizes the weak* / weak
convergence (convergence in distribution) on the space of probability measures, whereas
TV induces the much stronger norm topology. Pinsker gives
`δ(P,Q) <= sqrt(KL(P||Q)/2)`, and with `M=(P+Q)/2` also
`δ(P,Q) <= sqrt(2 JS(P,Q))`; conversely `δ -> 0` makes `JS -> 0`. On a compact space of
diameter `B`, `W(P,Q) <= B·δ(P,Q)`. Thus KL convergence implies TV/JS convergence, TV and
JS induce the same topology, and either of them implies `W` convergence, while the reverse
implications are false in general.

**The motivating diagnostic finding (about existing GAN training).** For the GAN objective
the inner-optimal discriminator is `D* = P_r/(P_r+P_g)`, and substituting it back shows the
generator is minimizing `2·JS(P_r,P_g) − 2 log 2`. When the two supports are disjoint or
meet only on a measure-zero set — exactly the manifold situation above — three things are
provable and have been observed:
1. There exists a smooth, perfectly accurate discriminator with `∇_x D* = 0` on both
   supports; the discriminator can be driven to perfection.
2. In that regime `JS(P_r,P_g) = log 2` is constant, `KL` is `+∞` both ways, and total
   variation is `1`, so the JS loss is flat and its gradient is zero.
3. As the trained `D` approaches `D*` (`||D−D*|| < ε`), the generator gradient
   `||∇_theta E_z[log(1−D(g_theta(z)))]||` is bounded by `M·ε/(1−ε) -> 0` — it vanishes.
   This forces a delicate balance: a discriminator that is too good gives no signal, one
   that is too weak gives an inaccurate one.

The common workaround is the `−log D` generator step, `∇_theta E_z[−log D(g_theta(z))]`.
Its inner-optimal form equals `∇_theta[ KL(P_g||P_r) − 2 JS(P_g||P_r) ]`: the JS term
carries the wrong sign (pushing the distributions apart), and the KL is the *reverse* KL,
which charges almost nothing for dropping modes and a huge amount for atypical samples —
matching the observed mode-dropping of GANs. Worse, because `D*` is a singular density ratio
that fails to exist under disjoint supports, the gradient acquires exploding variance as `D`
sharpens, and the updates become unstable; empirically the generator gradient norms grow as
the discriminator is trained, and momentum-based optimizers were seen to make this worse.

**Why "use another f-divergence" does not help.** GAN objectives can be generalized to any
f-divergence (Nowozin et al. 2016, f-GAN), but every f-divergence is a functional of the
density ratio `dP_r/dP_g`, which is ill-defined or saturated precisely when the supports do
not overlap. Switching f-divergences does not change the topology and does not escape the
flat-loss / vanishing-gradient regime.

**Integral Probability Metrics.** A different family (Müller 1997):
`d_F(P,Q) = sup_{f∈F} E_P[f] − E_Q[f]`. Different function classes `F` give radically
different metrics: `F = {bounded in [−1,1]}` recovers twice total variation, the same
strong topology up to a constant; the unit ball of an RKHS gives Maximum Mean Discrepancy
(Gretton et al. 2012), which needs no auxiliary network
(kernel trick) but costs `O(samples^2)` and in high dimensions needs very large batches to
be a reliable statistic (Ramdas et al. 2014). Energy-based GANs (Zhao et al. 2016) can be
shown to optimize total variation at the optimal energy. Different choices of the function
class `F` thus yield IPMs that recover quite different distances and topologies.

# Baselines

**GAN (Goodfellow et al. 2014).** Minimax game `min_G max_D
E_{x~P_r}[log D(x)] + E_z[log(1 − D(g(z)))]`. At the inner optimum `D*=P_r/(P_r+P_g)`, the
generator minimizes `2·JS − 2 log 2`. The `−log D` variant is used in practice to avoid
early saturation. Gap: the JS objective is flat with zero/vanishing gradient on
non-overlapping supports, requires careful D/G capacity balancing, gives no loss curve that
tracks sample quality, and exhibits mode collapse.

**f-GAN (Nowozin et al. 2016).** Generalizes GAN to minimize any f-divergence via a
variational bound `sup_T E_{P_r}[T] − E_{P_g}[f*(T)]`. Gap: still a density-ratio quantity,
so it inherits the same disjoint-support pathology as JS.

**MMD / Generative Moment Matching Networks (Gretton et al. 2012; Li et al. 2015).** IPM
with `F` the unit ball of an RKHS; closed-form via the kernel trick, no auxiliary net to
train. Gap: `O(samples^2)` cost, needs large batches to be reliable in high dimensions, and
low-bandwidth kernels saturate similarly to TV — limited scalability for `64x64` images.

**Energy-based GAN (Zhao et al. 2016).** Discriminator is a non-negative energy; generator
pushes its energy down. At the optimal energy its signal is tied to total variation, the
same strong topology obtained by bounded-score IPMs. Gap: TV has the same strong topology
as JS, so EBGAN cannot be trained to discriminator optimality without the same saturation.

**DCGAN (Radford et al. 2015).** Not a new objective but the standard architecture: a
transpose-convolution generator and strided-convolution discriminator with batch
normalization and ReLU/LeakyReLU, trained with the ordinary GAN loss. It is the de facto
backbone and the natural body to reuse for any new GAN objective.

# Evaluation settings

The natural yardstick is unconditional image generation. The standard dataset is
LSUN-Bedrooms (Yu et al. 2015), natural indoor-bedroom photographs; CIFAR-10 and image
folders are also standard. Images are center-cropped/resized to `64x64x3` and normalized to
`[−1,1]`. The prior is a `100`-dimensional Gaussian. The reference architecture is DCGAN
(transpose-conv generator, strided-conv discriminator), with secondary configurations that
stress any new objective: a generator without batch normalization and constant filter count,
and a 4-layer ReLU MLP with 512 hidden units. Comparison is against the standard GAN procedure
using the `−log D` trick. Diagnostic quantities of interest are training-loss curves and
their correlation with visual sample quality, sensitivity to generator/discriminator
balancing, and the presence or absence of mode collapse.

# Code framework

The primitives are a data pipeline, a noise prior, generic network modules, optimizers, and
an adversarial training loop. The open slots are the scalar objective, any admissibility
constraint on the scoring network, and the inner/outer update schedule.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Data pipeline: images -> [-1,1], 64x64
def make_dataloader(root, image_size=64, batch_size=64):
    ds = dset.ImageFolder(root, transform=transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

# Noise prior
def sample_noise(batch_size, nz=100):
    return torch.randn(batch_size, nz, 1, 1)

# Generator body: a DCGAN-style transpose-conv stack
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        # TODO: fill with the existing transpose-conv sampler body
        pass
    def forward(self, z):
        # TODO: map noise to an image sample
        pass

# The second network scores a sample. Its output type and any constraint on
# the represented function are the open design questions.
class ScoringNetwork(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        # TODO: fill with a convolutional scoring body and output head
        pass
    def forward(self, x):
        # TODO: return the score used by the training objective
        pass

# Objective and constraint slots.
def scoring_network_loss(net, real, fake):
    # TODO: loss for updating the scoring network
    pass

def generator_loss(net, fake):
    # TODO: loss for updating the generator through the scoring network
    pass

def enforce_constraint(net):
    # TODO: keep the scoring function inside the admissible class
    pass

# Training loop skeleton; the inner/outer schedule is a knob.
def train(generator, scoring_net, dataloader, nz=100, n_inner=1):
    opt_g = optim.SGD(generator.parameters(), lr=1e-4)      # TODO: which optimizer
    opt_n = optim.SGD(scoring_net.parameters(), lr=1e-4)    # TODO: which optimizer
    for real in dataloader:
        # TODO: how many scoring-net steps per generator step?
        for _ in range(n_inner):
            enforce_constraint(scoring_net)
            z = sample_noise(real.size(0), nz)
            fake = generator(z).detach()
            loss_n = scoring_network_loss(scoring_net, real, fake)
            opt_n.zero_grad(); (loss_n).backward(); opt_n.step()

        z = sample_noise(real.size(0), nz)
        fake = generator(z)
        loss_g = generator_loss(scoring_net, fake)
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()
```
