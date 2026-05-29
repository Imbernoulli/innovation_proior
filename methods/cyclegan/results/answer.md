# CycleGAN: Unpaired Image-to-Image Translation

## Problem

Learn to translate images between two domains X and Y given only two **unaligned** collections {x_i} and {y_j} — no information about which x corresponds to which y. Aligned pairs are expensive (segmentation), require an artist (stylization), or simply do not exist (horse→zebra has no ground-truth output).

## Key idea

An adversarial loss alone can make G(x) look like domain Y, but it only matches the output *distribution* — it cannot tie an individual input to a meaningful output (infinitely many maps induce the same output distribution; in practice the optimizer mode-collapses). Add structure by requiring the translation to be **cycle-consistent**: learn a forward map G: X→Y *and* an inverse F: Y→X simultaneously, and require that going there and back returns the original image, F(G(x)) ≈ x and G(F(y)) ≈ y. Recoverability rules out information-dropping collapse; it does not by itself rule out every invertible measure-preserving scramble, so the convolutional/residual architecture and low-distortion losses are part of the inductive bias that makes the learned map preserve content.

## Objective

Adversarial loss (per direction):

  L_GAN(G, D_Y, X, Y) = E_y[log D_Y(y)] + E_x[log(1 − D_Y(G(x)))]

For fixed G, the pointwise optimum of a·log d + b·log(1−d) is D_Y*(y)=p_data(y)/(p_data(y)+p_g(y)). Substituting D_Y* gives

  V(G,D_Y*) = −log4 + KL(p_data ‖ (p_data+p_g)/2) + KL(p_g ‖ (p_data+p_g)/2)
            = −log4 + 2·JSD(p_data ‖ p_g),

so the adversarial term matches the target marginal distribution and nothing more.

Cycle-consistency loss (L1, both directions):

  L_cyc(G, F) = E_x[‖F(G(x)) − x‖₁] + E_y[‖G(F(y)) − y‖₁]

L1 is used for reconstruction because pixelwise L2 targets the conditional mean and tends to average plausible images into blur; L1 targets the median and is less prone to washed-out reconstructions.

Full objective:

  L = L_GAN(G, D_Y, X, Y) + L_GAN(F, D_X, Y, X) + λ·L_cyc(G, F),
  G*, F* = argmin_{G,F} max_{D_X,D_Y} L,    λ = 10.

Optional identity loss (anchors color for painting↔photo): L_idt = E_y[‖G(y) − y‖₁] + E_x[‖F(x) − x‖₁], weight 0.5λ.

In practice each L_GAN is replaced by its **least-squares** form (non-saturating, sharper): D minimizes E_y[(D(y)−1)²] + E_x[D(G(x))²], G minimizes E_x[(D(G(x))−1)²].

## Architecture and training

- **Generator** (Johnson-style residual transformer): 7×7 conv → two stride-2 downsampling convs → 6 residual blocks (128px) or 9 (256px+) → two fractionally-strided upsampling convs → 7×7 conv → Tanh. Instance normalization, reflection padding. Residual blocks impose an identity prior (output = input + learned change).
- **Discriminator**: 70×70 PatchGAN, 4×4 convolutions with C64/C128/C256 at stride 2, C512 at stride 1, then a stride-1 output conv; LeakyReLU(0.2), instance norm except the first layer, fully convolutional, judges local texture realism.
- **Stabilization**: least-squares GAN loss; 50-image history buffer for discriminator updates; divide D objective by 2.
- **Optimization**: Adam, lr 2e-4, β₁=0.5, batch size 1, weights ~ N(0, 0.02), λ=10 (0.5λ identity). Constant lr for 100 epochs, then linear decay to 0 over 100 more.
- Interpretation: two coupled adversarial autoencoders F∘G: X→X and G∘F: Y→Y whose bottleneck is an image in the other domain.

## Code

```python
import torch
import torch.nn as nn
import itertools
import functools
import random


def use_conv_bias(norm_layer):
    if isinstance(norm_layer, functools.partial):
        return norm_layer.func == nn.InstanceNorm2d
    return norm_layer == nn.InstanceNorm2d


def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class ResnetBlock(nn.Module):
    """Residual block: out = x + r(x). Identity prior keeps scene structure."""
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """Downsample -> residual blocks -> upsample. Instance norm, reflection pad, Tanh."""
    def __init__(self, in_nc, out_nc, ngf=64, n_blocks=9,
                 norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)):
        super().__init__()
        use_bias = use_conv_bias(norm_layer)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_nc, ngf, 7, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_down = 2
        for i in range(n_down):
            m = 2 ** i
            model += [nn.Conv2d(ngf * m, ngf * m * 2, 3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * m * 2), nn.ReLU(True)]
        m = 2 ** n_down
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * m, norm_layer, use_bias)]
        for i in range(n_down):
            m = 2 ** (n_down - i)
            model += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(ngf * m // 2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN: C64-C128-C256-C512 -> 1-channel patch-score map."""
    def __init__(self, in_nc, ndf=64, n_layers=3,
                 norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)):
        super().__init__()
        use_bias = use_conv_bias(norm_layer)
        kw, padw = 4, 1
        seq = [nn.Conv2d(in_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        mult, mult_prev = 1, 1
        for n in range(1, n_layers):
            mult_prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * mult), nn.LeakyReLU(0.2, True)]
        mult_prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * mult, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    """Least-squares GAN loss: regress to 1 (real) / 0 (fake)."""
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, pred, target_is_real):
        target = pred.new_tensor(1.0 if target_is_real else 0.0).expand_as(pred)
        return self.loss(pred, target)


class ImagePool:
    """50-image history buffer to decorrelate discriminator updates."""
    def __init__(self, pool_size=50):
        self.pool_size, self.num, self.images = pool_size, 0, []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for image in images:
            image = image.detach().unsqueeze(0)
            if self.num < self.pool_size:
                self.num += 1
                self.images.append(image)
                out.append(image)
            elif random.uniform(0, 1) > 0.5:
                i = random.randint(0, self.pool_size - 1)
                out.append(self.images[i].clone())
                self.images[i] = image
            else:
                out.append(image)
        return torch.cat(out, 0)


class CycleGAN:
    # G_A = G (X->Y), G_B = F (Y->X), D_A = D_Y, D_B = D_X
    def __init__(self, in_nc=3, out_nc=3, lambda_cyc=10.0, lambda_idt=0.5, lr=2e-4):
        if lambda_idt > 0:
            assert in_nc == out_nc, "identity loss requires matching input/output channels"
        self.G_A = ResnetGenerator(in_nc, out_nc)
        self.G_B = ResnetGenerator(out_nc, in_nc)
        self.D_A = NLayerDiscriminator(out_nc)
        self.D_B = NLayerDiscriminator(in_nc)
        for net in (self.G_A, self.G_B, self.D_A, self.D_B):
            init_weights(net)
        self.fake_A_pool, self.fake_B_pool = ImagePool(), ImagePool()
        self.criterionGAN = GANLoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        self.lambda_cyc, self.lambda_idt = lambda_cyc, lambda_idt
        self.opt_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                      lr=lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                      lr=lr, betas=(0.5, 0.999))

    def forward(self, real_A, real_B):
        self.fake_B = self.G_A(real_A)       # G(x)
        self.rec_A = self.G_B(self.fake_B)   # F(G(x)) ~ x
        self.fake_A = self.G_B(real_B)       # F(y)
        self.rec_B = self.G_A(self.fake_A)   # G(F(y)) ~ y

    def backward_G(self, real_A, real_B):
        lam, lam_i = self.lambda_cyc, self.lambda_idt
        loss_idt = 0.0
        if lam_i > 0:
            loss_idt = (self.criterionIdt(self.G_A(real_B), real_B) * lam * lam_i +
                        self.criterionIdt(self.G_B(real_A), real_A) * lam * lam_i)
        loss_G = (self.criterionGAN(self.D_A(self.fake_B), True) +
                  self.criterionGAN(self.D_B(self.fake_A), True))
        loss_cyc = (self.criterionCycle(self.rec_A, real_A) * lam +
                    self.criterionCycle(self.rec_B, real_B) * lam)
        (loss_G + loss_cyc + loss_idt).backward()

    def backward_D(self, netD, real, fake):
        loss = (self.criterionGAN(netD(real), True) +
                self.criterionGAN(netD(fake.detach()), False)) * 0.5
        loss.backward()
        return loss

    def optimize(self, real_A, real_B):
        self.forward(real_A, real_B)
        for p in itertools.chain(self.D_A.parameters(), self.D_B.parameters()):
            p.requires_grad = False
        self.opt_G.zero_grad(); self.backward_G(real_A, real_B); self.opt_G.step()
        for p in itertools.chain(self.D_A.parameters(), self.D_B.parameters()):
            p.requires_grad = True
        self.opt_D.zero_grad()
        self.backward_D(self.D_A, real_B, self.fake_B_pool.query(self.fake_B))
        self.backward_D(self.D_B, real_A, self.fake_A_pool.query(self.fake_A))
        self.opt_D.step()
```
