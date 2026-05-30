# pix2pix: Image-to-Image Translation with Conditional Adversarial Networks

## Problem

Map an input image to a corresponding output image of the same scene in a different representation (labels↔photo, edges→photo, grayscale→color, map↔aerial, day→night, thermal→color, inpainting). One general-purpose recipe — same architecture and objective — that works across all of these by simply retraining on different paired data, with no per-task loss engineering.

## Key idea

Per-pixel regression blurs: the L2-optimal output under a multimodal `p(y|x)` is the conditional mean, an average of plausible images. So learn the loss adversarially. A conditional GAN's discriminator sees the (input, output) *pair*, so it enforces not just realism but **correspondence to the input**. Add an L1 term to anchor the output to the ground truth; L1 (conditional median) blurs less than L2 (conditional mean). The L1 term already captures the **low frequencies**, so the discriminator only needs to police **high-frequency** detail — which is local — so it can be a **PatchGAN** that classifies each `N × N` patch and averages (a learned local texture/MRF loss: fewer parameters, faster, applies to any image size). The generator is a **U-Net**: an encoder-decoder with skip connections so aligned low-level structure bypasses the bottleneck. Noise is supplied only as **dropout** (explicit noise input gets ignored).

## Objective

Conditional GAN term — discriminator sees both `x` and the output:

```
L_cGAN(G, D) = E_{x,y}[ log D(x, y) ] + E_{x,z}[ log(1 − D(x, G(x, z))) ]
```

L1 reconstruction term:

```
L_L1(G) = E_{x,y,z}[ ‖ y − G(x, z) ‖_1 ]
```

Full objective:

```
G* = arg min_G max_D  L_cGAN(G, D) + λ · L_L1(G),     λ = 100
```

Training: alternate one step on `D`, one on `G`; non-saturating generator update (train `G` to maximize `log D(x, G(x,z))`); halve the `D` objective to slow `D`; Adam, lr `2e-4`, `β1=0.5`, `β2=0.999`; weights `~ N(0, 0.02²)`; random jitter (`256→286→` crop `256`) and mirroring; dropout kept on at test time, batchnorm uses test-batch statistics (instance norm at batch size 1).

## Architecture

Modules: `Ck` = Conv(4×4, stride 2)-BatchNorm-ReLU with k filters; `CDk` adds Dropout 0.5. No BatchNorm on the first `C64`. LeakyReLU(0.2) in the encoder and discriminator; plain ReLU in the decoder.

- **U-Net generator** (256×256): encoder `C64-C128-C256-C512-C512-C512-C512-C512` to a 1×1 bottleneck; U-Net decoder `CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128`, then a conv to the output channels and Tanh. Skip connections concatenate encoder layer `i` onto decoder layer `n−i` (this doubles the decoder input channels vs. a plain encoder-decoder).
- **70×70 PatchGAN discriminator**: `C64-C128-C256-C512`, then a conv to a 1-channel patch-score map, then sigmoid. Input channels = `input_nc + output_nc` (it receives the concatenation `[x, y]`). Receptive field is set by depth: `1×1` PixelGAN (`C64-C128` with 1×1 convs), `16×16`, `70×70`, `286×286` ImageGAN.

## Code

```python
import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    """One down/up level of the U-Net. forward() concatenates the block input onto
    its output (the skip), letting low-level structure bypass the bottleneck."""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=False)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=True):
        super().__init__()
        block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=None,
                                        norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=block,
                                            norm_layer=norm_layer, use_dropout=use_dropout)
        block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf,   ngf*2, submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                             submodule=block, outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN: classifies each NxN patch of [x, y]; output is a map of patch scores."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        kw, padw = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            seq += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, 2, padw, bias=False),
                    norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        seq += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, 1, padw, bias=False),
                norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf*nf_mult, 1, kw, 1, padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer("real", torch.tensor(real_label))
        self.register_buffer("fake", torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, pred, target_is_real):
        target = (self.real if target_is_real else self.fake).expand_as(pred)
        return self.loss(pred, target)


def optimize(G, D, criterionGAN, criterionL1, optG, optD, real_A, real_B, lambda_L1=100.0):
    fake_B = G(real_A)                                       # noise = dropout inside G

    # update D on conditional pairs [x, y]
    optD.zero_grad()
    pred_fake = D(torch.cat((real_A, fake_B), 1).detach())
    pred_real = D(torch.cat((real_A, real_B), 1))
    loss_D = (criterionGAN(pred_fake, False) + criterionGAN(pred_real, True)) * 0.5
    loss_D.backward(); optD.step()

    # update G: fool D on the pair + L1 anchor
    optG.zero_grad()
    pred_fake = D(torch.cat((real_A, fake_B), 1))
    loss_G = criterionGAN(pred_fake, True) + criterionL1(fake_B, real_B) * lambda_L1
    loss_G.backward(); optG.step()


# Setup: conditional D sees input_nc + output_nc channels.
# G  = UnetGenerator(input_nc, output_nc, num_downs=8, use_dropout=True)
# D  = NLayerDiscriminator(input_nc + output_nc, n_layers=3)   # 70x70 PatchGAN
# optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
# optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
```

## Why the design works

- **Conditioning the discriminator on `x`** turns "looks like a real image" into "is a real (input, output) pair," so the loss penalizes input/output mismatch. An unconditional `D` lets `G` collapse to producing the same realistic output regardless of input.
- **L1 over L2**: L2's optimum is the conditional mean (averages modes → blur); L1's is the conditional median (a single representative → sharper), and L1 is more robust.
- **Low/high-frequency split**: L1 already captures low frequencies, so the discriminator only needs to enforce high-frequency correctness. High-frequency structure is local → a patch discriminator suffices. A PatchGAN models the image as a Markov random field (independence beyond a patch diameter), has fewer parameters, runs convolutionally, and applies to arbitrarily large images.
- **U-Net skips**: input and output share aligned low-level structure; skip connections shuttle it directly across, instead of forcing it through the bottleneck.
- **Dropout as noise**: an explicit Gaussian `z` input gets ignored because the conditioning is nearly deterministic; dropout injects stochasticity that cannot be trivially routed around.
