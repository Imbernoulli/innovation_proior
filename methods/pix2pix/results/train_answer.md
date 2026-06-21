A large fraction of problems in image processing, graphics, and vision share one shape: take an input image $x$ and produce a corresponding output image $y$ of the same scene in a different representation — a semantic label map turned into a photo, an edge drawing into a photo, a grayscale frame into color, a daytime street into night, a thermal scan into RGB, a holed image into an inpainted one. Structurally these are the same task: predict pixels from pixels. Yet each has historically been attacked with its own special-purpose pipeline and, more painfully, its own hand-designed loss. Convolutional nets already give us a general tool for learning the *mapping*; what we lack is a general *loss*. We still have to tell the net, by hand, what "good" means for each task, and that is precisely the engineering I want to eliminate.

The default loss is per-pixel Euclidean distance, $\mathbb{E}_{x,y}\big[\lVert y - G(x)\rVert_2^2\big]$, and it reliably produces mush. The reason matters because it drives everything. For a given $x$ there are usually many plausible outputs — many photos consistent with one label map, many colorings consistent with one grayscale image — so the conditional distribution $p(y\mid x)$ is multimodal. The minimizer of $\mathbb{E}_{y\sim p(y\mid x)}\big[\lVert y - \hat y\rVert_2^2\big]$ over $\hat y$ is the conditional mean $\hat y = \mathbb{E}[y\mid x]$. The net, doing its job perfectly, outputs the *average* of all plausible sharp images; average a red car and a blue car pixelwise and you get a brownish smear. The blur is the L2-optimal answer to an ill-posed question, not an optimizer failure. Underneath sits a second flaw: per-pixel loss treats every output pixel as conditionally independent given $x$, so it is *unstructured* — it cannot say "these pixels together fail to form a coherent boundary." Swapping to L1, $\mathbb{E}\big[\lVert y - G(x)\rVert_1\big]$, helps because a separable absolute loss is minimized by a conditional *median* rather than a mean, so it is less pulled by outlying modes and blurs less — but "less blurry" is not "sharp," and it still washes out color when several colors are plausible. Hand-built structured losses (CRFs, SSIM, deep feature matching, Gram-matrix texture statistics) each bake in one fixed notion of structure and are task-specific expert work — exactly what we are trying to escape.

I propose pix2pix: a single conditional-adversarial recipe — one architecture, one objective — that a practitioner can point at any paired image-to-image dataset and train, with no per-task loss engineering. The central move is to stop writing the loss down and instead *learn* it. Stand up a discriminator $D$ trained to tell real data from $G$'s output, and train $G$ to fool it. Now $D$ is a learned loss, and a blurry image is *easy* for $D$ to flag as fake because real images are not blurry — so the moment $G$ produces mush, $D$ punishes it, pushing $G$ onto the real-image manifold where outputs are sharp by construction. The adversarial loss lets $G$ commit to a single sharp mode instead of averaging modes, curing the L2/L1 failure at its root. In practice the generator's half of the plain minimax game saturates early — when $D$ confidently rejects, $\log(1 - D(G(z)))$ is flat and $G$ gets no gradient exactly when it is doing worst — so I use the non-saturating form and train $G$ to maximize $\log D(G(z))$ instead, which shares the same fixed point but gives a strong gradient when $G$ is losing.

A plain GAN's $D$, however, sees only the output $y$: it asks "does this look like a real image?" and never "does this image correspond to *this* input $x$?" For translation that is fatal — $G$ could emit a perfectly realistic photo unrelated to the input label map and $D$ would be satisfied, so $G$ drifts toward a few realistic outputs that always pass and ignores $x$ entirely. Rather than patch this with an external regression term while leaving $D$ blind, I fix the loss itself: let $D$ see the *pair* $(x, y)$. Then $D$ classifies real $(\text{input}, \text{output})$ pairs against fakes, a realistic-but-mismatched photo is a *fake pair*, and $D$ learns to reject it — the discriminator polices correspondence, not just realism. With $G$ a map from input and noise to output, the conditional objective is

$$\mathcal{L}_{\mathrm{cGAN}}(G, D) = \mathbb{E}_{x,y}\big[\log D(x, y)\big] + \mathbb{E}_{x,z}\big[\log\big(1 - D(x, G(x, z))\big)\big],$$

with $G^* = \arg\min_G \max_D \mathcal{L}_{\mathrm{cGAN}}(G, D)$. The single change from the plain GAN — feeding $x$ into $D$ — converts "make something realistic" into "make something realistic *that goes with this input*."

The cGAN alone produces sharp outputs but wanders: the adversarial term only constrains the output to be *a* plausible matching image, nothing pins it near *the* ground-truth target, and GAN training is twitchy enough to invent plausible-but-wrong detail. So I add a reconstruction term to anchor the output — and I use L1, not L2, because L2's optimum is the conditional mean (heavy blur) while separable L1 is minimized by a conditional median and blurs less:

$$\mathcal{L}_{\mathrm{L1}}(G) = \mathbb{E}_{x,y,z}\big[\lVert y - G(x, z)\rVert_1\big].$$

The full objective is the adversarial term plus a weighted L1 term,

$$G^* = \arg\min_G \max_D \ \mathcal{L}_{\mathrm{cGAN}}(G, D) + \lambda\, \mathcal{L}_{\mathrm{L1}}(G),$$

with $\lambda \approx 100$, large because the per-pixel L1 magnitude is small and must actually anchor the result. There is a clean division of labor on color too: L1's per-pixel optimum under color uncertainty is the median, which trends desaturated, while the adversarial term learns that gray outputs are unrealistic and pushes the color distribution back toward the truth — L1 keeps color near the target, the GAN keeps it vivid.

The objective then reorganizes the discriminator. L1 is bad at high frequencies — that is the blur — but it is *fine* at low frequencies: the coarse layout, the big regions, the rough color all come out roughly right. So L1 already owns the low frequencies, and the *only* place the adversarial loss must add value is the high frequencies, the crisp edges and textures L1 smears. High-frequency structure is *local*: you can judge whether a small patch of texture is sharp or blurry from a small neighborhood alone, without seeing the whole image. So I restrict $D$ to classify $N \times N$ patches — a PatchGAN — running a fully convolutional $D$ whose every output unit has an $N \times N$ receptive field and scores one patch, then averaging the per-patch verdicts into the real/fake signal. By only ever looking within an $N \times N$ window, $D$ effectively models the image as a Markov random field (independence beyond a patch diameter), the same assumption behind classical texture and style models; it is, in effect, a learned local texture loss. The payoffs are concrete: far fewer parameters than a full-image classifier, faster, and — being fully convolutional over fixed-size patches — applicable to images of any size. Reasoning about the extremes fixes $N$: at $1\times 1$ (a PixelGAN) $D$ can only push the marginal color distribution toward vivid values, fixing grayness but not blur; at the full image (an ImageGAN) $D$ becomes a big deep network burdened with the long-range structure L1 already handles, harder to train and no better. A moderate patch around $70 \times 70$ — large enough to capture crisp local structure, small enough to stay light — is the sweet spot, with the receptive field set purely by depth.

For the generator I exploit the defining property of these tasks: input and output are renderings of the *same* scene, so their structure is roughly aligned — prominent edges sit in the same places, building outlines coincide. A standard encoder-decoder downsamples to a tiny bottleneck and upsamples back, forcing *all* information, including the low-level structure the input and output literally share, through the bottleneck to be rebuilt from a compressed code — throwing away detail handed to us for free. So I give that information a way around: skip connections joining encoder layer $i$ directly to decoder layer $n - i$ (with $n$ the total number of layers) by concatenating the encoder activations onto the decoder activations at that level. Low-level, high-resolution structure shuttles straight across at full detail while the bottleneck carries only the *transformation*. Because input and output are spatially aligned, the layer-$i$-to-layer-$(n-i)$ wiring delivers the encoder's detail to exactly the decoder stage reconstructing that same scale. This U-Net shape is the single biggest generator-side quality lever, and it helps even under a pure L1 loss by giving the decoder direct access to input detail. Finally the noise: feeding $z$ as an extra Gaussian input channel is pointless here, because the conditioning is so informative that the loss-minimizing map is nearly deterministic and $G$ simply learns to ignore the channel, which then carries no gradient and dies. So I inject stochasticity where it cannot be routed around — as dropout in several decoder layers, kept on at *test* time so it actually perturbs outputs. Even this yields only minor variation; capturing the full conditional entropy is a harder open problem, but functionally dropout is the noise and explicit $z$ is out.

The remaining details follow the DCGAN backbone: everything is built from convolution–BatchNorm–ReLU modules with $4\times 4$ filters at stride 2 so each conv halves (encoder/discriminator) or doubles (decoder) the spatial size; LeakyReLU (slope $0.2$) in the encoder and discriminator, plain ReLU in the decoder; no BatchNorm on the first $\texttt{C64}$; the first few decoder blocks carry dropout $0.5$; a final conv to the output channels followed by Tanh maps outputs into $[-1, 1]$. For a $256\times 256$ input the encoder halves eight times to a $1\times 1$ bottleneck, $\texttt{C64-C128-C256-C512-C512-C512-C512-C512}$, and the U-Net decoder, whose channels double because each block receives the upsampled features plus the skipped encoder features, is $\texttt{CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128}$. The $70\times 70$ PatchGAN is $\texttt{C64-C128-C256-C512}$ then a conv to a one-channel patch-score map, taking $\texttt{input\_nc}+\texttt{output\_nc}$ channels at its first layer since it receives the concatenation $[x, y]$. Optimization alternates one $D$ step then one $G$ step with the non-saturating $G$ update; I halve the $D$ objective when stepping $D$ so the discriminator does not run away from the generator; Adam with learning rate $2\times 10^{-4}$, $\beta_1 = 0.5$, $\beta_2 = 0.999$; weights initialized from $\mathcal{N}(0, 0.02^2)$; random jitter ($256\to 286$, crop back to $256$) and horizontal mirroring. At inference I keep dropout on and run BatchNorm on the test-batch statistics, which at batch size 1 is exactly instance normalization. One subtlety: BatchNorm on a $1\times 1$-spatial, batch-1 bottleneck zeroes it, effectively cutting the innermost layer — fine for the U-Net since the skips route around the bottleneck, but a plain encoder-decoder needs batch size greater than 1 to train.

The objective is self-consistent by construction. The L1 term supplies a good coarse, low-frequency anchor to the true target but cannot commit to sharp detail or vivid color; the cGAN term, with $D$ seeing $(x,y)$ and restricted to $N\times N$ patches, supplies exactly the high-frequency, sharp, colorful local detail L1 misses, conditioned on the input so it cannot be satisfied by an unmatched image. Dropping either term leaves a predictable hole — L1-only gives blur, cGAN-only gives sharp-but-sometimes-wrong-and-artifacty, unconditional-GAN-only gives realistic-but-uncorrelated-with-input. The full method is $\min_G \max_D \mathcal{L}_{\mathrm{cGAN}}(G,D) + \lambda\, \mathcal{L}_{\mathrm{L1}}(G)$ with a U-Net generator and a PatchGAN discriminator.

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
    """PatchGAN: classifies each NxN patch of [x, y]; output is a map of logits."""
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
        seq += [nn.Conv2d(ndf*nf_mult, 1, kw, 1, padw)]  # no sigmoid before BCEWithLogits
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
    fake_AB = torch.cat((real_A, fake_B.detach()), 1)         # detach G output only
    real_AB = torch.cat((real_A, real_B), 1)
    pred_fake = D(fake_AB)
    pred_real = D(real_AB)
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
