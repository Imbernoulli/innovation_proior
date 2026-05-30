Let me start from the thing that keeps biting me, which is the loss, not the architecture. I have a whole pile of image-to-image problems that are, structurally, the same problem: I'm handed an input image `x` and I want to produce an output image `y` of the same scene in some other representation — a label map turned into a photo, an edge drawing turned into a photo, a grayscale frame turned into color, a daytime street turned into night. Every one of these has been solved separately, with its own hand-tuned pipeline and its own bespoke loss. And the irony is that for the *mapping* I already have a general tool — a convolutional net learns it fine. What I don't have is a general *loss*. I still have to tell the net, by hand, what "good" means for each task. So let me chase that: can I avoid hand-designing the loss?

The obvious default loss is per-pixel Euclidean distance: train `G` to minimize `E_{x,y}[‖y − G(x)‖_2^2]`. I keep reaching for it and it keeps producing mush. Let me be precise about why, because the reason is going to drive everything. For a given input `x`, there are usually many plausible outputs — many photos consistent with one label map, many colorings consistent with one grayscale image. Call that conditional distribution `p(y|x)`. What does squared error actually ask `G` to output? The minimizer of `E_{y~p(y|x)}[‖y − ŷ‖_2^2]` over `ŷ` is the conditional mean `ŷ = E[y|x]`. So the net, doing its job perfectly, outputs the *average* of all the plausible sharp images. Average a red car and a blue car pixelwise and you get a brownish smear; average ten edge placements and you get a soft gray band where the edge should be. The blur isn't a bug in the optimizer — it's the L2-optimal answer to an ill-posed question. And there's a second problem hiding underneath: per-pixel loss treats every output pixel as independent given `x`. It has no way to say "these pixels together don't form a coherent boundary." It's an unstructured loss.

Could I just swap L2 for L1, `E[‖y − G(x)‖_1]`? It helps a little and it's worth understanding why. The minimizer of expected L1 error is the conditional *median*, not the mean. A median picks a representative single value rather than blending; under a bimodal `p(y|x)` the median sits at one of the modes (or between, but it doesn't smear the way the mean does), and it's robust to outliers. So L1 blurs *less* than L2. But "less blurry" is not "sharp." The median of many plausible high-frequency patterns is still soft — L1 still can't hallucinate a crisp edge it isn't sure the exact location of, and it'll still wash out color when several colors are plausible (it'll pick the median color, which trends gray). So L1 is a better default but not the answer.

People have built structured losses by hand — CRFs, SSIM, feature-matching in a fixed network, Gram-matrix statistics for texture. Each one bakes in a particular idea of what structure to preserve, and each is task-specific expert work. That's exactly the hand-engineering I'm trying to escape. I don't want to pick the structure; I want the structure to be learned from the data.

So flip the whole thing around. Instead of *me* writing down a distance that says "blurry is bad," what if I *train* a function to say it? That's the adversarial idea. Stand up a second network `D` whose job is to look at an image and decide: did this come from the real data, or did `G` make it? Train `D` to be as good as possible at that classification, and train `G` to fool it. Now `D` is a learned loss. And crucially, a blurry image is *easy* for `D` to flag as fake, because real photos aren't blurry — so as soon as `G` produces mush, `D` punishes it, and `G` is pushed toward outputs that lie on the real-image manifold, which are sharp by construction. The adversarial loss lets `G` commit to a single sharp mode instead of averaging modes. That's precisely the failure of L2/L1 cured at the root. The unconditional game is

```
min_G max_D  E_y[log D(y)] + E_z[log(1 − D(G(z)))].
```

I should note the practical wrinkle right away: that second term saturates. Early on `D` confidently calls `G`'s output fake, `D(G(z)) ≈ 0`, and `log(1 − D(G(z)))` is flat there — `G` gets almost no gradient exactly when it's doing worst. The standard repair is to train `G` to *maximize* `log D(G(z))` instead; same fixed point, but a strong gradient when `G` is losing. I'll carry that non-saturating form along.

Now, there's a hole in just dropping a GAN onto this. A plain GAN's `D` only sees the output `y`. It asks "does this look like a real photo?" — it does *not* ask "does this photo correspond to *this* input `x`?" For a translation task that's a disaster. `G` could produce a perfectly realistic photo that has nothing to do with the input label map, and `D` would be happy. In fact I'd expect `G` to drift toward generating one or a few realistic outputs that always pass, ignoring `x` entirely — the input has no teeth in the loss. People have patched this before by keeping `D` unconditional and bolting on an L2 term to tie output to input, but then the *correspondence* is enforced only by the regression term, and `D` is doing nothing about it.

Let me fix the loss itself instead of patching around it. The whole point of an adversarial loss is that `D` learns what I want. So let `D` see the *pair* `(x, y)`. Now `D` is classifying not "is this a real photo?" but "is this a real (input, output) pair?" — a real edge-map-and-its-photo, versus an edge map paired with whatever `G` cobbled up. A realistic photo that doesn't match the edges is now a *fake pair*, and `D` can learn to reject it. The discriminator is judging correspondence, not just realism. That's the conditional GAN, and writing it out, with `G` now a map from input and noise to output:

```
L_cGAN(G, D) = E_{x,y}[ log D(x, y) ]
             + E_{x,z}[ log( 1 − D(x, G(x, z)) ) ].
```

`G` minimizes this against a `D` that maximizes it: `G* = arg min_G max_D L_cGAN(G, D)`. The single change from the plain GAN — feeding `x` into `D` — is what turns "make something realistic" into "make something realistic *that goes with this input*." (I'll keep the unconditional `L_GAN`, with `D(y)` and `D(G(x,z))`, around as a control so I can later check that conditioning `D` actually matters and isn't just decoration.)

Is the cGAN alone enough? When I imagine training it, I expect sharp outputs — `D` kills blur — but I also worry about it wandering. The adversarial loss only constrains the output to be *a* plausible matching image; nothing pins it close to *the* ground-truth target, and GAN training is twitchy, so I'd expect occasional structural artifacts where `G` invents plausible-but-wrong detail. The earlier hand-built systems found it useful to mix in a traditional regression term to anchor the output, and I think they were right — but I'll mix in L1, not L2. I already worked out why: L2's optimum is the conditional mean (heavy blur), L1's is the median (a single representative, less blur). I want the regression term to gently pull `G`'s output toward the ground truth without itself reintroducing a lot of blur, and L1 does that better. So:

```
L_L1(G) = E_{x,y,z}[ ‖ y − G(x, z) ‖_1 ].
```

And the full objective is the adversarial term plus a weighted L1 term,

```
G* = arg min_G max_D  L_cGAN(G, D) + λ · L_L1(G),
```

with `λ` large — around 100 — because the L1 magnitude per pixel is small and I want it to actually anchor the result. There's a nice side effect I notice while thinking about color: L1's per-pixel optimum under color uncertainty is the median color, which trends toward a desaturated gray; the adversarial term, by contrast, can learn that gray outputs are unrealistic and push the output color distribution back toward the true one. So the two terms split labor on color too — L1 keeps it near the truth, the GAN keeps it vivid.

Now the question that reorganizes the architecture. I've got a learned adversarial loss *and* an L1 term. Am I asking `D` to do too much? Let me look at what each term is actually good at across the frequency spectrum. L1 (and L2) are bad at high frequencies — that's the blur, the fine crisp detail they can't commit to. But are they bad at *low* frequencies? No. The coarse layout, the big regions, the rough color — L1 gets those roughly right; the blur is a high-frequency phenomenon, not a where-is-the-building phenomenon. So the L1 term is already doing a fine job on the low-frequency content of the image.

That means I don't need `D` to police the whole image. The low frequencies are handled. The *only* place the adversarial loss needs to add value is the high frequencies — the crisp edges and textures that L1 smears. So restrict `D`'s job to exactly that: enforce high-frequency correctness, and let L1 own the low frequencies. This division is going to let me make `D` much smaller and simpler than a full-image classifier.

How small? High-frequency structure is *local*. Whether a patch of texture looks real, whether an edge is crisp, whether a small region is internally coherent — you can judge all of that by looking at a small neighborhood; you don't need to see the whole image to know a 30×30 patch of "photo" is sharp and a 30×30 patch of "blur" is not. The long-range stuff — does this region belong here, is the global layout right — is exactly what L1 already covers. So let `D` classify *patches*: have it look at each `N × N` patch of the (input, output) and decide real-or-fake at that scale, then average its verdict over all patches. Concretely I run a fully convolutional `D` over the image; each output unit has an `N × N` receptive field and scores one patch, and I average the responses into the final real/fake signal. I'll call it a PatchGAN.

This has an interpretation I like. By only ever looking within an `N × N` window and assuming patches farther apart than that don't need to be jointly judged, `D` is effectively modeling the image as a Markov random field — independence beyond a patch diameter. That's the same assumption that underlies classical texture and style models, where realistic *local* statistics, tiled, give realistic texture. So my discriminator is, in effect, a learned texture/style loss, and the cGAN+L1 split is "L1 for global structure, learned texture loss for local detail." And the practical payoffs are concrete: a patch discriminator has far fewer parameters than a full-image one, it runs faster, and — because it's fully convolutional and only ever looks at fixed-size patches — I can apply it to images of *any* size, even larger than I trained on. It also means I don't burden `D` with modeling long-range dependencies it would struggle to learn; I've offloaded that to L1.

What should `N` be? This is a real knob and I should reason about the extremes. Shrink it all the way to `1 × 1` — a "PixelGAN" that judges each pixel's color independently. That can't say anything about spatial sharpness (a single pixel has no spatial extent), but it *can* push the marginal color distribution toward realistic, vivid values — so I'd expect it to fix the grayness without fixing the blur. Grow `N` to a small patch like `16 × 16` and now `D` can see local spatial structure: outputs get sharp. Push it to the full image (an "ImageGAN") and in principle `D` sees everything — but now `D` is a big deep network with many parameters and a hard training problem, and it's being asked to also model the long-range structure that L1 already handles, so I'd expect it to be harder to train and not actually better. So I expect a sweet spot at a moderate patch — large enough to capture crisp local structure, small enough to stay light and trainable. I'll center on something like `70 × 70` and keep the others as a study.

Now the generator. The defining feature of these problems: a high-resolution input grid maps to a high-resolution output grid, and the two are renderings of the *same* scene, so their structure is roughly *aligned* — the prominent edges in a grayscale image sit at the same places as in its color version; the building outlines in a label map sit where the building is in the photo. The standard image-prediction network is an encoder-decoder: downsample through convs to a tiny bottleneck, then upsample back. But stare at what that does. Every bit of information — including all that low-level structure the input and output literally share — is forced to squeeze through the bottleneck and be reconstructed on the far side from a compressed code. That's throwing away something I'm being handed for free: the location of edges in the input is the location of edges in the output, and I'm making the net rediscover it through a pinhole.

So give the information a way around the bottleneck. Add skip connections: connect encoder layer `i` directly to decoder layer `n − i` (with `n` the total number of layers) by concatenating the encoder activations onto the decoder activations at that level. Low-level, high-resolution structure shuttles straight across the net at full detail; the bottleneck only has to carry the *transformation*, not re-transmit the shared structure. This is the U-Net shape. Because input and output are spatially aligned, the layer-`i`-to-layer-`n−i` correspondence is exactly the right wiring — the skip delivers the encoder's spatial detail to the decoder stage that's reconstructing that same spatial scale. I expect this to matter a lot: a plain encoder-decoder should struggle to produce realistic detail precisely because it can't pass the detail through, and adding the skips should be the single biggest quality lever on the generator side — and not specific to the adversarial setting either, since even under a pure L1 loss the skips should help by giving the decoder direct access to input detail.

There's the noise `z` to deal with. A GAN is supposed to be a generative model — `z` is what makes outputs stochastic. But here `x` already determines `y` almost completely, and if I feed `z` as an extra Gaussian input channel alongside `x`, I expect `G` to just learn to ignore it: the conditioning is so informative that the loss is minimized by a near-deterministic map, the noise channel carries no useful gradient, and it dies. So providing `z` as an input is largely pointless. If I want any stochasticity at all, I should inject it where it can't be trivially routed around — as dropout in the generator, applied at several decoder layers, and kept on at *test* time too so it actually perturbs outputs. I'll be honest that even this yields only minor output variation; capturing the full conditional entropy is a harder open problem. But functionally, dropout is my noise, and `z` as an explicit input is out.

Let me settle the remaining architectural details, taking the DCGAN recipe as the backbone: build everything from convolution–BatchNorm–ReLU modules, use strided convolutions to down/upsample instead of pooling. I'll use `4 × 4` filters at stride 2 throughout, so each conv halves (encoder/discriminator) or doubles (decoder) the spatial size. Leaky ReLUs (slope 0.2) in the encoder and discriminator — the downsampling/critic path — and ordinary ReLUs in the decoder. The very first layer (the first `C64`) gets no BatchNorm, since there's nothing to normalize meaningfully at the input. The decoder's first few blocks use dropout (that's my noise). After the last decoder layer, one more conv maps to the output channel count (3 for color, 2 for the ab-channels in colorization) followed by a Tanh to put outputs in `[−1, 1]`. The discriminator ends with a conv to a 1-channel map (one score per patch) and a sigmoid, and — because it's conditional — it takes `input_channels + output_channels` channels at its first layer, since I feed it the concatenation `[x, y]`.

Spelling out the generator depth for a `256 × 256` input: the encoder halves the resolution eight times,

```
encoder:  C64 - C128 - C256 - C512 - C512 - C512 - C512 - C512
```

(no BatchNorm on the first `C64`), down to a `1 × 1` bottleneck, and the decoder mirrors it back up. As a plain encoder-decoder the decoder is

```
decoder:  CD512 - CD512 - CD512 - C512 - C256 - C128 - C64
```

where `CDk` is a conv-BN-Dropout-ReLU block (dropout 0.5) — the first three decoder blocks carry dropout. To make it a U-Net I add the skip from encoder layer `i` to decoder layer `n − i` by concatenation, which *doubles* the channel count entering each decoder block (the upsampled features plus the skipped encoder features), so the U-Net decoder widths become

```
U-Net decoder:  CD512 - CD1024 - CD1024 - C1024 - C1024 - C512 - C256 - C128
```

For the discriminator, the `70 × 70` PatchGAN is `C64 - C128 - C256 - C512` then a conv to 1 channel and a sigmoid (no BatchNorm on the first `C64`, leaky ReLU 0.2 everywhere). I can dial the receptive field by changing the depth: a `1 × 1` PixelGAN is just `C64 - C128` with `1 × 1` convs; `16 × 16` is `C64 - C128`; the full `286 × 286` ImageGAN is `C64 - C128 - C256 - C512 - C512 - C512`. Depth sets the receptive field, which sets the patch size.

Optimization follows the standard GAN protocol: alternate one gradient step on `D`, then one on `G`. Use the non-saturating generator update — train `G` to maximize `log D(x, G(x,z))`. One tweak I want: divide the `D` objective by 2 when stepping `D`, which slows `D`'s learning relative to `G` so the discriminator doesn't run away from the generator. Adam, learning rate `2 × 10^{-4}`, `β1 = 0.5`, `β2 = 0.999` — the low `β1` is the usual GAN stabilizer. Initialize weights from `N(0, 0.02^2)`, train from scratch. For augmentation, random jitter — upsize `256 → 286` and random-crop back to `256` — plus horizontal mirroring. At inference I do something slightly unusual: I keep dropout *on* (it's my noise source), and I run BatchNorm using the statistics of the test batch rather than accumulated training statistics; with batch size 1 that's exactly instance normalization, which behaves well for image synthesis. There's a subtlety with batch size 1 and the bottleneck: BatchNorm on a `1 × 1`-spatial, batch-1 activation zeroes it, effectively cutting the innermost layer — fine for the U-Net because the skips route around the bottleneck anyway, but it means a plain encoder-decoder needs a batch size greater than 1 to train at all.

Let me check the objective is actually self-consistent by re-deriving the split one more time, because the whole design hinges on it. The full loss is `min_G max_D L_cGAN(G,D) + λ L_L1(G)`. The L1 term `λ E‖y − G(x,z)‖_1` is minimized, per pixel, by the conditional median — it nails the *coarse, low-frequency* content (right region, right rough color) but cannot commit to sharp high-frequency detail or vivid color, because the median of many plausible fine patterns is soft and the median of many plausible colors is gray. The `L_cGAN` term has `D` see `(x, y)`, so it scores correspondence-and-realism; restricting `D` to `N × N` patches and averaging makes it a high-frequency / local-texture critic, since fine detail is local while global structure is already covered by L1. So `D` supplies exactly the high-frequency, sharp, colorful detail L1 misses, conditioned on the input so it can't be satisfied by an unmatched but realistic image; L1 supplies the low-frequency anchoring to the true target. The two are complementary by construction, and dropping either one should leave a predictable hole — L1-only gives blur, cGAN-only gives sharp-but-sometimes-wrong-and-artifacty, and the unconditional GAN-only gives realistic-but-uncorrelated-with-input (since `D` can't see `x`, correspondence has nothing enforcing it). That's the whole method.

Now the code, grounded in a standard implementation. The generator is built recursively from a skip-connection block — innermost first, then wrapped outward — so each block downsamples, hands off to its inner submodule, upsamples, and (unless it's the outermost block) concatenates its input onto its output, which is exactly the U-Net skip:

```python
import torch
import torch.nn as nn
import functools


class UnetSkipConnectionBlock(nn.Module):
    """One down/up level of the U-Net. The skip is the concat in forward():
    the block's input x is carried straight across to its output, so low-level
    structure bypasses the bottleneck."""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        # 4x4 stride-2 conv downsamples; transposed conv upsamples (DCGAN backbone)
        downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)           # leaky in the encoder path
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)                        # plain ReLU in the decoder path
        upnorm = norm_layer(outer_nc)

        if outermost:                                 # first/last level: no BN on outer, Tanh out
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Tanh()]
        elif innermost:                               # bottleneck: no submodule, no skip-concat input
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=False)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            # noise = dropout on the decoder side of the inner levels
            model = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)        # <-- the skip connection


class UnetGenerator(nn.Module):
    """U-Net G(x): encoder-decoder with skips. For a 256x256 input, num_downs=8
    so the encoder reaches a 1x1 bottleneck, and the decoder mirrors it back."""
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        # build innermost first, wrap outward
        block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_nc=None, submodule=None,
                                        norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):                 # the dropout-carrying middle levels
            block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_nc=None, submodule=block,
                                            norm_layer=norm_layer, use_dropout=use_dropout)
        block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf,   ngf*2, submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                             submodule=block, outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)
```

The discriminator is the PatchGAN: a few strided convs whose stacked receptive field is `N × N`, ending in a 1-channel map of per-patch scores (no final spatial pooling — running it convolutionally *is* the averaging-over-patches, since each output cell scores one patch):

```python
class NLayerDiscriminator(nn.Module):
    """PatchGAN: classifies each NxN patch of [x, y] as real/fake. n_layers sets
    the receptive field (n_layers=3 -> 70x70). Output is a map of patch scores."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        kw, padw = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]  # no BN on first C64
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            seq += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, 2, padw, bias=False),
                    norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        seq += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, 1, padw, bias=False),  # stride 1: grow RF
                norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf*nf_mult, 1, kw, 1, padw)]   # -> 1-channel patch-score map
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)
```

The GAN loss compares the patch-score map against an all-real or all-fake target of the same shape (BCEWithLogits is the non-saturating real/fake cross-entropy; the all-ones target for `G` is what implements "maximize `log D`"):

```python
class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer("real", torch.tensor(real_label))
        self.register_buffer("fake", torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()
    def __call__(self, pred, target_is_real):
        target = (self.real if target_is_real else self.fake).expand_as(pred)
        return self.loss(pred, target)
```

And the training step assembles the full objective. The generator is `G(x)` (noise is the dropout inside `G`); the discriminator always sees the concatenation `[x, y]` — this is the conditioning that makes it judge correspondence:

```python
def optimize(G, D, criterionGAN, criterionL1, optG, optD, real_A, real_B, lambda_L1=100.0):
    fake_B = G(real_A)                                  # G(x); stochasticity via dropout

    # --- update D: real pair (x,y) vs fake pair (x, G(x)) ---
    optD.zero_grad()
    pred_fake = D(torch.cat((real_A, fake_B), 1).detach())   # condition D on x: feed [x, fake]
    pred_real = D(torch.cat((real_A, real_B), 1))            # feed [x, real]
    loss_D = (criterionGAN(pred_fake, False) + criterionGAN(pred_real, True)) * 0.5  # /2 slows D
    loss_D.backward(); optD.step()

    # --- update G: fool D on the pair, plus L1 anchor to ground truth ---
    optG.zero_grad()
    pred_fake = D(torch.cat((real_A, fake_B), 1))           # G wants D to call [x, fake] real
    loss_G_GAN = criterionGAN(pred_fake, True)              # non-saturating: target = real
    loss_G_L1  = criterionL1(fake_B, real_B) * lambda_L1    # L1 (median) low-frequency anchor
    (loss_G_GAN + loss_G_L1).backward(); optG.step()
```

Tracing the chain end to end: per-pixel regression blurs because squared error makes the net output the conditional mean of all plausible images, so I want a loss that lets the net commit to one sharp mode — an adversarial loss, where the discriminator learns to reject blur as fake; but a plain discriminator only checks realism, so I condition it on the input and have it judge the *pair* `(x, y)`, which makes it police correspondence; then I notice the L1 term I add for stability already nails the low frequencies, so the discriminator only needs to enforce high-frequency, local detail, which lets me shrink it to a patch-classifying PatchGAN (a learned local texture loss) that is light, fast, and size-agnostic; on the generator side, because input and output share aligned low-level structure, I let it bypass the bottleneck with U-Net skip connections; and since explicit noise gets ignored, I inject stochasticity as dropout — landing on `min_G max_D L_cGAN(G,D) + λ L_L1(G)` with a U-Net generator and a PatchGAN discriminator.
