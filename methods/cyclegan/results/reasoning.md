OK, let me think this through from scratch. I have two piles of images. One pile is photographs of landscapes; the other is paintings by Monet. Nobody has told me which photo corresponds to which painting — there is no correspondence, and for most of the pairs I'd care about, there couldn't be one, because Monet never stood in front of most of these exact scenes. I want a function that takes any landscape photo and renders it the way Monet would have, and another that takes a Monet and renders it as a photograph. Same story for horses and zebras, for summer and winter, for edge maps and shoes. The thing I keep coming back to is: with paired data this is easy. If I had {(x_i, y_i)} where y_i is literally the same scene as x_i rendered the other way, I'd just regress — fit G so that G(x_i) lands on y_i, maybe with an L1 penalty ‖G(x_i) − y_i‖₁ and an adversarial term to sharpen it, which is exactly what the paired translation framework does. But I don't have the pairs. I have a set X and a set Y and nothing connecting them. So the real question is: where does the supervision come from when there are no pairs?

The one thing I *do* have supervision over is the sets themselves. I know what a Monet looks like as a category, and I know what a photo looks like as a category. So I can at least ask: does my output look like it belongs to the target pile? And there's a tool built exactly for "make this thing indistinguishable from samples of that distribution" — the adversarial loss. Set up a discriminator D_Y that tries to separate real Monets from my generated G(x), and train G to fool it. Write it down:

  L_GAN(G, D_Y, X, Y) = E_y[log D_Y(y)] + E_x[log(1 − D_Y(G(x)))],

with G minimizing and D_Y maximizing. Let me actually convince myself what this buys me, because I want to know precisely what it pins down and what it doesn't. Fix G, so the distribution of G(x) is some fixed p_g over the Y-space. Then I'm choosing D_Y(y) ∈ [0,1] to maximize E_{y∼p_data}[log D_Y(y)] + E_{y∼p_g}[log(1 − D_Y(y))]. Pointwise, for each y I'm maximizing a·log d + b·log(1 − d) with a = p_data(y), b = p_g(y); derivative a/d − b/(1−d) = 0 gives d* = a/(a+b) = p_data(y)/(p_data(y) + p_g(y)). Plug that back into the generator's half of the game. The value becomes

  E_{p_data}[log p_data/(p_data+p_g)] + E_{p_g}[log p_g/(p_data+p_g)].

Add and subtract log 2 inside each term: this is −log4 + KL(p_data ‖ (p_data+p_g)/2) + KL(p_g ‖ (p_data+p_g)/2) = −log4 + 2·JSD(p_data ‖ p_g). The Jensen–Shannon divergence is zero only when p_g = p_data, and positive otherwise. So at the optimum the generator is driven to make p_g exactly equal to p_data(Y). Good — adversarial training really does force the distribution of my outputs to match the target domain. That's the part I wanted: "looks like a Monet" without me ever having to say what a Monet is.

So is that the whole method? Just train G with an adversarial loss against the pile of Monets and ship it? Let me stare at this for a second, because something feels too easy. The JSD result says: the *distribution* of outputs matches. It says nothing — literally nothing — about *which* input produced which output. The objective is a functional of the marginal distribution of G(x) alone; the input x only enters through that pushed-forward marginal. So suppose G* is a perfect distribution-matcher. Now take any permutation π of the target images that leaves p_data(Y) invariant — shuffle which output each input gets, as long as the overall bag of outputs is unchanged — and consider G' = π∘G*. The pushed-forward distribution of G'(x) is still exactly p_data(Y), so G' is *just as optimal* under the adversarial loss. But G' sends each x to a completely different y. There are infinitely many such maps, and the objective cannot tell them apart. So I have no guarantee whatsoever that x's content survives into G(x). The horse photo could come out as a perfectly realistic zebra that has nothing to do with the original horse's pose, the grass, the lighting — anything. That's a real wall: distribution-matching is genuinely blind to the input→output correspondence, and that correspondence is the entire point of *translation* as opposed to mere *generation*.

And it's worse than the theory makes it sound, because the degenerate map the optimizer actually drifts into isn't some innocuous permutation — it's the most degenerate one. In practice, optimizing this adversarial objective alone, the generator discovers that the cheapest way to fool D_Y is to send *everything* to one or a few outputs that the discriminator currently rates as very real. All inputs map to the same convincing zebra. That single output is in-distribution, the discriminator can't punish it on a per-sample basis, the gradient signal flattens out, and training just stalls there. Mode collapse. So the adversarial loss alone is both under-constrained in principle and unstable in practice. I can't fix this by tuning — the looseness is structural. I need to *add* something to the objective that rules out the bad maps.

What could that something be? With paired data the extra term is obvious: ‖G(x_i) − y_i‖₁ nails the output to the known target. Without pairs I have no y_i to compare against. I need a constraint that says "G(x) must be a faithful translation of *this* x" but expressed without ever naming the correct y. So: what property does a faithful translation have that a content-destroying or collapsed one lacks?

Here's the thing I keep circling. If G genuinely turned this photo into the Monet version of *this same scene*, then the scene is still in there — the composition, the layout, the objects — just re-rendered. The information about x hasn't been thrown away; it's been recoded. Whereas a collapsed G that maps every photo to the same canonical Monet has *destroyed* the information: given the output, you can't tell which photo it came from. That distinction — information preserved versus information destroyed — feels like exactly the lever I need. And there's a clean operational test for "information preserved": can I recover the input from the output?

Human translators already use a practical version of this check: translate a sentence back and see whether it lands on the original — back-translation and reconciliation. A tracker checks a match by tracking forward then backward and seeing if it returns to the start — forward-backward consistency. The common structure is that a good transformation is recoverable, and the recovery check is composition with an inverse. I don't have an inverse handy, but nothing stops me from *learning* one. So introduce a second mapping F: Y→X going the other way, and require that going there and back returns me to where I started: F(G(x)) ≈ x. Call that forward cycle consistency.

Let me check what this kills, because that's the whole reason I'm introducing F. Take the collapsed G that sends a thousand different horse photos all to the same zebra image z. Now F has to look at z and reconstruct... which of the thousand horses? It can't — z is the same input every time, so F(z) is one fixed image, and it can match at most one of the thousand originals; the reconstruction error averaged over the inputs is large. The cycle requirement directly penalizes information loss: if many x's collapse to the same G(x), no F can recover the distinct x's, so the cycle loss blows up. I should be careful not to overstate this. A perverse one-to-one, measure-preserving scramble could still have an inverse, so cycle consistency is not the same as paired supervision and it does not mathematically identify the semantic correspondence by itself. What it does is rule out the many-to-one and content-destroying maps that adversarial matching cannot see, and then the convolutional, residual, low-distortion biases of the networks make the simple content-preserving correspondence the one the optimizer can actually reach. Demanding reconstruction pushes G toward being approximately injective and information-preserving. That's precisely the structure the adversarial loss couldn't supply. I went looking for "tie the output to its input without naming the right output," and recoverability is that constraint stated without a single paired label.

But wait — if I only impose F(G(x)) ≈ x, I've regularized G but left F to do whatever it likes; F is constrained only on the *range* of G, and the moment I'm also training F adversarially against D_X to make F(y) look like a real X, that direction can collapse on its own exactly the way G could. The problem is symmetric in X and Y; there's no reason to privilege one direction. So impose the cycle in both directions: F(G(x)) ≈ x for every x, and G(F(y)) ≈ y for every y. Now G regularizes F and F regularizes G; the two maps are pinned to be mutual inverses, each closing the other's loophole. The cycle-consistency loss is the sum of both reconstruction errors:

  L_cyc(G, F) = E_x[ ‖F(G(x)) − x‖₁ ] + E_y[ ‖G(F(y)) − y‖₁ ].

Why measure reconstruction with the L1 norm and not L2? Because the reconstruction is an image, and squared error has a well-known pathology for images: when several outputs are roughly equally plausible, the L2-optimal prediction is their average, and averaging images produces a blurry mush. L1's optimum is the median, which is far less prone to that washing-out, so L1 reconstructions stay sharp. I briefly consider going fancier — replace the L1 reconstruction penalty with *another* adversarial loss, training a discriminator to tell F(G(x)) from x. It's more machinery, another min-max to balance, and when I think it through it doesn't obviously buy me anything: the cycle just needs the reconstruction to land back on the specific input, which a pixel-wise penalty states directly, whereas an adversarial term would only enforce that reconstructions look like plausible X's, not that they equal *this* x. The simpler L1 is the right call; keep it.

There's a nice way to see what I've actually built. F∘G is a map X→X that is supposed to be the identity, and G∘F is a map Y→Y supposed to be the identity — two autoencoders. But these aren't ordinary autoencoders: the bottleneck "code" through which X reconstructs itself is not an abstract latent vector, it's a full image *in the other domain*, and that intermediate image is forced by the adversarial loss to be distributed like a real member of the other domain. So it's an adversarial autoencoder where the prior imposed on the bottleneck is "looks like domain Y." The adversarial loss shapes the bottleneck; the cycle loss makes the autoencoder actually reconstruct. The two roles are complementary, which is reassuring — neither term is redundant.

Now assemble the full objective. Two adversarial terms, one per direction, plus the cycle term weighted by some λ:

  L(G, F, D_X, D_Y) = L_GAN(G, D_Y, X, Y) + L_GAN(F, D_X, Y, X) + λ·L_cyc(G, F),

and I solve the saddle problem

  G*, F* = argmin_{G,F} max_{D_X, D_Y} L(G, F, D_X, D_Y).

What's λ doing and how big should it be? The two adversarial terms are each an O(1) log-probability quantity, while the cycle term is an L1 pixel error which, with pixels normalized to [−1,1], lives on the order of a couple of units per pixel before it's driven down. These don't naturally sit at the same scale, and they're pulling in tension: the adversarial term will happily sacrifice fidelity to the input for an extra notch of realism (that's the very failure I'm guarding against), so the cycle term — which is the *only* thing carrying the content constraint — has to be the dominant signal, or the network drifts back toward content-destroying solutions. But not so dominant that reconstruction freezes the translation into a near-identity that ignores the adversarial push. Weighting the cycle term up by λ = 10 puts it firmly in charge of preserving content while still leaving the adversarial terms enough say to do the actual re-rendering. So λ = 10.

Before I touch architectures, the adversarial loss as written — the log-likelihood form — has a practical defect I should fix, because GAN instability is exactly what I can't afford here. With a sigmoid discriminator and cross-entropy, once a fake sample is on the correct side of the decision boundary the loss saturates: the generator gets a vanishing gradient even for fakes that are *far* from the real data, as long as they're nominally "classified correct." That starves G of signal precisely when it most needs to keep moving. The least-squares form fixes this: have D regress real to 1 and fake to 0 under squared error, and have G push its fakes' scores toward 1. Concretely, for a GAN term L_GAN(G, D, X, Y), train D to minimize E_y[(D(y) − 1)²] + E_x[D(G(x))²] and train G to minimize E_x[(D(G(x)) − 1)²]. Now the penalty grows with the *distance* to the target label, so a fake that's far from real — even one D already classifies correctly — still gets a gradient pulling it closer; samples don't get to coast on the right side of the boundary. It's more stable and the outputs come out sharper. (It also corresponds to minimizing a Pearson χ² divergence rather than JSD, but the operational reason I'm switching is the non-saturating gradient.)

One more stability worry specific to my setup: I'm going to train with a batch size of 1 (these are big images and the model is heavy). With batch size 1, the discriminator sees exactly one fake per step, so it can chase the generator's single latest output and the whole thing oscillates — D and G ping-ponging against each other's most recent move. The fix is to decorrelate D's updates from G's instantaneous state: keep a buffer of the 50 most recently generated images and update D on a sample drawn from that history rather than only on the current fake. D then learns against the generator's recent *distribution* of outputs, not its last twitch, which damps the oscillation. For each discriminator update I average the real and fake squared-error terms by multiplying their sum by 0.5; that keeps the D loss scale comparable to a single adversarial term instead of doubling it just because the discriminator sees both sides. Small knobs, but GAN training lives or dies on them.

Now the networks. The generator's job, the way I've framed it, is mostly to *keep* the input and change its appearance: the scene's geometry and layout should survive (the cycle loss is demanding exactly that), only the rendering changes. So the prior I want baked into the architecture is "output ≈ input plus a learned modification." That's the residual-block prior precisely — a block computes x + r(x), so the easy default is the identity and the network only has to learn the perturbation r. Stacking residual blocks lets me learn "mostly identity, with a structured change" cheaply and train it at depth without the gradient trouble plain deep stacks have. So I take the feed-forward transformer shape that already works well for style transfer and super-resolution: a 7×7 convolution to lift to features, two stride-2 convolutions to downsample (cutting spatial size so the heavy residual processing runs at low resolution where it's affordable and each filter sees a larger receptive field), a stack of residual blocks at that low resolution, two fractionally-strided (transpose) convolutions to upsample back, and a final 7×7 convolution to RGB with a Tanh so pixels land in [−1,1]. How many residual blocks? Enough capacity and receptive field for the image size: 6 blocks suffice at 128×128, but at 256×256 and up the field and capacity need to grow, so 9 blocks there. And I pad with reflection rather than zeros at the 7×7 convs, because zero-padding injects a dark border that shows up as boundary artifacts in the output; reflecting the image across the edge avoids that.

The normalization is a real choice, not a default. Batch normalization is out almost by definition here: batch size is 1, so "batch statistics" are a single sample's statistics — degenerate. But there's a deeper reason instance normalization is *right* and not just a fallback. Instance norm normalizes each image independently, per-channel, removing that particular image's contrast and intensity statistics. For translation that's exactly what I want stripped: the output's style should be a property of the *target domain*, not a leftover of the input image's own contrast and lighting. Normalizing per instance washes out the input's instance-specific appearance statistics and lets the network re-impose the target domain's. So instance normalization, in the generator and (except the first layer) in the discriminator.

For the discriminator, do I need to judge the whole image at once? The realism I'm enforcing is fundamentally about local texture and appearance — brushstroke statistics, the stripe texture of zebra fur, the grain of a photo. Those are local, high-frequency properties; the low-frequency global structure is already being held in place by the cycle term. So instead of a discriminator that ingests the whole image and emits one number, use a fully-convolutional patch discriminator that classifies each overlapping local patch real-or-fake and averages. This has far fewer parameters, focuses capacity exactly on the high frequencies that matter, and — being fully convolutional — runs on any input size. The patch size is a tradeoff: too small and it can't see enough texture to judge realism, too large and it's back to a full-image discriminator with its parameter cost and artifacts; a 70×70 receptive field is the sweet spot for texture quality. Architecturally that's 4×4 convolutions: C64, C128, and C256 use stride 2; C512 uses stride 1; a final stride-1 convolution emits the 1-channel patch-score map. With kernel 4 and padding 1, the receptive field grows 4 → 10 → 22 → 46 → 70 pixels, so the advertised 70×70 patch size is exact.

One application-specific wrinkle shows up when I think about painting→photo. With only the adversarial-plus-cycle objective, G and F are free to shift the overall color cast — push a Monet's daytime palette toward a photographic sunset, say — because a global tint shift is equally consistent with "looks like the target domain" and is invisible to the cycle loss as long as it can be undone on the way back. I'd like to anchor the color. The anchor is almost tautological: if I feed G an image that is *already* in the target domain Y, G has nothing to do — it should return it unchanged. So add an identity term: penalize ‖G(y) − y‖₁ for real y and ‖F(x) − x‖₁ for real x. This tells each generator "when handed something already in your output domain, be the identity," which pins down the color/tint that the adversarial-plus-cycle objective otherwise leaves floating. I weight it at 0.5λ — present enough to anchor the palette, light enough not to fight the translation — and only switch it on for the tasks (like painting↔photo) where the color drift actually shows up.

Training specifics fall out of all this: Adam with learning rate 2×10⁻⁴ and β₁ = 0.5 (the lower first-moment decay is the usual GAN setting, since the loss landscape is non-stationary and you don't want too much momentum dragging you along a stale direction), batch size 1, weights initialized from N(0, 0.02), λ = 10 for the cycle term and 0.5λ for the identity term when used. Keep the learning rate constant for the first 100 epochs, then decay it linearly to zero over the next 100 — let it explore at full rate while the two adversarial games are still finding their balance, then anneal to settle.

Let me make sure the pieces compose into one training step before I write the code. Each iteration: push the current x through G to get fake_y = G(x) and through F again to reconstruct rec_x = F(fake_y); push y through F to get fake_x = F(y) and through G to reconstruct rec_y = G(fake_x). The generators' loss is the two least-squares adversarial terms (make D rate fake_y and fake_x as real) plus λ times the two cycle reconstruction terms plus, if enabled, the identity terms — all summed, and I step G and F *together* with one optimizer, since they're coupled. Then the discriminators: D_Y sees real y and a fake_y drawn from its history buffer, D_X sees real x and a buffered fake_x, each trained with the least-squares real-vs-fake loss, objective halved, stepped with their own optimizer. The buffer query and the detach of the fakes when training D (so generator gradients don't leak into the discriminator step) are the bookkeeping that keeps the two games separate. That's the whole loop.

I can write the pieces down now.

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


# ---- Generator: Johnson-style residual transformer ----
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        # x + r(x): identity prior, learn only the appearance change
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),  # reflect, not zero -> no dark-border artifacts
            nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_nc, out_nc, ngf=64, n_blocks=9,
                 norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)):
        super().__init__()
        use_bias = use_conv_bias(norm_layer)
        # lift to features with a 7x7 conv
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_nc, ngf, 7, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        # downsample x2 so the heavy residual stack runs at low res / large field
        n_down = 2
        for i in range(n_down):
            m = 2 ** i
            model += [nn.Conv2d(ngf * m, ngf * m * 2, 3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * m * 2), nn.ReLU(True)]
        # residual blocks: "mostly identity + small change"; 6 for 128px, 9 for 256px+
        m = 2 ** n_down
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * m, norm_layer, use_bias)]
        # fractionally-strided convs upsample back to full resolution
        for i in range(n_down):
            m = 2 ** (n_down - i)
            model += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(ngf * m // 2), nn.ReLU(True)]
        # 7x7 conv to RGB, Tanh -> pixels in [-1, 1]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ---- Discriminator: 70x70 PatchGAN ----
class NLayerDiscriminator(nn.Module):
    def __init__(self, in_nc, ndf=64, n_layers=3,
                 norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)):
        super().__init__()
        use_bias = use_conv_bias(norm_layer)
        kw, padw = 4, 1
        seq = [nn.Conv2d(in_nc, ndf, kw, stride=2, padding=padw),  # no norm on C64
               nn.LeakyReLU(0.2, True)]
        mult, mult_prev = 1, 1
        for n in range(1, n_layers):
            mult_prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * mult), nn.LeakyReLU(0.2, True)]
        mult_prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * mult, 1, kw, stride=1, padding=padw)]  # 1-channel patch-score map
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


# ---- least-squares adversarial loss (non-saturating, sharper) ----
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()  # LSGAN: regress to 1 (real) / 0 (fake)

    def __call__(self, pred, target_is_real):
        target = pred.new_tensor(1.0 if target_is_real else 0.0).expand_as(pred)
        return self.loss(pred, target)


# ---- 50-image history buffer to decorrelate D updates (batch size 1) ----
class ImagePool:
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
            elif random.uniform(0, 1) > 0.5:        # 50%: return an old image, store the new one
                i = random.randint(0, self.pool_size - 1)
                out.append(self.images[i].clone())
                self.images[i] = image
            else:                                    # 50%: return the current image
                out.append(image)
        return torch.cat(out, 0)


class CycleGAN:
    # naming map to the formulation: G_A = G (X->Y), G_B = F (Y->X),
    #                                D_A = D_Y,      D_B = D_X
    def __init__(self, in_nc=3, out_nc=3, lambda_cyc=10.0, lambda_idt=0.5, lr=2e-4):
        if lambda_idt > 0:
            assert in_nc == out_nc, "identity loss requires matching input/output channels"
        self.G_A = ResnetGenerator(in_nc, out_nc)   # G:  X -> Y
        self.G_B = ResnetGenerator(out_nc, in_nc)   # F:  Y -> X
        self.D_A = NLayerDiscriminator(out_nc)      # D_Y: judges domain Y
        self.D_B = NLayerDiscriminator(in_nc)       # D_X: judges domain X
        for net in (self.G_A, self.G_B, self.D_A, self.D_B):
            init_weights(net)
        self.fake_A_pool, self.fake_B_pool = ImagePool(), ImagePool()
        self.criterionGAN = GANLoss()
        self.criterionCycle = nn.L1Loss()           # L1 -> sharp reconstructions
        self.criterionIdt = nn.L1Loss()
        self.lambda_cyc, self.lambda_idt = lambda_cyc, lambda_idt
        # G and F are coupled -> optimize them jointly
        self.opt_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                      lr=lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                      lr=lr, betas=(0.5, 0.999))

    def forward(self, real_A, real_B):
        self.fake_B = self.G_A(real_A)              # G(x)
        self.rec_A = self.G_B(self.fake_B)          # F(G(x))  -> should be x
        self.fake_A = self.G_B(real_B)              # F(y)
        self.rec_B = self.G_A(self.fake_A)          # G(F(y))  -> should be y

    def backward_G(self, real_A, real_B):
        lam, lam_i = self.lambda_cyc, self.lambda_idt
        # identity anchor: feeding a target-domain image should change nothing (pins color)
        loss_idt = 0.0
        if lam_i > 0:
            loss_idt = (self.criterionIdt(self.G_A(real_B), real_B) * lam * lam_i +
                        self.criterionIdt(self.G_B(real_A), real_A) * lam * lam_i)
        # adversarial: make the fakes pass as real in the target domain
        loss_G = (self.criterionGAN(self.D_A(self.fake_B), True) +
                  self.criterionGAN(self.D_B(self.fake_A), True))
        # cycle consistency: come back to where you started (this is what ties output to input)
        loss_cyc = (self.criterionCycle(self.rec_A, real_A) * lam +
                    self.criterionCycle(self.rec_B, real_B) * lam)
        (loss_G + loss_cyc + loss_idt).backward()

    def backward_D(self, netD, real, fake):
        loss = (self.criterionGAN(netD(real), True) +
                self.criterionGAN(netD(fake.detach()), False)) * 0.5   # average real/fake terms
        loss.backward()
        return loss

    def optimize(self, real_A, real_B):
        self.forward(real_A, real_B)
        # --- generators ---
        for p in itertools.chain(self.D_A.parameters(), self.D_B.parameters()):
            p.requires_grad = False
        self.opt_G.zero_grad(); self.backward_G(real_A, real_B); self.opt_G.step()
        # --- discriminators, on buffered (history) fakes ---
        for p in itertools.chain(self.D_A.parameters(), self.D_B.parameters()):
            p.requires_grad = True
        self.opt_D.zero_grad()
        self.backward_D(self.D_A, real_B, self.fake_B_pool.query(self.fake_B))
        self.backward_D(self.D_B, real_A, self.fake_A_pool.query(self.fake_A))
        self.opt_D.step()
```

Stepping back through the chain: I had two unordered image collections and no pairs, so I leaned on the one supervision I had — "belongs to the target pile" — via an adversarial loss, and proved it matches output distributions but is blind to the input→output correspondence and collapses in practice. To pin the correspondence without ever naming the right output, I demanded recoverability — learn a reverse map F and require F(G(x)) ≈ x and G(F(y)) ≈ y — which penalizes the information loss behind collapse while still relying on the network biases to choose a simple correspondence among possible invertible ones. I measured reconstruction in L1 to stay sharp and weighted it by λ = 10 so content dominates. I stabilized the adversarial games with a least-squares loss and a 50-image history buffer, built the generator as a residual transformer (identity prior, instance norm, reflection padding) and the discriminator as a 70×70 PatchGAN judging local texture, and added an identity term to anchor color where it drifts. Two coupled adversarial autoencoders whose bottleneck is an image in the other domain — that's the method.
