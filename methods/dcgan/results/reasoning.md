Let me start from what's actually bugging me, because I want to name the problem precisely before I touch any architecture. Supervised CNNs are eating computer vision alive — backprop, ReLU, a big labeled dataset, and a deep convolutional stack just works. But the labels are the whole cost. There's effectively unlimited unlabeled imagery sitting around, and I have no clean way to turn it into the kind of reusable features that supervised CNNs learn as a side effect of classification. That's the thing I want: features learned with no labels that I can peel off and reuse downstream.

So what unsupervised objective produces good convolutional features? Autoencoders are the obvious answer — reconstruct the input through a bottleneck — but reconstruction has a built-in disease. The loss is per-pixel, and when several outputs are equally plausible, the loss-minimizing thing to do is *average* them, which smears detail into blur. I keep seeing this: pixel-wise mean-squared reconstruction produces over-smoothed images, and over-smoothed images mean the features upstream were pushed toward encoding low-frequency averages rather than the sharp structure I care about. So I'd rather not anchor my feature learner to a pixel-reconstruction loss at all.

The adversarial game is the alternative that doesn't have that anchor. Two networks: a generator `G` turns a noise vector `z ~ p_z` into an image, and a discriminator `D` outputs a probability that an image is real rather than generated. They play `min_G max_D E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]`. In practice `D` is just a binary classifier trained with cross-entropy — real labelled 1, fake labelled 0 — and `G` is trained to make `D` call its fakes real. Notice what's *not* in there: no explicit density, no partition function, no per-pixel reconstruction term. The learning signal for `G` is "fool an adversary," and the learning signal for `D` is "catch a forger." If this works, both networks have to build a hierarchy of image features just to play, and those features are exactly what I want to harvest — the discriminator especially, since it's literally a CNN classifier.

So why hasn't everyone already done this with a deep convolutional generator and discriminator? Because it doesn't train. When people take the CNN architectures that work for supervised classification and drop them into the adversarial game, the thing falls apart — the generator produces garbage, or it collapses, every `z` mapping to the same output image. That collapse is the signature failure: the generator finds one image that `D` currently can't reject and just emits it for every input, the game stalls there. The instability is bad enough that the strongest convolutional-GAN result so far, the Laplacian-pyramid approach, deliberately *avoids* training one big convolutional generator — instead it trains a stack of small conditional models, each upscaling the image one resolution band at a time. That's a workaround, and it has its own cost: chaining several independent models injects noise at every stage, so the objects come out wobbly, and a pyramid built for synthesis isn't a single clean feature extractor I can reuse. I want one convolutional generator and one convolutional discriminator that I can train directly and stably. So the real problem isn't the objective — the adversarial objective is fine and I'll keep it untouched. The problem is the *architecture and training recipe*. Let me go find what's destabilizing the naive CNN GAN and fix each piece.

First destabilizer: pooling. The supervised CNNs I'd copy from are full of max-pooling — a fixed, hand-chosen operator that downsamples by taking the max in each window. In a discriminator that's probably survivable, but think about what the generator needs: it has to go the *other* way, from a low-resolution feature map up to a full image, so it needs *upsampling*. The naive thing is a fixed unpooling or a nearest-neighbor resize followed by a convolution. But a fixed resize is exactly the kind of hand-chosen, non-learnable operator that I suspect is fighting the game — it forces a particular upsampling structure that the gradient can't reshape. There's a cleaner idea floating around from the all-convolutional-net work: get rid of pooling entirely and let *strided convolutions* do the spatial resampling, so the network learns its own downsampling instead of being handed one. For the discriminator that's a direct fit — a convolution with stride 2 halves the spatial size and the operator is learned. For the generator I want the transpose of that: a *fractionally-strided* convolution (a transposed convolution — people sometimes loosely call it a deconvolution) that *doubles* the spatial size, and again the upsampling operator is learned rather than fixed. So: no pooling anywhere. Discriminator downsamples with strided convs; generator upsamples with fractionally-strided convs. Now the spatial resampling is part of what the game can tune, top to bottom.

Second destabilizer: the fully-connected head. Supervised CNNs stick big fully-connected layers on top of the convolutional features. The trend in the best classifiers is to throw those out — the strongest version being global average pooling, which collapses each final feature map to one scalar and feeds those straight to the output, killing most of the parameters and acting as a regularizer. Let me try the most aggressive version: global average pooling on top of the discriminator. When I do that, the model gets *more stable* — but it converges noticeably slower. So the extreme is a stability win and a speed loss. I want the stability without paying that much speed. The compromise: remove the fully-connected *hidden* layers, but connect the highest convolutional features directly to the input and output. Concretely, the generator's very first operation takes the 100-dimensional noise `z` and does a matrix multiply — you could call that fully connected — but I immediately *reshape* the result into a small-spatial-extent, many-channel 4D tensor and treat that as the start of the convolution stack. And the discriminator's last convolutional layer I just flatten and feed into a single sigmoid. So there are no fully-connected hidden layers in between; the only matrix-multiply-like things are the projection-and-reshape at the generator's input and the flatten-to-one-scalar at the discriminator's output. That keeps the parameter count and the regularizing effect of dropping FC layers, without the full convergence hit of global average pooling.

Third destabilizer, and this is the big one: getting a *deep* generator to start learning at all, and keeping it from collapsing. Deep nets in the supervised world got rescued by batch normalization — normalize the input to each unit over the minibatch to zero mean and unit variance, with a learned scale and shift, so the signal entering each layer is well-conditioned regardless of initialization, and gradients flow through depth instead of stalling. Two reasons that's exactly what a GAN needs. One, poor initialization is lethal here: if the generator's early layers output badly-scaled activations, the deep stack never gets off the ground, and batchnorm is the thing that lets a deep generator *begin* learning. Two — and this is the one I care about most — batchnorm seems to directly fight mode collapse. The collapse failure is the generator funneling every `z` to a single output point; but batchnorm normalizes activations *across the batch*, so if every sample in the batch is heading toward the same point, the normalization statistics expose that and the gradient pushes back. Empirically, adding batchnorm is what prevents the generator from collapsing all samples to one point. So: batchnorm in both networks.

But when I naively put batchnorm on *every* layer, the samples start oscillating and the model goes unstable again. So it's not "batchnorm everywhere," it's "batchnorm in the right places." Where does it hurt? Two layers. The generator's *output* layer: if I normalize the generated batch right before it becomes the image, I'm forcing the batch of generated images to have a particular mean and variance, which fights the actual pixel statistics the generator is trying to match and fights the bounded output range — the output layer should be free to produce whatever the data distribution actually looks like. And the discriminator's *input* layer: if I batch-normalize the raw pixels coming in, I'm normalizing away part of the real-versus-fake signal at the pixel level before the discriminator even gets to look — the input statistics are *information*, and normalizing them across a batch that mixes scales is exactly the oscillation source. So the rule lands as: batchnorm in both nets, but *not* on the generator's output layer and *not* on the discriminator's input layer. Everywhere else, batchnorm.

Now the activations, and I want to reason about each rather than copy defaults. In the generator, ReLU through the stack — clean non-vanishing gradients on the active side, standard for deep nets. But the *output* layer is special: it's producing pixels. I pre-scale all my training images to `[-1, 1]` (the only preprocessing I do), so the generator's output should live in `[-1, 1]` too. A bounded activation that saturates to that range is the natural choice — tanh. And there's a real benefit beyond matching the range: because tanh saturates, the generator learns *quickly* to push values to the extremes and cover the full color space of the training distribution, rather than timidly hovering near gray. So ReLU everywhere in `G` except a tanh output.

In the discriminator, the original adversarial work used maxout, but let me think about what the discriminator's job is in *this* coupled system. The generator learns *only* through the gradient that the discriminator passes back about the fakes. If the discriminator uses plain ReLU and a chunk of its units die — stuck on the negative side with zero gradient — then the signal it hands back to the generator is impoverished, and the generator's learning degrades for reasons that have nothing to do with the generator. So I want the discriminator's gradient to stay alive everywhere. Leaky ReLU does exactly that: `max(αx, x)` with a small slope `α` on the negative side, so gradient always flows, no dead units. I'll use a leak slope of `0.2`. This matters more the higher the resolution, because deeper, higher-res discriminators have more units to lose to death. So leaky ReLU throughout `D`.

Now the optimizer, and here the moving-target nature of the game bites. Previous GAN work used plain momentum SGD. Let me reach for Adam instead — adaptive per-parameter step sizes are well suited to a loss landscape this badly conditioned. The suggested default learning rate is `0.001`; when I use it the training is too aggressive and unstable, so I drop it to `0.0002`. But the more interesting knob is the momentum term `β1`. The default is `0.9`, which means each gradient step is heavily averaged with a long history of past gradients. In a *stationary* optimization that's great — it smooths noise. But the adversarial game is *non*-stationary: every time `D` updates, the loss surface that `G` sees changes, and vice versa. A heavy momentum of `0.9` keeps dragging the update in the direction of *stale* gradients computed against an opponent that has already moved, and that lag is what I see as training oscillation and instability. Lowering `β1` to `0.5` shortens that memory so each update tracks the *current* opponent much more closely — and that stabilizes training. I'll keep `β2` at its `0.999` default. So Adam, `lr = 0.0002`, `β1 = 0.5`, `β2 = 0.999`.

A couple of remaining settings. Initialization: a GAN's first few steps are fragile, so I initialize all weights from a zero-centered normal with a small standard deviation, `0.02`, to start the game gently rather than with large random activations that would blow the early dynamics up. Minibatch SGD with batch size `128`. The latent `z` is a `100`-dimensional uniform vector — a low-dimensional manifold the generator expands into an image. That's the full recipe, and the whole thing was reached by walking layer by layer through what destabilizes the naive CNN GAN: no pooling (strided / fractionally-strided convs), no FC hidden layers (project-reshape in, flatten-sigmoid out), batchnorm everywhere except the two boundary layers, ReLU+tanh in `G`, leaky-ReLU in `D`, Adam with low `β1`.

Let me lay out the actual generator concretely, since the resolution math has to close. I want a `64×64` RGB image out of a `100`-dimensional `z`. Start by projecting and reshaping `z` to a `4×4` spatial map with many channels — say `8f` channels where `f` is a width unit. Then four fractionally-strided convolutions, each with a `4×4` kernel, stride `2`, padding `1`, which doubles the spatial size and (for the first three) halves the channels: `4×4×8f → 8×8×4f → 16×16×2f → 32×32×f → 64×64×3`. After each of the first four maps: batchnorm then ReLU. After the last: just tanh, no batchnorm — that's the output-layer exception. The transposed conv with kernel 4, stride 2, padding 1 exactly doubles a spatial dimension, so `4→8→16→32→64`, which lands on `64×64`. Good, the arithmetic closes.

The discriminator mirrors it. Take the `64×64×3` image, apply convolutions with `4×4` kernel, stride `2`, padding `1`, each halving the spatial size and growing the channels: `64×64×3 → 32×32×f → 16×16×2f → 8×8×4f → 4×4×8f`. Leaky-ReLU(0.2) after every conv; batchnorm after every conv *except the first* — that's the input-layer exception. Then a final convolution with `4×4` kernel, stride `1`, padding `0` collapses the `4×4` map to a single `1×1` value, and a sigmoid turns it into the probability. Stride-2 conv with kernel 4, padding 1 exactly halves, so `64→32→16→8→4`, and the final valid-stride conv reads off the one scalar. Mirror image of the generator, as it should be.

Let me write it as real code. The width unit is `ngf` for the generator and `ndf` for the discriminator, both `64`; `nz = 100`, `nc = 3`.

```python
import torch
import torch.nn as nn
import torch.optim as optim

nz, ngf, ndf, nc = 100, 64, 64, 3

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # project-and-reshape z (nz x 1 x 1) into a 4x4 map with 8*ngf channels.
            # no pooling/unpooling: fractionally-strided convs learn their own upsampling.
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),            # batchnorm: let the deep gen start; fight collapse
            # 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # 32x32 -> 64x64; output layer: NO batchnorm, tanh to cover the [-1,1] color range
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input layer: NO batchnorm (don't normalize away the raw-pixel real/fake signal)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),                   # leaky: keep gradient alive -> G keeps getting signal
            # 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> 1x1 single scalar, sigmoid = P(real). No FC hidden layer; just flatten-to-one.
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.main(x).view(-1)

def weights_init(m):
    # gentle start: small zero-centered normal init
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

netG, netD = Generator(), Discriminator()
netG.apply(weights_init); netD.apply(weights_init)

criterion = nn.BCELoss()
# Adam, low lr and low beta1: track the moving opponent instead of stale gradients
optD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

real_label, fake_label = 1.0, 0.0

def train_step(real):
    b = real.size(0)
    # --- update D: maximize log D(real) + log(1 - D(G(z))) ---
    netD.zero_grad()
    out_real = netD(real)
    loss_real = criterion(out_real, torch.full((b,), real_label))
    loss_real.backward()
    z = torch.randn(b, nz, 1, 1)
    fake = netG(z)
    out_fake = netD(fake.detach())                       # detach: don't push G on the D step
    loss_fake = criterion(out_fake, torch.full((b,), fake_label))
    loss_fake.backward()
    optD.step()
    # --- update G: maximize log D(G(z)) (non-saturating: label fakes as real) ---
    netG.zero_grad()
    out = netD(fake)
    loss_g = criterion(out, torch.full((b,), real_label))
    loss_g.backward()
    optG.step()
    return loss_real + loss_fake, loss_g
```

That's the whole thing. The causal chain: I wanted reusable features from unlabeled images without a blur-inducing reconstruction loss, which pointed at the adversarial game; the game's objective was fine but the *convolutional* version of it wouldn't train, so I walked through the destabilizers one at a time — fixed pooling became learned strided / fractionally-strided convolution, fully-connected heads became a project-reshape input and a flatten-sigmoid output, depth-and-collapse were fixed by batchnorm placed everywhere except the two boundary layers where it fights the pixel statistics, dead discriminator units were fixed by leaky-ReLU so the generator keeps receiving gradient, the generator output was bounded by tanh to match the `[-1,1]` color range, and the non-stationary game was tamed by Adam with a short-memory `β1 = 0.5` and a small learning rate. The result is a single convolutional generator-discriminator pair that trains stably, and whose discriminator features are exactly the reusable representation I set out to find. What I'd want to validate next: that those discriminator features, fed to a simple linear classifier, transfer to a labeled benchmark, and that the latent space is smooth and semantically linear enough to walk and do arithmetic in.
