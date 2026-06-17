Let me start from the thing that keeps biting me: I want to know how different two images are, in a way that matches what a person would say, and the distance I actually reach for — per-pixel Euclidean, `||x − x0||²`, the same thing PSNR is a rescaling of — is just wrong in a way I can demonstrate to myself in two lines. Take an image and blur it. The local averages may not move far in squared-error terms, yet the lost edges and texture are obvious. Now take the same image and shift it by one pixel, or warp it a hair. A person sees the same picture; the per-pixel distance is huge, because the high-frequency content no longer lines up. So the measure I have is both too forgiving of perceptible blur and too sensitive to imperceptible geometry. The disease is obvious once I name it: per-pixel `ℓ2` treats every pixel as an independent coordinate, so it is blind to structure, to the correlations between neighboring pixels that carry all the perceptual content. I need a distance that respects structure.

The hand-built perceptual metrics were exactly an attempt at this, and I should be precise about what they bought and where they stall, because I'm going to react against them. SSIM gives up on comparing pixels and compares *local windowed statistics*: over a sliding window it forms a luminance term `(2 μ_x μ_y + C1)/(μ_x² + μ_y² + C1)`, a contrast term `(2 σ_x σ_y + C2)/(σ_x² + σ_y² + C2)`, and a structure term `(σ_xy + C3)/(σ_x σ_y + C3)`, then multiplies them, `l^α c^β s^γ`, with small stabilizers `C1=(k1 L)²`, `C2=(k2 L)²`. That's a genuine step — it notices a contrast change or a structural disruption that `ℓ2` would average away, and on photometric distortions it tracks people better than `ℓ2`. But stare at what it is: a fixed algebraic combination of a few first- and second-order local statistics. There is nothing in it that knows what *blur* looks like as opposed to what a *small warp* looks like; it was never designed for the case where a tiny spatial shift makes the windows misalign, and that is precisely where it disagrees with humans. And there is nothing to *adapt*. If I had a pile of human judgments and wanted the metric to fit them better, SSIM gives me no knobs but `k1, k2, α, β, γ` — a handful of global scalars. It is a shallow, frozen function, and the frozen-ness is the wall: it cannot absorb what people actually do.

So the shallow metrics are out, and the natural reaction is "make it deep and learn it." But here a second wall is already in front of me, and it is important to walk into it now rather than later, because it is the obvious thing to try. The straightforward supervised route is: collect a big dataset of human pairwise preferences — show a reference `x` and two distortions `x0, x1`, record which one a person calls closer — and train a network end to end to predict that preference directly. I want to believe this works; it is the textbook move. But the label structure makes me nervous. Human similarity is *context-dependent* and *pairwise*. Is a red circle more similar to a red square or to a blue circle? It depends which "respect of similarity" the viewer is holding in mind, and the judgment is a comparison between *two* things, not a property of one. The judgments may not even satisfy the axioms of a metric. A network fit directly to triplets could learn the quirks of the sampled comparisons rather than a reusable distance. I need a route that does not ask pairwise labels to create the whole representation from scratch.

What else is on the table? There's this thing the image-synthesis people have been doing that I keep seeing work qualitatively. Gatys does style transfer, Johnson does fast style transfer and super-resolution, Dosovitskiy and Brox do image generation — and the common ingredient is that none of them measure error in pixel space. They take a network `φ` pretrained for ImageNet classification, freeze it, and measure distance between the *feature maps* it produces. Johnson writes it plainly as a feature reconstruction loss, `ℓ_feat = (1/(C_j H_j W_j)) · ||φ_j(ŷ) − φ_j(y)||²₂` — squared error between the activations at some layer `j` of a fixed VGG. They use it as a training loss and the outputs look dramatically better than per-pixel losses give: sharper, more plausible texture, fewer of the washed-out averages that `ℓ2` produces. They *call* it a "perceptual loss." And the reason this is intriguing rather than just another heuristic is the same intuition that's been surfacing elsewhere: a network trained to predict — to classify — has to build internal representations that encode the structure of images that actually matters, edges then textures then parts then objects. Distance in *that* space might be perceptual not because anyone designed it to be, but as a side effect of the representation being good at a hard predictive task. There's even the parallel from neuroscience that these same task-trained representations model visual-cortex activity, and that the better the task representation, the better the cortical model. So the hypothesis forming is: perceptual similarity might not be a special function of its own that I have to fit; it might be a *consequence* of a representation tuned to be predictive about the world.

That reframes the whole problem and avoids the direct-fit trap. Instead of asking pairwise human labels to create the entire representation, I take an embedding that already exists, that was trained for a different task and may already organize images in a perceptually useful way, and I measure distance in it. The supervised human data, if I use it at all, isn't used to learn the representation from scratch; it's used at most to *calibrate* a representation I already have. That's the move: borrow the emergent structure, don't try to manufacture it.

But "measure distance in feature space" the way the synthesis papers do is too sloppy to be the answer, and pinning down why exposes the design I actually need. They compare *raw* activations at *one hand-picked layer* with squared error. Three problems. First, the layer choice is arbitrary — early layers carry low-level detail (edges, color), late layers carry semantics (parts, objects), and perception clearly uses both; committing to one layer throws away the rest. So I should use *several* layers and combine them. Second, and this is the subtle one, the raw activations across channels and across layers have wildly different magnitudes. If I take a plain squared L2 over a feature map, a handful of channels that happen to fire large will dominate the sum, and the distance is no longer comparable from layer to layer — it's just measuring whichever channels are loudest. I need to put the channels on equal footing before I compare them. Third, nobody checked whether this "perceptual loss" is actually perceptual against human data, or whether the choice of network even matters; it's asserted from output quality. I want the thing to be *measured* and, where possible, *calibrated*.

Take the magnitude problem first, because the fix shapes the formula. At each spatial location of a feature map I have a vector across channels, `y^l_{hw} ∈ ℝ^{C_l}`. The issue is its overall length varies enormously between locations, channels, and layers. The clean way to neutralize that is to normalize each such vector to unit length in the channel dimension before comparing: `ŷ^l_{hw} = y^l_{hw} / ||y^l_{hw}||₂`. Now every location contributes a *direction*, not a magnitude, and the comparison across layers is on the same footing. Let me check what the squared L2 of normalized differences actually computes, because it should reduce to something familiar. With unit vectors `a, b`, `||a − b||² = ||a||² + ||b||² − 2 a·b = 2 − 2 cos∠(a,b)`. So comparing channel-normalized features with squared L2 is, up to a constant and factor, *cosine distance* in feature space. Good — that's a principled, magnitude-free comparison, and it means the un-calibrated version of my metric is exactly "cosine distance between deep features," which is a clean baseline I can fall back on. In code the normalization needs a floor so a zero-activation location doesn't divide by zero: `ŷ = y / (||y||₂ + ε)` with a tiny `ε`, say `1e-10`, well below any real activation norm.

Now combine layers and reduce to a scalar. For each layer `l`, I have the per-location squared difference of normalized features. I want a single number, so I average over the spatial positions `(h, w)` of that layer — that makes it a patch-level quantity, robust to exactly where in the patch the difference sits — and then I sum the per-layer contributions to fold in low-level and high-level evidence together. Writing it out, with all channel coefficients equal to 1:

  `d(x, x0) = Σ_l (1 / (H_l W_l)) Σ_{h,w} || ŷ^l_{hw} − ŷ0^l_{hw} ||²₂`.

That is the cosine-distance-across-layers metric, and it is the thing I can compute with no human data at all, straight out of any frozen pretrained net. Before I learn anything, this gives me the clean control I need: compare the same normalized feature-distance recipe across VGG, AlexNet, SqueezeNet, self-supervised and unsupervised trunks, and a random-weight trunk. If the useful object is really the representation rather than ImageNet labels specifically, trained visual networks should give a much more plausible ordering of distortions than the shallow formulas, while a random network should be the sanity check that architecture alone is not enough. The definition of the distance does not depend on those comparisons. They only tell me whether the borrowed embedding is worth calibrating. For a usable loss, a small fast backbone such as AlexNet is a sensible default because the formula only needs the intermediate feature maps and their channel counts.

Cosine distance over features gives me the nonlearned baseline. Can the human data calibrate it without asking the triplets to learn the representation itself? I can keep the representation frozen and learn the *smallest possible* correction on top of it. Look at the metric again: it weights every channel of every layer equally. But surely not every channel is equally relevant to perception — some directions in feature space carry variation a person would call a real difference, others carry variation people ignore. The minimal learnable correction is therefore a single non-negative coefficient per channel on the squared channel disagreement:

  `d(x, x0) = Σ_l (1 / (H_l W_l)) Σ_{h,w,c} α^l_c (ŷ^l_{hwc} − ŷ0^l_{hwc})²`, with `α^l_c ≥ 0`.

The paper writes the same object as a channel scale `w_l` inside the squared norm, `||w_l ⊙ (ŷ^l_{hw} − ŷ0^l_{hw})||²₂`; in implementation the stored quantity is the equivalent `α = w²`, learned directly as a `1×1` convolution applied to the squared per-location differences. Setting every stored coefficient to 1 recovers cosine distance exactly, so this is a strict generalization that can only help if there is signal in the data. And the parameter count is tiny — for VGG it is 64+128+256+512+512 = 1472 numbers, for AlexNet 64+192+384+256+256 = 1152. That's the point: I am not fitting a similarity *function* with millions of parameters that can memorize the context-dependent labels; I am calibrating a fixed feature space with a couple of thousand scalars. Dropout before the `1×1` layer is light regularization while fitting the calibration.

One constraint on the stored linear coefficients is forced, not optional. The package applies the learned `1×1` layer to squared channel differences; if one of those coefficients became negative, increasing a feature difference in that channel would *lower* the distance. That would destroy the monotonic meaning of the score. So the learned coefficients must be nonnegative. The clean way to enforce this during training is projection: after each gradient step, clamp any negative `1×1` weight back to zero. It is a projected-gradient step onto the nonnegative orthant, and it costs nothing.

Now, how do I actually fit these coefficients to the human triplets without falling back into the trap of regressing the raw preference? The data is a triplet `(x, x0, x1, h)` where `h = 0` means the person chose `x0`, `h = 1` means the person chose `x1`, and aggregated judgments can sit between them. From my metric I get two distances, `d0 = d(x, x0)` and `d1 = d(x, x1)`. The naive thing is a ranking loss: force a fixed margin, push `d0` and `d1` apart by some constant whenever the human prefers one. Let me think about whether a *constant* margin is right. It isn't — some pairs are nearly tied for people (a genuinely ambiguous triplet) and some are obvious, and a fixed margin punishes the metric for not separating the ambiguous ones as hard as the obvious ones. The margin should depend on the pair. So instead of hard-coding the relationship between `(d0, d1)` and the preference, I let a small network `G` learn it: feed it the distance pair and have it output a probability `ĥ = G(d0, d1) ∈ (0, 1)` that the human prefers `x1`, then train with binary cross-entropy against the recorded preference,

  `L(x, x0, x1, h) = − h log G(d0, d1) − (1 − h) log(1 − G(d0, d1))`.

`G` can be tiny — two 32-channel `1×1`/fully connected layers with LeakyReLU, then a one-channel layer and a sigmoid — and its job is only to map a pair of scalar distances to a preference probability, learning the pair-dependent margin instead of me imposing one. When humans split on a triplet (say the two judges disagree), I just set the target `h` to the fraction, e.g. `0.5`, and BCE handles the soft label correctly. Gradients flow back through `G` and through `d0, d1` into the linear calibration coefficients (and, if I let them, into the backbone). I'll fit it with Adam at learning rate `1e-4`, `β1 = 0.5`, batch size 50, five epochs at the initial rate and five more with linear decay, projecting the learned `1×1` coefficients back to `≥ 0` after every step.

That raises the obvious question of *how much* of the network to train. Three configurations sit on a spectrum. I can freeze the backbone `F` entirely and learn only the linear calibration — "lin." I can initialize `F` from the pretrained classifier and fine-tune everything — "tune." Or I can throw away the pretrained weights and train `F` from scratch on the human data — "scratch." The safest default follows from the earlier wall: the representation is the asset, and the human triplets are pairwise and context-dependent. If I train the trunk from scratch, I am back to asking these labels to create the whole representation. If I fine-tune the trunk, I may specialize a broad visual representation to the particular distortions in the triplets. So the clean default is the frozen trunk plus the cheap linear calibration, while `tune` and `scratch` stay as ablations that test how much representation learning the triplets can support.

There's one more practical wrinkle before this can be used as a metric or as a loss. The inputs should live in `[-1, 1]`; if callers have `[0, 1]` tensors, the forward pass can map them by `2x - 1` with a `normalize=True` flag. Then a fixed per-channel affine standardization goes before the trunk, with shift `(-0.030, -0.088, -0.188)` and scale `(0.458, 0.448, 0.450)`. This is part of the metric, not an optional downstream trick: the frozen classifier features only make sense on the range and channel statistics they expect. Once the inputs are on that range, the metric is differentiable end to end — backbone features, channel normalization, the `1×1` calibration layers, spatial average, and layer sum — so it can be added as `λ · d(x, x_target)` to another objective and backpropagated through the image being optimized.

At this point the implementation should be almost mechanical. The trunk stays frozen in the default path. Inputs pass through the fixed scaling layer before the trunk. Every feature map is channel-normalized with a tiny `1e-10` floor. Squared normalized differences go through dropout plus a no-bias `1×1` calibration layer, then a spatial average and a sum across layers. The preference head receives exactly five scalar maps, `(d0, d1, d0 - d1, d0 / (d1 + 0.1), d1 / (d0 + 0.1))`, and maps them through two 32-channel `1×1` layers with LeakyReLU activations, a final `1×1`, and a sigmoid whose target is the `h=1` choice. Adam updates the linear calibration and the preference head at `1e-4` with `β1 = 0.5`, and immediately after the step I project the learned `1×1` weights back to nonnegative values.

I can write the code now. The backbone is any frozen ImageNet classifier sliced into its `relu` stages; here I'll write it for AlexNet's five `conv`/`relu` blocks (`64, 192, 384, 256, 256` channels) since that's the fast default for a loss, but VGG (`64, 128, 256, 512, 512`) and SqueezeNet drop in by swapping the slices and channel counts.

```python
import torch
import torch.nn as nn
from torchvision import models as tv
import inspect
import os


def normalize_tensor(x, eps=1e-10):
    # channel-wise unit normalization: turns each per-location feature vector into a direction,
    # so loud channels can't dominate and per-layer distances are comparable
    norm = torch.sqrt((x ** 2).sum(dim=1, keepdim=True))
    return x / (norm + eps)


def spatial_average(x, keepdim=True):
    # patch-level reduction: average the per-location distance over (H, W) of this layer
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    # fixed per-channel standardization the frozen backbone expects on its input
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


class AlexBackbone(nn.Module):
    # pretrained classifier, frozen, exposed as 5 intermediate feature stages
    def __init__(self, requires_grad=False):
        super().__init__()
        feats = tv.alexnet(pretrained=True).features
        self.slices = nn.ModuleList()
        for a, b in [(0, 2), (2, 5), (5, 8), (8, 10), (10, 12)]:
            s = nn.Sequential()
            for x in range(a, b):
                s.add_module(str(x), feats[x])
            self.slices.append(s)
        if not requires_grad:                          # freeze: the representation is the asset
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        out, h = [], x
        for s in self.slices:
            h = s(h)
            out.append(h)
        return out                                     # list of (N, C_l, H_l, W_l)


class NetLinLayer(nn.Module):
    # learned nonnegative per-channel coefficients as a 1x1 conv (no bias)
    def __init__(self, chn_in, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LPIPS(nn.Module):
    """Conceptual form:
    d = sum_l mean_hwc alpha_lc * (yhat0^l - yhat1^l)^2, with alpha_lc >= 0."""

    def __init__(self, net='alex', pretrained=True, version='0.1',
                 use_dropout=True, model_path=None, eval_mode=True):
        super().__init__()
        assert net == 'alex'
        self.version = version
        self.scaling_layer = ScalingLayer()
        self.net = AlexBackbone(requires_grad=False)   # frozen trunk (the 'lin' configuration)
        chns = [64, 192, 384, 256, 256]                # AlexNet conv1..conv5 channel counts
        self.L = len(chns)
        self.lin0 = NetLinLayer(chns[0], use_dropout)
        self.lin1 = NetLinLayer(chns[1], use_dropout)
        self.lin2 = NetLinLayer(chns[2], use_dropout)
        self.lin3 = NetLinLayer(chns[3], use_dropout)
        self.lin4 = NetLinLayer(chns[4], use_dropout)
        self.lins = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])
        if pretrained:
            if model_path is None:
                model_path = os.path.abspath(os.path.join(
                    os.path.dirname(inspect.getfile(self.__init__)), 'weights/v%s/%s.pth' % (version, net)))
            self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        if eval_mode:
            self.eval()

    def forward(self, in0, in1, normalize=False):
        if normalize:                                  # inputs given in [0,1] -> map to [-1,1]
            in0, in1 = 2 * in0 - 1, 2 * in1 - 1
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        val = 0
        for l in range(self.L):
            f0, f1 = normalize_tensor(outs0[l]), normalize_tensor(outs1[l])  # channel L2-normalize
            diff = (f0 - f1) ** 2                                            # squared normalized diff
            d_l = spatial_average(self.lins[l](diff))                        # learned channel sum + mean
            val = val + d_l                                                  # sum over layers
        return val


class Dist2LogitLayer(nn.Module):
    # small G mapping a distance pair to P(h=1, the second distorted patch is preferred)
    def __init__(self, chn_mid=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, chn_mid, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, chn_mid, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, 1, 1), nn.Sigmoid())

    def forward(self, d0, d1, eps=0.1):
        # feed the two distances and a few stable combinations of them
        return self.model(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super().__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = nn.BCELoss()

    def forward(self, d0, d1, judge_signed):
        target = (judge_signed + 1.0) / 2.0
        return self.loss(self.net(d0, d1), target)


def train_on_human_judgments(metric, rank_loss, triplet_loader, nepoch=5, nepoch_decay=5, lr=1e-4):
    metric.train()
    rank_loss.train()
    params = list(metric.lins.parameters()) + list(rank_loss.net.parameters())   # frozen trunk: linear coeffs + G
    opt = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    for epoch in range(1, nepoch + nepoch_decay + 1):
        for x, x0, x1, h in triplet_loader:
            d0, d1 = metric(x, x0), metric(x, x1)
            loss = rank_loss(d0, d1, h.view_as(d0) * 2.0 - 1.0)       # raw h=0 for x0, h=1 for x1
            opt.zero_grad(); loss.backward(); opt.step()
            for lin in metric.lins:                                  # project learned coeffs >= 0
                for m in lin.model:
                    if isinstance(m, nn.Conv2d):
                        m.weight.data.clamp_(min=0)
        if epoch > nepoch:                                           # linear decay, as in the training script
            new_lr = lr * max(0, 1 - (epoch - nepoch) / float(nepoch_decay))
            for group in opt.param_groups:
                group['lr'] = new_lr
```

And to use it as an auxiliary perceptual loss, I only need to feed valid images on the range the frozen features expect:

```python
lpips = LPIPS(net='alex', pretrained=True).eval()      # frozen, calibrated metric used as a loss

# x_hat and x_target are image tensors already on [-1, 1]; use normalize=True for [0, 1] inputs
loss_perceptual = lpips(x_hat, x_target)               # differentiable; backprops into x_hat
loss = loss_primary + lam * loss_perceptual
```

So the causal chain closes. Per-pixel `ℓ2` is blind to structure because it treats pixels as independent, so it over-penalizes imperceptible warps and under-penalizes blur; the hand-built metrics fix some of that with local statistics but are shallow, frozen formulas that break on spatial ambiguity and have nothing to adapt to human data; directly fitting a similarity predictor to human pairwise labels asks a context-dependent target to create an entire representation. The way through is to borrow a visual representation trained for another prediction task and use human judgments only to calibrate distances inside it. Channel-normalizing the features makes the comparison direction-based, so unit weights reduce to cosine distance; averaging over spatial positions and summing layers aggregates multi-scale evidence; learning a tiny set of nonnegative per-channel coefficients and fitting them through a five-input BCE preference head gives just enough adaptation while keeping the trunk fixed. The result is a cheap, differentiable, calibrated perceptual distance that drops in as either an evaluation metric or a training loss.
