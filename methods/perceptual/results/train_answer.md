I want a distance $d(x, x_0)$ on image patches whose ordering agrees with what a person would judge similar, and every cheap candidate lies about it in a specific way. Per-pixel $\ell_2$ (or its log form, PSNR) treats each pixel as an independent coordinate, so it is fooled in both directions: blurring a photo can leave the average squared residual small even though the perceptual damage is obvious, and shifting the whole image by one pixel produces a large $\ell_2$ error for essentially no perceptual change. SSIM and its relatives (MS-SSIM, FSIM) patch this a little by comparing local luminance, contrast, and structure statistics inside sliding windows,
$$\mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)},$$
but they are shallow, fixed functions of low-order local statistics tuned by a couple of hand-set constants, and they were never fit to what people actually say. The real obstacle is that human similarity depends on high-order structure rather than local pixel statistics, is context-dependent (a red circle can be judged closer to a red square or to a blue circle depending on which respect of similarity is in view), and may not even satisfy the axioms of a distance metric. Regressing a function directly onto pairwise human judgments over the whole space of natural images looks hopeless — the context-dependence and the dimensionality would sink a model with no useful prior to lean on. What I need instead is a representation in which the *cheap* distance, plain Euclidean, is already most of the way to perceptual, so that fitting to human data becomes a small correction rather than learning everything from scratch.

The clue for where such a representation might live comes from image synthesis. Generators trained under a per-pixel $\ell_2$ loss produce blurry outputs, because squared pixel error is minimized by averaging over the many plausible high-frequency completions of an ambiguous region rather than committing to one of them. Replacing the pixel loss with a distance computed in the feature space of a pretrained classification network — the "feature-reconstruction" or perceptual loss, $\ell_{\text{feat}}^{\phi,j}(\hat y, y) = \frac{1}{C_jH_jW_j}\|\phi_j(\hat y)-\phi_j(y)\|_2^2$ — removes the blur and yields sharp, natural-looking images. The diagnostic that makes this worth chasing further is that optimizing an image to match a target's activations at an *early* layer of such a network reproduces the target almost exactly, while matching a *deep* layer preserves content and overall layout but not exact color, shape, or texture. That is a hint that Euclidean distance *inside these feature spaces* already behaves perceptually — even though the network was only ever trained to classify ImageNet, never to judge similarity — with the layer ladder running from low-level appearance to high-level semantics. It is a hint read off a qualitative reconstruction diagnostic, not a result measured against human judgments, so the natural next step is to test the idea directly against people and then correct whatever it gets wrong.

I call the resulting distance LPIPS, Learned Perceptual Image Patch Similarity. Take a frozen, pretrained convolutional trunk $F$ (VGG, AlexNet, or SqueezeNet) with $L$ chosen layers, push both patches through it, and start from the naive idea of summing squared feature differences across layers. That naive version is wrong in a specific, checkable way: the channels of a convolutional layer differ wildly in typical activation magnitude, so a summed squared difference is dominated by whichever channels happen to fire loudest, independent of whether they encode anything a person would call "different." A two-channel toy makes this concrete: comparing $a=(10.0,0.1)$ against $b=(10.0,-0.1)$ (the quiet channel flips sign — a real change in which feature is active) and against $c=(5.0,0.1)$ (the loud channel merely halves in gain, same direction, quiet channel untouched) gives raw squared distances of $0.04$ and $25.0$ respectively — the metric calls $a$ roughly six hundred times closer to $b$, the one that actually changed direction, than to $c$, which didn't. The fix is to unit-normalize the activation vector across channels at every spatial location before comparing, $\hat y^l_{hw} = y^l_{hw}/(\|y^l_{hw}\|_2+\epsilon)$, so what gets compared is feature *direction*, not raw energy. On the same toy numbers this reverses the ordering to the correct one, and in general, since $\|u-v\|_2^2 = 2-2\langle u,v\rangle$ for unit vectors, comparing normalized activations makes summed squared difference an affine function of cosine similarity — a gain-invariant cosine distance in feature space.

Within a layer I average the per-location squared differences over the $H_l\times W_l$ spatial positions, so each layer contributes one patch-level scalar rather than a number that grows with feature-map size, and I sum over layers, since the early-vs-deep diagnostic says different layers carry different aspects of similarity that a perceptual judgment ought to combine. That gives the unlearned baseline
$$d(x,x_0)=\sum_l \frac{1}{H_lW_l}\sum_{h,w}\big\|\hat y^l_{hw}-\hat y_0{}^l_{hw}\big\|_2^2,$$
which weights every channel in every layer equally — literally a summed cosine distance. It is unlikely every channel matters equally to a human: some encode texture statistics people ignore, others track salient edges, and there is no first-principles way to tell which is which from the network alone. What I do have is 2AFC human-judgment data, and the right use of it is a *recalibration* of this feature space, not a new function learned on top of it — asking the judgments to learn a whole new mapping would run straight back into the context-dependence problem I started by trying to avoid. So I attach one non-negative coefficient $w^l_c$ per channel per layer, scaling the normalized-feature difference before the squared norm is taken,
$$d(x,x_0)=\sum_l\frac{1}{H_lW_l}\sum_{h,w}\big\|w^l\odot(\hat y^l_{hw}-\hat y_0{}^l_{hw})\big\|_2^2=\sum_l\frac{1}{H_lW_l}\sum_{h,w}\sum_c a^l_c\big(\hat y^l_{hwc}-\hat y_0{}^l_{hwc}\big)^2,\qquad a^l_c=(w^l_c)^2\ge 0.$$
Setting every $a^l_c=1$ collapses this back exactly to the unlearned cosine baseline, so calibration is a strict generalization of it: the learned metric can only move away from the baseline if the human data pays for the move. The coefficients have to stay non-negative for a structural reason, not a cosmetic one — each term $(\hat y_c-\hat y_{0,c})^2$ measures how much channel $c$ disagrees between the two patches, and a negative coefficient would let *more* disagreement in some channel *lower* the reported distance, which is perceptual nonsense (checkable directly: flipping one coefficient negative turns an increase in channel disagreement from $0.8\to1.2$ into a decrease to $-0.6$). In the implementation the per-channel weighting is just a $1\times1$ convolution over the squared-difference map — $C_l$ channels in, one channel out, no bias — whose weights are clamped to be $\ge 0$ after every optimizer step.

To fit those coefficients, each training example is a triplet $(x,x_0,x_1,h)$ from a two-alternative forced choice: a reference patch, two distorted versions, and which one a human judged closer, $h\in\{0,1\}$ (or $0.5$ when two judges split on the same triplet). Rather than push $d_0=d(x,x_0)$ and $d_1=d(x,x_1)$ apart by a fixed margin — wrong, because how much closer one patch is varies enormously across triplets, from nearly tied to obviously different, and a uniform margin fights the data — I learn a small network $G$ that maps the distance pair to a judgment probability $\hat h=G(d_0,d_1)\in(0,1)$. I feed it not just $d_0,d_1$ but also their difference and ratios, $[d_0,d_1,d_0-d_1,d_0/(d_1+\epsilon),d_1/(d_0+\epsilon)]$, so it doesn't have to rediscover relative closeness on its own. Two 32-unit fully-connected LeakyReLU(0.2) layers and a sigmoid output are enough, since $G$ only has to learn a soft, possibly asymmetric decision boundary in a low-dimensional space. Everything — the per-channel coefficients and $G$ — trains end to end by binary cross-entropy,
$$\mathcal L(x,x_0,x_1,h) = -h\log G(d_0,d_1) - (1-h)\log\big(1-G(d_0,d_1)\big),$$
with Adam, learning rate $10^{-4}$, $\beta_1=0.5$, five epochs at that rate then five epochs of linear decay, batch size 50, and the non-negativity projection applied after every step. The trunk $F$ itself stays frozen throughout: the perceptual usefulness of these features is an emergent property of training on real-image structure, and I only want a small correction on top of it — for VGG the learned coefficients number about 1472 scalars, for AlexNet's five conv layers $64+192+384+256+256=1152$ — not a retraining that would bend the representation toward the idiosyncrasies of this particular distortion set. Before the trunk, a fixed (non-learned) scaling layer subtracts a per-channel shift and divides by a per-channel scale matched to ImageNet training statistics, so patches enter the trunk in the regime it was trained in; skipping this quietly moves the features off-distribution and degrades everything without an obvious symptom.

Because the calibration is fit on traditional and CNN-generated distortions, the real test of transfer is held-out outputs of actual algorithms — super-resolution, frame interpolation, video deblurring, colorization — that the calibration never saw, and a light, frozen-trunk linear correction (the "lin" variant, as opposed to fine-tuning the whole trunk or training one from scratch) is the version of this design most likely to still hold up there; a heavier fit would specialize harder to the calibration set at the cost of exactly that transfer. This also makes a prediction about *where* the perceptual behavior comes from in the first place: if it were a special consequence of ImageNet classification, self-supervised and unsupervised networks of the same architecture should do noticeably worse, and random weights of that architecture — same convolutions, same receptive fields, no learned filters — should do worst of all, since arbitrary filter directions have no reason to align with structure that is actually dense in natural images. If instead any network trained to model real-image structure lands in a similar strong band while only the untrained one falls away, perceptual similarity is not a special function of classification at all — it is close to a generic byproduct of training a deep net on real data, largely independent of the exact task used to train it.

```python
import torch
import torch.nn as nn


def normalize_tensor(feat, eps=1e-10):
    # unit-normalize across channels at each location -> cosine comparison of feature direction
    norm = torch.sqrt(torch.sum(feat ** 2, dim=1, keepdim=True))
    return feat / (norm + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    # fixed ImageNet-style input normalization used before the feature trunk
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    # per-channel learned coefficient as a 1x1 conv -> 1 channel, no bias
    def __init__(self, chn_in, use_dropout=True):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LPIPS(nn.Module):
    def __init__(self, trunk, chns, lpips=True, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.net = trunk                      # frozen pretrained backbone; forward -> list of L feats
        self.chns = chns
        self.L = len(chns)
        self.lpips = lpips                    # True: learned coefficients; False: cosine baseline
        if lpips:
            self.lins = nn.ModuleList([NetLinLayer(c, use_dropout=use_dropout) for c in chns])

    def forward(self, in0, in1, normalize=False):
        if normalize:                         # flip if inputs are in [0,1] -> [-1,1]
            in0, in1 = 2 * in0 - 1, 2 * in1 - 1
        out0 = self.net(self.scaling_layer(in0))
        out1 = self.net(self.scaling_layer(in1))
        val = 0
        for kk in range(self.L):
            f0, f1 = normalize_tensor(out0[kk]), normalize_tensor(out1[kk])
            diff = (f0 - f1) ** 2
            if self.lpips:
                res = spatial_average(self.lins[kk](diff), keepdim=True)
            else:
                res = spatial_average(diff.sum(dim=1, keepdim=True), keepdim=True)
            val = val + res
        return val


class Dist2LogitLayer(nn.Module):
    # G: distance pair -> judgment probability, via [d0, d1, d0-d1, d0/d1, d1/d0]
    def __init__(self, chn_mid=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, chn_mid, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, chn_mid, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, d0, d1, eps=0.1):
        return self.model(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super().__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = nn.BCELoss()

    def forward(self, d0, d1, judge):         # judge in [-1,1]; per in [0,1]
        per = (judge + 1.) / 2.
        self.logit = self.net(d0, d1)
        return self.loss(self.logit, per)


def clamp_weights(net):                       # project learned 1x1-conv coefficients >= 0
    for m in net.modules():
        if hasattr(m, 'weight') and getattr(m, 'kernel_size', None) == (1, 1):
            m.weight.data = torch.clamp(m.weight.data, min=0)
```
