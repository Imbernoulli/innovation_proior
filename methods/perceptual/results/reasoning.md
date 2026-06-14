Let me start from the thing that keeps biting me: I want to know whether two images look alike, and every cheap distance I have lies about it. Per-pixel `l2` — `d(x,y) = mean_i (x_i - y_i)^2` — treats the image as a bag of independent pixels, and that independence assumption is exactly where it breaks. Blur a photo: the average squared residual can stay small while the perceptual damage is obvious, because the high-frequency structure that makes the image look sharp has been washed out. Shift the whole image one pixel to the right: enormous `l2`, essentially zero perceptual change. So `l2` is measuring something — energy of the residual — that is only loosely coupled to what my visual system reports. PSNR is just `l2` on a log axis; same disease. The hand-designed structural metrics are the field's attempt to patch this. SSIM compares local windows by luminance, contrast, and structure,

```
SSIM(x,y) = [(2 mu_x mu_y + C1)(2 sigma_xy + C2)] / [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)],
```

and it genuinely beats raw pixel error because at least it looks at correlations within a window instead of pixel-by-pixel. But it is a fixed, shallow function of low-order local statistics with a couple of hand-tuned constants `C1, C2`. It has no notion of high-order structure, and it is known to fall apart when there is geometric ambiguity — a small spatial warp confuses the local-statistics comparison. None of these has ever been fit to what humans actually say. So the real obstacle is staring at me: human similarity is about *high-order* structure, it is *context-dependent* — I can simultaneously hold "same color" and "same shape" as competing senses of similar — and it might not even be a proper metric, since human judgments can be asymmetric and break the triangle inequality. If I try to brute-force regress a function straight onto pairwise human judgments over the whole space of images, the context-dependence and the dimensionality look like they'll sink me. I need a representation in which the *easy* distance — plain Euclidean — is already most of the way to perceptual, so I'm only making a small correction rather than learning the whole thing from scratch.

Where would such a representation come from? Here is the suggestive fact from image synthesis. When people train an image generator under a per-pixel `l2` loss, the outputs come out blurry — over-smoothed. The reason is the same independence sin in another guise: squared pixel error is minimized by *averaging* over all the plausible high-frequency completions, so the generator hedges and produces the mean, which is mush. The fix that swept the field was to stop measuring the error in pixel space and measure it in the feature space of a pretrained classification network instead — VGG features, say — and suddenly the outputs are sharp and natural. Johnson and colleagues wrote that feature-reconstruction loss as

```
ell_feat(y_hat, y) = (1 / (C_j H_j W_j)) || phi_j(y_hat) - phi_j(y) ||_2^2,
```

the squared Euclidean distance between the activations `phi_j` at some layer `j`. And the diagnostic that comes with it is the part I should really chew on: if you optimize an image to minimize this at an *early* VGG layer, the result is visually indistinguishable from the target; if you minimize it at a *deep* layer, the content and overall layout survive but the exact color, texture, and shape do not. Read that again. That means Euclidean distance *in these feature spaces* already behaves perceptually — small feature distance at low layers really does mean "looks the same," and the layers form a low-to-high ladder from appearance to semantics. The networks were never trained to do this. They were trained to classify ImageNet. The perceptual behavior is a side effect of learning a representation that is predictive about real images.

So my candidate is almost embarrassingly simple: take a fixed network, push both patches through, and measure Euclidean distance between their feature stacks. But "almost" is doing a lot of work, and I should not just declare victory — the perceptual-loss people themselves flagged a problem with using raw feature distance as an *objective*: optimizing it alone gives high-frequency garbage, because many non-natural images map to the same feature vector, so they always combined it with other terms. That worry is about *generating* an image to minimize the distance (you can exploit the null space of the feature map). For *evaluating* a distance between two given natural images it's much less of an issue — I'm not synthesizing anything, I'm scoring a pair. Still, it tells me the raw feature distance is not automatically the right perceptual metric; it's a starting point that needs shaping.

Let me actually look at what "Euclidean distance between feature stacks" computes and find where it's crude. At layer `l` I have two feature maps `y^l, y0^l`, each of shape `H_l x W_l x C_l`. The naive thing is `sum over h,w,c (y^l_{hwc} - y0^l_{hwc})^2`, then sum that over layers. First problem: the channels of a convolutional layer have wildly different activation magnitudes — some filters fire hard on common textures, others barely move — so this sum is dominated by a few high-energy channels, and what it's really comparing is whether those few loud channels matched in *gain*, not whether the overall *pattern* of activation agrees. That's not what I want; perceptual similarity should be about which features are present, the direction of the activation vector across channels, not its raw length. The fix is forced by that observation: at each spatial location, unit-normalize the activation vector across the channel dimension before comparing. Write `y_hat^l_{hw} = y^l_{hw} / ||y^l_{hw}||_2`, the channel vector scaled to unit length (with a tiny `eps` in the denominator so a near-dead location doesn't blow up). Now the per-location comparison is between two unit vectors, i.e. it's governed by their angle — a cosine-like comparison of which features are active, insensitive to overall activation energy. And notice what plain summed-squared-difference of unit vectors is: `||y_hat - y0_hat||^2 = 2 - 2 <y_hat, y0_hat>`, an affine function of cosine similarity. So after channel-normalization, this *is* a cosine distance in feature space, which is exactly the gain-invariant thing I argued I wanted.

Stack it up across the patch and across layers. Within a layer, average the per-location squared differences over the `H_l W_l` spatial positions, so every layer contributes a patch-level scalar rather than a number that grows with feature-map size; the convolutional trunk and its pooling already provide the limited local tolerance, and the average is just the clean way to pool the remaining local disagreements. Then sum across layers, because the early-vs-deep diagnostic told me different layers carry different aspects of similarity — low-level appearance down low, semantics up high — and a perceptual judgment uses both. So the distance so far is

```
d(x, x0) = sum_l (1 / (H_l W_l)) sum_{h,w} || y_hat^l_{hw} - y0_hat^l_{hw} ||_2^2.
```

This is the unlearned baseline. It weights every channel in every layer equally — it is literally summed cosine distance. And I can already guess it's leaving accuracy on the table, because surely not every channel is equally relevant to *human* perception. A channel that lights up on some texture statistic a human ignores should count for less than one that tracks a salient edge. But which channels matter? I have no first-principles way to know — and crucially, I now *have* a way to find out, because the whole point of collecting 2AFC human judgments is to ground exactly this.

The feature space is already doing the hard representational work, so the human data should not be asked to learn a giant new function on top of it. That route would run straight back into context-dependence and overfitting. What the judgments can plausibly ground is a recalibration: one coefficient per channel per layer, applied to the amount of squared disagreement in that channel. If I write the mathematical distance as a channel scale `w^l in R^{C_l}` before the squared norm, the coefficient on `(Delta_c)^2` is `(w_c^l)^2`; if I implement the same idea as a linear layer on the squared-difference map, I learn the non-negative coefficient on `(Delta_c)^2` directly:

```
d(x, x0) = sum_l (1 / (H_l W_l)) sum_{h,w} || w^l ⊙ ( y_hat^l_{hw} - y0_hat^l_{hw} ) ||_2^2
         = sum_l (1 / (H_l W_l)) sum_{h,w} sum_c a^l_c ( y_hat^l_{hwc} - y0_hat^l_{hwc} )^2,
           with a^l_c = (w^l_c)^2 and a^l_c >= 0.
```

Setting the coefficient `a^l = 1` for all `l` recovers the cosine-distance baseline exactly, which is the right sanity check: the learned metric is a strict generalization of the unlearned one, so calibration can only help if the data supports it. And I should pin down *why* the learned linear coefficient has to be non-negative, because that constraint is not cosmetic. Each term `(y_hat_c - y0_hat_c)^2` is the amount by which channel `c` *disagrees* between the two patches. If the coefficient multiplying that squared disagreement were negative, then making a feature *more* different would *decrease* the total distance — a patch that diverges harder in some feature could be scored as more similar. That's perceptual nonsense: more disagreement in any feature must never pull two patches closer. So the linear coefficients on squared disagreements must be `>= 0`. I'll enforce that by projection during training — after each gradient step, clamp any negative coefficient back to zero.

How do I actually fit the channel coefficients to 2AFC data? Each datum is a triplet `(x, x0, x1, h)` with `h in {0,1}` encoding the chosen side under a fixed convention. I can compute `d0 = d(x, x0)` and `d1 = d(x, x1)` with my current coefficients. The crude thing would be a fixed-margin ranking loss: push `d0` and `d1` apart by some constant margin in the direction `h` says. But a fixed margin is wrong, because how *much* closer one patch is varies enormously across triplets — some pairs are nearly tied, some are night-and-day, and forcing the same margin everywhere fights the data. I'd rather *learn* the mapping from a distance pair to a judgment probability. So put a tiny network `G` on top that takes `(d0, d1)` and outputs `h_hat in (0,1)`, the predicted probability of label `h=1` under that convention. Two small fully-connected-ReLU layers and a final unit with a sigmoid is plenty — `G` only has to learn a soft, possibly-asymmetric decision boundary in the two-dimensional `(d0, d1)` space. I'll feed it not just `d0, d1` but also their difference and ratios, `[d0, d1, d0 - d1, d0/(d1+eps), d1/(d0+eps)]`, since the human decision is plausibly driven by *relative* closeness, and handing `G` the difference and ratio directly saves it from having to discover them. Then train both the channel coefficients (through `d0, d1`) and `G` end to end against the judgments. Because `h_hat` is a probability and `h` is a binary (or, when two judges split, fractional) target, the natural objective is binary cross-entropy:

```
L(x, x0, x1, h) = - h log G(d0, d1) - (1 - h) log( 1 - G(d0, d1) ).
```

When the two human judges on a training pair disagree, I set the target to `0.5` rather than forcing a hard label — the loss should treat a genuinely ambiguous pair as ambiguous. (I did briefly weigh that fixed-margin ranking objective; the learned `G` should win, precisely because it stops demanding a uniform margin where the data has none.)

Now, what exactly do I let learn? I keep the feature trunk `F` *frozen* and learn only the linear channel coefficients and `G`. This is the cleanest and most defensible choice, and the reason is the whole thesis: the perceptual structure I'm exploiting is the *emergent* property of the classification-trained features, and the linear layer is a small recalibration of that space — for VGG it's only on the order of a thousand-some scalars (one per channel summed over the chosen layers), a featherweight correction. If instead I unfroze `F` and fine-tuned it on the low-level perceptual task, I'd be retraining the very representation whose accidental perceptual goodness is the thing I want to preserve, and I'd risk bending it toward the idiosyncrasies of this particular distortion set and away from the broad structure that made it work. The conservative read is: calibrate, don't overwrite. (There are obviously fine-tune and train-from-scratch variants worth checking, but the linear calibration of a frozen, pretrained trunk is the one I trust to generalize.)

The tensor implementation should operate on the squared-difference map because that is what the network already has in hand after channel-normalization. The formula scales the difference by `w_c` before the norm, which contributes `w_c^2 (Delta_c)^2`; the code can equivalently learn the non-negative coefficient `a_c = w_c^2` multiplying each squared-difference channel. That map from `(Delta_c)^2` to a single per-location number is just a 1x1 convolution: `C_l` input channels, `1` output channel, no bias, applied to `(y_hat^l - y0_hat^l)^2`. So I don't need a bespoke weighting op; I compute `diffs = (y_hat^l - y0_hat^l)^2`, run a `1x1` conv whose weights are the learned non-negative channel coefficients, then spatially average. The non-negativity projection is then just clamping that conv's weights at zero after each step. Folding it this way also lets me drop dropout in front of the conv during training as a mild regularizer on the learned coefficients without touching the distance formula.

One more thing the trunk needs at its input. The pretrained backbones were trained on ImageNet-normalized inputs, so to keep their features in the regime they were trained in, I pass each patch through a fixed scaling layer first — subtract a per-channel shift and divide by a per-channel scale matched to the training statistics — before the trunk. It's a fixed, non-learned normalization; getting it wrong (e.g. not scaling at all) just moves the features off-distribution and quietly degrades everything, which is the kind of bug that's easy to ship and hard to notice.

So now I have a complete object: fixed trunk with input scaling, channel-wise unit-normalize each layer's activations, take squared differences, weight them per channel by learned non-negative linear coefficients (a `1x1` conv, equivalent to the squared channel scale in the norm expression), spatially average per layer, sum over layers — and fit those coefficients plus the little `G` to 2AFC judgments by BCE with non-negativity projected after each step. The validation has to ask whether this is tracking perception rather than becoming just a fourth image-quality score.

The validation I'd run is: score each method by how well its ordering of triplets agrees with the human votes, where the right ceiling isn't 100% — humans themselves disagree. If a fraction `p` of people pick one side, then a human agent picking with the same distribution `p` agrees with the crowd `p^2 + (1-p)^2` of the time in expectation (probability both pick side one, `p*p`, plus both pick side two, `(1-p)(1-p)`), and an oracle that always picks the majority gets `max(p, 1-p)`. So the human-consistency number, somewhere below 100%, is the bar. And I'd want a *second*, less subjective perceptual test to make sure 2AFC isn't just measuring some cognitively-penetrable quirk — a just-noticeable-difference test ("are these two the same or different?"), scored by whether the metric ranks confusable pairs as close, where there's a single objective answer per pair. If a metric trained/calibrated for 2AFC also does well on JND, that's evidence it's tracking something real about perception rather than fitting the 2AFC protocol.

Now I need to ask what would be responsible if the simple feature distance actually works. The deep classification features should beat `l2` and SSIM out of the box — that's basically the premise I started from, the perceptual-loss observation, just measured against people instead of eyeballed. The non-obvious question is whether the cause is ImageNet classification specifically, the architecture, or simply having trained weights at all. A randomly-initialized network of the same architecture still has the convolutional structure, the multi-scale receptive fields, the nonlinearities — so if architecture alone were enough, random weights would already give a good perceptual distance. I doubt it will, because random filters point in arbitrary directions and won't concentrate on the structure that's actually dense in natural images; the channel directions would be perceptually meaningless. At the other extreme, take networks of the *same* architecture trained *without* class labels — self-supervised tasks like solving jigsaw puzzles, cross-channel/colorization prediction, inpainting, predicting motion in video, adversarial feature learning, even a purely data-dependent unsupervised stacked-k-means initialization. None of these ever saw a class label. If perceptual similarity were a special consequence of *classification* in particular, these should be much worse. But by the logic of why the features are perceptual in the first place — they're representations tuned to be *predictive about the structure of natural images* — any task that forces the network to model that structure should induce a similar space. So my prediction is: supervised, self-supervised, and unsupervised-with-real-data all land in a similar, strong band; random weights do not. Then the conclusion would be that perceptual similarity is not its own special function; it is an *emergent* property of essentially any representation that has been trained to predict real-image structure, shared across architectures and across levels of supervision. That's a much stronger and stranger claim than "VGG features make a good loss," and it's why I'd call the effectiveness of these deep features unreasonable: the thing falls out almost regardless of *how* you trained the net, as long as you trained it on something real.

Let me also be honest about where I'd expect the calibration to help and where it might not. Calibrating the channel coefficients on the traditional and CNN-based distortions should improve agreement on those, since that's literally what it's fit to. The real test is transfer to held-out outputs of *actual* algorithms — super-resolution, frame interpolation, deblurring, colorization — sampled as patch triplets, which is the real-world use case. A small, frozen-trunk linear calibration is the kind of light touch I'd expect to transfer safely; the heavier I learn (fine-tune the trunk, or train from scratch on this distortion set), the more I'd worry about specializing to the calibration distortions at the cost of transfer. So the design bias toward "minimal learned correction on top of a frozen, broadly-trained representation" isn't just elegance — it's the version most likely to still work on distortions it never saw.

Let me write it as the code I'd actually ship, filling the single empty slot from the harness — the function that turns two feature stacks into a scalar, plus the tiny calibration machinery it learns.

```python
import torch
import torch.nn as nn


def normalize_tensor(feat, eps=1e-10):
    # unit-normalize across the CHANNEL dimension at each spatial location:
    # compare feature *direction* (cosine), not raw activation energy.
    norm = torch.sqrt(torch.sum(feat ** 2, dim=1, keepdim=True))
    return feat / (norm + eps)


def spatial_average(x, keepdim=True):
    # one scalar per layer; normalize for feature-map size.
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    # fixed (non-learned) input normalization so patches enter the trunk in the
    # ImageNet regime the trunk was trained on.
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    # non-negative per-channel coefficients as a 1x1 conv -> 1 output channel,
    # no bias: out_{hw} = sum_c a_c * (squared-diff map)_{c,hw}.
    def __init__(self, chn_in, use_dropout=True):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PerceptualDistance(nn.Module):
    """Channel-normalized, per-channel-calibrated distance in a frozen trunk's feature space."""

    def __init__(self, trunk, chns, lpips=True, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.trunk = trunk                 # frozen pretrained backbone, exposes L feature maps
        self.L = len(chns)
        self.lpips = lpips                 # True: learned channel coefficients; False: cosine baseline
        if lpips:
            self.lins = nn.ModuleList([NetLinLayer(c, use_dropout=use_dropout) for c in chns])

    def forward(self, x0, x1):
        f0 = self.trunk(self.scaling_layer(x0))   # [f0^0, ..., f0^{L-1}]
        f1 = self.trunk(self.scaling_layer(x1))
        d = 0
        for l in range(self.L):
            a0 = normalize_tensor(f0[l])          # unit-normalize channels (cosine comparison)
            a1 = normalize_tensor(f1[l])
            diff = (a0 - a1) ** 2                  # per-channel squared disagreement
            if self.lpips:
                per_layer = spatial_average(self.lins[l](diff))   # sum_c a_c * diff_c, then space-avg
            else:
                per_layer = spatial_average(diff.sum(dim=1, keepdim=True))  # w=1: cosine distance
            d = d + per_layer                     # sum over layers
        return d                                   # scalar (per batch element) distance


class Dist2Logit(nn.Module):
    # tiny G: maps a distance pair to a judgment probability via [d0, d1, d0-d1, d0/d1, d1/d0].
    def __init__(self, chn_mid=32, eps=0.1):
        super().__init__()
        self.eps = eps
        self.model = nn.Sequential(
            nn.Conv2d(5, chn_mid, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, chn_mid, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, d0, d1):
        feats = torch.cat((d0, d1, d0 - d1, d0 / (d1 + self.eps), d1 / (d0 + self.eps)), dim=1)
        return self.model(feats)


def train_calibration(dist, G, data_loader):
    # learn linear channel coefficients (inside dist) and G; trunk stays frozen.
    params = [p for n, p in dist.named_parameters() if 'lins' in n] + list(G.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    for x, x0, x1, h in data_loader:        # h in [0,1]; split judges -> 0.5
        d0 = dist(x, x0)
        d1 = dist(x, x1)
        h_hat = G(d0, d1)
        loss = bce(h_hat.view_as(h), h)     # -h log h_hat - (1-h) log(1-h_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for m in dist.modules():            # project coefficients >= 0: disagreement cannot reduce distance
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                m.weight.data.clamp_(min=0)
```

So the causal chain, start to finish. Per-pixel `l2`/PSNR lie about perception because they assume pixel independence — blur and shift fool them — and the hand-designed structural metrics (SSIM and kin) only patch this with shallow local statistics and break on geometric distortion, with nothing learned from people. The synthesis community had stumbled on the key lever: distance in a pretrained classification network's feature space behaves perceptually — early-layer feature distance means "looks the same," and the layers ladder from appearance to semantics — and it does so as an accident of learning a representation predictive about real images. So I measure distance there, not in pixels. Raw feature distance is dominated by a few high-energy channels, so I unit-normalize across channels at each location, turning the per-location comparison into a gain-invariant cosine of feature direction; then I spatially average within a layer and sum across layers to use both low- and high-level cues. The unlearned cosine baseline still weights every channel equally, which cannot be the final perceptual story, so I add the single learned correction the human data can ground: non-negative per-channel coefficients on squared feature disagreement, realized as a `1x1` conv and fit — along with a tiny network `G` mapping a distance pair to a judgment probability — by binary cross-entropy on 2AFC triplets, with the coefficients projected to be non-negative after each step and the trunk kept frozen so the calibration stays a light touch on top of the emergent representation. And the deeper payoff I'd expect to see when validating: the perceptual behavior shows up across architectures and across supervised, self-supervised, and even unsupervised training, but not for random weights — perceptual similarity isn't a special function, it's what you get for free from almost any representation trained to predict the structure of the visual world.
