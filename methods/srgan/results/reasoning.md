Let me start from the thing that's actually wrong, because I think the field has been optimizing the wrong quantity and I want to be precise about it before I build anything. The task is single-image super-resolution: given a low-resolution image `I^LR`, produce a high-resolution `I^SR` that matches the true `I^HR`. Everyone trains a network to minimize the per-pixel mean squared error between the output and the ground truth, because minimizing MSE also maximizes PSNR, and PSNR is the number everyone reports. The networks keep getting deeper and faster and the PSNR keeps inching up. And yet the outputs, at large upscaling factors, look *bad* — washed out, smooth, the fine texture gone. So either the perceptual problem is just hard, or the objective itself is causing the smoothing. I suspect the latter, and I want to nail down why.

Think about what super-resolution actually is at `4×`. That's a `16×` increase in pixel count, and the high-frequency texture detail simply isn't present in the input — it has to be plausibly *invented*. So for a single `I^LR`, there isn't one correct `I^HR`; there's a whole *set* of high-resolution images, all sharp, all consistent with that input, differing in exactly the fine texture I can't recover from the low-res. The problem is underdetermined, and the ambiguity lives precisely in the high frequencies I care about most.

Now ask what minimizing pixel-wise MSE does in the face of that set. MSE penalizes squared pixel distance to the one ground-truth example I happen to have. If I train a network to minimize expected squared error, the minimizer at each input is the *conditional mean* — `E[I^HR | I^LR]` — the pixel-wise *average* over all those plausible sharp solutions. And here is the trap: averaging a bunch of sharp images that each have texture in *different* places gives you a smooth image with texture *nowhere*. The average of many crisp solutions is blurry. Worse, that average need not even be a natural image at all — real photographs lie on a thin manifold in pixel space, and the centroid of several manifold points generally sits *off* the manifold, in the smooth region between them. So the blur isn't a failure of capacity that a bigger network fixes; it's the *direct, intended* behavior of the per-pixel objective. The objective is asking for the average, and the average is smooth and off-manifold. Stack a hundred layers and MSE will still hand you the smooth centroid, just computed more precisely.

So I need to stop asking for the average. What I actually want is for the output to land on *one* of the plausible sharp solutions — any single point *on* the natural-image manifold — rather than at the smooth centroid between them. That reframing is the whole game: pull the reconstruction *onto the manifold of natural images* instead of toward the mean.

What could supply a "this looks like a real photo / get onto the manifold" signal? I need something that scores an image by how natural it looks, not by how close it is pixel-by-pixel to one reference. A fixed hand-designed naturalness prior would be brittle. But there's a learned alternative: train a discriminator network to distinguish real high-resolution photographs from my generator's super-resolved outputs, and train the generator to fool it. The discriminator *is* a learned "is this on the manifold of real images?" detector, and its gradient pushes the generator's outputs toward regions of image space that the discriminator can't tell apart from real photos — that is, toward the manifold. Crucially the adversarial signal doesn't care whether the texture is in exactly the right place pixel-wise; it only cares that the texture is *plausible*. So it actively rewards committing to one sharp solution rather than hedging toward the blurry average. That's exactly the opposite pressure from MSE. So: keep a generator `G` that maps `I^LR → I^SR`, add a discriminator `D` trained to separate real `I^HR` from `G(I^LR)`, and play the minimax game

    min_{θ_G} max_{θ_D}  E_{I^HR}[ log D(I^HR) ] + E_{I^LR}[ log(1 − D(G(I^LR))) ].

But the adversarial term alone is dangerous — nothing ties `G(I^LR)` to the *content* of this particular input. A generator could produce a gorgeous, perfectly natural-looking image that has nothing to do with the low-res input and still fool `D`. I need a *content* term that says "and it must be a super-resolution of *this* image." The obvious content term is pixel MSE again — but that's the very thing that smooths. So I'm stuck between needing a content anchor and not wanting pixel MSE's averaging.

Resolve it by changing the *space* in which I measure content similarity. The reason pixel MSE forces the blurry average is that it's defined on raw pixels, so any texture that's shifted or differs by a pixel gets fully penalized, and the only way to be safe everywhere is to smooth. But if I compare the two images in the *feature space* of a deep network instead — say the activations of a pre-trained VGG19 — those features are far more invariant to small pixel-space changes. Two images with the same content but differently-placed fine texture can have nearly identical high-level feature maps. So a feature-space content loss anchors the *content* (the structure, the objects, the layout) to the input while leaving the generator *free* to choose the exact texture — and that freedom is precisely what the adversarial term then uses to pick a sharp, on-manifold texture instead of the average. The two terms stop fighting: content in feature space, texture from the adversary.

Let me write both content options to compare. The pixel MSE content loss, with upscale factor `r`:

    l^SR_MSE = (1 / (r² W H)) Σ_{x=1}^{rW} Σ_{y=1}^{rH} (I^HR_{x,y} − G(I^LR)_{x,y})².

And the VGG feature-space content loss. Let `φ_{i,j}` be the feature map obtained by the `j`-th convolution (after its activation) before the `i`-th max-pooling layer of the pre-trained VGG19, which I hold fixed. Then

    l^SR_{VGG/i.j} = (1 / (W_{i,j} H_{i,j})) Σ_{x=1}^{W_{i,j}} Σ_{y=1}^{H_{i,j}} ( φ_{i,j}(I^HR)_{x,y} − φ_{i,j}(G(I^LR))_{x,y} )²,

where `W_{i,j}, H_{i,j}` are the spatial dimensions of that feature map. It's the same squared-error form, just evaluated on feature activations instead of pixels.

Which feature map — shallow or deep? A shallow layer like `φ_{2,2}` still encodes fairly low-level, near-pixel structure, so a content loss there still partly chains the texture to the reference and partly re-introduces the averaging problem. A deep layer like `φ_{5,4}` encodes high-level *content* — what's in the image, abstracted away from exact pixels — and is much more invariant to where the texture sits. If I anchor content with a deep feature loss, I'm telling the generator "keep the content of the input" while leaving the texture entirely to the adversarial term, which is exactly the division of labor I want. So the deep `φ_{5,4}` content loss is the natural choice for the most photo-realistic result, with the adversarial term owning the fine texture.

Now the adversarial loss seen from the generator's side. The textbook generator term is to minimize `log(1 − D(G(I^LR)))`, but early in training `D` rejects the fakes confidently, that term saturates, and the gradient to `G` vanishes. The standard fix is the non-saturating form: have the generator instead *minimize* `−log D(G(I^LR))` — same fixed point, but a strong gradient exactly when `D` is confident the sample is fake, which is when `G` most needs to move. So the generative loss is

    l^SR_{Gen} = Σ_n −log D(G(I^LR_n)).

Assemble the full objective. It's a weighted sum of a content loss and the adversarial loss — a *perceptual* loss:

    l^SR = l^SR_X + 10⁻³ · l^SR_{Gen},

with `l^SR_X` the chosen content loss (the deep VGG loss for the photo-realistic variant). The adversarial weight is small, `10⁻³`. The content term has to dominate — it's what keeps the output a faithful super-resolution of *this* input — while the adversarial term is a comparatively gentle nudge toward the manifold; let it dominate and the generator drifts toward pretty-but-unfaithful images. One scale issue to handle: the VGG feature activations live on a different numerical scale than pixel intensities, so I rescale the VGG feature maps by `1/12.75`; because the feature loss squares differences, that is equivalent to multiplying the VGG loss by `1/(12.75²) ≈ 0.006`, bringing it onto a scale comparable to the pixel MSE loss before the `10⁻³` adversarial weight is applied.

With the objective settled, design the generator. I want it *deep*, because the mapping from low-res to plausible high-res texture is complex, and depth is what models complex mappings — but deep plain networks are hard to train. Residual blocks with skip connections fix that: a skip connection lets a block default to the identity and only learn the residual, so the optimizer isn't forced to represent the trivial identity mapping through a stack of convolutions, and depth becomes trainable. So make the generator a deep residual network. Each residual block: a `3×3` convolution with 64 feature maps, batch normalization, a Parametric ReLU (the negative-slope is learned rather than fixed — a small extra degree of freedom in the activation), then another `3×3`/64 convolution and batch norm, then an elementwise sum with the block's input. Stack `B = 16` such blocks. After the block stack, one more `3×3`/64 convolution and batch norm, and then a *long* skip connection that adds back the features from *before* the block stack — so the residual blocks collectively learn a correction on top of the early features, the same identity-relief logic applied at the whole-stack scale.

Then upscaling, and I won't pre-upsample the input bicubically and convolve in high-res space — that's wasteful and bakes in an interpolation. Instead learn the upscaling *inside* the network with sub-pixel convolution: a convolution produces `r²` times as many channels in low-resolution space, and a pixel-shuffle operation rearranges those channels into a `2×`-larger spatial grid. Two such `2×` sub-pixel blocks (each: `3×3` conv producing the extra channels, pixel-shuffle `×2`, Parametric ReLU) give the total `4×`. Doing the heavy convolution in low-res space and only expanding spatially at the end is far cheaper, and the upscaling filters are learned rather than fixed. A first `9×9`/64 convolution with Parametric ReLU sits before the residual stack to embed the input with a large receptive field, and a final `9×9`/3 convolution with a `tanh` maps back to a 3-channel image in `[−1, 1]`.

The discriminator follows the known stable convolutional recipe for adversarial training: avoid pooling, use strided convolutions to downsample, use leaky ReLU (slope `0.2`) throughout, and use batch normalization. Concretely, eight `3×3` convolutional layers whose channel count climbs `64 → 128 → 256 → 512` (doubling, VGG-style), with a stride-2 convolution halving the spatial resolution each time the channel count doubles; then two dense layers and a final sigmoid producing the probability that the input is a real high-resolution image.

A couple of training choices that matter. The adversarial game starting from a randomly-initialized generator is a recipe for bad local optima and high-frequency artifacts — `D` can win too easily and push `G` somewhere strange. So first train the generator *alone* on the MSE content loss (this is just a strong PSNR-oriented residual super-resolver — call it SRResNet), and use those weights to *initialize* the generator when I start the full adversarial training. Starting from a competent super-resolver means the adversarial term only has to refine texture, not learn super-resolution from scratch. For the optimizer I use Adam; note I keep the momentum term `β1` at the usual `0.9` rather than lowering it the way one does for a from-scratch adversarial generator — here the generator is pre-trained and the content loss dominates, so the optimization is far better-behaved than a pure GAN and standard momentum is fine. I alternate one generator update with one discriminator update. At test time I put batch normalization in evaluation mode so the output depends deterministically on the input.

Let me write it as real code.

```python
import math
import torch
import torch.nn as nn
import torchvision

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, s=1, bn=False, act=None):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, k, s, padding=k // 2)]
        if bn: layers.append(nn.BatchNorm2d(cout))
        if act == 'prelu':  layers.append(nn.PReLU())
        if act == 'leaky':  layers.append(nn.LeakyReLU(0.2))
        if act == 'tanh':   layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    # two 3x3/64 convs, BN, PReLU; elementwise add the input (identity relief)
    def __init__(self, c=64):
        super().__init__()
        self.c1 = ConvBlock(c, c, 3, bn=True, act='prelu')
        self.c2 = ConvBlock(c, c, 3, bn=True, act=None)
    def forward(self, x):
        return x + self.c2(self.c1(x))

class SubPixelBlock(nn.Module):
    # learn the upscaling in LR space: conv to r^2 channels, pixel-shuffle, PReLU
    def __init__(self, c=64, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(c, c * scale ** 2, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU()
    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))

class Generator(nn.Module):            # SRResNet body; the same net is the GAN generator
    def __init__(self, n_blocks=16, scale=4):
        super().__init__()
        self.head = ConvBlock(3, 64, 9, act='prelu')                 # large-receptive-field embed
        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(n_blocks)])
        self.body_tail = ConvBlock(64, 64, 3, bn=True, act=None)
        n_up = int(math.log2(scale))                                 # 4x -> two 2x sub-pixel blocks
        self.up = nn.Sequential(*[SubPixelBlock(64, 2) for _ in range(n_up)])
        self.tail = ConvBlock(64, 3, 9, act='tanh')                  # HR in [-1, 1]
    def forward(self, lr):
        h = self.head(lr)
        x = self.body_tail(self.body(h))
        x = x + h                                                    # long skip over the residual stack
        return self.tail(self.up(x))

class Discriminator(nn.Module):        # DCGAN-style: strided conv, leaky-ReLU, BN, no pooling
    def __init__(self):
        super().__init__()
        cfg = [(64,1,False),(64,2,True),(128,1,True),(128,2,True),
               (256,1,True),(256,2,True),(512,1,True),(512,2,True)]   # 8 conv layers, channels 64->512
        layers, cin = [], 3
        for cout, s, bn in cfg:
            layers += [nn.Conv2d(cin, cout, 3, s, 1)]
            if bn: layers += [nn.BatchNorm2d(cout)]
            layers += [nn.LeakyReLU(0.2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                                # 96x96 HR crop -> 512x6x6
            nn.Linear(512 * 6 * 6, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 1))                                       # sigmoid folded into BCEWithLogits
    def forward(self, img):
        return self.classifier(self.features(img))

def truncated_vgg(i=5, j=4):
    vgg = torchvision.models.vgg19(pretrained=True).features
    maxpools, convs, cut = 0, 0, None
    for idx, layer in enumerate(vgg):
        if isinstance(layer, nn.Conv2d):   convs += 1
        if isinstance(layer, nn.MaxPool2d): maxpools += 1; convs = 0
        if maxpools == i - 1 and convs == j:
            cut = idx; break                                         # include up to (and incl.) this conv's activation
    net = nn.Sequential(*list(vgg.children())[:cut + 2]).eval()      # +2 to include the ReLU after the conv
    for p in net.parameters(): p.requires_grad = False
    return net

vgg54 = truncated_vgg(5, 4)
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
VGG_LOSS_SCALE = 1.0 / (12.75 ** 2)   # feature maps are rescaled by 1/12.75

def vgg_input_from_tanh(x):
    # SR/HR tensors stay in [-1, 1]; torchvision VGG expects ImageNet-normalized RGB.
    x = (x + 1.0) / 2.0
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std

def content_loss(sr, hr):
    return VGG_LOSS_SCALE * mse(vgg54(vgg_input_from_tanh(sr)),
                                vgg54(vgg_input_from_tanh(hr)))   # feature-space, not pixel-space

def train_step(lr, hr, G, D, opt_g, opt_d):
    # --- D: real HR vs generated SR ---
    with torch.no_grad():
        sr_for_d = G(lr)
    opt_d.zero_grad()
    d_real = D(hr); d_fake = D(sr_for_d)
    d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
    d_loss.backward(); opt_d.step()
    # --- G: perceptual loss = content (VGG feature space) + 1e-3 * non-saturating adversarial ---
    opt_g.zero_grad()
    sr = G(lr)
    d_fake = D(sr)
    adversarial = bce(d_fake, torch.ones_like(d_fake))   # non-saturating: -log D(G(LR))
    g_loss = content_loss(sr, hr) + 1e-3 * adversarial
    g_loss.backward(); opt_g.step()
    return g_loss, d_loss
```

The causal chain: per-pixel MSE provably hands back the conditional mean over all plausible high-res solutions, and that average is smooth and lies off the natural-image manifold — so the smoothing is the objective's fault, not the network's; the cure is to pull the output onto the manifold with a learned adversarial "looks-real" signal, which rewards committing to one sharp texture instead of averaging; but the adversarial term alone is unanchored, so I add a *content* loss measured in deep VGG feature space — invariant enough to pixel shifts that it pins the content while leaving the texture to the adversary, so the two terms cooperate rather than fight; the generator is a deep residual network with a long skip and learned sub-pixel upscaling, pre-trained on MSE so the adversarial phase only refines texture; the discriminator follows the stable strided-conv recipe; and the perceptual loss is `content + 10⁻³ · (−log D(G(I^LR)))`. What I'd want to validate next: that against the smooth MSE baselines this produces visibly sharper, on-manifold texture at `4×` — and since PSNR will, by construction, *drop* relative to the MSE model, that human perceptual judgment (a mean-opinion-score test) is the metric that actually moves in the right direction.
