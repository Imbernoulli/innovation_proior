# SRGAN — photo-realistic super-resolution via a perceptual loss

## The problem it solves

Recover photo-realistic fine texture in single-image super-resolution at large (`4×`) upscaling
factors. The reigning per-pixel MSE objective maximizes PSNR but produces smooth, perceptually
unsatisfying images — because for an underdetermined SR problem the MSE minimizer is the *pixel-wise
average* (conditional mean) over all plausible sharp high-resolution solutions, and that average is
blurry and lies *off* the natural-image manifold. The fix is to change the objective, not the network
capacity.

## The key idea

Replace the pixel objective with a **perceptual loss** that pulls the output onto the natural-image
manifold and anchors content without re-introducing the averaging:

- **Adversarial loss** — a discriminator trained to tell real HR photos from super-resolved outputs
  supplies a learned "looks like a real photograph" gradient, pushing the generator toward a single
  sharp on-manifold solution rather than the smooth average.
- **Content loss in VGG feature space** (not pixel space) — comparing deep VGG19 feature maps is much
  more invariant to pixel-space shifts, so it pins the *content* of the input while leaving the *texture*
  free for the adversarial term to make plausible. The two terms cooperate instead of fighting.

## The objective

Perceptual loss = content loss + adversarial loss:

    l^SR = l^SR_X + 10^{-3} · l^SR_Gen.

Content-loss options:

    l^SR_MSE      = (1/(r²WH)) Σ_x Σ_y ( I^HR_{x,y} − G(I^LR)_{x,y} )²                        (pixel MSE)
    l^SR_{VGG/i.j}= (1/(W_{i,j}H_{i,j})) Σ_x Σ_y ( φ_{i,j}(I^HR)_{x,y} − φ_{i,j}(G(I^LR))_{x,y} )²

where `φ_{i,j}` is the feature map from the `j`-th convolution (after activation) before the `i`-th
maxpool of a frozen VGG19. The photo-realistic variant uses the deep `φ_{5,4}` (VGG54). VGG features
are rescaled (≈ `1/12.75`) to match the MSE scale.

Adversarial (non-saturating, for good gradients): `l^SR_Gen = Σ_n −log D(G(I^LR_n))`, the generator
side of the minimax game

    min_{θ_G} max_{θ_D}  E_{I^HR}[log D(I^HR)] + E_{I^LR}[log(1 − D(G(I^LR)))].

## Architecture

- **Generator (SRResNet):** Conv `9×9`/64 + PReLU; `B = 16` residual blocks (each: Conv `3×3`/64 → BN →
  PReLU → Conv `3×3`/64 → BN → elementwise add); Conv `3×3`/64 → BN; a **long skip** adding the
  pre-block features; two **sub-pixel convolution** (PixelShuffle) `×2` blocks (Conv → PixelShuffle →
  PReLU) for `4×` learned upscaling; Conv `9×9`/3 → Tanh.
- **Discriminator (DCGAN-style):** eight `3×3` conv layers, channels `64→128→256→512`, strided
  convolutions for downsampling (no pooling), LeakyReLU(0.2), batch norm (except first), then dense
  layers and a sigmoid.

## Training

LR via bicubic downsampling (`r = 4`); HR scaled to `[−1,1]`, LR to `[0,1]`. Adam, `β1 = 0.9`.
Pre-train the generator alone on MSE (the SRResNet), then **initialize the GAN generator with those
weights** to avoid bad local optima and high-frequency artifacts; train the adversarial phase
alternating G and D updates (`k = 1`). Batch norm in eval mode at test time.

## Working code

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
        if act == 'prelu': layers.append(nn.PReLU())
        if act == 'leaky': layers.append(nn.LeakyReLU(0.2))
        if act == 'tanh':  layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.c1 = ConvBlock(c, c, 3, bn=True, act='prelu')
        self.c2 = ConvBlock(c, c, 3, bn=True, act=None)
    def forward(self, x): return x + self.c2(self.c1(x))

class SubPixelBlock(nn.Module):
    def __init__(self, c=64, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(c, c * scale ** 2, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU()
    def forward(self, x): return self.act(self.shuffle(self.conv(x)))

class Generator(nn.Module):
    def __init__(self, n_blocks=16, scale=4):
        super().__init__()
        self.head = ConvBlock(3, 64, 9, act='prelu')
        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(n_blocks)])
        self.body_tail = ConvBlock(64, 64, 3, bn=True, act=None)
        self.up = nn.Sequential(*[SubPixelBlock(64, 2) for _ in range(int(math.log2(scale)))])
        self.tail = ConvBlock(64, 3, 9, act='tanh')
    def forward(self, lr):
        h = self.head(lr)
        x = self.body_tail(self.body(h)) + h          # long skip over residual stack
        return self.tail(self.up(x))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [(64,1,False),(64,2,True),(128,1,True),(128,2,True),
               (256,1,True),(256,2,True),(512,1,True),(512,2,True)]
        layers, cin = [], 3
        for cout, s, bn in cfg:
            layers += [nn.Conv2d(cin, cout, 3, s, 1)]
            if bn: layers += [nn.BatchNorm2d(cout)]
            layers += [nn.LeakyReLU(0.2)]; cin = cout
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(6), nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 1))
    def forward(self, img): return self.classifier(self.features(img))

def truncated_vgg(i=5, j=4):
    vgg = torchvision.models.vgg19(pretrained=True).features
    maxpools, convs, cut = 0, 0, None
    for idx, layer in enumerate(vgg):
        if isinstance(layer, nn.Conv2d): convs += 1
        if isinstance(layer, nn.MaxPool2d): maxpools += 1; convs = 0
        if maxpools == i - 1 and convs == j: cut = idx; break
    net = nn.Sequential(*list(vgg.children())[:cut + 2]).eval()   # +2 includes the conv's ReLU
    for p in net.parameters(): p.requires_grad = False
    return net

vgg54 = truncated_vgg(5, 4)
mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
VGG_RESCALE = 1 / 12.75 ** 2

def content_loss(sr, hr):
    return VGG_RESCALE * mse(vgg54(sr), vgg54(hr))

def train_step(lr, hr, G, D, opt_g, opt_d):
    sr = G(lr)
    opt_d.zero_grad()
    d_real, d_fake = D(hr), D(sr.detach())
    (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))).backward()
    opt_d.step()
    opt_g.zero_grad()
    adversarial = bce(D(sr), torch.ones_like(d_fake))           # non-saturating -log D(G(LR))
    g_loss = content_loss(sr, hr) + 1e-3 * adversarial
    g_loss.backward(); opt_g.step()
    return g_loss

# Pre-train: first optimize G alone on l^SR_MSE (SRResNet), then init the GAN generator from it.
```

## Why it works

MSE returns the average of plausible solutions, which is smooth and off-manifold; the adversarial term
rewards landing on one sharp, on-manifold solution; and measuring content in deep VGG feature space —
invariant to pixel shifts — anchors *what* the image depicts while leaving *how* the texture looks to
the adversary, so the content and texture objectives no longer pull against each other. PSNR drops
relative to the MSE model by construction, but perceptual quality (mean-opinion-score) rises.
