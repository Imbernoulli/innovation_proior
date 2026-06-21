The task is single-image super-resolution: from a low-resolution input $I^{LR}$ produce a high-resolution image $I^{SR}$ that matches the true $I^{HR}$, and the hard regime is large upscaling — at $4\times$ the pixel count rises $16\times$ and the fine, high-frequency texture is essentially absent from the input, so it cannot be recovered but must be *invented* plausibly. The field's reflex is to train a deep convolutional network to minimize per-pixel mean squared error, because minimizing MSE also maximizes PSNR and PSNR is the number everyone reports. Networks have grown deeper and faster and PSNR has crept upward, yet the outputs at large factors look washed out and smooth, the texture gone. I want to be precise about why, because I think the objective itself is the culprit. For a single $I^{LR}$ there is no unique correct $I^{HR}$; there is a whole *set* of sharp high-resolution images all consistent with that input, differing exactly in the fine texture I cannot recover. Now ask what minimizing expected squared pixel error does in the face of that set: its minimizer at each input is the conditional mean $\mathbb{E}[I^{HR}\mid I^{LR}]$, the pixel-wise average over all those plausible sharp solutions. Averaging many crisp images whose texture sits in different places yields a smooth image with texture nowhere — and worse, real photographs lie on a thin manifold in pixel space, so the centroid of several manifold points generally sits *off* the manifold, in the smooth region between them. The blur is therefore not a deficiency of capacity that a bigger network repairs; it is the direct, intended behavior of the per-pixel objective. The closest prior remedies — interpolation, patch and sparse-coding methods, SRCNN, ESPCN, DRCN — all still optimize a pixel-wise loss and inherit the ceiling; even feature-space perceptual losses (Johnson et al., 2016; Bruna et al., 2016), the nearest ancestors in spirit, do not reach photo-realistic texture at $4\times$. The conclusion is that I must stop asking for the average and instead pull the reconstruction onto *one* point of the natural-image manifold.

I propose SRGAN: super-resolution trained against a learned *perceptual* loss combining an adversarial term with a deep feature-space content term, where the adversary supplies a "does this look like a real photograph" gradient and the content term anchors the output to the specific input. What can score an image by naturalness rather than by pixel distance to one reference is a discriminator network $D$ trained to distinguish real high-resolution photographs from the generator's outputs, with the generator $G$ trained to fool it; $D$ is then precisely a learned "is this on the manifold of real images" detector, and its gradient pushes $G$'s outputs toward regions of image space indistinguishable from real photos. Crucially the adversarial signal does not care whether the texture sits in exactly the right pixel — only that it is plausible — so it actively rewards committing to one sharp solution rather than hedging toward the blurry mean, the opposite pressure from MSE. This is the minimax game
$$\min_{\theta_G}\max_{\theta_D}\;\mathbb{E}_{I^{HR}}[\log D(I^{HR})]+\mathbb{E}_{I^{LR}}[\log(1-D(G(I^{LR})))].$$
But the adversarial term alone is unanchored: $G$ could produce a gorgeous natural image with nothing to do with this input and still fool $D$. I need a content term that says "and it must be a super-resolution of *this* image," yet the obvious choice — pixel MSE — is exactly what smooths. The resolution is to change the *space* in which content similarity is measured. Pixel MSE forces the average because, defined on raw pixels, it fully penalizes any texture shifted by a pixel, so the only safe answer everywhere is to smooth. If instead I compare the two images in the feature space of a frozen pre-trained VGG19, those activations are far more invariant to small pixel-space changes: two images with the same content but differently-placed fine texture have nearly identical high-level feature maps. So a feature-space content loss pins the *content* — structure, objects, layout — to the input while leaving the generator *free* to choose the exact texture, and that freedom is exactly what the adversarial term uses to pick a sharp, on-manifold texture. The two terms stop fighting: content in feature space, texture from the adversary.

The two content options, with upscale factor $r$, are the pixel MSE loss
$$l^{SR}_{MSE}=\frac{1}{r^2WH}\sum_{x=1}^{rW}\sum_{y=1}^{rH}\big(I^{HR}_{x,y}-G(I^{LR})_{x,y}\big)^2$$
and the VGG feature-space loss, where $\phi_{i,j}$ is the feature map from the $j$-th convolution (after its activation) before the $i$-th max-pooling layer of the held-fixed VGG19,
$$l^{SR}_{VGG/i.j}=\frac{1}{W_{i,j}H_{i,j}}\sum_{x=1}^{W_{i,j}}\sum_{y=1}^{H_{i,j}}\big(\phi_{i,j}(I^{HR})_{x,y}-\phi_{i,j}(G(I^{LR}))_{x,y}\big)^2.$$
The depth of the feature map matters. A shallow map like $\phi_{2,2}$ still encodes near-pixel structure, so a content loss there partly rechains the texture to the reference and reintroduces the averaging; a deep map like $\phi_{5,4}$ encodes high-level content abstracted from exact pixels and is far more invariant to where the texture sits. Anchoring content with the deep $\phi_{5,4}$ (VGG54) is therefore the natural choice for the most photo-realistic result, leaving the fine texture entirely to the adversary. For the generator's view of the adversarial term, the textbook form minimizes $\log(1-D(G(I^{LR})))$, but early in training $D$ rejects fakes confidently, that term saturates, and the gradient to $G$ vanishes; I use instead the non-saturating form, having $G$ minimize $-\log D(G(I^{LR}))$ — same fixed point, but a strong gradient exactly when $D$ is confident the sample is fake, which is when $G$ most needs to move. So $l^{SR}_{Gen}=\sum_n -\log D(G(I^{LR}_n))$. The full objective is the weighted sum
$$l^{SR}=l^{SR}_X+10^{-3}\cdot l^{SR}_{Gen},$$
with $l^{SR}_X$ the chosen (deep VGG) content loss. The adversarial weight is deliberately small, $10^{-3}$: the content term must dominate so the output stays a faithful super-resolution of *this* input, while the adversary is a gentle nudge toward the manifold — let the adversary dominate and $G$ drifts toward pretty-but-unfaithful images. One scale detail: VGG activations live on a different numerical scale than pixel intensities, so I rescale the feature maps by $1/12.75$; because the feature loss squares differences, this is equivalent to multiplying the VGG loss by $1/(12.75^2)\approx 0.006$, bringing it onto a scale comparable to pixel MSE before the $10^{-3}$ adversarial weight applies.

With the objective settled I design the generator to be deep, because the LR-to-plausible-texture mapping is complex and depth models complex mappings — but deep plain networks are hard to train, so each block is residual: a skip connection lets a block default to the identity and learn only a correction, so the optimizer is never forced to represent the trivial identity through a stack of convolutions. Each residual block is a $3\times3$/64 convolution, batch normalization, a Parametric ReLU (whose negative slope is *learned*, an extra activation degree of freedom), then another $3\times3$/64 convolution and batch norm, then an elementwise sum with the block input. I stack $B=16$ such blocks, follow with one more $3\times3$/64 convolution and batch norm, and add a *long* skip connection from before the block stack — the same identity-relief logic at the whole-stack scale, so the blocks collectively learn a correction on top of the early features. For upscaling I refuse to pre-upsample bicubically and convolve in HR space, which is wasteful and bakes in a fixed interpolation; instead I learn the upscaling with sub-pixel convolution — a convolution produces $r^2$ times as many channels in LR space and a pixel-shuffle rearranges them into a $2\times$-larger grid — using two $2\times$ blocks (conv, pixel-shuffle, PReLU) for the total $4\times$. A first $9\times9$/64 convolution with PReLU embeds the input with a large receptive field, and a final $9\times9$/3 convolution with $\tanh$ maps back to a 3-channel image in $[-1,1]$. The discriminator follows the stable strided-convolution recipe: no pooling, strided convolutions to downsample, LeakyReLU(0.2) throughout, batch norm — eight $3\times3$ convolutions with channels climbing $64\to128\to256\to512$ (halving spatial resolution each time channels double), then two dense layers and a sigmoid. Two training choices matter: starting the adversarial game from a random generator invites bad local optima and high-frequency artifacts, so I first train the generator alone on the MSE content loss — a strong PSNR-oriented residual super-resolver, SRResNet — and initialize the GAN generator from those weights, so the adversarial phase only refines texture rather than learning super-resolution from scratch; and for the optimizer I use Adam keeping momentum $\beta_1=0.9$ rather than lowering it as one would for a from-scratch GAN, since here the generator is pre-trained and the content loss dominates, so the optimization is far better behaved. I alternate one generator update with one discriminator update, and put batch norm in eval mode at test time so the output is deterministic. The cost is that PSNR drops relative to the MSE model by construction, so the metric that actually moves is human perceptual judgment (a mean-opinion-score test).

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
            nn.Flatten(),                                      # 96x96 HR crop -> 512x6x6
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
VGG_LOSS_SCALE = 1.0 / (12.75 ** 2)

def vgg_input_from_tanh(x):
    x = (x + 1.0) / 2.0
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std

def content_loss(sr, hr):
    return VGG_LOSS_SCALE * mse(vgg54(vgg_input_from_tanh(sr)), vgg54(vgg_input_from_tanh(hr)))

def train_step(lr, hr, G, D, opt_g, opt_d):
    with torch.no_grad():
        sr_for_d = G(lr)
    opt_d.zero_grad()
    d_real, d_fake = D(hr), D(sr_for_d)
    (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))).backward()
    opt_d.step()
    opt_g.zero_grad()
    sr = G(lr)
    adversarial = bce(D(sr), torch.ones_like(d_fake))           # non-saturating -log D(G(LR))
    g_loss = content_loss(sr, hr) + 1e-3 * adversarial
    g_loss.backward(); opt_g.step()
    return g_loss

# Pre-train: first optimize G alone on l^SR_MSE (SRResNet), then init the GAN generator from it.
```
