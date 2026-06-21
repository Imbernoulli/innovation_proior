Generative adversarial networks can produce sharp images, but on large, diverse, many-class datasets such as ImageNet the gap to real photographs remains large in both fidelity and variety. Prior class-conditional ImageNet GANs reach an Inception Score around 52, while real data scores about 233. Two difficulties make this gap hard to close. First, adversarial training is a two-player game with a Nash equilibrium rather than a single loss minimum, so its dynamics are sensitive to architecture, optimizer, and capacity; it is not obvious that simply scaling up will help rather than destabilize training. Second, when a generator does produce high-fidelity samples it tends to do so only in a narrow region of its output space, and pushing for fidelity usually sacrifices variety, with no clean knob to navigate the trade-off.

The method I propose is BigGAN. It starts from a strong class-conditional GAN baseline and makes targeted changes in scale, conditioning, regularization, and sampling. The baseline is a ResNet-style generator and discriminator with self-attention in both, spectral normalization in both, a hinge adversarial objective, class-conditional BatchNorm in the generator, a projection discriminator, and different learning rates for the two networks. BigGAN scales this baseline up: it increases the batch size by a factor of eight, which covers many more data modes per gradient and gives both networks lower-variance, less mode-starved estimates of the adversarial game, and it increases width by about fifty percent to provide more capacity for the complex dataset. To make the conditioning cheaper and more coherent, it uses a single shared class embedding that is linearly projected to the gain and bias of every conditional normalization layer, rather than a separate embedding table per layer. It also feeds the latent z into every generator residual block instead of only the first layer: z is split into equal chunks, and each block is conditioned on the concatenation of the shared class embedding and its own chunk of z. This skip-z lets the latent modulate features at every resolution.

BigGAN also exploits a freedom that GANs have but most other generative models do not: because the generator is never inverted, the sampling-time latent distribution need not match the training-time prior. The model is trained with z drawn from a standard normal, but at sampling time z is drawn from a truncated normal by clipping coordinates to a chosen threshold. Smaller thresholds keep z near the prior's high-density core, where the generator was best supervised, which improves fidelity at the cost of variety. This gives a continuous, post-hoc dial between diversity and fidelity. The dial is only safe if the generator is smooth and well-conditioned, so BigGAN applies a light modified orthogonal regularization that penalizes only the off-diagonal entries of the filter Gram matrix, decorrelating filter directions without constraining their norms. The resulting model reaches much stronger Inception Score and Fréchet Inception Distance on ImageNet at 128 to 512 pixels than prior class-conditional GANs.

```python
import functools
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import init
from scipy.stats import truncnorm

def power_iteration(W, u, eps=1e-12):
    with torch.no_grad():
        v = F.normalize(torch.matmul(u, W), eps=eps)
        u = F.normalize(torch.matmul(v, W.t()), eps=eps)
    sigma = torch.matmul(torch.matmul(v, W.t()), u.t()).squeeze()
    return sigma, u, v

class SN(object):
    def __init__(self, num_outputs, eps=1e-12):
        self.eps = eps
        self.register_buffer('u', torch.randn(1, num_outputs))
    def W_(self):
        W = self.weight.view(self.weight.size(0), -1)
        sigma, u, _ = power_iteration(W, self.u, self.eps)
        if self.training:
            with torch.no_grad():
                self.u[:] = u
        return self.weight / sigma

class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, bias=True, eps=1e-12, **kw):
        nn.Conv2d.__init__(self, in_c, out_c, kernel_size, padding=padding, bias=bias, **kw)
        SN.__init__(self, out_c, eps)
    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride, self.padding)

class SNLinear(nn.Linear, SN):
    def __init__(self, in_f, out_f, bias=True, eps=1e-12):
        nn.Linear.__init__(self, in_f, out_f, bias)
        SN.__init__(self, out_f, eps)
    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)

class SNEmbedding(nn.Embedding, SN):
    def __init__(self, n, d, eps=1e-12):
        nn.Embedding.__init__(self, n, d)
        SN.__init__(self, n, eps)
    def forward(self, x):
        return F.embedding(x, self.W_())

class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d):
        super().__init__()
        self.ch = ch
        self.theta = which_conv(ch, ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = which_conv(ch, ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = which_conv(ch, ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = which_conv(ch // 2, ch, kernel_size=1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.))
    def forward(self, x):
        theta = self.theta(x).view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = F.max_pool2d(self.phi(x), 2).view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = F.max_pool2d(self.g(x), 2).view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

class ccbn(nn.Module):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5):
        super().__init__()
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        self.bn = nn.BatchNorm2d(output_size, affine=False, eps=eps)
    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        return self.bn(x) * gain + bias

class GBlock(nn.Module):
    def __init__(self, in_ch, out_ch, which_conv, which_bn, upsample=True):
        super().__init__()
        self.in_channels, self.out_channels, self.upsample = in_ch, out_ch, upsample
        self.conv1, self.conv2 = which_conv(in_ch, out_ch), which_conv(out_ch, out_ch)
        self.bn1, self.bn2 = which_bn(in_ch), which_bn(out_ch)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.conv_sc = which_conv(in_ch, out_ch, kernel_size=1, padding=0)
    def forward(self, x, y):
        h = F.relu(self.bn1(x, y))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2)
            x = F.interpolate(x, scale_factor=2)
        h = self.conv1(h)
        h = self.conv2(F.relu(self.bn2(h, y)))
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x

class Generator(nn.Module):
    def __init__(self, dim_z=120, n_classes=1000, ch=96, shared_dim=128):
        super().__init__()
        which_conv, which_linear = SNConv2d, SNLinear
        in_ch = [16 * ch, 16 * ch, 8 * ch, 4 * ch, 2 * ch]
        out_ch = [16 * ch, 8 * ch, 4 * ch, 2 * ch, ch]
        self.num_slots = len(in_ch) + 1
        self.chunk = dim_z // self.num_slots
        self.dim_z = self.chunk * self.num_slots
        cond_dim = shared_dim + self.chunk
        self.shared = nn.Embedding(n_classes, shared_dim)
        bn_linear = functools.partial(which_linear, bias=False)
        which_bn = functools.partial(ccbn, input_size=cond_dim, which_linear=bn_linear)
        self.linear = which_linear(self.chunk, in_ch[0] * 4 * 4)
        self.blocks = nn.ModuleList([
            GBlock(in_ch[i], out_ch[i], which_conv, which_bn)
            for i in range(len(in_ch))])
        self.attn = Attention(2 * ch, which_conv)
        self.attn_after = 3
        self.output = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(), which_conv(ch, 3))
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                init.orthogonal_(m.weight)
    def forward(self, z, y):
        zs = torch.split(z, self.chunk, 1)
        emb = self.shared(y)
        conds = [torch.cat([emb, zs[i + 1]], 1) for i in range(len(self.blocks))]
        h = self.linear(zs[0]).view(z.size(0), -1, 4, 4)
        for i, blk in enumerate(self.blocks):
            h = blk(h, conds[i])
            if i == self.attn_after:
                h = self.attn(h)
        return torch.tanh(self.output(h))

class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, which_conv, preactivation=True, downsample=True):
        super().__init__()
        self.preactivation, self.downsample = preactivation, downsample
        self.conv1, self.conv2 = which_conv(in_ch, out_ch), which_conv(out_ch, out_ch)
        self.learnable_sc = in_ch != out_ch or downsample
        if self.learnable_sc:
            self.conv_sc = which_conv(in_ch, out_ch, kernel_size=1, padding=0)
    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv_sc(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x
    def forward(self, x):
        h = F.relu(x) if self.preactivation else x
        h = self.conv2(F.relu(self.conv1(h)))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h + self.shortcut(x)

class Discriminator(nn.Module):
    def __init__(self, n_classes=1000, ch=96):
        super().__init__()
        which_conv = SNConv2d
        in_ch = [3, ch, 2 * ch, 4 * ch, 8 * ch, 16 * ch]
        out_ch = [ch, 2 * ch, 4 * ch, 8 * ch, 16 * ch, 16 * ch]
        self.blocks = nn.ModuleList([
            DBlock(in_ch[i], out_ch[i], which_conv,
                   preactivation=(i > 0), downsample=(i < 5))
            for i in range(len(in_ch))])
        self.attn = Attention(ch, which_conv)
        self.linear = SNLinear(16 * ch, 1)
        self.embed = SNEmbedding(n_classes, 16 * ch)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                init.orthogonal_(m.weight)
    def forward(self, x, y):
        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if i == 0:
                h = self.attn(h)
        h = torch.sum(F.relu(h), [2, 3])
        out = self.linear(h) + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out

def loss_hinge_dis(d_fake, d_real):
    return F.relu(1. - d_real).mean(), F.relu(1. + d_fake).mean()

def loss_hinge_gen(d_fake):
    return -d_fake.mean()

def ortho(model, strength=1e-4, blacklist=()):
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim < 2 or any(p is b for b in blacklist):
                continue
            w = p.view(p.shape[0], -1)
            grad = 2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w)
            p.grad.data += strength * grad.view(p.shape)

def truncated_z_sample(batch, dim_z, truncation=0.5):
    return truncation * truncnorm.rvs(-2, 2, size=(batch, dim_z))

class EMA(object):
    def __init__(self, source, target, decay=0.9999):
        self.source, self.target, self.decay = source, target, decay
        self.s = source.state_dict()
        self.t = target.state_dict()
        for k in self.s:
            self.t[k].data.copy_(self.s[k].data)
    def update(self):
        with torch.no_grad():
            for k in self.s:
                self.t[k].data.copy_(self.t[k].data * self.decay + self.s[k].data * (1 - self.decay))

def train_step(G, D, GD, x, y, z_, y_, ema, cfg):
    for _ in range(cfg['num_D_steps']):
        D.optim.zero_grad()
        z_.normal_()
        y_.random_(0, cfg['n_classes'])
        d_fake, d_real = GD(z_, y_, x, y, train_G=False)
        d_real_loss, d_fake_loss = loss_hinge_dis(d_fake, d_real)
        (d_real_loss + d_fake_loss).backward()
        D.optim.step()
    G.optim.zero_grad()
    z_.normal_()
    y_.random_(0, cfg['n_classes'])
    d_fake = GD(z_, y_, train_G=True)
    loss_hinge_gen(d_fake).backward()
    if cfg['G_ortho'] > 0:
        ortho(G, cfg['G_ortho'], blacklist=list(G.shared.parameters()))
    G.optim.step()
    ema.update()
```
