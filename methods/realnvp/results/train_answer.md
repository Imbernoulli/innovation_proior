We want to learn an exact probabilistic model $p(x)$ of high-dimensional, highly structured continuous data — natural images are the canonical case — and we want one model family that delivers four things at once: an exact log-likelihood we can evaluate and train on directly, sampling that is exact and parallel rather than a long sequential chain, exact and efficient inference of the latent code of any data point, and a latent space that is usable and semantically organized. No existing family gives all four. Undirected models like RBMs and DBMs write down an energy easily but carry an intractable partition function, so training (contrastive divergence), evaluation (annealed importance sampling), and sampling (MCMC) all rest on approximations with unbounded mixing times and correlated samples — exact anything is off the table. Variational autoencoders sample and infer cleanly, but they maximize a lower bound rather than the likelihood, their inference is only approximate, and the Gaussian decoder imposes a fixed-form $L_2$ reconstruction cost that rewards low-frequency content over high-frequency detail and visibly blurs the samples. Fully autoregressive models factor $p(x)=\prod_i p(x_i\mid x_{<i})$ and are exact and flexible, but to draw one image they must sample $D$ coordinates one after another in a fixed order — non-parallelizable, slow at image scale, with no latent representation and a nagging sensitivity to the chosen ordering. GANs give sharp samples and escape the fixed reconstruction cost, but they abandon the likelihood entirely, so density and diversity are not measurable, training is unstable, and there is no encoder from $x$ back to $z$. The common thread is clear: VAE blur comes from a fixed reconstruction cost, autoregressive slowness from the sequential factorization, GAN's missing likelihood and encoder from abandoning maximum likelihood. We want the maximum-likelihood principle — for an exact objective and an exact encoder — but without a fixed reconstruction cost and without a sequential bottleneck.

There is exactly one corner of probability where maximum likelihood costs nothing extra. Suppose the generator $g$ that turns a latent $z$ into an image $x$ is not an arbitrary net but a *bijection*, with inverse $f=g^{-1}$ carrying $x$ back to $z$. Then inference is literally $z=f(x)$, exact and direct, with no approximate posterior, and the change-of-variables identity gives the density in closed form: pushing an infinitesimal volume $dx$ through $f$ lands it on a volume $|\det(\partial f/\partial x^{\mathsf T})|\,dx$, mass is conserved, $p_X(x)\,dx=p_Z(z)\,dz$, so

$$\log p_X(x) = \log p_Z(f(x)) + \log\bigl|\det\!\bigl(\partial f(x)/\partial x^{\mathsf T}\bigr)\bigr|.$$

Pick a simple prior $p_Z$ such as a unit Gaussian and we get the exact log-likelihood of any $x$, train by maximizing it directly, sample by drawing $z\sim p_Z$ and returning $x=f^{-1}(z)$, and infer by $z=f(x)$ — all four wishes from one identity. The reason this had not scaled is the determinant: for an arbitrary $f$ on $\mathbb{R}^D$ the Jacobian determinant is an $O(D^3)$ operation, badly conditioned and hopeless per step on thousand-dimensional images. So the whole problem collapses to a single design question — can we build a bijection $f$ that is genuinely expressive, trivially invertible, *and* whose Jacobian determinant we can read off cheaply?

I propose Real NVP — real-valued non-volume-preserving transformations — which answers all three by forcing the Jacobian to be triangular by construction. The determinant of a triangular matrix is the product of its diagonal, an $O(D)$ read-off, and the way to get triangularity without a sequential inverse is the *affine coupling layer*. Split the $D$ coordinates into two blocks. Copy the first block untouched, and transform the second block with an affine map whose scale and shift are arbitrary functions of the first block only:

$$y_{1:d}=x_{1:d},\qquad y_{d+1:D}=x_{d+1:D}\odot\exp\!\bigl(s(x_{1:d})\bigr)+t(x_{1:d}),$$

with $s,t:\mathbb{R}^d\to\mathbb{R}^{D-d}$ and $\odot$ the elementwise product. The top-left Jacobian block is the identity (that half is copied) and the top-right block is zero (the copied half does not depend on the second half), so

$$\frac{\partial y}{\partial x^{\mathsf T}}=\begin{bmatrix} I_d & 0\\[2pt] \partial y_{d+1:D}/\partial x_{1:d}^{\mathsf T} & \operatorname{diag}\!\bigl(\exp(s(x_{1:d}))\bigr)\end{bmatrix}$$

is lower-triangular. The bottom-right block is diagonal because coordinate $j$ of the second block is $x_{d+1:D,j}\exp(s(x_{1:d})_j)$ plus an offset constant in $x_{d+1:D}$, so its determinant is $\prod_j\exp(s(x_{1:d})_j)=\exp(\sum_j s(x_{1:d})_j)$ and

$$\log|\det| = \sum_j s(x_{1:d})_j.$$

Three design choices carry this. First, the scale is $\exp(s)$ rather than a raw multiplier: the exponential is always positive so the per-coordinate scaling never hits zero and invertibility is guaranteed, and it turns a product of scales into a clean *sum* of the raw $s$ outputs in the log-det — no logarithm, no absolute value. Second, the determinant sees only the diagonal bottom-right block; the off-diagonal block holds all the derivatives of $s$ and $t$ with respect to the first half and sits strictly below the diagonal, so $s$ and $t$ can be arbitrarily deep convolutional nets at zero cost to the determinant. Third — and this is the advance over the earlier additive coupling of NICE, $y_{d+1:D}=x_{d+1:D}+m(x_{1:d})$, whose bottom-right block is the identity and whose determinant is therefore exactly $1$ — making the transform affine instead of merely additive turns $\det=1$ into $\det=\exp(\sum s)\neq1$. The additive map is *volume-preserving*: it can move and curve mass but never locally contract or expand it, which is exactly what density estimation needs (pile mass where the data concentrates, thin it where data is sparse), and NICE could only recover that with a single bolted-on diagonal scaling at the very end. The affine coupling makes every layer non-volume-preserving, so the stack reshapes volume throughout and the global diagonal is no longer needed. The inverse is just as cheap and needs no inverse of $s$ or $t$: read off $x_{1:d}=y_{1:d}$, feed it through $s$ and $t$, and undo the affine map,

$$x_{1:d}=y_{1:d},\qquad x_{d+1:D}=\bigl(y_{d+1:D}-t(y_{1:d})\bigr)\odot\exp\!\bigl(-s(y_{1:d})\bigr).$$

Forward and inverse cost the same single pass through $s$ and $t$, so sampling is exactly as fast as inference and the autoregressive bottleneck dissolves.

One coupling freezes half the coordinates, so we compose, and composition keeps both gifts. By the chain rule Jacobians multiply, and since $\det(AB)=\det(A)\det(B)$ the log-dets simply add across layers, while inverses reverse, $(f_b\circ f_a)^{-1}=f_a^{-1}\circ f_b^{-1}$. We alternate which half is frozen so every coordinate is eventually transformed; a coordinate that were only ever copied would stay a raw Gaussian forever. To respect image structure we implement the partition with a binary mask $b$ — the frozen half is wherever $b=1$ — and write the coupling in masked form

$$y = b\odot x + (1-b)\odot\bigl(x\odot\exp(s(b\odot x))+t(b\odot x)\bigr),$$

feeding $b\odot x$ into $s$ and $t$ so the conditioner literally sees only the frozen pixels and triangularity is preserved. Two masks do the work: a *spatial checkerboard* with $b=1$ where $i+j$ is odd, so each transformed pixel conditions on its immediate neighbors where the correlation lives, and a *channel-wise* mask with $b=1$ on the first half of the channels, so whole channels condition on whole channels. The nets $s,t$ are rectified residual convolutional networks.

Two numerical guardrails make a deep stack of these exp-scaled layers trainable. If $s$ ever emits a large positive value, $\exp(s)$ and its gradient explode into NaNs, so rather than cap expressiveness I bound the raw scale and let the model learn its range: $s=(\text{learned per-channel scale})\cdot\tanh(\text{raw conv output})$, the $\tanh$ confining the raw signal to $(-1,1)$ and the learned factor, wrapped in weight normalization, free to grow under control. The translation $t$ needs no such guard since it is additive. Beyond that, batch normalization is folded into the flow as one more bijector. It is a per-dimension affine map $x\mapsto(x-\tilde\mu)/\sqrt{\tilde\sigma^2+\epsilon}$ with diagonal Jacobian, so it owes a determinant term like any other layer: its forward log-det is $-\tfrac12\sum_i\log(\tilde\sigma_i^2+\epsilon)$ and its inverse $x\mapsto x\sqrt{\tilde\sigma^2+\epsilon}+\tilde\mu$ contributes $+\tfrac12\sum_i\log(\tilde\sigma_i^2+\epsilon)$. To stay robust at small batch size we use running averages $\tilde\mu_{t+1}=\rho\tilde\mu_t+(1-\rho)\hat\mu_t$ and likewise for $\tilde\sigma^2$, backpropagating only through the current-batch statistics. This directly tames the $\exp(s)$ instability and lets the coupling stack go deep.

Running the full $D$-dimensional image through every layer at full resolution is wasteful and ignores that images are multi-scale, so the architecture squeezes and factors. A *squeeze* reshapes an $s\times s\times c$ tensor into $(s/2)\times(s/2)\times 4c$ by stacking each $2\times2\times c$ spatial block into the channel axis; it is a pure permutation of elements, absolute determinant $1$, contributing nothing to the log-det, and it both enlarges the convolutional receptive field and makes channel-wise masking meaningful by packing adjacent pixels into channels. The per-scale rhythm is three checkerboard couplings at full resolution, then a squeeze, then three channel-wise couplings (with the channel partition chosen so it is not redundant with the checkerboard). Then we *factor out* half the coordinates as finished latents and recurse on the rest: with $h^{(0)}=x$,

$$\bigl(z^{(i+1)},h^{(i+1)}\bigr)=f^{(i+1)}\!\bigl(h^{(i)}\bigr),\qquad z^{(L)}=f^{(L)}\!\bigl(h^{(L-1)}\bigr),\qquad z=\bigl(z^{(1)},\dots,z^{(L)}\bigr).$$

This is not just a shortcut: it cuts compute, memory, and parameters so a larger model trains; because peeled-off units are scored against the prior right where they are factored out, every scale gets a direct training signal (the deep-supervision effect); and it builds a coarse-to-fine hierarchy, since units factored out earlier must be Gaussianized before those factored out later, separating local from global features. Hidden width in $s,t$ doubles after each squeeze as resolution halves, and the last scale, with nothing left to factor into, just applies a few checkerboard couplings.

Finally the objective meets the data domain. Pixels are integers with $k=256$ levels but the model lives in unbounded continuous space, so for input $x\in[0,1]$ we dequantize as $r=(255x+u)/256$ with $u\sim U[0,1]$, then keep the logit off the boundaries with $v=\alpha+(1-2\alpha)r$ at $\alpha=.05$ and set $y=\operatorname{logit}(v)$; since $v=\sigma(y)$, the per-dimension preprocessing log-det is $\operatorname{softplus}(y)+\operatorname{softplus}(-y)+\log(1-2\alpha)$. The prior is the isotropic unit Gaussian, $\log p_Z(z)=\sum_i-\tfrac12(z_i^2+\log 2\pi)$. The full objective is exactly the change-of-variables log-likelihood with the total log-det accumulated across every coupling layer ($\sum s$), every batch-norm layer ($-\tfrac12\sum\log(\tilde\sigma^2+\epsilon)$), and the preprocessing transform, with $z=f(x)$ scored under the Gaussian. For discrete data we subtract $D\log k$ from the log-likelihood, equivalently divide $-[\log p_Z(z)+\text{sldj}-D\log k]$ by $D\log 2$ for bits per dimension, and maximize directly with Adam — exact likelihood, parallel sampling via $x=f^{-1}(z)$, exact inference via $z=f(x)$, and a latent space organized coarse-to-fine.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_norm = nn.BatchNorm2d(channels)
        self.in_conv = WNConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_conv = WNConv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x
        x = self.in_conv(F.relu(self.in_norm(x)))
        x = self.out_conv(F.relu(self.out_norm(x)))
        return x + skip


class ResNet(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_blocks,
                 kernel_size=3, padding=1, double_after_norm=False):
        super().__init__()
        self.in_norm = nn.BatchNorm2d(in_ch)
        self.double_after_norm = double_after_norm
        self.in_conv = WNConv2d(2 * in_ch, mid_ch, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_ch, mid_ch, kernel_size=1, padding=0, bias=True)
        self.blocks = nn.ModuleList([ResidualBlock(mid_ch) for _ in range(num_blocks)])
        self.skips = nn.ModuleList([
            WNConv2d(mid_ch, mid_ch, kernel_size=1, padding=0, bias=True)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.BatchNorm2d(mid_ch)
        self.out_conv = WNConv2d(mid_ch, out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_norm(x)
        if self.double_after_norm:
            x = 2. * x
        x = self.in_conv(F.relu(torch.cat((x, -x), dim=1)))
        x_skip = self.in_skip(x)
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip = x_skip + skip(x)
        return self.out_conv(F.relu(self.out_norm(x_skip)))


def squeeze_2x2(x, reverse=False, alt_order=False):
    if reverse:
        b, c4, h, w = x.size()
        c = c4 // 4
        if alt_order:
            x = x.view(b, 4, c, h, w).permute(0, 2, 1, 3, 4)
            x = x[:, :, [0, 2, 3, 1]].contiguous().view(b, c4, h, w)
        return x.view(b, c, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).contiguous().view(
            b, c, 2 * h, 2 * w)

    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, 4 * c, h // 2, w // 2)
    if alt_order:
        x = x.view(b, c, 4, h // 2, w // 2)
        x = x[:, :, [0, 3, 1, 2]].permute(0, 2, 1, 3, 4).contiguous().view(
            b, 4 * c, h // 2, w // 2)
    return x


def checkerboard_mask(h, w, reverse=False, device=None):
    cb = [[((i % 2) + j) % 2 for j in range(w)] for i in range(h)]
    mask = torch.tensor(cb, dtype=torch.float32, device=device)
    if reverse:
        mask = 1 - mask
    return mask.view(1, 1, h, w)


class Rescale(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        return self.weight * x


class FlowBatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-4, rho=0.99):
        super().__init__()
        self.eps = eps
        self.rho = rho
        self.register_buffer("running_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, num_channels, 1, 1))

    def _stats(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            mean = self.rho * self.running_mean + (1. - self.rho) * batch_mean
            var = self.rho * self.running_var + (1. - self.rho) * batch_var
            with torch.no_grad():
                self.running_mean.copy_(mean.detach())
                self.running_var.copy_(var.detach())
            return mean, var
        return self.running_mean, self.running_var

    def forward(self, x, sldj, reverse=False):
        _, _, h, w = x.size()
        mean, var = (self.running_mean, self.running_var) if reverse else self._stats(x)
        log_var = torch.log(var + self.eps)
        if reverse:
            x = x * torch.exp(0.5 * log_var) + mean
            if sldj is not None:
                sldj = sldj + 0.5 * h * w * log_var.view(-1).sum()
            return x, sldj
        x = (x - mean) * torch.exp(-0.5 * log_var)
        if sldj is not None:
            sldj = sldj - 0.5 * h * w * log_var.view(-1).sum()
        return x, sldj


class CouplingLayer(nn.Module):
    """Affine coupling: y_change = x_change * exp(s) + t."""
    def __init__(self, in_channels, mid_channels, num_blocks, channel_wise, reverse_mask):
        super().__init__()
        self.channel_wise = channel_wise
        self.reverse_mask = reverse_mask
        cond_in = in_channels // 2 if channel_wise else in_channels
        self.st_net = ResNet(cond_in, mid_channels, 2 * cond_in, num_blocks,
                             double_after_norm=not channel_wise)
        self.rescale = nn.utils.weight_norm(Rescale(cond_in))
        self.flow_bn = FlowBatchNorm(in_channels)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            x, sldj = self.flow_bn(x, sldj, reverse=True)

        if not self.channel_wise:                              # spatial checkerboard
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, x.device)
            s, t = self.st_net(x * b).chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))                    # keep exp(s) numerically sane
            s, t = s * (1 - b), t * (1 - b)
            if reverse:
                x = (x - t) * s.mul(-1).exp()
                if sldj is not None:
                    sldj = sldj - s.view(s.size(0), -1).sum(-1)
            else:
                x = x * s.exp() + t
                sldj = sldj + s.view(s.size(0), -1).sum(-1)
        else:                                                  # channel-wise split
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)
            s, t = self.st_net(x_id).chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            if reverse:
                x_change = (x_change - t) * s.mul(-1).exp()
                if sldj is not None:
                    sldj = sldj - s.view(s.size(0), -1).sum(-1)
            else:
                x_change = x_change * s.exp() + t
                sldj = sldj + s.view(s.size(0), -1).sum(-1)
            x = (torch.cat((x_id, x_change), dim=1) if self.reverse_mask
                 else torch.cat((x_change, x_id), dim=1))
        if not reverse:
            x, sldj = self.flow_bn(x, sldj, reverse=False)
        return x, sldj


class _RealNVP(nn.Module):
    """One scale: 3 checkerboard couplings -> squeeze -> 3 channel couplings ->
    squeeze+split -> recurse on half the dims."""
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super().__init__()
        self.is_last = scale_idx == num_scales - 1
        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, False, False),
            CouplingLayer(in_channels, mid_channels, num_blocks, False, True),
            CouplingLayer(in_channels, mid_channels, num_blocks, False, False),
        ])
        if self.is_last:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, False, True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, False),
            ])
            self.next = _RealNVP(scale_idx + 1, num_scales,
                                 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for c in self.in_couplings:
                x, sldj = c(x, sldj, reverse)
            if not self.is_last:
                x = squeeze_2x2(x)
                for c in self.out_couplings:
                    x, sldj = c(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
                x = squeeze_2x2(x, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
        else:
            if not self.is_last:
                x = squeeze_2x2(x, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
                x = squeeze_2x2(x)
                for c in reversed(self.out_couplings):
                    x, sldj = c(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
            for c in reversed(self.in_couplings):
                x, sldj = c(x, sldj, reverse)
        return x, sldj


class RealNVP(nn.Module):
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super().__init__()
        self.register_buffer('data_constraint', torch.tensor([0.9]))
        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def _preprocess(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.            # dequantize
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()                          # logit
        ldj = F.softplus(y) + F.softplus(-y) + self.data_constraint.log()
        return y, ldj.view(ldj.size(0), -1).sum(-1)

    def _postprocess(self, y):
        x = y.sigmoid()
        x = (2. * x - 1.) / self.data_constraint
        return ((x + 1.) / 2.).clamp(0., 1.)

    def forward(self, x, reverse=False):
        sldj = None
        if not reverse:
            x, sldj = self._preprocess(x)
        return self.flows(x, sldj, reverse)

    def sample(self, z):
        y, _ = self.forward(z, reverse=True)
        return self._postprocess(y)


class RealNVPLoss(nn.Module):
    """Change-of-variables NLL with an isotropic unit-Gaussian prior."""
    def __init__(self, k=256):
        super().__init__()
        self.k = k

    def log_likelihood(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = (prior_ll.view(z.size(0), -1).sum(-1)
                    - np.log(self.k) * np.prod(z.size()[1:]))
        return prior_ll + sldj

    def forward(self, z, sldj):
        return -self.log_likelihood(z, sldj).mean()

    def bits_per_dim(self, z, sldj):
        dims = np.prod(z.size()[1:])
        return (-self.log_likelihood(z, sldj) / (dims * np.log(2))).mean()
```
