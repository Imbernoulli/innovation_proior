# RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

## Problem

Given two consecutive frames $I_1, I_2$, estimate a dense per-pixel displacement field. The prior recipe — a coarse-to-fine cascade (whether hand-crafted variational or learnable CNN) — loses small fast-moving objects (gone at coarse levels), cannot recover from errors made at coarse resolution, and is expensive to train. The goal: maintain a single high-resolution flow field, refine it with a recurrent operator that sees both small and large displacements, and train the whole thing end-to-end.

## Key idea

1. **All-pairs 4D correlation volume as the data term.** Extract per-pixel features $g_\theta(I_1), g_\theta(I_2) \in \mathbb{R}^{H\times W\times D}$ at $1/8$ resolution and compute the inner product of *every* pair of feature vectors,
   $$C_{ijkl} = \sum_h g_\theta(I_1)_{ijh}\,g_\theta(I_2)_{klh},\qquad C \in \mathbb{R}^{H\times W\times H\times W},$$
   as a single matrix multiplication $C = g_\theta(I_1)\,g_\theta(I_2)^\top$ (then divided by $\sqrt{D}$). Computed once; independent of the number of refinement steps. No bounded window, no warping.

2. **Correlation pyramid (pool only the target axis).** Average-pool the *last two* dimensions of $C$ with kernels $\{1,2,4,8\}$ and equal stride to form $\{C^1,C^2,C^3,C^4\}$, with $C^k$ of shape $H\times W\times H/2^k\times W/2^k$. Keeping the first two (source) dimensions at full resolution preserves small fast objects; pooling the target dimensions packs large-displacement context into a small grid. Equivalently (linearity of the inner product), $C^m_{ijkl} = \langle g^{(1)}_{ij}, \text{avgpool}(g^{(2)})_{kl}\rangle$, which gives an $O(NM)$ on-demand implementation if the $O(N^2)$ volume is ever a bottleneck.

3. **Lookup.** Given the current flow $\mathbf{f}$, map each source pixel $\mathbf{x}$ to $\mathbf{x}'=\mathbf{x}+\mathbf{f}$ and read correlation in a local grid $\mathcal{N}(\mathbf{x}')_r = \{\mathbf{x}'+\mathbf{dx}: \|\mathbf{dx}\|_1\le r\}$ via bilinear sampling, on *every* pyramid level (level $k$ indexed at $\mathbf{x}'/2^k$). A constant radius $r$ means coarser levels cover a larger real-world range ($k{=}4$, $r{=}4$ ⇒ ±256 px). Concatenate all levels.

4. **Recurrent update operator (learned descent step).** Maintain $\mathbf{f}_0=\mathbf{0},\mathbf{f}_1,\dots$ and apply $\mathbf{f}_{k+1}=\mathbf{f}_k+\Delta\mathbf{f}$, where a **weight-tied** convolutional GRU proposes $\Delta\mathbf{f}$ from (correlation lookup, current flow, image context, hidden state):
   $$z_t=\sigma(\text{Conv}_{3\times3}([h_{t-1},x_t],W_z)),\quad r_t=\sigma(\text{Conv}_{3\times3}([h_{t-1},x_t],W_r)),$$
   $$\tilde h_t=\tanh(\text{Conv}_{3\times3}([r_t\odot h_{t-1},x_t],W_h)),\quad h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.$$
   Tied weights + gated bounded updates drive the sequence to a fixed point $\mathbf{f}_k\to\mathbf{f}^*$, decouple iteration count from training, and improve generalization. The full model uses a separable GRU ($1\times5$ then $5\times1$) for a larger receptive field at low parameter cost. A context network reads $I_1$ only and is injected so the operator can regularize within motion boundaries.

5. **Convex upsampling.** Upsample the $1/8$-resolution flow by predicting, per output pixel, softmax weights over a $3\times3$ neighborhood of coarse pixels (mask of shape $H/8\times W/8\times(8\times8\times9)$) and taking the convex combination — sharp at motion boundaries, unlike bilinear.

6. **End-to-end & supervision.** Every operation is differentiable, so the feature encoder trains directly on flow error (no surrogate embedding loss). Supervise the *entire* sequence of predictions with discounted $L_1$:
   $$\mathcal{L}=\sum_{i=1}^N \gamma^{N-i}\,\|\mathbf{f}_{gt}-\mathbf{f}_i\|_1,\qquad \gamma=0.8.$$
   Only the $\Delta\mathbf{f}$ branch carries gradient (the $\mathbf{f}_k$ branch is detached). For video, warm-start $\mathbf{f}_0$ from the previous frame's flow (forward-projected, occlusion gaps nearest-neighbor filled).

## Training details

PyTorch; AdamW; gradient clipping (norm 1.0); OneCycle LR (linear, `pct_start=0.05`). Feature encoder: 6 residual blocks, downsampling to $1/8$, $D=256$ (full) / $128$ (small, bottleneck blocks); InstanceNorm in the feature encoder, BatchNorm in the context encoder. Unroll 12 updates during training; evaluate with 32 updates (Sintel) / 24 (KITTI); converges and does not diverge even at 200. Pretrain on FlyingChairs → FlyingThings3D, then finetune (Sintel: S+T+K+H; KITTI: K). Photometric, spatial, and occlusion (random-erase in $I_2$) augmentation.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- utilities ----------------
def bilinear_sampler(img, coords):
    """grid_sample wrapper using pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device),
                            torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


# ---------------- feature / context encoder ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='instance', stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        ng = planes // 8
        norm = {'group': lambda c: nn.GroupNorm(ng, c),
                'batch': nn.BatchNorm2d, 'instance': nn.InstanceNorm2d,
                'none': lambda c: nn.Sequential()}[norm_fn]
        self.norm1, self.norm2 = norm(planes), norm(planes)
        if stride == 1:
            self.downsample = None
        else:
            self.norm3 = norm(planes)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride), self.norm3)

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='instance', dropout=0.0):
        super().__init__()
        self.norm_fn = norm_fn
        n1 = {'group': lambda: nn.GroupNorm(8, 64), 'batch': lambda: nn.BatchNorm2d(64),
              'instance': lambda: nn.InstanceNorm2d(64), 'none': lambda: nn.Sequential()}[norm_fn]
        self.norm1 = n1()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(96, 2)
        self.layer3 = self._make_layer(128, 2)
        self.conv2 = nn.Conv2d(128, output_dim, 1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride):
        layers = (ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride),
                  ResidualBlock(dim, dim, self.norm_fn, stride=1))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            b = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [b, b], dim=0)
        return x


# ---------------- all-pairs 4D correlation + pyramid + lookup ----------------
class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels, self.radius = num_levels, radius
        self.corr_pyramid = []
        corr = CorrBlock.corr(fmap1, fmap2)
        b, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(b * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)          # pool only target dims
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        b, h1, w1, _ = coords.shape
        out = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            centroid = coords.reshape(b * h1 * w1, 1, 1, 2) / 2 ** i
            coords_lvl = centroid + delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            corr = bilinear_sampler(corr, coords_lvl)
            out.append(corr.view(b, h1, w1, -1))
        return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        b, dim, ht, wd = fmap1.shape
        f1 = fmap1.view(b, dim, ht * wd)
        f2 = fmap2.view(b, dim, ht * wd)
        corr = torch.matmul(f1.transpose(1, 2), f2)         # C = g1 g2^T : all pairs
        corr = corr.view(b, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


# ---------------- update operator (motion encoder + separable ConvGRU + heads) ----------------
class FlowHead(nn.Module):
    def __init__(self, in_dim=128, hid=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hid, 3, padding=1)
        self.conv2 = nn.Conv2d(hid, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super().__init__()
        d = hidden_dim + input_dim
        self.convz1 = nn.Conv2d(d, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(d, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(d, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(d, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(d, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(d, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx)); r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1))); h = (1 - z) * h + z * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx)); r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1))); h = (1 - z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super().__init__()
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, cor_planes, hidden_dim=128):
        super().__init__()
        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hid=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion = self.encoder(flow, corr)
        inp = torch.cat([inp, motion], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)                          # scale to balance gradients
        return net, mask, delta_flow


# ---------------- full model ----------------
class RAFT(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=128, corr_levels=4, corr_radius=4):
        super().__init__()
        self.hidden_dim, self.context_dim = hidden_dim, context_dim
        self.corr_radius = corr_radius
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=hidden_dim + context_dim, norm_fn='batch')
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.update_block = BasicUpdateBlock(cor_planes, hidden_dim=hidden_dim)

    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1                              # flow = coords1 - coords0

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1).view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1, image2 = image1.contiguous(), image2.contiguous()

        fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = CorrBlock(fmap1.float(), fmap2.float(), radius=self.corr_radius)

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init                    # warm-start

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()                       # detach f_k branch
            corr = corr_fn(coords1)                          # lookup
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow                   # f_{k+1} = f_k + delta f
            flow_predictions.append(self.upsample_flow(coords1 - coords0, up_mask))

        if test_mode:
            return coords1 - coords0, flow_predictions[-1]
        return flow_predictions


# ---------------- sequence loss ----------------
def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    n = len(flow_preds)
    flow_loss = 0.0
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n):
        w = gamma ** (n - i - 1)                             # later iterates weighted more
        flow_loss += w * (valid[:, None] * (flow_preds[i] - flow_gt).abs()).mean()
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {'epe': epe.mean().item(),
               '1px': (epe < 1).float().mean().item(),
               '3px': (epe < 3).float().mean().item(),
               '5px': (epe < 5).float().mean().item()}
    return flow_loss, metrics


# ---------------- optimizer ----------------
def fetch_optimizer(model, lr=4e-4, wdecay=1e-4, num_steps=100000, eps=1e-8):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay, eps=eps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, num_steps + 100, pct_start=0.05,
        cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler
```

A smaller variant (≈1M parameters) replaces the residual blocks with bottleneck blocks, uses $D=128$ features, hidden/context dims $96/64$, a single $3\times3$ ConvGRU, radius $r=3$, and bilinear (rather than convex) upsampling; it still outperforms much larger prior coarse-to-fine networks.
