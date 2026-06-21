Two frames are placed in front of me and I want, for every pixel of the first, the 2D vector pointing to where it went in the second. Forty years of optical flow, and the same cases still break everything: a bird flapping across the frame, a car edge against a guardrail, motion blur smearing the texture away. The classical view treats this as energy minimization — write $E(\mathbf{f}) = E_{\text{data}}(\mathbf{f}) + \lambda E_{\text{reg}}(\mathbf{f})$, where the data term says flow should map visually similar regions onto each other and the regularizer says nearby pixels should move similarly, then keep a single dense field and push it downhill. I genuinely like one thing in that picture: you maintain *one* estimate and you *refine* it, which is honest about the fact that flow is something you converge to, not something you read off in one shot. But there is a rot at its center. To keep the energy differentiable, the data term gets a first-order Taylor expansion of image intensities, $I_2(\mathbf{x}+\mathbf{f}) \approx I_2(\mathbf{x}) + \nabla I_2 \cdot \mathbf{f}$, which is only valid when $\mathbf{f}$ is small. So large motion must be estimated at low resolution — shrink the image until displacement is small in pixels, solve, upsample, refine. Coarse-to-fine. And that cascade has two diseases I know by heart: a small fast object is simply *gone* at the coarse level where large motion is estimated, so it can never be recovered, and an error committed at a coarse level is almost impossible to undo at finer levels, because each fine level only ever estimates a small residual around whatever the coarse level handed it.

The deep-learning methods promised escape but did not deliver it. Look at what the best of them actually are — PWC-Net, LiteFlowNet, VCN, FlowNet2: a learnable feature pyramid, a warping layer, a partial cost volume at each pyramid level, decode a flow update, go up a level, repeat. That is the classical coarse-to-fine cascade in a CNN costume, and it inherited the exact same two diseases — it still loses the bird, still cannot recover from a coarse mistake — while costing on the order of millions of training iterations. DCFlow showed a full 4D cost volume over learned features is cheap to build, but its SGM cost-volume processing has no gradient, so its feature network must be trained on a surrogate triplet loss rather than directly on flow error. What is forcing coarse-to-fine in the first place is *only* the Taylor linearization; the pyramid exists to keep the linearized data term valid. If I never linearize — if I let the data term directly see large displacement — there is no reason left to go coarse-to-fine at all, and I can maintain a single high-resolution flow field from start to finish.

I propose RAFT: Recurrent All-Pairs Field Transforms. The whole method follows from refusing the linearization and replacing the hand-crafted energy with three learned pieces — an all-pairs correlation volume as the data term, a recurrent learned update operator as the optimizer, and a learned convex upsampler — all of it differentiable end-to-end. Start with the data term, which is "visual similarity between a source pixel and a candidate target pixel." If I want to see *all* displacements at once, I should compute similarity between a source pixel and *every* target pixel — not a window, everything. Extract per-pixel features $g_\theta(I_1), g_\theta(I_2) \in \mathbb{R}^{H\times W\times D}$ at $1/8$ resolution and form

$$C_{ijkl} = \sum_h g_\theta(I_1)_{ijh}\,g_\theta(I_2)_{klh} = \langle g_\theta(I_1)_{ij},\, g_\theta(I_2)_{kl}\rangle,$$

a 4D object of shape $H\times W\times H\times W$: the data term for every possible displacement of every pixel. FlowNetC declared this intractable in 2015 and bounded the displacement to a window, but that was a mistake of perspective — $C_{ijkl}$ is an inner product over channels, so flattening the spatial dimensions makes the full correlation just $C = g_\theta(I_1)\,g_\theta(I_2)^\top$, a single $(HW)\times(HW)$ matrix multiply, one of the most optimized operations on a GPU. It is computed once, since it does not depend on the flow estimate, and amortized over every refinement step. The raw inner products scale with $D=256$, which makes the magnitudes large and the downstream operator twitchy, so I divide by $\sqrt{D}$ — cheap insurance. There is no bounded window, no maximum displacement, no warping.

For each source pixel I now have a full 2D response map over the whole target image, but the update operator cannot ingest the entire $HW$-dimensional response per pixel each step; most of it is irrelevant. What it needs is the similarity in the *vicinity of where it currently thinks the pixel went*. Given current flow $\mathbf{f}$, map source pixel $\mathbf{x}$ to its estimated correspondence $\mathbf{x}'=\mathbf{x}+\mathbf{f}$ and read correlation in a local grid $\mathcal{N}(\mathbf{x}')_r = \{\mathbf{x}'+\mathbf{dx}: \mathbf{dx}\in\mathbb{Z}^2,\ \|\mathbf{dx}\|_1\le r\}$, bilinearly sampling the volume since $\mathbf{x}'$ is real-valued. But a fixed radius $r$ around the current guess only sees displacements within $r$ pixels of that guess — initialize at zero with a true motion of 200 pixels and the lookup reads near the origin, nowhere near the answer. A local window has re-smuggled in the small-displacement problem. The resolution is to pool the volume, and to pool it *asymmetrically*. The correlation $C_{ijkl}$ has two index pairs: $(i,j)$ for the source pixel, $(k,l)$ for the target. Average-pool only the *last two* (target) dimensions with kernels $\{1,2,4,8\}$ and matching strides to build a pyramid $\{C^1,C^2,C^3,C^4\}$, $C^k$ of shape $H\times W\times H/2^k\times W/2^k$. Keeping the source dimensions $(i,j)$ at full resolution is what protects the bird — a 3-pixel-wide fast object still has its own response maps and is never blurred away, exactly the failure of the cascade, which pooled source and target together. Then do the local lookup on *every* level, indexing $C^k$ at $\mathbf{x}'/2^k$ with the same radius $r$. A constant $r$ across levels means a wildly different real-world range per level: at the coarsest level three stride-2 poolings shrink each cell to span $8\times 8 = 64$ original pixels, so with $r=4$ that level reaches $\pm 256$ pixels. Concatenating all levels, a single cheap local lookup simultaneously carries fine nearby similarity and coarse evidence about matches hundreds of pixels away. The move is: don't shrink the *image* to reach large motion, shrink the *displacement axis of the cost volume* while keeping the image sharp. This is also principled rather than convenient — pulling $g^{(1)}_{ij}$ out of the pooling sum by linearity of the inner product gives $C^m_{ijkl} = \langle g^{(1)}_{ij}, \text{avgpool}(g^{(2)})_{kl}\rangle$, so pooling the correlation over the target window is identical to correlating with pooled target features, which hands me an $O(NM)$ on-demand implementation if the $O(N^2)$ volume ever became a memory bottleneck.

Now the engine. I kept "one field, refined iteratively" but deleted the analytic descent direction, because there is no explicit energy to differentiate anymore — the hand-crafted data+smoothness energy has been replaced by a learned correlation volume. So I *learn* the update step, in the spirit of learning-to-optimize: a first-order optimizer is a sequence of update steps, so learn the step from data. Maintain $\mathbf{f}_0=\mathbf{0}, \mathbf{f}_1, \mathbf{f}_2, \dots$ and apply $\mathbf{f}_{k+1}=\mathbf{f}_k+\Delta\mathbf{f}$, where a learned operator looks at the correlation around the current guess and proposes an increment; the smoothness prior, once hand-written into $E_{\text{reg}}$, becomes whatever the operator learns. What should that operator be? A plain conv-ReLU stack applied fifty times will drift — nothing bounds the state, nothing decides when to stop changing a pixel that is already correct. The sequence-modeling lesson (TrellisNet's weight tying, DEQ's observation that weight-tied stacks converge to a fixed point) says the ingredients for convergence are tied weights and bounded, gated updates. A GRU is exactly a gated update with an update gate that decides per element how much state to overwrite and sigmoids/tanh that bound activations, so a converged pixel can simply stop updating ($z\to 0$) while one still far from its match keeps moving. Make the operator a convolutional GRU, replacing the fully-connected layers with $3\times3$ convolutions:

$$z_t=\sigma(\text{Conv}_{3\times3}([h_{t-1},x_t],W_z)),\quad r_t=\sigma(\text{Conv}_{3\times3}([h_{t-1},x_t],W_r)),$$
$$\tilde h_t=\tanh(\text{Conv}_{3\times3}([r_t\odot h_{t-1},x_t],W_h)),\quad h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.$$

The gating *is* the convergence mechanism. Crucially, I *tie the weights across iterations* — every step is the same GRU. This is not a parameter-saving trick: it forces the network to learn a single update rule that must work no matter how many times applied, which shrinks the hypothesis space (better generalization, less overfitting to synthetic data) and decouples the iteration count from training, so I train with a modest unroll and run as many steps as I like at test time. Untying the weights would balloon the parameters by an order of magnitude *and* generalize worse. The operator's input $x_t$ is three things: the correlation lookup features (the data evidence), the current flow (so it knows where it is), and features from $I_1$ only via a context network $h_\theta$ — because the smoothness prior needs to know where the object boundaries are, and that lives in image appearance, not in the correlation, so injecting context lets the operator propagate motion within a region and stop at its edge. The hidden state is initialized from the context too (split into a $\tanh$ hidden part and a $\text{ReLU}$ input part), and the flow head produces a 2-channel residual $\Delta\mathbf{f}$, not absolute flow, because the design is about refining a maintained estimate. To give the operator a larger receptive field for propagating motion across an object without paying for a dense $5\times5$ conv, I factor it: run the GRU twice per step, once with $1\times5$ and once with $5\times1$ convolutions, a separable pass that yields a $5\times5$-ish reach cheaply.

Two training subtleties for the recurrent chain. First, in $\mathbf{f}_{k+1}=\mathbf{f}_k+\Delta\mathbf{f}$, backpropagating through both branches sends gradients through the whole history of additions and destabilizes training; the fix is to carry gradient only through the $\Delta\mathbf{f}$ branch and detach $\mathbf{f}_k$, so each step learns to produce a good increment given the current flow as a constant — exactly the optimizer-step semantics I want. Second, I supervise the *entire* sequence rather than only the last iterate, so intermediate steps are pressured to be real refinements, using a discounted $L_1$ loss

$$\mathcal{L}=\sum_{i=1}^N \gamma^{N-i}\,\|\mathbf{f}_{gt}-\mathbf{f}_i\|_1,\qquad \gamma=0.8,$$

with $L_1$ rather than $L_2$ because flow ground truth has outliers and occlusion artifacts I do not want dominating, and $\gamma=0.8$ letting early steps count enough to learn from without swamping the final-step signal. The last piece is upsampling the $1/8$-resolution flow to full resolution. Bilinear interpolation across a motion boundary averages two motions into a smear, killing the sharpness I worked to preserve. So I learn the upsampling: from the hidden state, predict a mask of shape $H/8\times W/8\times(8\times8\times9)$, i.e. for each of the $8\times8$ fine pixels in a coarse cell, 9 weights over the $3\times3$ coarse neighbors; softmax the 9 into a convex combination, take the weighted sum of the $\times8$-scaled coarse flow, and reshape. Because the weights are predicted per pixel, the network can put all the weight on the correct side at a boundary — no averaging across the edge — and I scale the mask by $0.25$ before the softmax to keep gradient magnitudes balanced. Every operation here — matmul, pooling, bilinear lookup, GRU, convex upsampling, loss — is differentiable, so the feature encoder trains *directly on flow error* rather than a surrogate embedding loss, learning to be exactly the features that make the flow accurate. And because every step shares weights at a single high resolution with no warping, I can iterate to a fixed point at test time and warm-start $\mathbf{f}_0$ from the previous frame's flow when processing video — something the cascade, lacking any single coherent field, cannot do.

I find it cleanest to represent the flow as the difference of two coordinate grids: $\text{coords0}$ is the identity grid, $\text{coords1}$ starts equal to it, and the flow is always $\text{coords1} - \text{coords0}$. Then "look up correlation at the current correspondence" is "look up at $\text{coords1}$," "apply the update" is $\text{coords1} \mathrel{+}= \Delta\mathbf{f}$, and warm-start is just adding the initial flow to $\text{coords1}$.

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

A smaller variant of about 1M parameters replaces the residual blocks with bottleneck blocks, uses $D=128$ features, hidden/context dimensions $96/64$, a single $3\times3$ ConvGRU, radius $r=3$, and bilinear rather than convex upsampling; it still outperforms much larger prior coarse-to-fine networks. For the full model I train with InstanceNorm in the feature encoder and BatchNorm in the context encoder, unroll 12 updates during training, evaluate with 32 (Sintel) or 24 (KITTI) updates — the field converges and does not diverge even at 200 — pretraining on FlyingChairs then FlyingThings3D before finetuning, with photometric, spatial, and occlusion (random-erase in $I_2$) augmentation.
