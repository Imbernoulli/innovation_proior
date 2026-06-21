Let me start from where the leverage actually is. The thing I keep noticing is that improvements to the ImageNet classification backbone don't stay in classification — they transfer. A better backbone makes detection, segmentation, pose, video, all of it better, because everything downstream is increasingly built on learned visual features. So the single highest-value thing I can do is make the backbone better. And "better" right now mostly means "scaled up": since 2014, deeper and wider convolutional nets keep buying accuracy.

But scaling up the way it's usually done is wasteful, and that waste is the problem I want to attack. I have two reference points. One is the uniform-3×3 stack — beautifully simple, one primitive repeated, but it evaluates at enormous compute cost and carries about three times the parameters of the old eight-layer net, most of them stranded in a fat fully-connected head. The other is the parallel-branch design — heterogeneous Inception modules with cheap 1×1 dimension reductions — which hits comparable accuracy at around five million parameters, a twelve-fold parameter reduction, and much lower compute. That efficiency is what makes it usable where memory and compute are tight. So the efficient design is clearly the right starting point. The trouble is it's *complex*, and complexity makes it brittle to modify: if I just scale it up the obvious way, doubling every filter-bank width, I pay a 4× increase in both compute and parameters — quadratic, because doubling input and output channels of a conv squares the cost — and that's prohibitive when the accuracy gain is modest. And the original design never documented *why* each choice was made, so I can't safely adapt it without flying blind.

So I won't just turn knobs. I want principles — rules for spending compute so that every added multiply-add earns its keep. There's an information-flow constraint first. A feed-forward net is a directed acyclic graph from input to classifier; for any cut separating inputs from outputs, some amount of information passes through. If I compress the representation too hard at some cut — an extreme bottleneck — I've thrown away information that no downstream layer can recover. So the representation size should decrease *gently* from input to output, never in a violent early drop. I'll be careful: dimensionality is only a rough proxy for information content, since it ignores correlation structure, but as a guideline, avoid representational bottlenecks, especially early in the network.

Second, the opposite-direction pressure: higher-dimensional representations are easier to process locally. More activations per tile means features can disentangle, and the network trains faster. So locally — within a module — I want room, more channels.

Third, and this is the one that makes efficiency possible: spatial aggregation can be done over a *lower-dimensional* embedding with little or no loss of representational power. The reason is correlation. In a vision network, adjacent units are strongly correlated, so if I reduce the channel dimension right before a spatial convolution, I lose little — the signal was compressible — and I even speed up learning. This is exactly what the 1×1 reductions in the efficient architecture are already exploiting, and now I can see why they work rather than just that they do.

Fourth, balance. For a fixed compute budget, I get the best return by increasing depth and width *in parallel*, not dumping the whole budget into one. So when I spend, I spend on both.

These principles are guidelines, not theorems — I'd expect violating them to hurt and respecting them to help, and that's the level of confidence I have. They re-engineer the expensive pieces one at a time.

The most expensive pieces are the large-filter convolutions. A 5×5 convolution with the same number of filters as a 3×3 costs 25/9 ≈ 2.78× as much — disproportionate. The 5×5 buys a larger receptive field, capturing dependencies between units further apart, so naively shrinking the filter costs expressiveness. But maybe I don't have to shrink it — maybe I can *factorize* it. Zoom into the computation of a single 5×5 output: it's a small fully-connected network sliding over each 5×5 input tile. Since I'm building a vision network, I should exploit translation invariance *inside* that mini-network too, and replace its fully-connected component by a two-layer convolutional one: a 3×3 convolution, then another layer on top of the 3×3 output grid. Slide that over the input and the whole thing collapses to two stacked 3×3 convolutions — and two stacked 3×3s have exactly the receptive field of one 5×5. What does it cost? With no channel expansion the two 3×3s cost (9+9) versus the 5×5's 25, a (9+9)/25 ratio — a 28% reduction in compute, and because the net is fully convolutional (each weight is one multiply per activation), the parameter count drops by exactly the same 28%.

Two worries before I commit. Does the replacement lose expressiveness? And — since I'm factorizing what is essentially the linear part of the computation — should the first of the two layers be *linear*, no activation, so I'm purely re-expressing one linear map as a composition? That second idea is seductive but I should test it rather than assume. I run the control: a factorized module with a linear first layer plus ReLU, against one with two ReLU layers. The two-ReLU version wins, settling around 77.2% top-1 where the linear-plus-ReLU version stalls near 76.2%. So the extra nonlinearity is not free expressiveness I can discard — it enlarges the space of functions the module can represent, especially once I batch-normalize the activations. Keep ReLU in both layers. The factorization is real, and it's strictly better than the 5×5.

If a 5×5 reduces to two 3×3s, then any filter larger than 3×3 is suspect — it can always be rewritten as a sequence of 3×3s. So can I push *below* 3×3? Factor a 3×3 into two 2×2 convolutions? That only saves 11%, disappointing. But there's a sharper move: asymmetric convolutions. Replace an n×n by a 1×n followed by an n×1. Take the 3×3 case: a 3×1 convolution followed by a 1×3 convolution slides a two-layer network with the same 3×3 receptive field, and for equal input and output filter counts it's 33% cheaper than the 3×3 — much better than the 2×2's 11%. And the saving only grows as n grows: in principle an n×n becomes a 1×n then n×1 with savings that increase dramatically with n. But I have to be empirical about *where* to use this. When I try asymmetric factorization on the early layers it works poorly — the early layers seem to need genuine 2D filters. On medium-sized grids, though, feature maps between roughly 12×12 and 20×20, it works very well. So I'll factorize n×n on the medium grids, and I'll pick n = 7 there, using a 1×7 followed by a 7×1.

Now a different expensive piece: reducing the grid size. Traditionally you pool to shrink the spatial grid. But there's a tension with principle one. Say I'm at a d×d grid with k filters and I want to land at d/2 × d/2 with 2k filters. The textbook order is: first a stride-1 convolution producing 2k filters on the full d×d grid, then pool — but that convolution on the large grid dominates the cost at about 2·d²·k² operations. To cut cost I could flip the order: pool first to d/2 × d/2, then convolve — now it's 2·(d/2)²·k², a quarter of the cost. But flipping the order means the convolution runs *after* I've already collapsed the representation to (d/2)²·k, and that's a representational bottleneck — exactly what principle one says to avoid — so the network comes out less expressive. So one order is too expensive and the other is too lossy. The way out is to refuse to choose: run two parallel stride-2 branches, a pooling branch P and a convolution branch C, both stride 2, and concatenate their filter banks. The convolution branch supplies new learned filters so there's no bottleneck, while the pooling branch is cheap — together it's cheaper than the expensive option and avoids the bottleneck of the cheap one. That's my grid reduction.

Let me now reconsider the auxiliary classifiers, because I've been assuming the received story and the data doesn't fit it. The story is that intermediate classifier heads push useful gradient down into the lower layers and combat vanishing gradients, helping low-level features evolve faster. But when I watch training with and without the side head, the two curves are virtually identical through the whole early phase — if the head were feeding the lower layers a helpful gradient, I'd expect *faster* early convergence, and I see none. The head only matters near the end of training, where the version with it edges slightly higher. And when there are two side heads, removing the lower one does nothing to final quality. None of that is consistent with "evolves low-level features." What it *is* consistent with is regularization: the head helps late, like a regularizer that keeps the model from settling into a worse generalizing solution, and removing the redundant lower one is harmless. A test of that reading: if it's a regularizer, then making the side branch itself better-regularized should help the *main* classifier. And indeed, batch-normalizing the side head's layers (or adding dropout there) improves the main classifier — about 0.4% absolute top-1 from BN'ing the side head — which also hands me weak evidence that batch normalization itself acts as a regularizer. So I keep one auxiliary head, batch-normalized, and treat it as a regularizer, not a gradient pump.

Now the architecture, assembled from these pieces, at 299×299×3 input. The stem is a stack of small convolutions and pooling, and here's the first place factorization pays at the very front: instead of a single 7×7 convolution, I use three stacked 3×3s — same receptive-field idea as the 5×5→two-3×3 move, applied to 7×7, cheaper with more nonlinearity. So: a 3×3 stride-2 conv down to 149×149×32, a 3×3 conv to 147×147×32, a padded 3×3 to 147×147×64, a 3×3 stride-2 pool to 73×73×64, a 1×1 conv to 80, a 3×3 conv to 192, and a pool, landing at 35×35×192.

Then the body in three blocks of modules separated by grid reductions, and which module variant I use where is governed by the principles. On the 35×35 grid I place three traditional modules: a 1×1 branch, a reduced 5×5 branch, a reduced two-3×3 branch, and a pooled projection branch. The 5×5 branch is already cheap because the 1×1 reduction squeezes it first; the wider path that wants a 5×5-sized receptive field uses the two stacked 3×3s. These modules end at 35×35×288. Then a grid reduction to 17×17 with 768 filters. On the 17×17 grid, the medium-grid regime, I place five modules using the 1×7/7×1 asymmetric factorization. Then another grid reduction to 8×8 with 1280 filters. On the coarsest 8×8 grid I place two modules of a third kind — ones with *expanded* filter banks, where instead of stacking a 1×3 then 3×1 I put them in *parallel* and concatenate, widening the module's output. Why widen specifically here? Principle two: high-dimensional representations are easier to process locally, and the coarsest grid is exactly where producing a high-dimensional, sparse representation matters most — the spatial extent is tiny, so the ratio of local 1×1 processing to spatial aggregation is at its highest, and that's where the extra width does the most good. These two modules end at 8×8×2048. Then an 8×8 average pool, dropout to keep the classifier from leaning on any one final feature too hard, a linear layer to logits, and softmax over 1000 classes. The whole thing is 42 layers deep but costs only about 2.5× the batch-normalized predecessor, and stays well under the uniform-3×3 net.

One more expensive-piece reconsideration, this one on the *objective* rather than the architecture. Training uses softmax cross-entropy against a one-hot target. Write the predicted probabilities `p(k) = exp(z_k) / Σ_i exp(z_i)` from logits `z`, and the loss `ℓ = −Σ_k q(k) log p(k)` against ground-truth distribution `q`. The gradient with respect to a logit is clean: `∂ℓ/∂z_k = p(k) − q(k)`, which is bounded in [−1, 1]. With a hard one-hot target `q(k) = δ_{k,y}`, minimizing the loss means maximizing the log-likelihood of the single correct label, and that maximum is only *approached* as the correct logit `z_y` grows much larger than all the others — it's never attained at finite logits. Two things go wrong as the model chases that. It overfits: assigning full probability to the training label for every example gives no guarantee of generalization. And it drives the gap between the largest logit and the rest to blow up, and with the gradient bounded by 1, a model that's already saturated near the one-hot target has little gradient left to adapt — it becomes over-confident and stiff.

So I want to discourage the model from putting *all* its mass on one label. Replace the hard target with a smoothed one: mix in a small amount of a fixed, example-independent label distribution `u(k)`. With smoothing parameter ε,

  `q'(k) = (1 − ε) δ_{k,y} + ε u(k)`,

and the natural choice for `u` is the label prior; on ImageNet I just take it uniform, `u(k) = 1/K`, so `q'(k) = (1 − ε) δ_{k,y} + ε/K`. Now the loss can never reward driving one logit to infinity, because every `q'(k)` has a positive lower bound ε/K — if `p` collapsed onto a single label, the cross-entropy against `q'` would actually grow. To see exactly what it's doing, expand the cross-entropy against the mixture:

  `H(q', p) = (1 − ε) H(q, p) + ε H(u, p)`.

So smoothing is just adding a second loss `ε·H(u, p)` to the usual `H(q, p)` — a term that penalizes the predicted distribution for deviating from the prior `u`, with relative weight `ε/(1−ε)`. Since `H(u, p) = D_KL(u‖p) + H(u)` and `H(u)` is constant, that second term is, up to a constant, the KL divergence from the prior `u` to the prediction `p` — it pulls predictions gently toward the prior. With K = 1000 and ε = 0.1 this gives a consistent ~0.2% absolute improvement on both top-1 and top-5. Cheap, and it directly fixes the over-confidence I diagnosed.

Now training. I batch-normalize after every convolution — that's what lets these factorized stacks train at all, and as I noted it doubles as a regularizer. I also keep dropout before the main classifier with keep probability 0.8. For the optimizer, momentum SGD with decay 0.9 works, but the best models come from RMSProp with decay 0.9 and ε = 1.0 — the per-parameter adaptive scaling helps with the heterogeneous module structure. Learning rate 0.045, decayed by an exponential factor of 0.94 every two epochs. The deep factorized stack can throw occasional large gradients, so I clip gradients at a threshold of 2.0 to keep training stable. And I evaluate on a running (exponential moving) average of the parameters rather than the raw latest weights, which smooths out the noise of the small per-replica batch. Training runs across about fifty GPU replicas at batch size 32 each, for roughly 100 epochs.

The full recipe I'm landing on is the factorized architecture *plus* these four cumulative refinements: the batch-normalized auxiliary head, RMSProp, label smoothing, and the 7×7-into-three-3×3 stem factorization — each one earned its place by a principle or a control experiment. What I'd want to validate is the single-frame top-1/top-5 against the batch-normalized predecessor and against the denser, heavier successors, at a fraction of their compute and parameters, plus that ensembling a few of these with multi-crop pushes top-5 lower still.

Let me write it as code, filling the slots from the skeleton. The base primitive is conv-without-bias → BN → ReLU. The repeated multi-branch module comes in three flavors — the two-3×3 version for the 35×35 grid, the 1×7/7×1 version for the 17×17 grid, the expanded-parallel version for the 8×8 grid — each a set of branches concatenated on the channel axis. The grid-reduction module is the parallel stride-2 conv-and-pool. The auxiliary head is the batch-normalized side classifier on the last 17×17 stage. The objective is label-smoothed cross-entropy. I'll mirror the standard structure, where these module flavors are named A/B/C and the reductions B/D.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):                       # base primitive: conv -> BN -> ReLU
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):                   # 35x35 grid: reduced 5x5 + double 3x3
    def __init__(self, in_ch, pool_features):
        super().__init__()
        self.b1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b5_1 = ConvBN(in_ch, 48, kernel_size=1)            # reduce before spatial
        self.b5_2 = ConvBN(48, 64, kernel_size=5, padding=2)
        self.b3_1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b3_2 = ConvBN(64, 96, kernel_size=3, padding=1)    # two stacked 3x3
        self.b3_3 = ConvBN(96, 96, kernel_size=3, padding=1)    #   = one 5x5 RF
        self.bp = ConvBN(in_ch, pool_features, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b5 = self.b5_2(self.b5_1(x))
        b3 = self.b3_3(self.b3_2(self.b3_1(x)))
        bp = self.bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([b1, b5, b3, bp], 1)


class GridReductionB(nn.Module):               # 35x35 -> 17x17: parallel stride-2 conv & pool
    def __init__(self, in_ch):
        super().__init__()
        self.b3 = ConvBN(in_ch, 384, kernel_size=3, stride=2)
        self.b3d_1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b3d_2 = ConvBN(64, 96, kernel_size=3, padding=1)
        self.b3d_3 = ConvBN(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        b3 = self.b3(x)
        b3d = self.b3d_3(self.b3d_2(self.b3d_1(x)))
        bp = F.max_pool2d(x, 3, stride=2)
        return torch.cat([b3, b3d, bp], 1)      # conv branches + pool branch, no bottleneck


class InceptionC(nn.Module):                   # 17x17 grid: n x n factorized into 1x7 + 7x1
    def __init__(self, in_ch, c7):
        super().__init__()
        self.b1 = ConvBN(in_ch, 192, kernel_size=1)
        self.b7_1 = ConvBN(in_ch, c7, kernel_size=1)
        self.b7_2 = ConvBN(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.b7_3 = ConvBN(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.b7d_1 = ConvBN(in_ch, c7, kernel_size=1)
        self.b7d_2 = ConvBN(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.b7d_3 = ConvBN(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.b7d_4 = ConvBN(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.b7d_5 = ConvBN(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.bp = ConvBN(in_ch, 192, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b7 = self.b7_3(self.b7_2(self.b7_1(x)))
        b7d = self.b7d_5(self.b7d_4(self.b7d_3(self.b7d_2(self.b7d_1(x)))))
        bp = self.bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([b1, b7, b7d, bp], 1)


class GridReductionD(nn.Module):               # 17x17 -> 8x8
    def __init__(self, in_ch):
        super().__init__()
        self.b3_1 = ConvBN(in_ch, 192, kernel_size=1)
        self.b3_2 = ConvBN(192, 320, kernel_size=3, stride=2)
        self.b7_1 = ConvBN(in_ch, 192, kernel_size=1)
        self.b7_2 = ConvBN(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.b7_3 = ConvBN(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.b7_4 = ConvBN(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        b3 = self.b3_2(self.b3_1(x))
        b7 = self.b7_4(self.b7_3(self.b7_2(self.b7_1(x))))
        bp = F.max_pool2d(x, 3, stride=2)
        return torch.cat([b3, b7, bp], 1)


class InceptionE(nn.Module):                   # 8x8 grid: expanded (parallel 1x3 & 3x1) for width
    def __init__(self, in_ch):
        super().__init__()
        self.b1 = ConvBN(in_ch, 320, kernel_size=1)
        self.b3_1 = ConvBN(in_ch, 384, kernel_size=1)
        self.b3_2a = ConvBN(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.b3_2b = ConvBN(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.b3d_1 = ConvBN(in_ch, 448, kernel_size=1)
        self.b3d_2 = ConvBN(448, 384, kernel_size=3, padding=1)
        self.b3d_3a = ConvBN(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.b3d_3b = ConvBN(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.bp = ConvBN(in_ch, 192, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b3 = self.b3_1(x)
        b3 = torch.cat([self.b3_2a(b3), self.b3_2b(b3)], 1)     # parallel, not stacked -> wider
        b3d = self.b3d_2(self.b3d_1(x))
        b3d = torch.cat([self.b3d_3a(b3d), self.b3d_3b(b3d)], 1)
        bp = self.bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([b1, b3, b3d, bp], 1)


class AuxHead(nn.Module):                      # batch-normalized side classifier = regularizer
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv0 = ConvBN(in_ch, 128, kernel_size=1)
        self.conv1 = ConvBN(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv1(self.conv0(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.fc(torch.flatten(x, 1))


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        # stem: 7x7 factorized into three 3x3, then reductions to 35x35x192
        self.Conv2d_1a = ConvBN(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a = ConvBN(32, 32, kernel_size=3)
        self.Conv2d_2b = ConvBN(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b = ConvBN(64, 80, kernel_size=1)
        self.Conv2d_4a = ConvBN(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)      # 35x35x256
        self.Mixed_5c = InceptionA(256, pool_features=64)      # 35x35x288
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = GridReductionB(288)                    # -> 17x17x768
        self.Mixed_6b = InceptionC(768, c7=128)
        self.Mixed_6c = InceptionC(768, c7=160)
        self.Mixed_6d = InceptionC(768, c7=160)
        self.Mixed_6e = InceptionC(768, c7=192)
        self.aux = AuxHead(768, num_classes) if aux_logits else None
        self.Mixed_7a = GridReductionD(768)                    # -> 8x8x1280
        self.Mixed_7b = InceptionE(1280)                       # 8x8x2048
        self.Mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)                       # keep probability 0.8
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.Conv2d_1a(x); x = self.Conv2d_2a(x); x = self.Conv2d_2b(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.Conv2d_3b(x); x = self.Conv2d_4a(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.Mixed_5b(x); x = self.Mixed_5c(x); x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x); x = self.Mixed_6c(x); x = self.Mixed_6d(x); x = self.Mixed_6e(x)
        aux = self.aux(x) if (self.aux is not None and self.training) else None
        x = self.Mixed_7a(x); x = self.Mixed_7b(x); x = self.Mixed_7c(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.dropout(x)
        return self.fc(x), aux


def label_smoothed_ce(logits, target, num_classes, eps=0.1):
    # H(q', p) = (1-eps) H(q, p) + eps H(u, p), u uniform = 1/K
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)      # H(q, p)
    smooth = -logp.mean(dim=1)                                 # H(u, p) up to const
    return ((1 - eps) * nll + eps * smooth).mean()


model = InceptionV3(num_classes=1000, aux_logits=True)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045, alpha=0.9, eps=1.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)   # every 2 epochs
```

The whole thing is one idea applied with discipline: spend compute where principles say it pays. Factorize expensive filters into stacks of 3×3 and asymmetric 1×n / n×1 convolutions where the grid scale supports it; reduce dimension before spatial aggregation; reduce the grid with parallel conv-and-pool branches so there's never a bottleneck; widen only on the coarsest grid where high-dimensional local processing helps most; regularize with dropout, a batch-normalized auxiliary head, and a label-smoothed objective; and train it with RMSProp, exponential decay, gradient clipping, and a parameter EMA. That is the full candidate I would now test against the batch-normalized predecessor under the same ImageNet evaluation protocol.
