The leverage here is unusually clear: improvements to the ImageNet classification backbone do not stay in classification — they transfer to detection, segmentation, pose, and video, because everything downstream is increasingly built on the same learned features. So the highest-value move is to make the backbone better, and right now "better" mostly means "scaled up," since deeper and wider convolutional nets keep buying accuracy. The trouble is that scaling the usual way is wasteful, and that waste is what I want to attack. I have two reference points. One is the uniform-3×3 stack — one primitive repeated, conceptually beautiful, but it evaluates at enormous compute cost and carries about three times the parameters of the old eight-layer net, most of them stranded in a fat fully-connected head. The other is the parallel-branch design — heterogeneous Inception modules with cheap 1×1 dimension reductions — which reaches comparable accuracy at around five million parameters, a twelve-fold reduction, and much lower compute, which is exactly what makes it usable where memory and compute are tight. The efficient design is the right starting point, but it is complex, and complexity makes it brittle to modify: scaling it the obvious way, doubling every filter-bank width, costs a 4× increase in both compute and parameters, because doubling the input and output channels of a convolution squares its cost — prohibitive when the accuracy gain is modest. And the original design never documented why each choice was made, so I cannot safely adapt it without flying blind. I refuse to just turn knobs; I want principles for spending compute so that every added multiply-add earns its keep.

I propose Inception-v3: a 42-layer convolutional classifier that scales the parallel-branch architecture by re-engineering each expensive piece under four design principles, then training it with three further refinements. The principles are guidelines, not theorems — I expect violating them to hurt and respecting them to help. First, avoid representational bottlenecks, especially early: a feed-forward net is a directed graph from input to classifier, and across any cut some information passes through; compress too hard at some cut and downstream layers can never recover what was thrown away, so the representation size should decrease gently, never in a violent early drop. Second, higher-dimensional representations are easier to process locally — more activations per tile let features disentangle and the network train faster — so within a module I want room. Third, and this is what makes efficiency possible, spatial aggregation can be done over a lower-dimensional embedding with little loss, because adjacent units in a vision network are strongly correlated, so reducing the channel dimension right before a spatial convolution discards little; this is exactly what the 1×1 reductions already exploit, and now I can see why. Fourth, balance: for a fixed budget, grow depth and width in parallel, not one alone.

The most expensive pieces are the large-filter convolutions. A 5×5 convolution with the same number of filters as a 3×3 costs $25/9 \approx 2.78\times$ as much — disproportionate — but it buys a larger receptive field, so naively shrinking it costs expressiveness. Instead I factorize it. The computation of a 5×5 output is a small fully-connected network sliding over each 5×5 input tile; since I am building a vision network I can exploit translation invariance inside that mini-network too, replacing its fully-connected part by a two-layer convolutional one — a 3×3 followed by another layer on the 3×3 output grid — which slides out to two stacked 3×3 convolutions, and two stacked 3×3s have exactly the receptive field of one 5×5. With no channel expansion the two 3×3s cost $9+9$ against the 5×5's $25$, a $(9+9)/25$ ratio: a 28% reduction in compute, and because the net is fully convolutional (one weight per multiply per activation), parameters drop by the same 28%. One seductive idea is to make the first of the two layers linear, so I am merely re-expressing a linear map as a composition. I tested it rather than assumed: a factorized module with a linear first layer plus ReLU settles near 76.2% top-1, while one with two ReLU layers reaches about 77.2%. The extra nonlinearity is not free expressiveness to discard — it enlarges the function space, especially once activations are batch-normalized — so I keep ReLU in both layers. Once a 5×5 reduces to two 3×3s, any larger filter is suspect; and below 3×3 the sharp move is asymmetric factorization, replacing an $n \times n$ by a $1 \times n$ followed by an $n \times 1$. For $3 \times 3$, a $3\times1$ then $1\times3$ is 33% cheaper at equal filter counts — far better than the 11% from $3\times3 \to$ two $2\times2$ — and the saving grows with $n$. But this works poorly on the early layers, which seem to need genuine 2D filters, and very well on medium grids (roughly 12×12 to 20×20), so I apply it there with $n=7$: a $1\times7$ then $7\times1$.

Reducing the grid is the next expensive piece, and it pits cost against the no-bottleneck principle. Going from a $d \times d$ grid with $k$ filters to $d/2 \times d/2$ with $2k$ filters, the textbook order — stride-1 convolution to $2k$ filters on the full grid, then pool — costs about $2 d^2 k^2$ and is dominated by that large-grid convolution. Flipping the order, pool first then convolve, costs $2 (d/2)^2 k^2$, a quarter as much, but now the convolution runs after the representation has already collapsed to $(d/2)^2 k$ — a representational bottleneck. One order is too expensive, the other too lossy, so I refuse to choose: run a stride-2 convolution branch and a stride-2 pooling branch in parallel and concatenate their banks. The convolution branch supplies new learned filters so there is no bottleneck, while the pooling branch is cheap; together it beats the expensive option and avoids the cheap one's bottleneck. On the coarsest 8×8 grid I widen the modules instead of stacking: I put the $1\times3$ and $3\times1$ in parallel and concatenate rather than chaining them, because principle two says high-dimensional local processing matters most where the spatial extent is smallest and the ratio of 1×1 processing to spatial aggregation is highest.

I also reconsider the auxiliary classifiers, because the received story does not fit the data. The story is that intermediate heads push useful gradient into the lower layers and combat vanishing gradients; but the training curves with and without the side head are virtually identical through the whole early phase — if it were feeding helpful gradient I would see faster early convergence — and it only edges the main classifier higher near the end, and removing the lower of two heads does nothing to final quality. That is the signature of a regularizer, not a gradient pump. The test: if it regularizes, making the side branch itself better-regularized should help the main classifier — and batch-normalizing the side head buys about 0.4% absolute top-1, which also hands me weak evidence that batch normalization is itself a regularizer. So I keep one batch-normalized auxiliary head and treat it as such.

The last expensive piece is the objective. With softmax probabilities $p(k) = \exp(z_k)/\sum_i \exp(z_i)$ and loss $\ell = -\sum_k q(k)\log p(k)$, the gradient with respect to a logit is clean, $\partial\ell/\partial z_k = p(k) - q(k)$, bounded in $[-1,1]$. Against a hard one-hot target $q(k) = \delta_{k,y}$, minimizing the loss maximizes the log-likelihood of the single correct label, and that maximum is only approached as $z_y$ grows unboundedly larger than the rest — never attained at finite logits. Chasing it overfits (full mass on the training label guarantees nothing about generalization) and blows up the gap between the top logit and the others, leaving a saturated model with little gradient left to adapt — over-confident and stiff. So I smooth the target, mixing in a fixed example-independent distribution $u$ with weight $\varepsilon$:
$$q'(k) = (1-\varepsilon)\,\delta_{k,y} + \varepsilon\, u(k),$$
and on ImageNet I take $u$ uniform, $u(k) = 1/K$, so $q'(k) = (1-\varepsilon)\delta_{k,y} + \varepsilon/K$. Now no logit can be driven to infinity for reward, because every $q'(k)$ has a positive floor $\varepsilon/K$. Expanding the cross-entropy against the mixture,
$$H(q', p) = (1-\varepsilon)\,H(q, p) + \varepsilon\, H(u, p),$$
shows smoothing simply adds $\varepsilon\,H(u,p)$ to the usual loss; since $H(u,p) = D_{\mathrm{KL}}(u\,\|\,p) + H(u)$ and $H(u)$ is constant, that term is, up to a constant, the KL divergence from the prior to the prediction, pulling predictions gently toward $u$. With $K = 1000$ and $\varepsilon = 0.1$ this gives a consistent ~0.2% absolute improvement on top-1 and top-5.

Assembled at $299\times299\times3$ input, the stem factorizes the front 7×7 into three stacked 3×3s — a 3×3 stride-2 conv to $149^2\times32$, a 3×3 to $147^2\times32$, a padded 3×3 to $147^2\times64$, a stride-2 maxpool, a 1×1 to 80, a 3×3 to 192, and a maxpool, landing at $35\times35\times192$. The body is three blocks separated by grid reductions: three InceptionA modules on the 35×35 grid (a 1×1 branch, a cheap reduced 5×5 branch, a separate double-3×3 branch, and a pooled projection) to $35\times35\times288$; a GridReductionB to $17\times17\times768$; five InceptionC modules using the $1\times7$/$7\times1$ factorization ($c_7 = 128,160,160,192$) with the batch-normalized auxiliary head on the last stage; a GridReductionD to $8\times8\times1280$; and two expanded InceptionE modules with parallel $1\times3$/$3\times1$ branches to $8\times8\times2048$. Then an average pool to 2048, dropout with keep probability 0.8, a linear layer, and softmax over 1000 classes — 42 layers at only about 2.5× the batch-normalized predecessor's compute. I batch-normalize (eps 0.001) after every convolution, which is what lets these factorized stacks train at all; train with RMSProp (decay 0.9, $\varepsilon = 1.0$) since per-parameter adaptive scaling suits the heterogeneous modules, at learning rate 0.045 decayed by 0.94 every two epochs; clip gradients at 2.0 against the occasional large gradient from the deep stack; and evaluate on an exponential moving average of the parameters to smooth the noise of the small per-replica batch. The total loss is label-smoothed cross-entropy on the main head plus $0.4\times$ the same on the auxiliary head, trained across about fifty GPU replicas at batch 32 each for roughly 100 epochs. The whole thing is one idea applied with discipline: spend compute where the principles say it pays.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):                   # 35x35: reduced 5x5 + double 3x3
    def __init__(self, in_ch, pool_features):
        super().__init__()
        self.b1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b5_1 = ConvBN(in_ch, 48, kernel_size=1)
        self.b5_2 = ConvBN(48, 64, kernel_size=5, padding=2)
        self.b3_1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b3_2 = ConvBN(64, 96, kernel_size=3, padding=1)
        self.b3_3 = ConvBN(96, 96, kernel_size=3, padding=1)
        self.bp = ConvBN(in_ch, pool_features, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b5 = self.b5_2(self.b5_1(x))
        b3 = self.b3_3(self.b3_2(self.b3_1(x)))
        bp = self.bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([b1, b5, b3, bp], 1)


class GridReductionB(nn.Module):               # 35x35 -> 17x17
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
        return torch.cat([b3, b3d, bp], 1)


class InceptionC(nn.Module):                   # 17x17: 1x7 + 7x1
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


class InceptionE(nn.Module):                   # 8x8: expanded parallel 1x3 / 3x1
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
        b3 = self.b3_1(x); b3 = torch.cat([self.b3_2a(b3), self.b3_2b(b3)], 1)
        b3d = self.b3d_2(self.b3d_1(x)); b3d = torch.cat([self.b3d_3a(b3d), self.b3d_3b(b3d)], 1)
        bp = self.bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([b1, b3, b3d, bp], 1)


class AuxHead(nn.Module):
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
        self.Conv2d_1a = ConvBN(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a = ConvBN(32, 32, kernel_size=3)
        self.Conv2d_2b = ConvBN(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b = ConvBN(64, 80, kernel_size=1)
        self.Conv2d_4a = ConvBN(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = GridReductionB(288)
        self.Mixed_6b = InceptionC(768, c7=128)
        self.Mixed_6c = InceptionC(768, c7=160)
        self.Mixed_6d = InceptionC(768, c7=160)
        self.Mixed_6e = InceptionC(768, c7=192)
        self.aux = AuxHead(768, num_classes) if aux_logits else None
        self.Mixed_7a = GridReductionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)       # keep probability 0.8
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
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    smooth = -logp.mean(dim=1)
    return ((1 - eps) * nll + eps * smooth).mean()


model = InceptionV3(num_classes=1000, aux_logits=True)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045, alpha=0.9, eps=1.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
# total loss = label_smoothed_ce(main) + 0.4 * label_smoothed_ce(aux); clip grads at 2.0; EMA eval.
```
