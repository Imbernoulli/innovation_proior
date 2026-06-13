# Context

The aim is to train *substantially deeper* image-recognition networks than the current plain convolutional stacks allow, using the prevailing wisdom, load-bearing prior work, datasets, protocols, and framework code already available in late 2015.

## Research question

Depth is the lever that has driven every recent jump on ImageNet-scale recognition. A deep convolutional stack integrates low → mid → high-level features and the classifier end-to-end, and the "level" of features that can be expressed grows with the number of stacked layers. Empirically, each time the leading networks got deeper — eight layers, then sixteen to nineteen, then twenty-two — accuracy climbed. So the blunt question is: **is learning a better network as easy as stacking more layers?**

It is not, and the way it fails is the puzzle. The classical obstacle to depth was vanishing/exploding gradients *at the start* of training; once that was tamed (normalized initialization plus intermediate normalization), deep networks could begin to converge — and a new, counterintuitive failure surfaced. Beyond roughly 20–30 layers, adding layers makes a plain stacked network **worse on the training set**: training error saturates and then *rises* with depth. This is not overfitting — overfitting drives training error down while test error climbs; here both climb together. So it is not a generalization failure but an **optimization** failure, and it has no accepted explanation.

What a solution must achieve, precisely:
- Let a network keep gaining accuracy from greatly increased depth — well past 100 layers — instead of degrading.
- Preserve a clean controlled comparison: the main mechanism should add **no extra parameters, no meaningful extra compute, and no change to the training recipe**, so that any accuracy difference cannot be attributed to added capacity rather than the mechanism itself.
- Drop into existing frameworks and solvers with **no new layer type and no solver surgery**.

The sharp sub-question hiding inside the failure: a deeper network can always, *by construction*, reproduce a shallower one — copy the shallow weights into the bottom layers and set every added layer to the **identity** — so a solution at least as good provably exists inside the deeper network's parameter space. Why can the solver not find it (or anything as good) in feasible time? The situation is not a contradiction; it is *surprising* — a provably-existing solution the optimizer cannot reach — and that precise surprise is the clue.

## Background

By late 2015 the dominant lesson of large-scale vision is **depth wins**. The AlexNet breakthrough (Krizhevsky, Sutskever & Hinton, 2012) was pushed further every year by going deeper, and the leading ILSVRC entries all lean on "very deep" stacks. Two things are settled enough to build on:

1. **Vanishing/exploding gradients at initialization are largely addressed.** The classic difficulty (Bengio et al. 1994; Glorot & Bengio 2010) — signals and gradients shrink or blow up multiplicatively through many nonlinear layers, so early layers receive no usable signal — has been countered by (a) variance-preserving weight initialization (Glorot/Xavier; the orthogonal-init analysis of Saxe et al. 2013; and the ReLU-aware initialization of He et al. 2015 that gets the factor of 2 right), and (b) intermediate normalization, above all Batch Normalization. Together these let networks of a few tens of layers *start* converging under plain SGD with backprop.

2. **The standard recipe is mature.** ReLU nonlinearities (Nair & Hinton 2010), heavy data and color augmentation, SGD with momentum and step learning-rate decay, weight decay, and a global-average-pooling head instead of a giant fully-connected one (Network-in-Network, Lin et al. 2013) are all common practice.

The fresh pain point — exposed *because* (1) worked — is the degradation described above. It has been reported (He & Sun, "Convolutional neural networks at constrained time cost," CVPR 2015; and the Highway Networks work) and is, so far, nameless and unexplained. The prevailing patches are indirect: auxiliary classifiers bolted onto intermediate layers to inject gradient (GoogLeNet; Deeply-Supervised Nets, Lee et al. 2014), and gated shortcuts (Highway Networks). None has shown clean accuracy *gains* beyond ~100 layers.

A background observation matters here: the **identity-by-construction argument** above reframes degradation as a conditioning problem, not a representation problem — the capacity is there, the optimizer just cannot navigate to it. Several established methods from outside neural nets are part of the available toolkit. In image retrieval, VLAD (Jégou et al. 2012) encodes descriptors by their *residual* to dictionary centers, and the Fisher Vector (Perronnin & Dance 2007) is a probabilistic version; encoding residual vectors beats encoding raw vectors in vector quantization (Jégou et al. 2011). In numerical PDE solving, the **Multigrid** method (Briggs et al. 2000) and **hierarchical-basis preconditioning** (Szeliski 1990, 2006) solve for the difference between a coarse and a fine scale and converge far faster than solvers that ignore that structure.

## Baselines

**VGG — Very Deep Convolutional Networks (Simonyan & Zisserman, ICLR 2015).** Build depth from one uniform primitive: stacks of **3×3 convolutions** (two stacked 3×3 give a 5×5 receptive field with fewer parameters and an extra nonlinearity; three give 7×7). Two design rules govern width: (i) same output map size → same number of filters; (ii) halve the map → **double the filters**, keeping per-layer time complexity roughly constant. It ends with large fully-connected layers and softmax. VGG-19 has 16 conv + 3 FC weight layers and costs ~19.6 billion FLOPs, with most parameters in the FC head. *Gap:* VGG is the proof that uniform 3×3 stacking scales, but it stops at ~19 layers, is heavy (FC head, high FLOPs), and offers no mechanism for going much deeper — a naively deeper VGG hits the degradation wall. The reusable seed is the 3×3-and-doubling philosophy, which a deeper method can keep while dropping the FC head for global average pooling and slimming each layer.

**GoogLeNet / Inception (Szegedy et al., CVPR 2015).** Instead of a uniform stack, use **Inception modules**: parallel 1×1, 3×3, 5×5 conv and pooling branches whose outputs concatenate, with 1×1 convs as cheap dimension-reduction "bottlenecks" so a 22-layer net stays affordable. To fight vanishing gradients during training it attaches **auxiliary classifiers** at intermediate depths. *Gap:* the modules are hand-engineered and heterogeneous, and the auxiliary classifiers are a band-aid for the optimization difficulty of depth rather than a fix for it. One established trick here is the **1×1 reduce → 3×3 → 1×1 restore** bottleneck shape: the expensive spatial convolution runs at a *reduced* channel count, with cheap 1×1 convs squeezing in and expanding back out.

**Batch Normalization (Ioffe & Szegedy, ICML 2015).** During training, normalize each layer's pre-activation over the current mini-batch to zero mean / unit variance, then apply a learnable scale γ and shift β:
`ŷ = γ · (x − μ_B)/√(σ²_B + ε) + β`.
This stabilizes the layer-to-layer activation distribution, permits much larger learning rates, lets ~30-layer nets train in far fewer steps, and acts as a mild regularizer (often replacing dropout). *Role (not a competitor):* BN is the tool that makes the puzzle **sharp**. With BN after every conv, forward activations have healthy non-zero variance and backward gradients have healthy norms; so when a deep plain net *still* trains worse, the failure **cannot** be blamed on vanishing or exploding signals. BN clears away the easy explanation and leaves "optimization difficulty, not signal magnitude" as what remains.

**He initialization / PReLU (He et al., ICCV 2015).** A variance-preserving initialization derived for ReLU networks (weights with variance ~2/fan), keeping signal variance stable through ReLU stacks where Xavier under-scales by a factor reflecting ReLU's halved variance. *Role:* the initialization for every plain/deep net considered, part of why such nets converge from scratch.

**Highway Networks (Srivastava, Greff & Schmidhuber, 2015) — concurrent.** Borrow LSTM-style gating to let information skip layers:
`y = H(x, W_H) · T(x, W_T) + x · C(x, W_C)`,
with a data-dependent **transform gate** T and **carry gate** C (often C = 1 − T), each carrying its own learned parameters; very deep nets train by sometimes "carrying" inputs unchanged. *Gap (the direct contrast):* the gates have parameters and are data-dependent, so when a gate closes (T → 1, carry → 0) the layer reverts to an ordinary non-residual transform — the skip path is **not guaranteed open**, and can shut off exactly where the depth most needs it. And Highway has not demonstrated accuracy *gains* from extremely increased depth (e.g. beyond 100 layers).

**Shortcut connections — the longer lineage.** Connecting non-adjacent layers is old: early MLP practice added a linear skip from input to output (Bishop 1995; Ripley 1996; Venables & Ripley 1999); methods for centering responses/gradients used shortcuts (Schraudolph 1998; Raiko et al. 2012; Vatanen et al. 2013); and auxiliary-classifier nets wire intermediate layers to losses. *Gap:* these treat the shortcut as a gradient-injection or signal-centering trick, and none has been shown to address the degradation problem.

**Smaller pieces in use.** ReLU (Nair & Hinton 2010) as the in-block nonlinearity; global average pooling and 1×1 convs (Network-in-Network, Lin et al. 2013); the AlexNet augmentation / 10-crop testing recipe (Krizhevsky et al. 2012); dropout (Hinton et al. 2012), deliberately omittable once BN regularizes.

## Evaluation settings

These datasets, metrics, and protocols are the natural yardsticks.

- **ImageNet (ILSVRC 2012 classification), 1000 classes.** ~1.28M training images, 50k validation, 100k test (test scored by a held-out server). Metrics: **top-1 and top-5 error rate**. The training-set *error* itself is a first-class diagnostic here, because the degradation phenomenon is precisely a training-error effect — the decisive plots are training-error-vs-iteration curves, not just final accuracy.
- **CIFAR-10.** 50k training / 10k test images, 10 classes, 32×32. Used to probe the *optimization behavior* of extremely deep networks cheaply (including aggressively deep regimes) rather than to chase state-of-the-art; classification error on the test set is the metric.
- **Training protocol (the mature recipe to hold fixed).** SGD with mini-batch 256, momentum 0.9, weight decay 1e-4; learning rate starting at 0.1 and divided by 10 when error plateaus; BN after each conv and before activation; He init; train from scratch; no dropout. ImageNet augmentation: shorter side randomly sampled in [256, 480] for scale jitter, 224×224 random crop or horizontal flip, per-pixel mean subtraction, standard color augmentation. CIFAR augmentation: 4-pixel padding then 32×32 random crop / flip.
- **Testing protocol.** Standard 10-crop testing for comparison studies; fully-convolutional multi-scale testing (shorter side in {224, 256, 384, 480, 640}) for best results, both inherited from AlexNet/VGG.
- **Downstream yardsticks (already established).** PASCAL VOC 2007/2012 detection (mAP@.5) and MS COCO detection (mAP@.5 and mAP@[.5,.95]) under the Faster R-CNN framework — the natural way to test whether better classification features transfer.
- **Controlled-comparison principle.** The decisive experiment is a *plain network vs. its modified twin at identical depth, width, parameter count, and FLOPs*; any dimension-matching projection should be reported as a separate, small exception rather than hidden inside the main claim.

## Code framework

The available code is a bare deep-CNN image-classification harness. The libraries supply convolution, batch normalization, ReLU, an SGD-with-momentum optimizer, a step learning-rate scheduler, a cross-entropy loss, and the standard data-augmentation transforms. The scaffold is the harness with one empty architecture slot: the block builder and the network class are stubs.

```python
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    # the VGG primitive; bias=False because a BN right after carries the shift
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    # cheap channel-mixing / dimension-changing conv (no spatial extent)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv_bn_relu_block(inplanes, planes, stride=1):
    # TODO: the layer block we'll design.
    # A block is some stack of conv -> BN -> ReLU that a stage will repeat.
    # How many convs it contains, and how they connect, remains open.
    raise NotImplementedError


class Net(nn.Module):
    # TODO: the architecture we'll design.
    # Fixed by the VGG-style philosophy we're inheriting: a cheap stem that
    # downsamples fast, then four stages that stack the block above, halving the
    # spatial map and doubling the channels at each stage transition, then a
    # global-average-pool head into a single classification FC. What the block is
    # and how the stages wire it together is the open slot.
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        # stem: 7x7/64 stride 2, then 3x3 max-pool stride 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # four stages; halve spatial / double channels at each transition (VGG rule)
        self.stage1 = self._make_stage(block, 64, layers[0])
        self.stage2 = self._make_stage(block, 128, layers[1], stride=2)
        self.stage3 = self._make_stage(block, 256, layers[2], stride=2)
        self.stage4 = self._make_stage(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # global average pool (NIN head)
        self.fc = nn.Linear(512, num_classes)         # single classification FC

        # He init for convs (ReLU-aware, variance-preserving); BN starts as identity
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, block, planes, blocks, stride=1):
        # TODO: stack `blocks` copies of the block at this stage's width, with
        # the first one handling the stride-2 downsample / channel change.
        raise NotImplementedError

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# --- training harness ---
model = Net(block=conv_bn_relu_block, layers=[2, 2, 2, 2])     # depth = block counts per stage
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=..., gamma=0.1)

# standard VGG/AlexNet augmentation pipeline (scale jitter, crop, flip, mean-subtract)
# train_transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(),
#                            ColorJitter(...), ToTensor(), Normalize(mean, std)])

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / total
```
