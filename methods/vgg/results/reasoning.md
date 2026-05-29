OK, let me think this through from scratch. ConvNets have just won large-scale image classification — the 2012 network buried the hand-crafted-feature pipelines on ImageNet, and the 2013 winners pushed the error down further. Everyone is now in the business of tweaking that 2012 recipe. And when I look at what the tweaks actually were, two stand out. The 2013 winners shrank the first convolutional layer — from an 11×11 filter taken at stride 4 down to a 7×7 filter at stride 2 — and they processed the image densely and at multiple scales. Both helped. But both change the early *spatial sampling* and the *evaluation*, not the thing I keep coming back to when I stare at these networks: they're all about the same depth. Eight weight layers, give or take. Nobody has actually asked the clean question — if I hold the whole recipe fixed and just keep stacking more layers, does accuracy keep climbing? Is depth, by itself, the lever?

So that's the thing I want to isolate: depth as a single controlled variable. Build a family of nets that differ *only* in how many layers they have, everything else identical, and watch the error as the family gets deeper.

The trouble is, the moment I try to do that with the existing layer design, it falls apart. Take the 2012 net's first layer: an 11×11×3 filter at stride 4. The stride-4 part already bothers me — the very first operation throws away three out of every four pixels in each direction. The deconvolutional visualizations from the 2013 work made this concrete: finer first-layer sampling recovers cleaner low-level structure, which is exactly why dropping to 7×7 stride 2 helped. So I don't want big strides early; I want stride 1 so nothing is discarded up front. But then the receptive fields of those big filters overlap massively and the spatial map barely shrinks, so I have to pool aggressively to keep compute sane, and the big filters themselves are enormous: an 11×11 filter over C input channels and C output channels is 121·C² weights per layer. If I want to make the net *deep* by stacking layers like that, the parameter count detonates and the resolution collapses within a couple of pools. Big-filter layers are simply not a substrate you can stack to depth 16 or 19. The depth question and the filter-design question are tangled together, and I can't answer the first without first fixing the second.

So let me attack the filter. What is the *smallest* convolution filter that still means anything? A 1×1 filter sees a single location — it can't tell left from right or up from down, it only mixes channels at a point. The smallest filter that captures the notion of left/right, up/down, and center is 3×3. That's the atom. Let me see how far I can get if I refuse to use anything bigger than 3×3 anywhere in the net.

The obvious worry: a 3×3 filter sees almost nothing — a tiny 3×3 patch. Big filters were used precisely to see more context at once. Don't I lose all the reach? Let me actually trace the reach through a stack and not hand-wave it. Take stride-1 convolutions and, crucially, put no pooling between them. A unit in the first 3×3 layer sees a 3×3 patch of the input. Now a unit in the second 3×3 layer sees a 3×3 neighborhood of *first-layer units*. The two outermost of those three first-layer units are centered one step apart on each side, and each of them reaches one further pixel out, so along one axis the footprint runs from the leftmost pixel seen by the left unit to the rightmost pixel seen by the right unit: 3 + (3−1) = 5. Two stacked 3×3 layers see a 5×5 patch of the input. Stack a third: 5 + (3−1) = 7. So three stacked 3×3 layers have the reach of a single 7×7 filter. The pattern is clean — each extra 3×3 stride-1 layer adds 2 to the side, so k of them give (2k+1)×(2k+1). Three give 7×7, five give 11×11. I can match *any* big filter's receptive field just by stacking enough 3×3 layers. The reach isn't lost at all; it's assembled.

So now the real question sharpens: I have a choice between a single 7×7 layer and a stack of three 3×3 layers that see the same 7×7 of input. They have the same receptive field. What, concretely, do I gain or lose by choosing the stack?

Let me count parameters, because that's the thing I was scared would blow up. Assume the same number of channels C going in and out at each layer (and ignore biases). A single 7×7 conv layer mapping C channels to C channels has 7²·C² = 49·C² weights. The stack of three 3×3 layers, each C→C, has 3·(3²·C²) = 3·9·C² = 27·C² weights. So the stack is *cheaper*: 27 versus 49. The single big filter needs 49/27 ≈ 1.81 times as many parameters — about 81% more — for the *same* receptive field. That's the opposite of what I feared; small filters stacked deep use fewer parameters than one big filter, not more. Same arithmetic for the 5×5 case: one 5×5 is 25·C², two stacked 3×3 are 2·9·C² = 18·C², so the 5×5 carries 25/18 ≈ 1.39 times as many weights, ~39% more. Stacking small filters is a parameter saving, and the saving grows with filter size.

There's a second thing I get, and it's the one that actually excites me. The single 7×7 layer applies exactly one nonlinearity — one ReLU — to that 7×7 receptive field. The stack of three 3×3 layers applies *three* ReLUs over the same field. So for the same reach I've turned one linear-then-rectify into a composition of three rectified transforms. The decision function over a 7×7 patch can be far more nonlinear, more discriminative, than a single rectified linear map of it. Said another way: a 7×7 filter is forced to be a single linear map of its patch; the three-3×3 stack is a *factored* 7×7 — it can only express 7×7 transforms that decompose into three 3×3 stages — but with a nonlinearity injected between the stages. The factorization is a regularizer on the big filter (fewer, structured parameters) and the injected nonlinearities buy expressiveness. I get more discriminative power *and* fewer parameters at once. That's not a tradeoff; it's strictly the better deal for building depth.

And it dissolves the original tangle. The reason I couldn't make the old design deep was that depth meant big-filter layers, which meant exploding parameters and collapsing resolution. With 3×3 stride-1 (and padding 1 so the spatial size is preserved across the conv, only pooling ever downsamples), each added layer is cheap and harmless to resolution. Depth is now a free knob. I can finally build the controlled family.

Let me lay out the generic skeleton, then grow it. Input is a fixed 224×224 RGB crop; the only preprocessing is subtracting the training-set mean RGB per pixel. The body is stacks of 3×3 conv layers (stride 1, pad 1, ReLU after each), separated by max-pooling. I'll use 2×2 pooling at stride 2 — the minimal pool that halves resolution — and five of them, so the spatial size goes 224 → 112 → 56 → 28 → 14 → 7, a factor of 2⁵ = 32 down to a 7×7 map. Channel width starts small at 64 and doubles after each pool — 64 → 128 → 256 → 512 — capping at 512 so it doesn't run away; the doubling roughly compensates for the halving resolution, keeping per-layer compute balanced. Then on top, three fully-connected layers: 4096, 4096, and 1000 (one per class), then softmax. ReLU on all hidden layers. The head is identical across the whole family; only the conv body's depth changes.

Now the family, varying depth as the single knob. Start with the shallowest net I can train cleanly — call it A: one conv3 per block in the early blocks, two in the later blocks, 8 conv layers plus the 3 FC = 11 weight layers. Then thicken: B adds a second conv3 to each of the first two blocks → 13 weight layers. Then push deeper in the later blocks: D adds a third conv3 to the last three blocks → 16 weight layers. Then E adds a fourth conv3 to each of those three blocks → 19 weight layers. Each step is *only* more 3×3 layers; everything else is frozen. That's the depth axis, clean.

Before I trust the parameter story for the whole net, let me sanity-check that depth really doesn't blow up the count, because that was the whole motivation. Where do the weights actually live? Not in the conv stacks — those are cheap, as I just showed. They live in the first fully-connected layer, which flattens the 7×7×512 final map into a 4096 vector: 512·7·7·4096 = 102,760,448 weights, about 102.8M, the bulk of the model. Against that, going from 11 to 19 weight layers barely moves the total: the family runs about 133M, 133M, 134M, 138M, 144M parameters from shallowest to deepest. Depth grows by 8 layers and the parameter count grows by ~8%, almost all of it irrelevant to depth. And to put it in perspective, even the deepest at 144M is no larger than a shallow contemporary net with big first-layer filters and wide convs (that one was ~144M too). So I've decoupled depth from parameter count, which is exactly what I needed to study depth fairly.

While I'm fixing layer design, there's an orthogonal knob worth a column in the family. A convolution at a location is a linear map of its patch, and only linearly separable patches get clean treatment — the Network-in-Network line made this point and answered it with a tiny per-location MLP, which is just stacked 1×1 convolutions with a nonlinearity between them. A 1×1 conv keeps the receptive field at a single location: it's a linear projection across channels, and with the same number of input and output channels it's even dimensionality-preserving, so on its own it adds nothing geometric. But follow it with a ReLU and it adds a nonlinear stage to the decision function *without touching the receptive field*. So I can probe a clean sub-question: how much is the gain from added depth really about added *nonlinearity* versus added *spatial context*? Make a net C identical to D in depth (16 weight layers) but where the three layers I added in the late blocks are 1×1 instead of 3×3. C and D have the same number of layers, the same number of nonlinearities; they differ only in whether those extra layers see a 3×3 spatial neighborhood or just a point. If C beats the shallower B, added nonlinearity helps on its own. If D beats C, spatial context matters beyond nonlinearity. That comparison is built into the family by construction — I'll let the experiments speak, but the architecture is designed so the question is answerable.

One more thing the older recipe carried that I should reconsider rather than copy: Local Response Normalization, the lateral inhibition between channels from the 2012 net. I'll keep one variant of the shallow net with LRN (A-LRN) precisely so I can check whether it earns its keep. My expectation is it won't move accuracy on this dataset while costing memory and compute, in which case I drop it from all the deeper nets and keep them clean. No reason to carry machinery that doesn't pay.

Now, can I even train a 19-layer net? This is the wall I have to expect. Deep nets are notoriously sensitive to initialization — start the weights badly and the gradient is unstable, the signal either vanishes or explodes through that many layers, and learning just stalls. If I initialize a 19-layer net from scratch with naive Gaussian weights, there's a real chance it never gets off the ground. So I need a way in. The family itself gives me one: net A is shallow enough to train from random initialization — weights drawn from a zero-mean Gaussian with small variance (10⁻², biases zero). Once A is trained, I can use it to seed the deeper nets: initialize the first four conv layers and the last three FC layers of a deeper net with A's learned weights, leave the middle layers random. The well-conditioned learned layers at the two ends keep the gradients sane while the random middle catches up, and I don't freeze the borrowed layers — I let them keep learning. In standalone code, when I am not explicitly loading that warm-start, I still want the same stability principle in initializer form: choose the variance from the layer dimensions so activations and gradients do not systematically shrink or grow as they pass through the stack.

For the optimization itself I follow the standard deep-ConvNet recipe, since I'm deliberately changing only the architecture. Minimize multinomial logistic regression — softmax over the 1000 classes with cross-entropy — by mini-batch SGD with momentum. Batch 256, momentum 0.9. The fully-connected layers hold over 100M parameters and will overfit, so regularize: L2 weight decay with multiplier 5×10⁻⁴, and dropout at 0.5 on the first two FC layers. Learning rate starts at 10⁻² and I divide it by 10 whenever validation accuracy stops improving; it ends up dropping three times over a run of about 370K iterations (~74 epochs). This keeps the optimization recipe conventional while the architecture is the only real experimental variable.

Now feeding the net. Input is a 224×224 crop, but a training image is bigger and the object inside can be any size. Let S be the smallest side of the isotropically rescaled training image — the training scale — and crop the 224×224 from that (one crop per image per SGD step, plus random horizontal flip and a random RGB color shift for augmentation). If S = 224 the crop spans the whole smallest side and captures whole-image statistics; if S ≫ 224 the crop is a small window onto the image and sees an object part up close. So S sets the apparent scale of objects the net trains on.

Which S? I could fix it — train at S = 256 (the common choice) or at S = 384. But objects in the real world appear across a wide range of sizes, and a net trained at a single scale only ever sees one. So instead of committing to one scale, sample it: for each training image draw S uniformly from a range — I'll use [256, 512]. Now a single net is trained to recognize objects across a span of scales; it's training-set augmentation by scale jittering, free robustness to size. To get there cheaply I don't train the jittered net from scratch — I fine-tune all layers of a single-scale net (pre-trained at fixed S = 384, which I in turn warm-started from the S = 256 net with a smaller learning rate of 10⁻³). So the scales bootstrap each other.

Test time. The image at test has a smallest side Q — the test scale, not necessarily equal to S. The old way to classify is to crop many fixed windows out of the image, run each through the net, and average — but that re-runs the whole net per crop, which is wasteful, and the number of crops bounds how densely you sample the image. There's a cleaner idea I can borrow: the only thing in my net that demands a fixed 224×224 input is the fully-connected head, because a linear layer has a fixed-size weight matrix. But a fully-connected layer over a fixed spatial extent *is* a convolution. The first FC consumes the 7×7×512 map, so it's exactly a 7×7 convolution; the next two FC layers consume a 1×1 spatial input, so they're 1×1 convolutions. Rewrite the three FC layers as those conv layers and the whole net becomes fully convolutional, with no fixed-size constraint anywhere. Apply it to the *whole* uncropped image at side Q and the output is no longer a single 1000-vector but a spatial *map* of class scores — one 1000-vector per sliding position, computed in a single shared-computation pass instead of one forward pass per crop. To collapse that map back to one prediction for the image, spatially average (sum-pool) the score map over its grid; average that with the result on the horizontally flipped image. One efficient pass, dense over the image, no redundant recomputation.

The dense and the multi-crop views aren't even rivals — they treat the image boundary differently. A cropped window gets zero-padded at its edges when convolved; the dense whole-image pass, by contrast, fills those same border positions with real neighboring image content (the padding for an interior window comes from its actual surroundings, via both the convolutions and the pooling), so dense evaluation effectively captures more context at the borders. Different boundary conditions mean the two can make different errors, so averaging dense with a multi-crop pass (say a 5×5 grid of crops with 2 flips per scale) is a reasonable test-time ensemble if I can afford the extra compute.

And since I trained with scale jittering, the net is comfortable across a range of Q, so I shouldn't test at just one. Run several test scales and average the class posteriors. For a net trained at fixed S, a big train/test scale gap hurts, so test near it: Q ∈ {S−32, S, S+32}. For the jittered net trained over [S_min, S_max], I can spread wider: Q ∈ {S_min, 0.5(S_min+S_max), S_max}. Averaging over scales aggregates more independent decisions and is exactly what the scale-jittered training prepared the net to exploit.

So the whole thing comes together as one idea pursued stubbornly: refuse anything bigger than a 3×3 filter, discover that stacking them assembles any receptive field while *saving* parameters and *adding* nonlinearity, realize that this finally makes depth a clean and cheap knob, and then just turn that knob — 11, 13, 16, 19 weight layers — on an otherwise frozen, conventional ConvNet, warm-starting the deep ones from the shallow one to get them training, jittering scale at train and test time, and evaluating the net fully-convolutionally over the whole image. Everything is in service of asking, fairly, whether depth is the thing that drives accuracy.

Let me write the reusable code around the depth family. The body is built from a compact description of each configuration — a list where an integer is "a 3×3 conv layer with that many output channels" and 'M' is a max-pool — so the A/B/D/E family is one builder with depth as data. The 1×1 branch is a diagnostic variation; the standard implementation stays with the clean 3×3 family.

```python
from typing import Union, cast

import torch
import torch.nn as nn


# An int N means "3x3 conv to N channels, stride 1, pad 1, then ReLU";
# "M" means 2x2 max-pool at stride 2. Depth is data in the config.
cfgs: dict[str, list[Union[str, int]]] = {
    # A: 11 weight layers (8 conv + 3 FC)
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # B: 13 weight layers (a second 3x3 added to the first two blocks)
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # D: 16 weight layers (the extra late-block layers are 3x3: nonlinearity AND spatial context)
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
          512, 512, 512, "M", 512, 512, 512, "M"],
    # E: 19 weight layers (a fourth 3x3 in each of the last three blocks)
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
          512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(cfg: list[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    # Build the conv body: stride-1, pad-1 3x3 convs preserve resolution across each conv;
    # only the 2x2/stride-2 max-pools downsample. A ReLU follows every conv.
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            out_channels = cast(int, v)
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Five 2x2 pools take 224 -> 7, so the conv body ends at 7x7x512. The head is identical for
        # every depth; most of the model's parameters live in the first FC (512*7*7 -> 4096).
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),        nn.ReLU(True), nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Fan-scaled conv initialization keeps signal magnitudes stable through the stack;
        # FC weights use the small Gaussian convention, with zero biases.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)              # 224x224x3 -> 7x7x512
        x = self.avgpool(x)               # keep the classifier shape fixed
        x = torch.flatten(x, 1)
        return self.classifier(x)        # -> 1000 class scores


def build_vgg(cfg: str = "D", batch_norm: bool = False, **kwargs) -> VGG:
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)


# Training: minimize softmax cross-entropy with SGD + momentum, weight decay, dropout on the FC head.
def make_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

loss_fn = nn.CrossEntropyLoss()  # multinomial logistic regression objective

def train_step(model, images, labels, optimizer):
    optimizer.zero_grad()
    loss = loss_fn(model(images), labels)   # images are random 224 crops from images rescaled to S,
    loss.backward()                         # with S jittered in [256, 512], plus flips + color shift
    optimizer.step()
    return loss.item()


# Dense evaluation: re-express the FC head as convolutions so the net is fully convolutional and can
# run over a whole image at test scale Q (the 7x7-input FC -> a 7x7 conv; the two 1x1-input FCs -> 1x1
# convs). The output is a spatial class-score map; average it (and the flipped image, and several Q) to
# get one 1000-vector per image.
def to_fully_convolutional(model, num_classes=1000):
    fc1, fc2, fc3 = [m for m in model.classifier if isinstance(m, nn.Linear)]
    kwargs = {"device": fc1.weight.device, "dtype": fc1.weight.dtype}
    conv1 = nn.Conv2d(512, 4096, kernel_size=7, **kwargs)          # first FC consumes 7x7 -> 7x7 conv
    conv2 = nn.Conv2d(4096, 4096, kernel_size=1, **kwargs)         # next two FCs consume 1x1 -> 1x1 convs
    conv3 = nn.Conv2d(4096, num_classes, kernel_size=1, **kwargs)
    with torch.no_grad():
        conv1.weight.copy_(fc1.weight.view(4096, 512, 7, 7))
        conv1.bias.copy_(fc1.bias)
        conv2.weight.copy_(fc2.weight.view(4096, 4096, 1, 1))
        conv2.bias.copy_(fc2.bias)
        conv3.weight.copy_(fc3.weight.view(num_classes, 4096, 1, 1))
        conv3.bias.copy_(fc3.bias)
    return nn.Sequential(model.features, conv1, nn.ReLU(inplace=True),
                         conv2, nn.ReLU(inplace=True), conv3)


def predict_dense(model, image, num_classes=1000):
    net = to_fully_convolutional(model, num_classes)
    net.eval()
    with torch.no_grad():
        score_map = net(image)                           # N x 1000 x H' x W' over the whole image
    return score_map.mean(dim=(2, 3))                    # sum/avg-pool the map -> one 1000-vector
```

I start with the question of whether depth alone drives accuracy, but the prevailing big-filter, big-stride layers can't be stacked deep without exploding parameters and destroying resolution, so I drop to the smallest meaningful filter, 3×3; tracing receptive fields shows a stack of them assembles any larger filter's reach, and counting weights shows the stack does it with fewer parameters (27C² vs 49C² for a 7×7's worth) while inserting extra ReLUs for more nonlinearity; that turns depth into a cheap, clean knob, so I build a family that holds everything fixed and only adds 3×3 layers (11→13→16→19), warm-start the deep nets from the shallow one or use fan-scaled initialization to get them to train, add 1×1 layers in one variant to separate nonlinearity from spatial context, jitter the input scale at train and test time for size robustness, and evaluate fully-convolutionally over the whole image so a single pass yields a dense, averageable score map.
