OK, let me think this through from scratch. The thing everyone wants right now is depth. LeNet had five layers, VGG nineteen, and only in the last year has anything crossed a hundred — Highway nets, residual nets. And depth genuinely buys you something: more layers, more abstraction, better features. So I want hundreds of layers. But the moment I try to just stack a hundred plain convolutional layers, something goes wrong, and it's worth being precise about what.

The observation that bugs me is concrete. Take a network that works at, say, twenty layers. Now make it fifty by stacking thirty more layers on top. The fifty-layer net does *worse* — and not worse on the test set, worse on the *training* set. That rules out the easy explanation. It isn't overfitting; the model can't even fit the data it's shown. And that's strange, because there's a solution sitting right in front of the optimizer: take the twenty-layer net's weights for the first twenty layers, and make the extra thirty layers compute the identity. That network has exactly the twenty-layer net's training error, by construction. It exists. The optimizer just can't find it. So the problem is not capacity, it's optimization — the signal going forward and the gradient coming back have to thread through thirty extra nonlinear transformations, and along the way they smear out. By the time the gradient from the loss has been multiplied through fifty Jacobians it's a faint, distorted thing at the early layers.

So the real question isn't "how do I add capacity." It's: how do I wire the layers so that information going forward and gradient coming back can travel a *short* distance between any early layer and any late layer, even when the network is nominally very deep?

Let me look at what the field has tried, because there's clearly a pattern. Highway networks were first past a hundred layers. Their move is a gate: a highway layer outputs y = H(x)·T(x) + x·C(x), where H is the usual transformation, T is a learned "transform" gate and C a "carry" gate, and they usually tie C = 1 − T. When the gate T is near zero, the layer just carries x forward unchanged — an open "highway" for the signal. That works, and it's the first proof that the trick is *bypassing* the nonlinearity. But the gate is learned and it costs parameters, and the highway is only as open as the gate decides to make it. There's no guaranteed clean path; the network has to learn to open its own doors.

ResNet does away with the gate entirely. Throw away the learned T and C and just hard-wire the bypass to be the identity. A residual block computes x_ℓ = H_ℓ(x_{ℓ−1}) + x_{ℓ−1}. The reframing is that the layer no longer learns the whole target mapping H; it learns the *residual* F(x) = H(x) − x, the correction on top of "just copy the input." The identity-copy solution that the plain net couldn't find is now the *default* — set F to zero and you copy. And the identity shortcut costs nothing: no parameters, no real compute. The reason this should help the gradient I want to actually write out rather than take on faith, because the gradient flow is the whole reason I care. If the shortcut is a clean identity all the way through (which is what the pre-activation BN–ReLU–Conv ordering buys you, by keeping any nonlinearity off the skip path), then I can unroll the recursion. x_L = x_{L−1} + F(x_{L−1}) = x_{L−2} + F(x_{L−2}) + F(x_{L−1}) = … so for any earlier ℓ,

  x_L = x_ℓ + Σ_{i=ℓ}^{L−1} F(x_i).

Differentiate the loss with respect to the early activation:

  ∂L/∂x_ℓ = ∂L/∂x_L · ∂x_L/∂x_ℓ = ∂L/∂x_L · ( 1 + ∂/∂x_ℓ Σ_{i=ℓ}^{L−1} F(x_i) ).

Look at what that standalone "1" does. The gradient that arrives at the deepest layer is delivered to layer ℓ with a coefficient of 1 on it — undiminished — plus a correction from the residuals. There's no long product of Jacobians multiplying it that could drive it to zero. To convince myself the "1" really is depth-independent: imagine F were exactly zero (every block does nothing). Then x_L = x_ℓ for any ℓ, ∂x_L/∂x_ℓ = 1 exactly, and the gradient reaches the very first layer at full strength no matter whether L − ℓ is 2 or 2000. The plain net in the same situation would have ∂x_L/∂x_ℓ = a product of L − ℓ weight Jacobians, which generically shrinks or explodes geometrically in the depth. That is the concrete difference, and it is why ResNets train past a thousand layers. So identity shortcuts do solve the flow problem.

But two things nag at me about the residual formulation, and they're connected.

First, the combine operation is a *sum*. The carried-forward signal x_{ℓ−1} and the freshly computed features H_ℓ(x_{ℓ−1}) get added into one tensor of the same width. Addition is lossy in a specific way: once you've added two feature-maps element-wise, you can't recover either one — they can partially cancel, overwrite, interfere. The identity term is preserving information and the H term is contributing new information, and they're forced to share the same channels and merge. That can impede information flow exactly where I need it clean: the network is being asked to carry old state forward *and* write new state into the same additive slot.

Second — and this is what really makes me suspicious — stochastic depth. Someone trained a 1202-layer ResNet by *randomly dropping whole residual blocks during training*: with some probability a block is replaced by the identity for that minibatch. And it didn't just survive, it trained *better*. Sit with that. You can delete layers at random and the thing improves. What that screams is **redundancy**: in a deep residual net, lots of layers are doing very little, and each of those layers is nonetheless carrying its own full-width set of weights — a whole convolutional kernel bank that mostly re-derives or re-carries what's already there. The state of a deep ResNet starts to look like an unrolled recurrent net where each step has its own private parameters but is mostly passing information along. That's an enormous parameter waste. And there's a second thing stochastic depth tells me, almost as an aside: when you drop the layers *between* two surviving layers, those two surviving layers become directly connected. So the very thing that helps — short, direct connections between non-adjacent layers — is being manufactured, randomly and temporarily, by the dropout schedule.

Let me also pull in Inception, because it's been sitting in the back of my mind. Inception modules combine features not by summing but by *concatenating* the outputs of different filters, on the grounds that this gives the next stage a more *diverse* set of inputs. And Szegedy's analysis notes that features within a single conv layer are highly correlated — you can squash a layer's channels down to fewer without losing much. So conventional layers are producing redundant feature-maps, and concatenation of differently-derived features is a known way to fight that redundancy.

Now let me put these three observations side by side, because together they point somewhere.

Observation one: short direct connections between early and late layers are what make deep nets trainable (Highway, ResNet, stochastic depth all converge on this).
Observation two: in a deep net, most layers are redundant — they don't need their own full-width parameters, because what they mostly do is preserve information that's already present.
Observation three: summation is a lossy, interfering way to combine "preserved" and "new" information, and concatenation is the known alternative that keeps them distinct.

Take observation one to its limit. If short connections are good, and stochastic depth is *accidentally* creating direct connections between non-adjacent layers, why be coy about it? Make them direct, all of them, permanently. Don't randomly connect layer ℓ to some earlier layer some of the time — connect every layer directly to *every* later layer, all of the time. That's the most short-paths you can possibly have: every pair (early, late) within a region is one hop apart.

But how do I "connect" them — what's the combine? Here observation three bites. If I connect all preceding layers to layer ℓ by *summing* them, I'm back in the additive world: everything gets merged into one width-matched tensor, old and new interfere, and I'd have to keep every layer at the same width to add them. Instead, **concatenate**. Layer ℓ's input is the concatenation of *all* preceding feature-maps:

  x_ℓ = H_ℓ( [x_0, x_1, …, x_{ℓ−1}] ),

where [·] stacks the feature-maps along the channel axis. Now nothing is overwritten. Every feature-map any earlier layer ever produced is still sitting there, intact and individually addressable, available as a direct input to every later layer. This is the cleanest possible answer to "preserve old information AND add new information": preserved information is literally preserved — appended, never summed away — and new information is just more channels appended on top.

And notice this *also* answers observation two, the redundancy, in a way that surprised me when I first saw it fall out. If every layer can read every previous feature-map directly, then no layer needs to *re-produce* or *re-carry* features that already exist somewhere behind it. The whole concatenated stack acts like a shared, growing global state — call it the network's collective knowledge — and each layer's only job is to read that state and *contribute a little new information* to it. It doesn't have to replicate the state forward, because the state is already globally accessible. That's completely different from a plain net or even a ResNet, where the full-width representation is effectively re-learned and re-written at every single block.

If each layer only needs to *contribute* a little, then each layer can be *tiny*. Let me make that quantitative, because it's the crux of whether this is a good idea or a parameter catastrophe — the connection count grows quadratically (L(L+1)/2 connections in an L-layer block instead of L), and if the parameter count grows the same way this is a bad trade. Suppose each H_ℓ produces only k new feature-maps — k small, like 12. Then if a block starts with k_0 channels, layer ℓ sees k_0 + k·(ℓ−1) input channels (everything before it) and emits k. Call k the growth rate: it's exactly how much each layer adds to the global state. Now, a 3×3 conv that maps c_in channels to c_out channels has about 9·c_in·c_out parameters. So layer ℓ costs roughly 9·k·(k_0 + k(ℓ−1)). Sum over ℓ = 1..L:

  total ≈ 9k · Σ_{ℓ=1}^{L} (k_0 + k(ℓ−1)) = 9k · ( L·k_0 + k·L(L−1)/2 ).

I want to actually check that closed form rather than trust the arithmetic. Take k = 12, k_0 = 24, L = 10. Summing 9·12·(24 + 12·ℓ) over ℓ = 0..9 term by term gives 84,240. The closed form 9·12·(10·24 + 12·10·9/2) = 9·12·(240 + 540) = 108·780 = 84,240. They agree, so the L(L−1)/2 telescoping is right and the leading term is k·L(L−1)/2, i.e. the parameter count is O(k²L²).

So "L²" is real, and at first glance it looks alarming. But it is multiplied by k², and k is *small*. Compare a conventional constant-width net of width W: each layer is a 9·W·W conv, and L of them give 9·W²·L = O(W²L). So the comparison is k²L² versus W²L. For k = 12 against W = 256, k²/W² = 144/65,536 = 0.002197, which is 1/455 — so the per-quadratic coefficient is on the order of 1/400 of the plain net's per-linear one. To see this isn't a sleight of hand at a single depth, take that same L = 10 example all the way through: the dense block's 84,240 parameters against a plain 10-layer width-256 net's 9·256²·10 = 5,898,240 is a ratio of 0.0143 — the dense block is about 70× smaller at this depth. The quadratic-in-L term hasn't yet swamped the tiny k² coefficient at the depths I'd actually use, and each layer is spending its parameters on *new* features, never on redundant carried copies. The efficiency *is* the feature reuse: re-learning is what costs parameters, and dense connectivity removes the need to re-learn. (I should stay honest that for L large enough the k²L² term must eventually overtake W²L; the claim is only that at practical block depths the coefficient keeps it well under the plain net, which the numbers above bear out.)

So far so good, but I've been quietly cheating: concatenation along the channel axis only works if all the feature-maps are the *same spatial size*. And a conv net absolutely must downsample — you can't keep everything at 32×32 forever; you need pooling to build spatial abstraction and to keep compute sane. The two requirements collide: dense concatenation wants constant spatial size, downsampling wants to shrink it.

The resolution is to not insist on one dense web over the entire network. Chop the network into a few **dense blocks**, each operating at a single fixed resolution, with full dense connectivity *inside* a block. Between consecutive blocks, put a **transition layer** whose job is exactly the downsampling that the block can't do internally: it takes the block's concatenated output, and reduces it spatially (and adjusts channels). Concretely a transition is a batch-norm, a 1×1 convolution to handle channels cheaply, then a 2×2 average pooling to halve the spatial size. So the picture is: stage at 32×32, transition (halve), stage at 16×16, transition, stage at 8×8, then a global average pool and a softmax head. Dense connectivity lives within each stage; the transitions stitch the stages together at decreasing resolution. For the composite function H itself I'll use BN–ReLU–Conv(3×3) — the pre-activation ordering, for the same clean-path reason it helped ResNet.

Let me check the gradient story actually holds with concatenation, because that was the original motivation and I don't want to have lost it. With x_ℓ = H_ℓ([x_0,…,x_{ℓ−1}]), I want to know what ∂x_ℓ/∂x_0 looks like. Concatenation is a linear copy: the input tensor to layer ℓ is u = [x_0, x_1, …, x_{ℓ−1}], and x_0 occupies a fixed contiguous channel slice of u, with ∂u/∂x_0 just a selection (an identity into those channels, zeros elsewhere). So ∂x_ℓ/∂x_0 = (∂H_ℓ/∂u) restricted to the x_0 slice — one Jacobian factor, the layer's own H, not a product over all the layers between 0 and ℓ. Concretely: take a tiny block, x_0 with 4 channels, two dense layers each adding k = 2. Layer 1 sees the 4 channels of x_0 and outputs x_1 (2 channels). Layer 2 sees [x_0, x_1] — 6 channels — and outputs x_2. The gradient ∂L/∂x_0 collects one term routed straight through layer 2's input slice for x_0 (no intervening layer), plus a term through x_1 (one layer), plus the term from x_0 being read by layer 1. The *shortest* of these paths from x_2's computation back to x_0 has length one, regardless of how many layers I stack in between, because x_0 is a literal slice of every later layer's input. The final loss still passes through the transitions after the block, especially when I compress, so I won't pretend every early map is a literal classifier input. But within each resolution the shortest path length collapses to one transformation, and across resolutions only a few transition layers sit between a feature and the top. That is even more direct than ResNet's additive "+1": there the short path is an additive identity that the optimizer must keep from being swamped by F; here it is an actual concatenation slice with its own dedicated, un-mergeable gradient route. A side effect, which I didn't set out to get: because every layer reaches the classifier through a small number of transitions and later stages rather than through every intervening convolution, the single loss at the top behaves like a form of implicit deep supervision, without a clutter of per-layer auxiliary objectives.

Now a problem of my own making. I said each layer emits only k maps — great, the *output* is tiny. But the *input* to layer ℓ is k_0 + k(ℓ−1) channels, which grows linearly with depth. Deep into a block, a layer is doing a 3×3 conv over a fat input stack even though it only produces twelve maps. The 3×3's cost is 9·(k_0+k(ℓ−1))·k, which grows with ℓ — that's the dominant per-layer expense and exactly what I'd like to cap. A 1×1 bottleneck is the standard fix for this shape of problem: before the 3×3 conv, insert a 1×1 conv that squeezes the fat input down to a fixed width c, then the 3×3 only ever sees c channels. So redefine H as BN–ReLU–Conv(1×1, →c)–BN–ReLU–Conv(3×3, →k). Set c proportional to k so it tracks the growth rate but never the depth; the usual choice is c = 4k. Let me actually check what that does to the costs, k = 12, k_0 = 24, across the first dozen layers. The 3×3 cost becomes 9·4k·k = 9·48·12 = 5184, and that's the *same number* whether ℓ is 0 (input 24) or 11 (input 156) — flat, depth-independent, which is the whole point of the bottleneck. The 1×1's own cost is 1·(k_0+k(ℓ−1))·4k, which I should not pretend is free: it's 1152 at the first layer and 7488 by the twelfth, so it does still grow linearly with depth. But notice the trade — without the bottleneck the *3×3* (the expensive spatial conv) was the thing growing with ℓ; with it, the part that grows is now a cheap 1×1, and the spatial conv is pinned. So the bottleneck doesn't make per-layer cost constant, but it moves the depth-dependence onto the cheapest operation and holds the dominant one fixed. Call this the bottleneck variant.

One more squeeze. Even with bottlenecks, the number of channels entering each successive *block* keeps climbing, because a block hands its whole concatenated output to the next stage. I can compress at the transitions: if a block ends with m feature-maps, let the transition emit only ⌊θ·m⌋ of them, with a compression factor 0 < θ ≤ 1, say θ = 0.5 to halve the channel count between blocks. Is throwing half the channels away reckless? The place to do it is exactly the transition, not inside the block: within a block I want every layer to keep direct access to the full shared state, while between resolutions I already need a 1×1 mixing layer before spatial pooling. That 1×1 can select and recombine the most useful information into fewer channels before the next block starts. θ = 1 keeps everything; θ = 0.5 is the deliberate compressed version. Bottleneck plus compression together is the variant I'll build as the default.

Before I commit to it, let me trace the channel arithmetic through a full network the way the code will, to make sure nothing diverges and the head sees a sensible width. Take the ImageNet config: growth k = 32, block_config (6, 12, 24, 16), stem output 64, θ = 0.5. The stem leaves 64 channels. Block 1 has 6 layers, so it adds 6·32 = 192 and outputs 64 + 192 = 256; the per-layer inputs inside it run 64, 96, …, 224 — visibly growing, which is the linear growth the bottleneck was for. Transition 1 compresses 256 → 128. Block 2 (12 layers) takes 128 → 128 + 12·32 = 512; transition 512 → 256. Block 3 (24 layers): 256 → 256 + 24·32 = 1024; transition 1024 → 512. Block 4 (16 layers): 512 → 512 + 16·32 = 1024, and there's no transition after the last block. So the final BN and the classifier see 1024 channels. That 1024 is a clean, finite number — the channel count doesn't blow up despite the dense concatenation, because the θ = 0.5 transitions keep halving the accumulated width between blocks. And the depth bookkeeping lands on a recognizable number: 6+12+24+16 = 58 dense layers, each with two convs (1×1 and 3×3) = 116, plus 1 stem conv + 3 transition convs + 1 classifier = 121 weight layers exactly. So this config is "DenseNet-121" feeding a 1024-wide head. The arithmetic closes with no off-by-one in the transitions or the final block; I'll build it.

Let me also sanity-check the conceptual difference from ResNet one more time, because superficially I've only changed a "+" to a concatenation. x_ℓ = H_ℓ([x_0,…,x_{ℓ−1}]) versus x_ℓ = H_ℓ(x_{ℓ−1}) + x_{ℓ−1}. But the consequences aren't superficial, and the numbers I worked above are what convince me. Summation requires width-matching, which forces every layer to be full-width and merges old and new into one slot; concatenation keeps every feature distinct and lets each layer be narrow because it only adds k maps to a shared state instead of re-writing a full-width state — that narrowness is exactly the k²/W² factor that came out to 1/455. The ResNet preserves information *implicitly*, by adding an identity it hopes the residual won't destroy; this design separates the preserved information (the concatenated history, untouched) from the added information (the k new maps), and the gradient trace showed that separation buys a literal length-one slice path to every earlier map rather than an additive term competing with F. So the change from "+" to concatenation is the thing doing the work, and the rest — small k, the parameter savings, the clean gradient slices, the implicit deep supervision — follows from it.

There's even a satisfying way to see this as the deterministic limit of stochastic depth. In stochastic depth, if every residual layer between two pooling layers happens to be dropped on some minibatch, the layers on either side become directly connected; over training, every such pair gets directly connected *some* of the time, at random. Dense connectivity just makes that connectivity permanent and deterministic — every pair within a block is always directly connected. Same short-path skeleton, but built in by design rather than conjured by a dropout schedule.

Now I can write the code. The per-layer transformation is the bottleneck composite: BN–ReLU–1×1-conv to 4k channels, then BN–ReLU–3×3-conv to k channels (optionally dropout). The dense block keeps a running list of all feature tensors produced so far, concatenates them as the input to each new layer, appends the new output, and finally returns the full concatenation. The transition is BN–ReLU–1×1-conv (halving channels for compression) plus 2×2 average pool. The whole network is a stem conv, then alternating dense blocks and transitions, a final BN, global average pooling, and a linear classifier.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    # H_ell as a bottleneck composite: the 1x1 squeezes the fat
    # concatenated input down to 4k so the 3x3's cost stops growing
    # with depth; the 3x3 emits exactly k new feature-maps (growth rate).
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, prev_features):
        # prev_features is the list [x_0, ..., x_{ell-1}]; concatenate it
        # so the layer reads the network's entire collective state so far.
        x = torch.cat(prev_features, 1)
        x = self.conv1(self.relu1(self.norm1(x)))     # 1x1 bottleneck -> 4k
        x = self.conv2(self.relu2(self.norm2(x)))     # 3x3          -> k
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x                                       # only k new maps


class DenseBlock(nn.ModuleDict):
    # Dense connectivity at one fixed spatial size: each layer takes all
    # preceding feature-maps and contributes k more. Input width grows as
    # num_input + i*k; output width is num_input + num_layers*k.
    def __init__(self, num_layers, num_input_features, growth_rate,
                 bn_size=4, drop_rate=0.0):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                               growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]                     # the running history
        for _, layer in self.items():
            new = layer(features)                      # read all, add k
            features.append(new)                       # nothing overwritten
        return torch.cat(features, 1)                  # full concatenation out


class Transition(nn.Sequential):
    # Between blocks: 1x1 conv (compress channels by theta) + 2x2 avg pool
    # (halve spatial size), so concatenation stays valid within each block
    # while the network still downsamples.
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    # DenseNet-BC: bottleneck layers + compression at transitions.
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.0,
                 compression=0.5, num_classes=1000):
        super().__init__()
        # Stem: standard ImageNet downsampling before the first block.
        self.features = nn.Sequential()
        self.features.add_module("conv0",
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                      padding=3, bias=False))
        self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module("pool0",
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate,
                               bn_size, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate   # grew by L*k
            if i != len(block_config) - 1:                          # not after last block
                out = int(num_features * compression)               # theta compression
                self.features.add_module("transition%d" % (i + 1),
                                         Transition(num_features, out))
                num_features = out
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        # Head: single classifier on top of all final feature-maps ->
        # gives every layer near-direct supervision through few transitions.
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)
```

So the causal chain, start to finish: deep plain nets fail to optimize because signal and gradient wash out over many layers, and the field's fix is short paths from early to late layers; ResNet's identity shortcut gives a clean additive gradient path but merges preserved and new information by summation and keeps full-width per-layer weights, while stochastic depth reveals that most of those layers are redundant and that direct non-adjacent connections are exactly what helps; pushing "short paths" to its limit means connecting every layer to every later one inside a fixed-resolution block, and replacing summation with concatenation so nothing is overwritten and preserved information is held distinct from added information; that lets each layer be narrow — adding only k feature-maps to a shared, globally readable state — which makes the model parameter-efficient because features are reused instead of re-learned; concatenation forces constant spatial size, so the net is split into dense blocks joined by downsampling transition layers; the growing per-layer input is tamed by a 1×1 bottleneck to 4k, and the growing per-block width by a θ compression at the transitions; and the whole thing trains easily because each layer has direct slice paths inside its block and only a small number of transitions between resolutions.
