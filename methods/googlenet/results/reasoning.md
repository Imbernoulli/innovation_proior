Let me start from the thing everyone is doing right now, because the pain is hiding inside the obvious move. The recipe that has been winning image recognition for two years is: take a convolutional net and make it bigger. Deeper — more levels of feature abstraction. Wider — more units per level. And it works; with enough labeled data, more size buys more accuracy, and it's the easiest, safest knob to turn. So the lazy plan writes itself: keep stacking, keep widening, push past the eight layers that won ImageNet two years ago. I want to actually train that plan in my head before I commit to it, because I have a nagging feeling that "just make it bigger" pays a bill that grows faster than the accuracy does.

Two bills, actually. The first is overfitting. More units means more parameters, and ImageNet is full of categories that are genuinely hard to tell apart — two breeds of dog that differ by a smudge of fur color. To separate those reliably you need a lot of correctly labeled examples, and that labeling is the expensive part; it takes expert raters. So if I let the parameter count balloon, I'm leaning harder and harder on a data supply that's costly to grow. That's a real bottleneck, not a hypothetical.

The second bill is the one I want to be precise about, because precision here is going to point at the fix. Take two convolutional layers chained back to back. Layer one produces some number of feature maps; call it $C_{\text{in}}$. Layer two convolves over them with $C_{\text{out}}$ filters of spatial size $k \times k$, over an output grid of $H \times W$ positions. The multiply-add count is

$$\text{cost} = H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot k^2.$$

Now "uniformly make it wider" means scale the number of filters everywhere by some factor $\alpha$. But $C_{\text{in}}$ of this layer *is* the $C_{\text{out}}$ of the previous one — widen both and the cost carries a factor of $\alpha$ from the inputs and another $\alpha$ from the outputs. It goes up like $\alpha^2$. Compute grows *quadratically* in the width. And here's the part that stings: if a chunk of that extra capacity ends up useless — filters whose weights drift to near zero, which absolutely happens — then I've paid the quadratic cost for nothing. The compute budget is finite, always, and increasingly I care about models that can actually run somewhere modest, not just sit in a lab. So uniform scaling is spending a quadratic amount of compute to buy a sub-quadratic amount of accuracy, and wasting a fraction of even that. There has to be a way to spend the compute where it earns its keep and nowhere else.

So let me reframe. The disease isn't "the network is big." The disease is that the network is *uniformly densely* big — every unit in a layer fully connected to every feature map below it, capacity sprayed evenly whether or not a given connection carries signal. The cure that suggests itself, if I take that diagnosis seriously, is *sparsity*: don't connect everything to everything; connect only where there's real statistical structure to exploit. A sparse network has fewer parameters (less overfitting) and fewer operations (less compute) for the same expressive reach — if I can find the right sparse pattern.

Is there any principled reason to believe a good sparse pattern exists and is findable? I keep coming back to a learnability result: roughly, if the data distribution can be modeled by a large but *very sparse* deep network, then you can build a good topology layer by layer — look at the correlation statistics of the activations in one layer, and cluster together the units whose outputs are highly correlated; each cluster becomes a unit in the next layer, wired only to the units it summarizes. That's a recipe, not just an existence claim. And it rhymes with something older and cruder that I trust as an intuition pump: the Hebbian slogan, neurons that fire together wire together. The strict theorem needs strong conditions I can't guarantee, but the fact that the careful result and the crude heuristic point the same way makes me believe the underlying idea survives outside the fine print. Group correlated activations; connect the groups. That would be the optimal *local* structure, repeated across the image because translation invariance means whatever's optimal at one location is optimal everywhere — which is just convolution again.

Great, so build the sparse net. Let me try to actually do it and watch where it breaks. I write out a network with non-uniform, clustered, sparse connection tables in the feature dimension. And immediately I hit the wall, and it's not a math wall, it's a hardware wall. Today's machines are catastrophically bad at non-uniform sparse computation. The arithmetic might drop by a hundred times, and it still loses, because the time goes into irregular memory access — scattered lookups, cache misses — not into the multiplies. Meanwhile the dense matrix-multiply routines keep getting faster, squeezing every drop out of the CPU and GPU, precisely because the access pattern is regular and predictable. That's not an accident of laziness; it's *why* the field abandoned the old sparse/random feature-map connection tables from the LeNet days and went back to full dense connections the moment GPUs showed up — dense is what the silicon rewards. So I have a cruel situation: the theory says sparse, the hardware says dense, and the hardware is not negotiable.

I'm stuck between those two for a while. Sparse is right in principle and wrong in practice; dense is wrong in principle and right in practice. The reflex is to pick one and eat the cost. But the interesting question is whether there's a third thing — a structure that is sparse *in the sense the theory wants* yet executes as *dense operations* the hardware loves.

The sparse-linear-algebra people offer the missing bridge. When you have to multiply by a genuinely sparse matrix and want it fast, the trick that wins in practice is not to chase individual nonzeros — it's to *cluster the nonzeros into relatively dense submatrices* and do dense multiplies on those blocks. You approximate the sparse object by a handful of dense ones. I don't have to choose between sparse-theory and dense-hardware. I can *approximate the optimal local sparse structure by readily available dense building blocks* — dense convs, dense pools, the things the libraries already make fast — arranged so that, taken together, they cover the clusters the correlation-clustering picture says should be there. Sparse in spirit, dense in execution. Now I need to turn that into an actual local block.

So: what does "cover the clusters by dense components" look like at one location? Go back to the layer-by-layer prescription. I'm looking at the units of the previous layer; each corresponds to some region of the image, and I want to cluster correlated ones and connect each cluster densely to a unit above. Where do the correlated units sit? In the low layers, close to the pixels, correlated units bunch up in *small local regions* — a cluster is essentially a single spot in the feature map across channels. A dense operator that reads exactly one spatial location across all channels and produces an output is a $1 \times 1$ convolution. So the tightest, most local clusters get covered by $1 \times 1$ convs.

But not every cluster is that tight. Some correlated groups are spread out over a slightly larger patch — those want a convolution over a bigger receptive field, $3 \times 3$. And some are spread wider still, wanting $5 \times 5$. There are fewer and fewer such clusters as the region grows, but they exist. So at a single location I don't want *one* operator, I want *several in parallel*, each sized to a different cluster scale: a $1 \times 1$, a $3 \times 3$, a $5 \times 5$. Why exactly those three sizes and not a continuum? Partly it's that odd kernel sizes centered on a pixel keep the output grid aligned with the input grid — even sizes force awkward half-pixel offsets and patch-alignment headaches. So I restrict to $1\times1$, $3\times3$, $5\times5$: enough scale coverage, no alignment grief. I'll admit this is convenience as much as necessity — there's nothing sacred about stopping at five — but it's a clean, hardware-friendly set.

And there's one more operator that every strong convnet of this era leans on and that I'd be foolish to drop: pooling. Max-pooling has been essential to basically every state-of-the-art result. If it's that load-bearing in the serial stacks, then a *parallel* pooling path inside my block — pool the input and carry that summary forward alongside the convs — ought to help too. It costs almost nothing and it'd be strange to leave it out.

So here's the block. Take the input feature map. Run four things on it in parallel: a $1\times1$ conv, a $3\times3$ conv, a $5\times5$ conv, and a $3\times3$ max-pool. Then concatenate all their outputs along the channel axis into one fat tensor, and that's what feeds the next block. The next block sees, side by side, features computed at several scales, and it can build its own abstractions by mixing across all of them. Multi-scale processing, learned rather than hand-set with fixed Gabor filters the way the old cortex-inspired models did, and repeated as many times as I want instead of being a two-layer affair. Stack these blocks, drop in a stride-2 pool between groups of them to halve the grid now and then, and I have a deep multi-scale net. As I go deeper and features get more abstract, I'd expect their spatial concentration to *spread out* — higher-level concepts aren't pinned to one pixel — so the share of $3\times3$ and $5\times5$ work should grow in the later blocks relative to the early ones. Nice, that even gives me a knob with a principled direction.

Let me now do the thing I always have to do: count the cost, because the entire point was efficiency and I have a bad feeling about that $5\times5$. Concretely, suppose this block sits where the input has $192$ channels on a $28\times28$ grid, and I want the $5\times5$ branch to output $32$ channels. Cost:

$$28 \cdot 28 \cdot 192 \cdot 32 \cdot 5^2 = 28\cdot28 \cdot 192 \cdot 32 \cdot 25 \approx 1.2 \times 10^{8}$$

multiply-adds. A hundred and twenty million — for *one branch* of *one block*. And that's with a modest $32$ outputs; the $5\times5$ over $192$ input channels is just brutal because cost is linear in $C_{\text{in}}$ and I'm feeding it all $192$. Now stack the problem on itself. The pooling branch is even worse in a sneaky way: a max-pool doesn't change the channel count, so it emits the *same* number of channels as its input. When I concatenate — convs plus the pool's passthrough — the output channel count of the block is *at least* the input channel count, and the conv branches pile more on top. So channels grow from block to block, monotonically, and since the $5\times5$ cost scales with input channels, the next block's $5\times5$ is even more expensive than this one's. A couple of stages in and the whole thing detonates. The naive block, even if it perfectly covers the optimal sparse structure, covers it so inefficiently that it's unusable. I've recreated the quadratic-compute disease I was trying to cure, just with prettier branches.

So what's actually expensive, precisely? The big-kernel convs, and the reason they're expensive is the $C_{\text{in}}$ factor — they're forced to look at every one of the many incoming channels. If I could shrink the channel count *just before* the expensive conv runs, the conv would get cheap, because its cost is linear in the channels it sees. Is there an operator that cheaply maps many channels down to few at each spatial location? That's exactly a $1\times1$ convolution: at each pixel it takes the $C_{\text{in}}$-vector of channel values and linearly projects it to a smaller $C_{\text{red}}$-vector. It has no spatial extent, so it's cheap — $H\cdot W\cdot C_{\text{in}}\cdot C_{\text{red}}$, with no $k^2$ — and it's a learned, sensible compression rather than a crude drop. The $1\times1$ conv, which I was already using for the tightest clusters, turns out to also be the perfect *dimension-reduction* tool: put one in front of each big conv to squeeze the channels down, let the $3\times3$ or $5\times5$ do its spatial work in the reduced space, and let the big conv expand back out to however many outputs I want.

Let me redo the cost for that $5\times5$ branch with a reduction in front. Reduce $192 \to 16$ with a $1\times1$, then $5\times5$ from $16 \to 32$:

$$\underbrace{28\cdot28\cdot192\cdot16\cdot1^2}_{\text{reduce}} \;+\; \underbrace{28\cdot28\cdot16\cdot32\cdot5^2}_{5\times5} \approx 2.4\times10^{6} \;+\; 1.0\times10^{7} \approx 1.2\times10^{7}.$$

About twelve million, against the hundred and twenty million from before. Ten times cheaper, and the output is the same shape — still $32$ channels on $28\times28$. The reduction is where almost all the savings live: I'm running the costly $5\times5$ at $16$ channels instead of $192$, a twelvefold cut on the operator that hurt most, and the $1\times1$ that buys me that cut is nearly free. The same medicine works on the $3\times3$ branch: a $1\times1$ reduce in front of it. And the pooling branch — the one that was bloating the channel count — gets a $1\times1$ *after* the pool, projecting its passthrough channels down to a small number so the concatenation doesn't keep ballooning. Now I control the output width of every branch independently, and the block's channel count stops growing out of control.

I should double-check I'm not fooling myself, because compressing channels feels like it should cost accuracy. Why is it okay to jam $192$ channels down to $16$ before the $5\times5$? The justification is the same one that makes embeddings work: a low-dimensional embedding of a patch can still carry most of the information in it — visual data is redundant, and a fair amount of it lives on a much lower-dimensional manifold than the raw channel count suggests. There's a real tension, though, and I want to name it so I don't over-apply the trick: a dense compressed representation is *harder to model* than a spread-out one — once you've squeezed information into few tightly-packed dimensions, the structure the next layer needs is more entangled. The sparse-structure theory wanted my representations to stay sparse, i.e. *not* compressed, at most places. So the rule that falls out is: keep things uncompressed almost everywhere, and compress *only* at the moments where signals have to be aggregated en masse — which is exactly right before an expensive aggregation like the $3\times3$ or $5\times5$ over many channels. Reduce only where you're about to pay, nowhere else. And a bonus: those $1\times1$ reducers carry a ReLU like every other conv, so each one also injects an extra nonlinearity for free — they're doing double duty, compressing *and* adding modeling power.

That's the block I'll commit to. Four parallel branches off the same input: a plain $1\times1$; a $1\times1$ reduce then a $3\times3$; a $1\times1$ reduce then a $5\times5$; and a $3\times3$ max-pool then a $1\times1$ projection. Concatenate the four along channels. Sparse in spirit — each branch covers a different cluster scale and the reductions keep the connectivity lean — but every operation is a dense conv or pool the libraries already run fast. Let me sketch it.

```python
class FeatureBlock(nn.Module):
    def __init__(self, in_channels, *widths):
        super().__init__()
        ch1, ch3red, ch3, ch5red, ch5, poolproj = widths
        # tightest, most local clusters: a single 1x1
        self.branch1 = ConvUnit(in_channels, ch1, kernel_size=1)
        # mid-scale clusters: reduce channels, then 3x3
        self.branch2 = nn.Sequential(
            ConvUnit(in_channels, ch3red, kernel_size=1),
            ConvUnit(ch3red, ch3, kernel_size=3, padding=1),
        )
        # wider clusters: reduce hard, then the expensive 5x5 at low width
        self.branch3 = nn.Sequential(
            ConvUnit(in_channels, ch5red, kernel_size=1),
            ConvUnit(ch5red, ch5, kernel_size=5, padding=2),
        )
        # parallel pooling, projected down so it stops bloating the width
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvUnit(in_channels, poolproj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )
```

Now let me build a real network out of these and pick actual widths, with a fixed compute budget hanging over me the whole time — I'm aiming to keep inference around a billion-and-a-half multiply-adds so this thing could plausibly run on ordinary hardware, not just be a trophy. First the stem. I *could* start stacking blocks from the pixels, but at full $224\times224$ resolution the blocks are memory-hungry, and honestly the earliest layers don't need the fancy multi-scale machinery — they're learning edges. So I keep a conventional front end: a $7\times7$ stride-2 conv down to $112\times112$ with $64$ channels, a $3\times3$ stride-2 max-pool, a response normalization, then a $1\times1$ conv and a $3\times3$ conv up to $192$ channels, another normalization, another stride-2 pool. That drops the grid to $28\times28\times192$ cheaply before any block runs. This is a pragmatic memory concession, not a principled boundary — there's nothing stopping me from using blocks lower down, the infrastructure just makes it wasteful right now.

A word on those normalizations, since they matter for honesty about the available tools. The stabilizer I have is local response normalization — normalize each unit against its neighbors in the channel dimension. It is already part of the large-convnet recipe, so I use it in the stem where the activations are widest and move on. The deeper stack will still be delicate, because gradients have a long way to travel.

Then the body: nine of these blocks, in three stages. Stage one at $28\times28$: two blocks (call them 3a, 3b), then a stride-2 pool to $14\times14$. Stage two at $14\times14$: five blocks (4a–4e), then a stride-2 pool to $7\times7$. Stage three at $7\times7$: two blocks (5a, 5b). For each block I choose the seven width knobs to spend more on the cheap branches early and shift toward the bigger-kernel branches as I go deeper — following the prediction that abstract features spread out spatially, so later stages want relatively more $3\times3$ and $5\times5$. Concretely the widths I land on, written as (in, $1\times1$, $3\times3$-reduce, $3\times3$, $5\times5$-reduce, $5\times5$, pool-proj):

```
3a (192,  64,  96, 128, 16,  32,  32)  -> 256
3b (256, 128, 128, 192, 32,  96,  64)  -> 480
4a (480, 192,  96, 208, 16,  48,  64)  -> 512
4b (512, 160, 112, 224, 24,  64,  64)  -> 512
4c (512, 128, 128, 256, 24,  64,  64)  -> 512
4d (512, 112, 144, 288, 32,  64,  64)  -> 528
4e (528, 256, 160, 320, 32, 128, 128)  -> 832
5a (832, 256, 160, 320, 32, 128, 128)  -> 832
5b (832, 384, 192, 384, 48, 128, 128)  -> 1024
```

Notice the reductions in action — block 4a takes $480$ channels in but squeezes to $16$ before its $5\times5$; without that the $5\times5$ over $480$ channels would be ruinous. The whole stack lands around the budget I wanted, and counting layers-with-weights this is about twenty-two deep, which is roughly three times the depth that won two years ago, at a *fraction* of the parameters.

Now the head, and this is where I get to delete the single biggest source of parameters in the old design. The classic template ends in fully-connected layers, and in those eight-layer nets the FC head holds the large majority of the parameters — tens of millions of weights, the prime overfitting suspect. After the last block I have a $7\times7\times1024$ feature map. The old move flattens that and runs it through dense layers. Instead I can average each of the $1024$ feature maps over its whole $7\times7$ spatial extent, collapsing to a $1024$-vector — global average pooling. It has *no parameters at all*. It's not just cheap; the averaging is itself a structural regularizer, forcing each final feature map to behave like a confidence for some concept rather than letting a dense layer memorize idiosyncratic spatial patterns. I'll keep one tiny linear layer after the pooling, $1024 \to 1000$, purely so the network is easy to retarget to a different label set later; it's a convenience, not load-bearing.

Even with the FC head gone, I should not pretend the overfitting problem is gone. The final $1024$ pooled features can still co-adapt with the classifier, and the convolutional body has plenty of capacity. Dropout is cheap at this point because the vector is small, so I keep a dropout ratio around $0.4$ right before the final linear. Global average pooling removes the worst parameter sink; dropout remains the guard against co-adaptation in the readout.

And now the part I've been deferring, the one that worries me most about going this deep with the tools I actually have. Twenty-two learned layers arranged as a plain forward stack is a long path. When I backpropagate, the gradient signal has to survive its trip all the way from the loss back to the early blocks, through a dozen-plus nonlinear layers, and I have real reason to fear it arrives faint. If the early blocks barely get a usable gradient, they barely learn, and the whole "depth buys abstraction" premise quietly fails at the bottom.

Let me think about what I actually know that bears on this. Shallower networks do *surprisingly well* on this task. That's a strange and useful fact: it means the features sitting in the *middle* of my deep net — roughly the depth of a whole shallow net that already works — must already be quite discriminative on their own. So here's a thought: if the middle features are good enough to classify from, why not *actually classify from them*, during training? Hang a small classifier off an intermediate block, give it the real labels, and add its loss to the main loss. Three things should happen at once. The intermediate classifier's gradient is injected right *at* that middle block, so it doesn't have to make the long faint journey from the top — the early-to-middle layers get a strong, direct training signal. It pressures those middle features to be genuinely discriminative rather than merely on-the-way-to-discriminative. And it acts as a regularizer, since the network now has to support classification from multiple depths. The vanishing-gradient worry and the "are the middle features good" worry get answered by the same device.

I have to be careful how I weight it, though. These side classifiers are a training crutch, not the thing I actually want to ship — the real prediction comes from the full-depth head. If I add their loss at full strength, I'd be telling the network the middle-depth answer matters as much as the final one, which would distort the deep representation toward shallow-classifiable features. So discount them hard: weight each auxiliary loss by $0.3$ in the total. Enough to inject gradient and shape features, not enough to take over. The total training objective is

$$L = L_{\text{main}} + 0.3\,L_{\text{aux}_1} + 0.3\,L_{\text{aux}_2},$$

and crucially these side heads exist *only* during training — at inference I rip them out entirely and just read the main head. Where to attach them? At the outputs of a couple of the middle blocks, deep enough that there are layers below them to benefit from the injected gradient, spread out so they cover different depths — say off block 4a and off block 4d.

What goes *in* a side head? It should be small and cheap — it's a regularizing crutch, not a second network. So: average-pool the intermediate map down to a small grid (a $5\times5$/stride-3 pool gets me to roughly $4\times4$), a $1\times1$ conv to squeeze to $128$ channels — there's my dimension-reduction trick again, shrinking before the dense layer — then flatten, a fully-connected layer to $1024$ with ReLU, a *heavy* dropout (around $0.7$, because it's a small head whose only jobs are gradient injection and regularization, so I lean hard on the regularizer), and a final linear to the $1000$ classes with its own softmax. Let me write the head and then assemble the whole thing.

```python
class SideHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvUnit(in_channels, 128, kernel_size=1)   # reduce before the FC
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)                     # heavy: it's a crutch

    def forward(self, x):
        x = self.avgpool(x)                               # 5x5/3 pool -> 4x4
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)
```

```python
class Net(nn.Module):
    def __init__(self, num_classes=1000, aux=True):
        super().__init__()
        self.aux = aux
        # conventional stem: plain convs + pooling + response norm
        self.conv1 = ConvUnit(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.0)
        self.conv2 = ConvUnit(64, 64, kernel_size=1)
        self.conv3 = ConvUnit(64, 192, kernel_size=3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.body = nn.ModuleList([
            FeatureBlock(192, 64, 96, 128, 16, 32, 32),       # 3a
            FeatureBlock(256, 128, 128, 192, 32, 96, 64),     # 3b
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            FeatureBlock(480, 192, 96, 208, 16, 48, 64),      # 4a
            FeatureBlock(512, 160, 112, 224, 24, 64, 64),     # 4b
            FeatureBlock(512, 128, 128, 256, 24, 64, 64),     # 4c
            FeatureBlock(512, 112, 144, 288, 32, 64, 64),     # 4d
            FeatureBlock(528, 256, 160, 320, 32, 128, 128),   # 4e
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            FeatureBlock(832, 256, 160, 320, 32, 128, 128),   # 5a
            FeatureBlock(832, 384, 192, 384, 48, 128, 128),   # 5b
        ])

        self.side_heads = nn.ModuleDict()
        if aux:
            self.side_heads["aux1"] = SideHead(512, num_classes)  # off 4a
            self.side_heads["aux2"] = SideHead(528, num_classes)  # off 4d

        # parameter-free head: global average pool -> dropout -> one linear
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.lrn1(self.pool1(self.conv1(x)))
        x = self.pool2(self.lrn2(self.conv3(self.conv2(x))))
        x = self.body[0](x)
        x = self.body[1](x)
        x = self.body[2](x)
        x = self.body[3](x)
        a1 = self.side_heads["aux1"](x) if (self.aux and self.training) else None
        x = self.body[4](x)
        x = self.body[5](x)
        x = self.body[6](x)
        a2 = self.side_heads["aux2"](x) if (self.aux and self.training) else None
        x = self.body[7](x)
        x = self.body[8](x)
        x = self.body[9](x)
        x = self.body[10](x)
        x = torch.flatten(self.final_pool(x), 1)
        x = self.classifier(x)
        if self.aux and self.training:
            return x, a2, a1
        return x
```

For training it's asynchronous SGD with momentum $0.9$, a fixed learning-rate schedule that decays the rate by $4\%$ every $8$ epochs, and I'd take a running average of the parameters to use as the final model — averaging late iterates smooths out the noise of stochastic optimization and tends to land on a flatter, better-generalizing point than any single iterate. The loss combines the three heads with the $0.3$ discount on the auxiliaries:

```python
def compute_loss(outputs, target):
    aux_weight = 0.3
    if isinstance(outputs, tuple):
        main, a2, a1 = outputs
        return (F.cross_entropy(main, target)
                + aux_weight * F.cross_entropy(a1, target)
                + aux_weight * F.cross_entropy(a2, target))
    return F.cross_entropy(outputs, target)
```

Let me trace the whole chain back to make sure it hangs together. The pain was that uniform dense scaling spends compute quadratically and parameters wastefully. The principled cure was sparsity, justified by correlation-clustering of co-firing units — but raw sparsity dies on real hardware. The escape was to approximate the sparse local structure with *dense* pieces, the way you'd block a sparse matrix into dense submatrices: cover correlated-unit clusters at several scales with parallel $1\times1$, $3\times3$, $5\times5$ convs plus a pool, concatenated. That naive block exploded in cost, so $1\times1$ convs came back in a second role — cheap channel reducers placed exactly where signals get aggregated en masse, cutting the expensive convs an order of magnitude and capping the channel growth, while doubling as extra nonlinearities. Stack nine such blocks behind a cheap conventional stem, replace the parameter-heavy fully-connected head with parameter-free global average pooling and a small dropout-regularized readout, and counter the faint gradient through twenty-two plain learned layers by hanging discounted, training-only auxiliary classifiers off the middle blocks, which double as regularizers and as a way to make the mid-level features directly discriminative. Sparse in spirit, dense in execution, deep but lean enough to actually run.
