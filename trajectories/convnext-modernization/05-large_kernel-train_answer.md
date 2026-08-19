Inverting the block paid off cleanly: FLOPs dropped from 5.27G to 4.64G
while accuracy nudged up slightly to 80.64%, so the rearrangement bought
back budget discipline for free. But nothing about that rung changed how
much of the image each layer actually sees — every convolution in this
ladder has stayed 3x3 since the beginning, the same as vanilla ResNet's
spatial convs. That's worth stopping on, because the single most
distinguishing property of self-attention isn't its channel-mixing
structure but its receptive field: even Swin's windowed variant restricts
attention to windows of at least 7x7, well beyond a 3x3 kernel's reach.
Large kernels aren't new to ConvNets; 3x3 became standard mainly because
stacking small kernels had efficient GPU implementations that a single
large kernel historically lacked — an engineering-cost argument, not
evidence a bigger per-layer receptive field is undesirable. Worth revisiting
now, given how much else about the block has already changed.

I can't just widen the kernel on the current block, though, because of
where the depthwise conv sits after rung 4: narrow input, 1x1 expand to 4x
width, depthwise conv *at that expanded width*, 1x1 project back down.
Depthwise FLOPs scale linearly with channel count, so the depthwise conv is
currently sitting at the most expensive place to grow it — sweeping kernel
size there means paying for channel width and spatial extent
simultaneously. There's a reorder that avoids this, and it mirrors
something already established in the Transformer block this whole ladder
has used as a reference: attention runs first, at the model's narrow
residual-stream width, and only afterward does the feedforward sublayer
expand to 4x and project back down — the expensive, structurally complex
op runs narrow, the cheap dense mixing runs wide. My block currently has
this backwards. So: move the depthwise conv from between the two 1x1s to
before the first one, narrow -> depthwise (now at narrow width) -> 1x1
expand -> 1x1 project. This is a mechanical parallel to attention-before-MLP,
and it's the efficiency unlock this rung needs — with the depthwise conv
scaled by the narrow width instead of the 4x-expanded one, growing its
kernel costs roughly a quarter of what it would under the rung-4 ordering
for the same kernel area.

I want to test the reorder on its own, at the unchanged 3x3 kernel, before
touching kernel size at all — the same discipline as separating
depthwise-alone from depthwise-plus-width earlier. My prediction: this
costs a modest amount of accuracy rather than gaining any, at least
temporarily. Under the old ordering the block's input first passes through
a dense 1x1 conv, mixing every channel, before any spatial op touches it;
under the new ordering the raw input goes straight into a depthwise conv
that (still at kernel 3) mixes each channel only with its own tiny
neighborhood, with no cross-channel mixing having happened yet. Same total
operations, just resequenced, so I'd expect any loss to be transient rather
than structural, but resequencing where information first combines this
deep in the stack plausibly disturbs what the optimizer had settled into,
and I don't expect that disturbance to be free.

With the reorder done, the real question: does a bigger depthwise receptive
field help, and how far do the gains extend? I want to sweep rather than
commit to one size, since my only real anchor is Swin's minimum window of
7x7, and that deserves comparison points on both sides rather than a blind
adoption. Sweeping 3, 5, 7, 9, 11, I expect a specific shape — accuracy
climbing as the kernel grows toward somewhere around 7, then flattening,
possibly turning over at the largest sizes — because a depthwise conv's
per-channel spatial context should eventually stop adding usable
information, whether because the effective receptive field from stacking
many layers already covers enough by that point, or because a wider kernel
spends more parameters on redundant, correlated neighboring positions, or
because optimizing a larger set of per-channel spatial weights gets harder
without a matching payoff. If the plateau lands near 7, that would suggest
convolution and windowed attention are converging on a similar answer to
"how much local context is enough" from opposite directions; if it lands
well off from 7, the two architectures' locality preferences don't actually
coincide, and matching Swin's window size elsewhere in this exploration was
closer to coincidence than principle. Whatever kernel size the sweep
settles at, that becomes the network's spatial extent for the rest of the
ladder — the last change to what a convolution can physically see; every
rung after this is per-layer micro-design rather than receptive field.

```python
# rung 5: move depthwise conv above the 1x1 expansion; sweep kernel size.
import torch.nn as nn

class LargeKernelBlock(nn.Module):
    """narrow -> depthwise(k) [narrow width] -> 1x1 expand 4x -> 1x1 project -> narrow."""
    expand_ratio = 4

    def __init__(self, dim, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        pad = kernel_size // 2
        # depthwise now runs at the NARROW boundary width, not the 4x-expanded width
        self.dw = nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=pad,
                             groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        hidden = dim * self.expand_ratio
        self.pw_expand = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pw_project = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.dw(x)))              # spatial mixing first, narrow width
        out = self.relu(self.bn2(self.pw_expand(out)))      # dense mixing does the heavy lifting
        out = self.bn3(self.pw_project(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

# sub-step 1: reorder only, kernel_size=3 (compare to rung 4's 80.64% / 4.64G)
# sub-step 2: sweep kernel_size in (3, 5, 7, 9, 11) with the reorder in place
KERNEL_SIZES_TO_SWEEP = (3, 5, 7, 9, 11)
```
