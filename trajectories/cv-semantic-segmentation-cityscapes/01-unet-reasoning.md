Let me start from where the cheap baseline stalls. I take a classification backbone, strip its global
pooling and fully-connected head, read off the deepest feature map, slap a 1×1 conv on it to turn the
2048 channels into 19 class scores, and bilinearly upsample to full resolution. It works, and the
features are semantically strong — but the prediction is *born* at a downsampled grid. A classification
net contracts resolution stage by stage; by its deepest stage it is at output stride 32. For the fixed
512×1024 crop that means the class scores live on a 16×32 grid — 512/32 = 16 rows, 1024/32 = 32
columns — and I am asking those 512 coarse cells to label 512×1024 ≈ 524k pixels. Each coarse cell owns
a 32×32 = 1024-pixel patch of the image, and inside that patch the single upsample can only interpolate
a smooth ramp between neighboring cells. It cannot manufacture a boundary that fell inside one cell.

Work out what that does to the classes I care about. A traffic pole in a Cityscapes frame is a few
pixels wide — call it 4 px. On the OS-32 grid that pole spans 4/32 = 0.125 of a cell; it never occupies
a cell of its own, so whatever cell it lands in is dominated by the road or building behind it and the
pole simply is not represented. A pedestrian's leg, maybe 10 px across, is a third of a cell — again
outbid by its background. Even a whole distant pedestrian, 20–30 px tall, is one or two cells: enough
to register "something", not enough to draw its silhouette. So the failure is structured, not random:
the segmentation comes out semantically right on the big planar regions — road, building, sky, car —
because those survive coarse sampling, but soft and wrong exactly on thin and small structures, which
are washed out below the grid before I ever get to upsample. Bilinear interpolation is a smooth spatial
average; it is the wrong tool for putting an edge back where pooling erased it, because the edge is not
missing from the interpolation, it is missing from the tensor being interpolated.

And the metric makes this failure expensive in a way it would not be for a pixel-accuracy score. mIoU
averages the per-class intersection-over-union with *equal weight* across all 19 classes, so a pole,
which occupies a fraction of a percent of the pixels, counts exactly as much as road, which occupies a
third of them. A single-upsample baseline that nails the big planar classes and loses the thin ones is
scoring near-perfect on a handful of high-area classes and near-zero on several low-area ones, and the
average punishes those zeros hard. So the very structure of the failure — big regions right, thin
structures gone — is the structure the metric penalizes most. That is the lever: recover the thin and
small classes and the class-averaged number moves far more than the recovered pixel count would suggest.

The root cause is that I asked one feature map to be two incompatible things at once. The *deep* map
is where the semantics live — "this region is a car, that is road" — but to get there the network
threw away the resolution: 32× of downsampling is what bought the receptive field and the abstraction.
The *shallow* maps still have the resolution — they are near full size, output stride 2 or 4 — but they
only see edges and textures, not classes. A single upsample of the deep map is me trying to recover
localization from a tensor that no longer contains it. The information about *where* the boundary sits
did not get folded into the deep map on the way down; it is still sitting in the shallow maps, which I
am ignoring entirely. So no amount of cleverness at the very end — a fancier upsample, a bigger final
conv — can help, because it is all operating on the coarse tensor. The fix has to reach back to where
the localization still lives.

Before I commit to a design let me lay out the options that could bring semantics and resolution back
together, because there are several and only one survives scrutiny. Option one: keep the single deep
map but replace bilinear upsample with a *learned* single upsample — a transposed convolution that
blows OS-32 straight up to full resolution. This is tempting because it is one line and adds a learnable
operator. But it is the same mistake dressed up: a transposed conv from a 16×32 map still has to invent
524k pixels of output from 512 input cells, and the true boundary information for those pixels is not in
the input. The conv can learn to sharpen the *ramps* it does produce, but it cannot know where a pole
was; the localization is genuinely absent, not merely unsharp. Option two: upsample the deep map in
several 2× steps instead of one 32× jump, with a couple of convs between steps, but *without* reaching
back to the shallow maps. This lets the intermediate convs refine as they go, which is strictly better
than a single blind jump — but every one of those refining convs is still working only from the coarse
deep tensor, so it is refining a hallucination. Progressive upsampling smooths the ramp more gracefully;
it still has no source of true high-frequency detail. Both options fail for the same reason, and the
reason names the fix: I need to *inject* the shallow maps' resolution back into the upsampling path,
because that is the only place the localization physically exists.

So the design is an expanding path that mirrors the contracting one and reaches sideways to grab the
shallow feature at each resolution as it climbs. The backbone is already a sequence of stages at 1, 1/2,
1/4, 1/8, 1/16 resolution — that is a natural *encoder* that contracts resolution while it builds up
semantics. I save the output of every encoder stage; those saved maps are the *skip* tensors. The
decoder is the mirror-image *expanding path*: at each level I take the lower-resolution decoder feature,
upsample it 2×, and concatenate it with the saved encoder map at this same resolution, then run a couple
of 3×3 convs to fuse the two. The upsampled decoder feature supplies the *semantics* — it descends from
the deepest, most context-aware part of the net — and the concatenated encoder skip supplies the
*high-frequency spatial detail* at this exact resolution. Stack these levels until I am back at full
resolution, then a final 1×1 conv to class scores. The shape is a "U": contract, then expand, with
horizontal wires across the U joining equal-resolution levels.

Let me make the depth concrete, because "how deep should the U go" is a real choice with an arithmetic
answer. If the encoder halves resolution each stage starting from the 512×1024 crop, the maps are
512×1024, 256×512, 128×256, 64×128, 32×64 — five resolution levels, the deepest at 1/16, a 32×64
bottleneck. Why stop the descent around 1/16 rather than driving all the way to 1/32 like the
classification backbone did? Two competing pressures. Going deeper widens the deepest units' receptive
field and abstraction, which is the semantic payoff; but the whole premise of the U is that I fuse
detail back on the way up, so I do not need the encoder alone to reach full abstraction — the decoder
convs keep working after each fusion. Meanwhile every extra descent-and-climb level is two more conv
blocks of compute, and the marginal semantic gain past 1/16 is small on a 512-tall crop. Five levels
down to 1/16 sits at the knee: deep enough that the bottleneck sees a wide swath of the scene, shallow
enough that the decoder has a short, well-conditioned climb with a real skip at every rung. That the
decoder is *symmetric* to the encoder — the same five resolution levels going up as came down — is not
an aesthetic choice; it is forced, because a skip fusion only works if the encoder map and the decoder
map at that level are the same size, so every up-level must have an exact-resolution partner to
concatenate with.

What is a "stage" made of, concretely? Each encoder stage is a small block of two 3×3 conv+norm+ReLU,
then a 2× downsample to hand off to the next stage; each decoder stage is the same two-conv block after
the upsample-and-concat. Why two 3×3 convs and not one, or a single 5×5? Two stacked 3×3s see a 5×5
receptive field — the second conv's 3×3 window spans three of the first conv's outputs, each of which
already saw 3×3 — but with two nonlinearities and 2·(9C²) parameters instead of one 5×5's 25C², so I
get the same spatial reach, more representational depth, and fewer weights. That is the standard reason
small stacked kernels beat one big kernel, and it matters doubly here because this encoder is trained
*from scratch* on a few thousand Cityscapes images: I want the parameter count controlled and the
per-layer normalization in place so the from-scratch optimization is well-conditioned. The norm after
each conv is not decoration; with the whole U learning end to end from random init, batch normalization is
what keeps the activations in range as the signal passes through a dozen-plus convolutions down and back
up.

Now the two load-bearing decisions inside a decoder level: how to fuse the skip, and how many skips.
Take fusion first. The two obvious ways to combine the upsampled decoder feature with the skip are
addition and concatenation. Addition is what a residual/lateral connection does: line the two tensors up
channel for channel and sum. But that presumes they are the *same kind of signal* living in the same
subspace, and here they are pointedly not — one is coarse semantics, the other is fine edge/texture
response. Summing them forces the network to represent both roles in one shared set of channels and
hope a later conv can disentangle them; worse, it demands the two tensors already have identical channel
counts, so I would have to 1×1-project one to match, throwing away capacity before I have used it.
Concatenation instead stacks them along the channel axis and lets the following 3×3 convs *learn*, from
data, how much of each to read and how to combine them per channel — no forced alignment, no assumed
shared subspace. The cost is that the fusing conv now sees more input channels: if the skip has C
channels and I upsample the decoder feature to C channels too, the concatenation is 2C channels wide,
which is exactly why the fusing block below is built to take `2 * skip_channels` inputs. That doubling
is the price of not pretending the two signals are the same thing, and it is cheap.

Second decision: a skip at *every* level, or just one fusion at the end? One skip at full resolution is
tempting — it is the level where the finest detail lives, and it would be less machinery. But trace the
error. If I only inject detail once, at the top, then every intermediate decoder level is reconstructing
structure purely from the blurred deep feature, and each of those levels hands its blur to the next; by
the time the single skip arrives at full resolution, the decoder feature it fuses with has already
accumulated several levels' worth of localization drift, and the final convs would have to *undo* that
drift at the same time as they sharpen — too much to ask of one fusion. Injecting the matching-resolution
encoder map at *each* upsampling step means every decoder level starts its fusion already re-anchored to
the true spatial structure at that scale, so the error does not compound down the expanding path; each
level's job shrinks to "refine locally around a correct scaffold" instead of "rebuild from a blur." That
per-level re-anchoring is the whole reason the U works where progressive-upsampling-without-skips did
not.

I briefly consider a richer fusion than plain concatenation — a learned gate that predicts, per pixel,
how much of the skip versus the deep feature to admit, so the network could down-weight the shallow map
wherever it is untrustworthy. Mechanically that is attractive: it directly targets the crisp-but-
mislabeled risk I am about to worry about. But it is premature. A gate adds parameters and, worse, adds
a second thing that has to be learned from the same few thousand images that the whole from-scratch
encoder is already straining to fit; a plain concatenation followed by two convs can already *represent*
a soft gate (the convs can learn small weights on the skip channels where it hurts), so the explicit
gate buys expressiveness I have no data budget to train and no evidence yet that I need. If the plain
concat turns out to sharpen the wrong boundaries, a gate is the natural next move — but I should find
that out from the metric first, not pay for it up front. So concatenation-then-conv it is.

One more small choice: the upsample operator inside a decoder level. The cheapest is plain bilinear
interp — but that is a fixed smoother, and I would like the expansion itself to be able to sharpen. A
transposed convolution can learn to sharpen, but a transposed conv with a stride that does not divide
its kernel evenly lays down a periodic checkerboard of overlap that shows up as grid artifacts in the
output — a known failure mode I would rather not fight. The clean middle is interpolate-then-conv: do a
parameter-free bilinear 2× upsample to fix the geometry, then a 1×1 (or small) conv to learn the
channel remapping and let the fusion that follows sharpen. That is the `InterpConv` I will use for the
upsample, and it is why the block's upsample maps the incoming `in_channels` down to `skip_channels`
before the concatenation — so the two halves of the concat arrive channel-balanced.

Let me write the decoder upsample-and-fuse block — the wire across the U:

```python
class UpConvBlock(nn.Module):
    def __init__(self, conv_block, in_channels, skip_channels, out_channels,
                 num_convs=2, upsample_cfg=dict(type='InterpConv')):
        super().__init__()
        # the fusing conv block consumes [skip ; upsampled] => 2*skip_channels in
        self.conv_block = conv_block(in_channels=2 * skip_channels,
                                     out_channels=out_channels,
                                     num_convs=num_convs)
        self.upsample = build_upsample_layer(cfg=upsample_cfg,
                                             in_channels=in_channels,
                                             out_channels=skip_channels)

    def forward(self, skip, x):
        x = self.upsample(x)             # bring deep feature up to skip's resolution
        out = torch.cat([skip, x], dim=1)  # fuse semantics (x) with detail (skip)
        out = self.conv_block(out)
        return out
```

and the whole U: run the encoder, stashing each stage's output, then walk the decoder back up,
consuming the stashed skips in reverse order.

```python
def forward(self, x):
    enc_outs = []
    for enc in self.encoder:          # contracting path
        x = enc(x)
        enc_outs.append(x)            # save skip at each resolution
    dec_outs = [x]
    for i in reversed(range(len(self.decoder))):   # expanding path
        x = self.decoder[i](enc_outs[i], x)        # upsample + fuse matching skip
        dec_outs.append(x)
    return dec_outs
```

Let me trace one pass to check the shapes actually line up, because a resolution mismatch anywhere in
the U silently breaks the concatenation. Encoder emits `enc_outs` at 512×1024, 256×512, 128×256,
64×128, 32×64 (levels 0..4). The decoder starts from the bottom, `x` at 32×64. The first up-level takes
`enc_outs[3]` (64×128) as its skip: `self.upsample(x)` bilinear-doubles 32×64 → 64×128 and remaps the
deep channels down to the skip's channel count, `torch.cat` stacks the 64×128 skip on the 64×128
upsampled feature — sizes match, the concat is `2 * skip_channels` wide — and the two 3×3 convs fuse to
64×128. Next level pairs the result with `enc_outs[2]` at 128×256, upsampling 64×128 → 128×256; then
256×512; then 512×1024, fusing `enc_outs[0]`. The `reversed(range(...))` index walk consumes skips 3,
2, 1, 0 in that order, which is exactly descending resolution-partner order — the loop is correct. The
final decoder feature is at 512×1024, full resolution, and goes straight into the 1×1 `conv_seg`. No
8× or 32× blind upsample anywhere: the largest single jump in the net is 2×, each one immediately
re-anchored by a skip. That is the structural difference from the baseline, made concrete.

It is worth being honest about what the U costs, because the whole bet is that the localization is
worth the compute. The decoder roughly mirrors the encoder, so as a first cut the net has about twice
the layers of a plain contracting backbone. But compute is not spread evenly across the levels: a conv
block's cost scales with its spatial resolution, so the full-resolution decoder level at 512×1024 is
16× more expensive per channel than the 128×256 level and 256× more than the 32×64 bottleneck. The
expanding path is therefore dominated by its top one or two levels — exactly the levels doing the
fine-boundary fusion I am paying for. This is the opposite compute profile from the baseline, which did
almost all its work at coarse resolution and then interpolated for free; here I am deliberately spending
the compute where the boundaries are. To keep the top-level cost bounded I hold the channel count *low*
at the fine levels and *high* at the coarse bottleneck — the usual arrangement, where channels double as
resolution halves — so the expensive full-resolution convs are also the narrowest. That channel-vs-
resolution trade is what makes a full-resolution decoder affordable at all; without it, two 3×3 convs
at 512×1024 on wide feature maps would dominate the entire forward pass.

What I expect, and how the metric should show it. Feeding the matching-resolution detail back in at
every step should fix exactly the failure I started from: the boundaries should sharpen, because the
final fusion levels are working at full resolution with the sharp encoder maps in hand rather than
blowing up a 16×32 map, and thin and small structures should survive, because their spatial signature
lives in the shallow skips and now gets carried straight into the decoder instead of being discarded
below the grid. On Cityscapes val mIoU the bet is that an encoder–decoder with per-level skip
connections clears the single-upsample baseline outright — that giving the decoder the localization it
was missing is worth more than the extra decoder compute. There is a second, subtler prediction I can
make from the metric pair. The evaluation also reports mIoU with multi-scale-and-flip test-time
augmentation, which mostly helps by letting the net see structures at a friendlier scale and averaging
out localization jitter. An encoder–decoder whose predictions ride on
fine-detail skips should benefit *noticeably* from that augmentation, because such a model's boundary
placement is precisely the thing that shifts when the input is rescaled or flipped — so I would expect a
sizeable single-scale-to-ms+flip gap here, a couple of points rather than a fraction — but I only have the
single-scale value to reason from now; the ms+flip column will tell.

The risk is the mirror image of the baseline's. The skips are *shallow* features — low-level
edge/texture responses with weak semantics — so if the fusing convs lean on them too hard the decoder
could sharpen the wrong boundaries: crisp but mislabeled, a clean edge drawn around the wrong class.
This is a genuine possibility, not a formality: the whole point of the skip is to trust the shallow
map's geometry, and geometry with weak semantics can be geometrically right and semantically wrong. The
hedge is the two-conv fusion at each level, with the deep semantic feature concatenated alongside the
skip rather than replaced by it — the convs see both, and the training signal can teach them to take
the *where* from the skip and the *what* from the deep feature. If this balance is wrong I would expect
it to show up as high-frequency label noise along boundaries rather than a global drop, and the fix
would be to widen the fusion or down-weight the skip; but the class-averaged metric should still come
out well ahead of the baseline because the dominant baseline error — whole thin/small structures
missing — is precisely what the skips restore. The concrete change is the symmetric encoder–decoder
with a skip concatenation at every resolution level; the full head is in the answer.
