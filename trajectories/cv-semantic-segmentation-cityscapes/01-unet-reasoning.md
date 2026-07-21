Start from where the cheap baseline stalls. Take a classification backbone, strip its global pooling
and fully-connected head, read off the deepest feature map, put a 1×1 conv on it to turn 2048 channels
into 19 class scores, and bilinearly upsample to full resolution. It works, and the features are
semantically strong — but the prediction is *born* at a downsampled grid. A classification net contracts
resolution stage by stage; by its deepest stage it is at output stride 32. For the fixed 512×1024 crop
that means the class scores live on a 16×32 grid — 512/32 = 16 rows, 1024/32 = 32 columns — and I am
asking those 512 coarse cells to label 512×1024 ≈ 524k pixels. Each cell owns a 32×32 = 1024-pixel patch,
and inside that patch the single upsample can only interpolate a smooth ramp between neighboring cells.
It cannot manufacture a boundary that fell inside one cell.

Work out what that does to the classes I care about. A traffic pole is a few pixels wide — call it 4 px,
so 4/32 = 0.125 of a cell; it never occupies a cell of its own and is dominated by the road or building
behind it. A pedestrian's leg, maybe 10 px, is a third of a cell, again outbid by its background. Even a
whole distant pedestrian, 20–30 px tall, is one or two cells — enough to register "something," not enough
to draw a silhouette. So the failure is structured, not random: semantically right on the big planar
regions — road, building, sky, car — that survive coarse sampling, soft and wrong exactly on thin and
small structures, washed out below the grid before I ever upsample. Bilinear interpolation is a smooth
spatial average; it is the wrong tool for putting an edge back where pooling erased it, because the edge
is not missing from the interpolation, it is missing from the tensor being interpolated.

And the metric makes this expensive. mIoU averages per-class intersection-over-union with *equal weight*
across all 19 classes, so a pole occupying a fraction of a percent of the pixels counts exactly as much
as road, which occupies a third of them. A baseline that nails the big planar classes and loses the thin
ones scores near-perfect on a handful of high-area classes and near-zero on several low-area ones, and
the average punishes those zeros hard. So the structure of the failure — big regions right, thin
structures gone — is exactly the structure the metric penalizes most: recover the thin and small classes
and the class-averaged number moves far more than the recovered pixel count would suggest.

The root cause is that I asked one feature map to be two incompatible things. The deep map is where the
semantics live — "this region is a car, that is road" — but to get there the network threw away the
resolution: 32× of downsampling is what bought the receptive field and the abstraction. The shallow maps
still have the resolution — output stride 2 or 4, near full size — but they see only edges and textures,
not classes. A single upsample of the deep map tries to recover localization from a tensor that no longer
contains it; the boundary information did not get folded into the deep map on the way down, it is still
sitting in the shallow maps I am ignoring. So no cleverness at the very end — a fancier upsample, a bigger
final conv — can help, because it all operates on the coarse tensor. The fix has to reach back to where
the localization still lives.

Two ways to bring semantics and resolution back together fail for the same instructive reason. A learned
single upsample — a transposed conv blowing OS-32 straight up to full resolution — still has to invent
524k output pixels from 512 input cells, and the true boundary information for those pixels is not in the
input; the conv can sharpen the *ramps* it produces but cannot know where a pole was. Progressive
upsampling in several 2× steps with a couple of convs between them refines as it climbs, strictly better
than one blind 32× jump — but every refining conv still works only from the coarse deep tensor, so it is
refining a hallucination. Both fail because they have no source of true high-frequency detail, and that
names the fix: *inject* the shallow maps' resolution back into the upsampling path, the only place the
localization physically exists.

So the design is an expanding path that mirrors the contracting one and reaches sideways to grab the
shallow feature at each resolution as it climbs. The backbone is already a sequence of stages at 1, 1/2,
1/4, 1/8, 1/16 resolution — a natural encoder that contracts resolution while it builds semantics. I save
every stage's output as the *skip* tensors. The decoder is the mirror-image expanding path: at each level
take the lower-resolution decoder feature, upsample it 2×, concatenate the saved encoder map at that same
resolution, and run a couple of 3×3 convs to fuse. The upsampled decoder feature supplies the semantics —
it descends from the deepest, most context-aware part of the net — and the concatenated skip supplies the
high-frequency spatial detail at this exact resolution. Stack levels until back at full resolution, then a
1×1 conv to class scores: a "U", contract then expand, with horizontal wires joining equal-resolution
levels.

How deep the U goes is a real choice with an arithmetic answer. Halving resolution from the 512×1024 crop
gives 512×1024, 256×512, 128×256, 64×128, 32×64 — five levels, the deepest at 1/16, a 32×64 bottleneck.
Why stop around 1/16 rather than driving all the way to 1/32? Going deeper widens the deepest units'
receptive field and abstraction, the semantic payoff; but the whole premise is that I fuse detail back on
the way up, so the encoder alone need not reach full abstraction — the decoder convs keep working after
each fusion. Meanwhile every extra descent-and-climb level is two more conv blocks, and the marginal
semantic gain past 1/16 is small on a 512-tall crop. Five levels sits at the knee. The decoder is
symmetric to the encoder not for aesthetics but because a skip fusion only works if the encoder map and
the decoder map at a level are the same size, so every up-level must have an exact-resolution partner to
concatenate with.

Each encoder stage is two 3×3 conv+norm+ReLU then a 2× downsample; each decoder stage is the same two-conv
block after the upsample-and-concat. Two stacked 3×3s see a 5×5 receptive field — the second conv's window
spans three of the first's outputs, each of which already saw 3×3 — but with two nonlinearities and
2·(9C²) parameters instead of one 5×5's 25C²: same spatial reach, more depth, fewer weights. That matters
doubly because this encoder trains *from scratch* on a few thousand Cityscapes images — I want the
parameter count controlled and per-layer normalization in place so the from-scratch optimization is
well-conditioned. With the whole U learning end to end from random init, batch normalization after each
conv is what keeps activations in range across a dozen-plus convolutions down and back up.

Two load-bearing decisions inside a decoder level. First, how to fuse the skip. Addition presumes the two
tensors are the same kind of signal in the same subspace, but here one is coarse semantics and the other
fine edge/texture response; summing forces both roles into one shared set of channels and demands
identical channel counts, so I would have to 1×1-project one to match, throwing away capacity before I
have used it. Concatenation stacks them along the channel axis and lets the following 3×3 convs *learn*,
from data, how much of each to read per channel — no forced alignment, no assumed shared subspace. The
cost is that the fusing conv sees 2C channels if the skip has C and I upsample the decoder feature to C
too, which is exactly why the fusing block takes `2 * skip_channels` inputs. That doubling is the price of
not pretending the two signals are the same thing, and it is cheap.

Second, a skip at every level or just one at the end. One skip at full resolution is tempting, but trace
the error. If I inject detail only at the top, every intermediate decoder level reconstructs structure
purely from the blurred deep feature and hands its blur to the next; by the time the single skip arrives,
the decoder feature it fuses with has accumulated several levels' worth of localization drift, and the
final convs would have to *undo* that drift while also sharpening — too much for one fusion. A
matching-resolution skip at *each* upsampling step means every level starts already re-anchored to the
true spatial structure at that scale, so error does not compound down the path; each level's job shrinks
to "refine locally around a correct scaffold" instead of "rebuild from a blur." That per-level
re-anchoring is why the U works where progressive-upsampling-without-skips did not.

A richer fusion than plain concatenation — a learned per-pixel gate deciding how much skip versus deep
feature to admit — is mechanically attractive, since it directly targets the crisp-but-mislabeled risk I
am about to worry about. But it is premature: a gate adds parameters that have to be learned from the same
few thousand images the from-scratch encoder is already straining to fit, and a plain concat followed by
two convs can already *represent* a soft gate (small learned weights on the skip channels where it hurts).
If plain concat sharpens the wrong boundaries, a gate is the natural next move — but I should find that out
from the metric first, not pay for it up front.

One more choice: the upsample operator inside a level. Plain bilinear is a fixed smoother; a transposed
conv can learn to sharpen but with a stride that does not divide its kernel evenly it lays down a
checkerboard of overlap that shows up as grid artifacts. The clean middle is interpolate-then-conv: a
parameter-free bilinear 2× to fix the geometry, then a small conv to learn the channel remapping and let
the following fusion sharpen. That is the `InterpConv`, and it is why the block's upsample maps the
incoming `in_channels` down to `skip_channels` before the concatenation, so the two halves arrive
channel-balanced. The upsample-and-fuse block and the U-shaped forward pass are in the answer.

Trace one pass, since a resolution mismatch anywhere silently breaks the concatenation. The encoder emits
at 512×1024, 256×512, 128×256, 64×128, 32×64. The decoder starts from the bottom at 32×64; the first
up-level takes the 64×128 skip — the upsample bilinear-doubles 32×64 → 64×128 and remaps the deep channels
down to the skip's count, `torch.cat` stacks the two 64×128 maps into a `2 * skip_channels`-wide tensor,
and two 3×3 convs fuse to 64×128. The next levels pair the result with 128×256, then 256×512, then
512×1024, consuming the skips in descending-resolution order. The final decoder feature is at full
resolution and goes straight into the 1×1 classifier. The largest single jump anywhere is 2×, each
immediately re-anchored by a skip — that is the structural difference from the baseline, made concrete.

The U roughly doubles the layers of a plain contracting backbone, but the compute is not spread evenly: a
conv block's cost scales with its spatial resolution, so the full-resolution decoder level at 512×1024 is
16× more expensive per channel than the 128×256 level and 256× more than the 32×64 bottleneck. The
expanding path is dominated by its top one or two levels — exactly the levels doing the fine-boundary
fusion I am paying for. This is the opposite compute profile from the baseline, which did almost all its
work at coarse resolution and interpolated for free. To keep the top-level cost bounded I hold the channel
count low at the fine levels and high at the coarse bottleneck — channels double as resolution halves — so
the expensive full-resolution convs are also the narrowest; without that, two 3×3 convs at 512×1024 on
wide maps would dominate the whole forward pass.

Feeding matching-resolution detail back in at every step should fix exactly the failure I started from:
boundaries sharpen because the final fusion levels work at full resolution with sharp encoder maps in hand
rather than blowing up a 16×32 map, and thin and small structures survive because their spatial signature
lives in the shallow skips and now rides straight into the decoder instead of being discarded below the
grid. The bet is that an encoder–decoder with per-level skip connections clears the single-upsample
baseline outright — that giving the decoder the localization it was missing is worth more than the extra
decoder compute. The evaluation also reports mIoU with multi-scale-and-flip test-time augmentation, which
helps by letting the net see structures at a friendlier scale and averaging out localization jitter; a
model whose predictions ride on fine-detail skips should benefit noticeably from that, because boundary
placement is precisely what shifts when the input is rescaled or flipped — but I only have the single-scale
value to reason from now.

The risk is the mirror image of the baseline's. The skips are *shallow* features — low-level edge/texture
with weak semantics — so if the fusing convs lean on them too hard the decoder could sharpen the wrong
boundaries: crisp but mislabeled. The hedge is the two-conv fusion at each level with the deep semantic
feature concatenated alongside the skip rather than replacing it — the convs see both and the training
signal can teach them to take the *where* from the skip and the *what* from the deep feature. If the
balance is wrong it should show up as high-frequency label noise along boundaries rather than a global
drop, and the fix would be to widen the fusion or down-weight the skip; but the class-averaged metric
should still come out well ahead, because the dominant baseline error — whole thin/small structures
missing — is exactly what the skips restore. The concrete change is the symmetric encoder–decoder with a
skip concatenation at every resolution level; the full head is in the answer.
