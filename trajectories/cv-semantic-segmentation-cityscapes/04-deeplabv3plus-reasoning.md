77.85, and 79.18 with multi-scale-and-flip — up +5.60 from the FCN head's 72.25, the biggest jump so far.
So context was indeed the dominant remaining problem, and the metric pair backs the mechanism: the ms+flip
gap moved only modestly, from +1.11 to +1.33, which is what a gain from scene-level reasoning rather than
boundary sharpness should do — the improvement is already there at single scale, not something test-time
augmentation had to rescue. Now look at *how* the pyramid gets its context and what that costs.

Each pyramid level adaptive-average-pools the feature map down to a tiny grid — 1×1, 2×2, 3×3, 6×6 — then
upsamples the summary back. Pooling to 2×2 means everything in each quadrant is averaged into one number
per channel: the *within-region* spatial detail of the context is gone, and the bilinear broadcast is a
smooth blanket that knows nothing about where the boundaries inside the quadrant are. So the pyramid buys
global reach by *destroying resolution* to get it, then upsampling — context that is right but spatially
coarse, and the coarseness shows most at object boundaries, exactly where a smooth blanket has nothing to
say.

Two problems are tangled here, and a fix for one is not a fix for the other. First: can I gather
multi-scale context *without* pooling resolution away — the same global reach, but a distinct value at
every pixel while I gather it? Second, separately: regardless of how I get context, the head still produces
its prediction at output stride 8 and bilinearly upsamples by 8, so the finest boundaries — thin poles, a
pedestrian's exact silhouette — stay blurry no matter how good the context, because an ×8 upsample of a
64×128 map cannot draw a sharp edge inside an 8×8 cell. The U-Net solved that second problem with a full
decoder; I dropped the decoder when I moved to the dilated backbone, and the pyramid never addressed it. I
fix the two with separate pieces.

Take the context problem. I want multi-scale context applied *densely*, at the feature map's own
resolution. The tool for "see farther without downsampling" is dilation — it is what made the backbone
output stride 8 in the first place. So instead of pooling into grids, run several 3×3 convs in parallel over
the OS-8 feature, each at a different dilation rate. A 3×3 at rate r places taps at {−r, 0, +r}, reaching ±r
feature-map cells = ±8r image pixels: rate 12 spans 192 image pixels, rate 24 spans 384, rate 36 spans 576
— wider than the 512-tall crop. Add a rate-1 branch (a plain 1×1) for the purely-local view and I have a
pyramid of *dilations* rather than pools — four branches sampling from ~1 pixel out to wider-than-the-crop,
all at full feature resolution, every output pixel keeping its location. The point is the within-region
detail pooling threw away: a rate-12 atrous conv sees far-apart context but still emits a *distinct* value
at every pixel, computed from that pixel's own wide neighborhood. Picture a pixel at the road/sidewalk seam.
Under the pooling pyramid its 2×2 and 3×3 context cells each hand it the average of a region straddling the
seam — ambiguous exactly where I need it sharpest. Under the atrous pyramid its rate-12 taps land some on
definite road, some on definite sidewalk, and the fusion resolves the seam from the pattern rather than
from a pre-averaged blur.

But check the arithmetic before trusting it. Push the rate too high and the 3×3's outer taps fall off the
feature map. At OS-8 the feature is 64×128. A rate-36 branch, at any row of the 64-tall map, wants vertical
taps at row±36 — but from center row r, r−36 ≥ 0 needs r ≥ 36 and r+36 ≤ 63 needs r ≤ 27, which cannot both
hold, so *every* row has at least one vertical tap on zero-padding; with two of three vertical taps dead,
the rate-36 conv degenerates toward a 1×1 reading its center tap. Even rate-24 keeps both vertical taps in
range only for rows 24…39, a thin central band. So the largest atrous branches do not deliver honest wide
context on the short axis of this crop. That names the fix: keep a single image-level global-average-pool
branch alongside the atrous branches, broadcast back, as the one *honest* whole-image summary — the same
global cue the pooling pyramid's 1×1 level gave — rather than pretending the rate-36 branch supplies it. The
module is: a 1×1 branch, three atrous branches at rates 12/24/36, and one global-pool branch, concatenated
and fused.

Why 12/24/36 and not the smaller ladder one reaches for first? The rate reads relative to the output
stride: a rate-r atrous conv reaches ±r *feature-map* cells, and at OS-8 each cell is 8 image pixels. The
canonical atrous ladder is tuned at output stride 16; run the same physical reach at OS-8 — half the cell
size, twice the cells per pixel — and the rates double, 12/24/36 rather than 6/12/18. Choosing OS-8 for the
finer prediction forces the larger rates, which is what pushed the biggest branch into the degeneration I
just checked; the three rates are geometrically spaced so the branches tile the scale axis evenly. Three
parallel 3×3 atrous convs over a 2048-channel map is heavy — a dense 3×3 from 2048 → 512 is ≈9.4M MACs per
location, times three. Depthwise-separable convolution cuts it: factor each into a depthwise 3×3 (one filter
per channel — the spatial/atrous part) and a pointwise 1×1 (the cross-channel mixing). The depthwise part
costs 3·3·2048 ≈ 18.4k and the pointwise 2048·512 ≈ 1.05M, totalling ≈1.07M against the dense conv's 9.4M —
about 9×, matching (9+C_out)/(9·C_out) for C_out = 512. Same receptive field, a fraction of the
multiply-adds, and the dilation lives entirely in the cheap depthwise part. So the three-branch context
module costs about what one dense branch would, which keeps the whole head — now two modules, context *and*
decoder — in the same rough compute class as the pooling-pyramid head it replaces.

Now the boundaries. The context module gives a strong OS-8 feature, but I am still one ×8 jump from the
labels and no context sharpness survives an ×8 blur. Rather than reinstate the whole U-Net, ask for the
*minimum* decoder that recovers boundaries here. The boundary information lives in the *shallow* OS-4
backbone stage — sharp but low-level. So add exactly one skip: bilinearly upsample the context feature OS-8
→ OS-4 (a 2× jump, not 8×), grab the OS-4 backbone feature, 1×1-conv it to squeeze channels, concatenate,
refine with a couple of separable 3×3 convs, then a final ×4 upsample. The U-Net skip idea, but a *single*
skip at OS-4 bolted onto the dilated backbone. One skip and not the full stack because the situations
differ: the U-Net needed five levels because its prediction was born at OS-16/32 and had to climb back
through resolutions the from-scratch encoder threw away; here the context feature is already OS-8, only
climbs to OS-4, and the dilated backbone kept the intermediate resolution. The boundary headroom left is the
difference between an ×8 blur and an ×4 blur plus one sharp skip, and a single OS-4 fusion captures most of
it; a second skip would recover ever-finer detail against ever-smaller returns.

A tempting shortcut for the boundaries: dilate the backbone one stage further, to output stride 4, and
shrink the final upsample from ×8 to ×4, no decoder at all. It fails on both cost and substance. On cost,
OS-4 runs the whole dilated tail *and the entire ASPP* over 128×256 instead of 64×128 — 4× the locations,
roughly 4× the head compute — and forces even larger atrous rates, deepening the degeneration. On substance,
an OS-4 backbone feature is still a *deep, context-heavy* feature smoothed by all the semantic processing
above it; making it denser does not give it sharp low-level edges. The decoder skip reaches back to the
*shallow* OS-4 stage, which still carries crisp edge and texture response the deep feature abstracted away —
a genuinely different, sharper signal the dilation cannot supply.

Why reduce the skip's channels before fusing? This is the crisp-but-mislabeled risk from the U-Net step,
acute here because the OS-4 feature is *low-level*. Concatenated at full width against the context feature,
its many channels could dominate the fusion and pull the prediction toward sharp-but-wrong, so I
1×1-reduce it to 48 channels, deliberately small: set against the 512-channel context feature that is
roughly a 10:1 ratio in favor of semantics, so the skip contributes geometry — *where* the edge is —
without outvoting the context on *what* the region is. Which shallow stage is itself a choice on a short
spectrum: OS-2 is sharpest but almost raw pixels, weak enough that even 48 channels could inject noise;
OS-8 is where the context already lives, no new detail; OS-4 is the sweet spot — fine enough that
upsampling the context to it is only a 2× step and the final upsample only ×4, yet processed enough that
its features are low-level *structure* rather than pixel noise.

The separable ASPP module and the decoder head are in the answer; trace the decoder shapes to be sure the
fusion lands. The context feature is 512 channels at OS-8 = 64×128; the OS-4 backbone feature is 256
channels at 128×256, which the `c1_bottleneck` 1×1 reduces to 48 channels at 128×256. `resize` lifts the
context 64×128 → 128×256, so both tensors are 128×256 and the concat gives 512 + 48 = 560 channels at OS-4;
the `sep_bottleneck`'s two separable 3×3 convs fold 560 → 512, the 1×1 classifier maps 512 → 19, and the
final ×4 resize lifts 128×256 → 512×1024. Every size closes, and the largest single upsample anywhere is
now ×4, not ×8 — the boundary blur is halved before the skip even contributes its sharp edges.

The prediction, against 77.85, tempered. Two things each add a little. The atrous pyramid gathers the same
multi-scale context as the pooling pyramid but *without* averaging away within-region resolution, so
context near boundaries is sharper — but the pooling pyramid already captured most of the *available*
context gain (that was the +5.60), so this delta is a refinement, not a second big jump. And the single
OS-4 skip recovers fine boundaries an ×8 upsample blurred, but acts almost entirely on the thin/boundary
classes — poles, fences, thin riders, traffic signs, the elongated things whose IoU is dominated by edges.
So I expect the gain here smaller than the PSPNet jump and concentrated on boundary classes rather than a
wholesale rise across all nineteen. Since this gain is boundary-flavored, and multi-scale/flip augmentation
is exactly what helps edge localization, I expect the ms+flip gap to hold or edge up rather than collapse.
The risk is diminishing returns made concrete: if dense atrous context is not meaningfully sharper than
pooled context on this crop — recall that the two largest atrous branches partly degenerate on the 64-tall
axis — then most of the headroom rests on the single OS-4 skip, and if that skip is too low-level it could
sharpen noise into crisp-but-wrong boundaries. The 48-channel reduction and the separable-conv fusion are
the hedges; if they fail it should show as ragged, over-sharp edges on small objects rather than a global
drop, and the class-averaged number should still edge up on the strength of the boundary classes. I keep
the intermediate-stage auxiliary head during training, as before, since the backbone is still the deep
dilated ResNet-50 and the new capacity sits at the top in the ASPP and decoder — a training-only scaffold,
dropped at inference. The concrete change is the depthwise-separable atrous spatial pyramid plus the single
OS-4 decoder; the full head is in the answer.
