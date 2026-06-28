Let me start from the thing that actually hurts. I train a convolutional network on digits, and it does well when the digit sits upright and centred. Then I drop the same digit into a random corner of a bigger canvas, shrink it, tilt it forty degrees, scatter a few clutter patches around it — and the network stumbles, even though to me it is obviously the same digit. Nothing about the *identity* of the thing changed; only its position, scale, and orientation did. And those are exactly uncorrelated with the label. So my network is burning capacity learning to be robust to nuisances instead of learning to read.

Why does this happen? Convolution is translation-equivariant: slide the input, the feature map slides with it. The only thing in the architecture that turns equivariance into any kind of *invariance* is max-pooling. But a pooling window is tiny — two by two. One pooling layer is invariant to a one-pixel jitter and nothing more. To get invariance to a big shift, never mind a change of scale or a rotation, I have to stack pooling on pooling on pooling and hope the hierarchy launders it out. And when people actually measure how a CNN's intermediate representations respond to a transformed input — fit the linear map between the representation of an image and the representation of its rotated or scaled copy — they find the maps are simply *not* invariant to large transformations. The invariance I was relying on is local, small, and hand-wired into a fixed pooling stencil. The network has no way to reorganise the spatial layout of a feature map in a way that depends on what it is looking at.

So what do people do instead? They augment. Rotate, scale, translate, warp every training image a dozen ways and feed all the copies in, so the network is forced to average over the nuisance. It works, sort of, but it is a crutch: it spends data and capacity memorising invariance with no guarantee it has covered the space, and at test time it does nothing to actually *fix* a tilted digit — it just hopes it has seen enough tilted digits. That is the wrong shape of solution. I do not want the network to tolerate a forty-degree digit; I want it to *un-tilt* the digit and hand a clean, upright version to the rest of the network. Pose normalisation, not pose tolerance.

Can I get pose normalisation for free? There is a family of methods that builds invariance into the *filters* — tie weights across a rotation group, or build a bank of scaled filters, so the response is invariant to the chosen group by construction. But that bakes a fixed, pre-chosen set of transformations into the feature extractor, and pays for each transformation in compute, and it can't decide per-image that *this* picture needs a thirty-degree correction and *that* one needs a two-times zoom. It manipulates the detectors, not the data. I think I want the opposite: leave the detectors alone, and manipulate the data — actively warp the feature map into a canonical pose, with the amount of warp decided per sample.

The capsule and transforming-parts line is closer in spirit — it represents objects as parts carrying explicit 2D affine transforms, and predicts those transforms. But it learns them from *supervision*: the transformations are handed to the network as input or target. I don't have ground-truth transformations. I only have the class label. Whatever module I build has to discover the right transformation purely from the downstream task loss.

And there's attention, the glimpse models — they do pick a region and process it, which is a kind of spatial selection. But hard-cropping is not differentiable (you can't take a derivative of "grab pixels 12 to 40"), so those models are trained with reinforcement learning, which is noisy and painful. The one differentiable attention I know of reads through a grid of axis-aligned Gaussian kernels — fine, but it's a translation-and-scale window, not rotation or shear or a free-form warp.

Let me state the target sharply. I want a single module that (1) looks at the incoming feature map and (2) decides, conditioned on that map, a spatial transformation to apply, then (3) applies that transformation to the whole map, producing a warped output map — and the whole thing must be differentiable so that the gradient of the ordinary task loss flows back through it and teaches it what transformation to pick, with no transformation labels at all. If it's differentiable, it's just another layer: I can drop it anywhere, train it with the same SGD, and the "how to warp this input" knowledge gets cached in its weights.

The hard word in there is *differentiable*. Picking a transformation and applying it to an image is, in the obvious implementation, a hard resampling — and that's where attention models had to reach for RL. So the entire problem reduces to: how do I make "warp a feature map by a predicted geometric transformation" a smooth function of the predicted parameters, smooth enough to backprop through? If I can crack the differentiable warp, everything else (predicting the parameters with a little network, training with the task loss) is routine.

Let me try the naive thing first and watch it break. Say I predict a transformation and I want to apply it forward: for each *input* pixel at location p, compute where it lands, q = transform(p), and write the input value there into the output. This is forward warping. Run it in my head on a simple zoom-out: many input pixels map to nearly the same output location, so they collide and overwrite each other, while if I zoom *in*, whole regions of the output get *no* input pixel mapping into them — holes. The output is a mess of collisions and gaps, and worse, "which output cell did this input pixel land in" involves a rounding to integer output coordinates, which is a step function — not differentiable in the transform parameters at all. Forward warping is the wrong direction.

Flip it. This is exactly the problem image resampling in graphics already solved, and they go backward. Don't push input pixels forward; iterate over the *output* pixels, and for each output pixel ask "where in the input did this come from?" — map the output coordinate *back* to a source coordinate, and read the input there. Every output pixel gets exactly one source coordinate, so there are no holes and no collisions: the output is completely and uniquely defined. The transform is defined output→input. Let me hold onto that: I will lay down a fixed regular grid of output pixel locations and, for each, compute a *source* location in the input to sample from.

Now pick the transformation class. Start with the affine — six numbers buy me translation, rotation, scale, shear, and crop, which already covers most of what hurts. Write the output grid point as `(x_t, y_t)` and the source point I want to read as `(x_s, y_s)`. Since I'm going output→input, the affine maps the output coordinate to the source coordinate:

    ( x_s )         ( x_t )
    (     ) = A_θ · ( y_t )
    ( y_s )         (  1  )

with

    A_θ = [ θ11  θ12  θ13 ]
          [ θ21  θ22  θ23 ].

The `1` in the input vector and the third column give me the translation `(θ13, θ23)`; the left 2×2 block does rotation, scale, and shear. So componentwise,

    x_s = θ11 · x_t + θ12 · y_t + θ13
    y_s = θ21 · x_t + θ22 · y_t + θ23.

One thing to settle before I go further: in what units are these coordinates? If I measure `x_t, y_t` in pixels, then the meaning of `A_θ` depends on the feature-map size — the identity transform would have different numbers at 28×28 than at 42×42, and a module dropped at a different resolution would mean something different. That's brittle. So I'll use height- and width-normalised coordinates: `−1 ≤ x_t, y_t ≤ 1` spans the output, `−1 ≤ x_s, y_s ≤ 1` spans the input. With `A_θ = [[1,0,0],[0,1,0]]` the formula gives `x_s = 1·x_t + 0·y_t + 0 = x_t` and `y_s = y_t`, so every output pixel reads from its own location in the input — that's the identity, and it's the same six numbers whether the map is 28×28 or 42×42. Good, that's resolution-independent as I wanted.

Before trusting the affine for the thing I actually care about — cropping — let me run a contraction through it with numbers rather than just assert it crops. Take `A_θ = [[0.5, 0, 0.3], [0, 0.5, 0]]`, a half-scale block (|det| = 0.25 < 1) with a translation. Walk the four corners of the output grid (the extremes `x_t, y_t ∈ {−1, +1}`) to their source coordinates:

- output corner `(−1, −1) → x_s = 0.5·(−1) + 0.3 = −0.2`, `y_s = 0.5·(−1) = −0.5`
- output corner `(+1, −1) → x_s = 0.5·(+1) + 0.3 = +0.8`, `y_s = −0.5`
- output corner `(−1, +1) → x_s = −0.2`, `y_s = +0.5`
- output corner `(+1, +1) → x_s = +0.8`, `y_s = +0.5`

So the full output grid, which spans `[−1,1]²`, reads from the input box `x_s ∈ [−0.2, 0.8]`, `y_s ∈ [−0.5, 0.5]` — a sub-rectangle that's half the width and half the height of the input, off-centre to the right. The output stretches that small box to fill the whole `[−1,1]²` frame. That is exactly a crop-and-zoom, and the `0.3` translated the crop window where I wanted it. Cropping falls out of the affine for free, which is the attention behaviour I wanted, no separate mechanism — and now I've actually seen it happen on coordinates rather than taken it on faith.

And I'm not locked into the full affine. If I only want attention — isotropic scale plus translation — I constrain it to

    A_θ = [ s  0  t_x ]
          [ 0  s  t_y ],

three parameters. Or I can go richer: a plane projective transform (eight parameters), a thin-plate spline, piecewise affine. The only requirement the whole scheme imposes on the transformation is that `(x_s, y_s)` be a differentiable function of the parameters `θ`, because that's the link the gradient has to cross to reach the parameter-predicting network. There's a clean way to see the general case: write the source grid as `T_θ = M_θ · B`, where `B` is a representation of the target grid (in the affine case `B` is just the regular output grid in homogeneous coordinates `(x_t, y_t, 1)`) and `M_θ` is a matrix built from `θ`. Then `B` itself could be made learnable — let the module learn the target grid layout suited to the task — but the affine with fixed `B = G` is the concrete case to nail down first.

So the grid generator gives me, for every output pixel `i`, a source coordinate `(x_s^i, y_s^i)`. Now the real obstacle: I have a fractional source coordinate — say `(x_s, y_s) = (12.3, 7.8)` — and the input is defined only on integer pixel locations. I need a value there, and I need it as a *differentiable* function of `x_s` and `y_s`, because that's the only path back to `θ`. Let me write the read as a sampling sum over all input pixels with some kernel that picks out the neighbourhood of `(x_s, y_s)`:

    V_i = Σ_n Σ_m  U_{nm} · k(x_s^i − m; Φ_x) · k(y_s^i − n; Φ_y),

where `U_{nm}` is the input value at integer location `(m, n)` (`m` indexing width, `n` height), `V_i` the output at pixel `i`, and `k` a 1-D sampling kernel separable across the two axes. This sums over the whole input, but the kernel will be zero except near `(x_s, y_s)`. (For a multi-channel feature map I apply the exact same sampling to every channel — the warp is one geometric object, and using the same grid across channels keeps the channels spatially aligned.)

What kernel? Try the obvious one: nearest neighbour. That's `k(d) = δ(round(d))` — round `x_s` to the nearest integer pixel and copy it:

    V_i = Σ_n Σ_m  U_{nm} · δ(⌊x_s^i + 0.5⌋ − m) · δ(⌊y_s^i + 0.5⌋ − n).

Differentiate this with respect to `x_s`. Nudging `x_s` by a hair doesn't change which integer it rounds to — `V_i` is flat — until `x_s` crosses a half-integer, where the selected pixel jumps discontinuously. So `∂V_i/∂x_s` is zero almost everywhere and undefined at the jumps. There is no gradient signal to tell the localisation network "move the sampling point a little this way." Dead end — nearest neighbour kills the very gradient I built the whole inverse-mapping machinery to get.

The fix is to make the read vary *continuously* as `x_s` moves between pixels. Bilinear interpolation does exactly that: weight each neighbouring pixel by how close `x_s` is to it, linearly. The 1-D weight for a pixel at integer `m` is `max(0, 1 − |x_s − m|)` — it is `1` when `x_s = m`, falls linearly to `0` one pixel away, and is `0` beyond. Separable across axes, the sampler becomes

    V_i = Σ_n Σ_m  U_{nm} · max(0, 1 − |x_s^i − m|) · max(0, 1 − |y_s^i − n|).

Only the (at most) four pixels surrounding `(x_s, y_s)` get nonzero weight, so in practice this is a four-tap read, not a sum over the whole map — I'll exploit that for speed later, but mathematically the full sum is fine. The point is that `V_i` is now piecewise-linear in `x_s` and `y_s`, so it has a gradient.

Let me actually compute the gradients, because this is the crux — if these don't exist and aren't useful, the module can't train. Three things need derivatives: the input `U` (so gradient flows to layers *before* the module), and the source coordinates `x_s`, `y_s` (so gradient flows *forward into the localisation network* through `θ`).

Gradient with respect to the input value `U_{nm}`. The sampler is linear in `U`, so the derivative is just the coefficient multiplying `U_{nm}`:

    ∂V_i/∂U_{nm} = max(0, 1 − |x_s^i − m|) · max(0, 1 − |y_s^i − n|).

That's the bilinear weight itself — clean, and it means gradient flows back to whatever produced the input feature map.

Gradient with respect to `x_s`. Hold the `y` factor fixed and differentiate the `x` factor. I need `d/dx_s [ max(0, 1 − |x_s − m|) ]`. Outside the support, where `|m − x_s| ≥ 1`, the term is clamped at zero, so the derivative is `0`. Inside the support the term is `1 − |x_s − m|`, and its derivative is `−d|x_s − m|/dx_s = −sign(x_s − m)`. Be careful with the sign. If `m ≥ x_s`, then `x_s − m ≤ 0`, so `sign(x_s − m) ≤ 0`, so `−sign = +1`. If `m < x_s`, then `x_s − m > 0`, so `−sign = −1`. Putting it together:

    ∂V_i/∂x_s = Σ_n Σ_m  U_{nm} · max(0, 1 − |y_s^i − n|) ·
                  {  0   if |m − x_s^i| ≥ 1
                    +1   if |m − x_s^i| < 1 and m ≥ x_s^i
                    −1   if |m − x_s^i| < 1 and m < x_s^i }.

It only sums over the support, and it weights each contribution by the input value and the orthogonal `y`-weight — so its sign and size tell the localisation network which way to slide the sampling point to reduce the loss. The `y_s` derivative is identical with `x` and `y` swapped. The absolute value has a kink at `x_s = m`, so strictly there's no derivative there; I just pick a sub-gradient at the kink, which is exactly what SGD is happy with.

I derived that derivative by hand and I don't trust hand-derivations of piecewise things, so let me check it against the function it's supposed to be the derivative of, on a concrete 1-D read. Drop the `y` factor (set `y_s` on a pixel so its weight is 1) and take a row of three input pixels at integer positions with values `U_0 = 2`, `U_1 = 5`, `U_2 = 1`. Sample at `x_s = 1.3`. The only pixels within distance 1 are `m=1` (weight `1−|1.3−1| = 0.7`) and `m=2` (weight `1−|1.3−2| = 0.3`); `m=0` is at distance 1.3, weight 0. So

    V(1.3) = 5·0.7 + 1·0.3 = 3.5 + 0.3 = 3.8.

Now my formula for `∂V/∂x_s` at `x_s = 1.3`: for `m=1`, `|1−1.3| = 0.3 < 1` and `m=1 < x_s=1.3`, so the branch is `−1`, contributing `U_1·(−1) = −5`. For `m=2`, `|2−1.3| = 0.7 < 1` and `m=2 ≥ x_s=1.3`, so the branch is `+1`, contributing `U_2·(+1) = +1`. Sum: `∂V/∂x_s = −5 + 1 = −4`.

Check it numerically. Nudge `x_s` to 1.31: weights `0.69` and `0.31`, `V = 5·0.69 + 1·0.31 = 3.45 + 0.31 = 3.76`. The slope is `(3.76 − 3.8)/0.01 = −0.04/0.01 = −4`. That matches my analytic `−4` exactly — as it should, since `V` is linear in `x_s` between integer pixels so the finite difference *is* the derivative there. And the sign reads correctly: moving the sample point right (toward pixel 2, value 1) lowers `V`, away from the bright pixel 1 (value 5). The sub-gradient does carry usable directional information. So this is a sub-differentiable sampler, and I've now confirmed both the value and the derivative on a real read.

Now close the loop to `θ`. The source coordinate is the affine of the output coordinate, so `∂x_s/∂θ` is trivial: `∂x_s/∂θ11 = x_t`, `∂x_s/∂θ12 = y_t`, `∂x_s/∂θ13 = 1`, and zero for the `θ2·` row; symmetrically for `y_s`. Chaining, the loss gradient reaches `θ` as

    ∂L/∂θ = Σ_i (∂L/∂V_i)(∂V_i/∂x_s^i)(∂x_s^i/∂θ) + (same with y),

and `θ` is just the output of an ordinary network, so from there it's standard backprop. The chain is unbroken from the task loss all the way to the parameter-predicting network — which is the whole point. No transformation labels anywhere; the geometry is learned purely because warping toward a canonical pose lowers the downstream loss.

That parameter-predicting network — call it the localisation network — can be anything: take the input feature map `U ∈ R^{H×W×C}` and run it through some hidden layers ending in a regression layer that emits `θ`. Fully connected or convolutional, doesn't matter, as long as the last layer produces the right number of parameters (six for affine, three for attention, eight for projective). Predicting `θ` in a structured low-dimensional form is a feature, not a limitation: it hands the localisation network an easy, well-posed regression instead of asking it to emit a dense flow field.

So the module has three pieces, in computation order: a localisation network maps `U → θ`; a grid generator maps the regular output grid `G` through `A_θ` to source coordinates `T_θ(G)`; and the bilinear sampler reads `U` at those coordinates to produce the warped output `V`. Self-contained, differentiable end to end, drop-in.

Let me think about whether this actually trains, because there's a subtle initialisation trap. If I initialise the localisation network's regression layer randomly, the very first forward pass predicts some arbitrary `θ` and the module warps the input into garbage — a wild rotation or a near-zero crop — before the rest of the network has learned anything, and the gradients it gets back are useless. The fix is to make the module a no-op at the start: initialise that final regression layer to output the identity transform — weights zero, bias `[1, 0, 0, 0, 1, 0]` so `A_θ = [[1,0,0],[0,1,0]]`. Then at step zero the module passes the input through untouched, the host network trains like a perfectly ordinary CNN, and the module only *gradually* learns to deviate from identity as deviating starts to help the loss. Normalised coordinates are what make this clean — the identity bias is the same six numbers at any resolution.

There's a second stability issue. The transform parameters are enormously high-leverage: a small change in `θ` can swing the whole sampled region across the image. If the localisation network learns at the same rate as everything else, it tends to overshoot — warp wildly, destabilise training. So I'll run the localisation network at a *lower* learning rate than the rest of the network (something like a tenth, or far less when it sits on top of a big pretrained backbone). Small, careful steps on the parameters that move the most.

What does this buy architecturally? The module is cheap — a small localisation net plus a four-tap read per output pixel — so the overhead is a few percent and I can place it freely. Put it right at the input and it pose-normalises the raw image. Put it deeper and it warps a feature map, with the localisation network now reading richer features to decide the transform. Stack several at increasing depth to warp increasingly abstract representations. Or run several *in parallel* on the same map: each can lock onto a different object or part — one settles on a bird's head, another on its body — and I concatenate their outputs. A useful side effect: if I let the output grid be smaller than the input, the module crops *and* downsamples in one step, so I can feed it a high-resolution image, attend to a small region, and pass a small canonical crop downstream — attention that actually saves computation rather than costing it. (The one caveat: with a fixed small bilinear kernel, downsampling by a large factor will alias, since four taps can't average a big region.) And because `θ` explicitly encodes the pose of whatever the module locked onto, I can even feed `θ` forward as a feature.

Let me write it as code, with the affine grid and bilinear sampler as the framework's `affine_grid` / `grid_sample`. These are the standard PyTorch primitives for the same output→input affine grid and bilinear read; I set the modern `align_corners` convention explicitly so the boundary convention is not hidden in a version default.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # --- the recognition network the warped map feeds into ---
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # --- localisation network: U -> hidden features -> theta ---
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        # regression head: emit the 6 affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )
        # start as a no-op: identity transform so the host net trains normally
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        # 1) localisation network predicts the transform parameters from x
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs).view(-1, 2, 3)          # A_theta, per sample

        # 2) grid generator: map the regular OUTPUT grid back to SOURCE coords
        #    (x_s, y_s)^T = A_theta (x_t, y_t, 1)^T, in normalised [-1,1] coords
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # 3) bilinear sampler: read x at the source coords; differentiable in
        #    both x (-> earlier layers) and the grid (-> theta -> localisation)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        x = self.stn(x)                                 # actively warp first
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      # then ordinary CNN
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

The causal chain, start to end: max-pooling gives a CNN only small, local, fixed spatial invariance, so large rotations/scales/warps and clutter still wreck it and augmentation only papers over it; I want active, per-sample pose normalisation instead, which means a module that *warps* the feature map and is trained by the task loss alone — and that forces the warp to be differentiable; forward warping is non-differentiable and full of holes, so I go output→input like graphics resampling, laying a regular output grid and mapping each output point back through an affine `A_θ` (in normalised coordinates, so identity is resolution-independent and cropping is just a contraction) to a fractional source coordinate; reading at a fractional coordinate must be smooth in that coordinate, which rules out nearest-neighbour (zero gradient) and selects bilinear interpolation, whose `∂V/∂U`, `∂V/∂x_s`, `∂V/∂y_s` I worked out and which therefore passes gradient back to both the earlier layers and, via `∂x_s/∂θ`, to a small localisation network that predicts `θ` from the map; initialise that head to the identity so it starts as a no-op and give it a lower learning rate so it doesn't overshoot — and the result is a cheap, drop-anywhere module that learns to spatially normalise its input from labels alone.
