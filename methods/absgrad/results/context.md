# Context: adaptive density control for 3D Gaussian radiance fields

## Research question

A scene is represented as a collection of anisotropic 3D Gaussian primitives and rendered by
differentiable splatting. Each primitive has a 3D center, covariance/scale/rotation, opacity, and
view-dependent color. Training starts from a sparse Structure-from-Motion point cloud and optimizes the
primitive parameters against posed photographs, but the initial point set is far too sparse to represent
the full scene. The representation must therefore change size during training: it must add primitives
where coverage is missing, subdivide primitives that cover too much, and remove primitives that become
transparent or excessively large.

The practical failure is visible in complex real scenes. Most regions render sharply, but grass,
foliage, uneven pavement, distant texture, and other high-frequency areas can remain soft. Inspecting
the learned ellipsoids in those regions shows a small number of large, flat splats covering an area that
needs many small primitives. The renderer can draw small splats, the loss exposes the photometric error,
and gradient descent continues to update the model, yet these large primitives are not selected for
subdivision. The open question is how to decide, during training and without changing the renderer,
loss, optimizer, dataset, or evaluation protocol, which primitives should be cloned, split, pruned, or
left untouched.

The criterion must satisfy two constraints. First, it has to be cheap, because density control runs many
times during training and the renderer already dominates cost. Second, it has to separate two different
growth cases: a small primitive in an under-covered area should be duplicated to increase coverage,
while a large primitive that is averaging over fine detail should be split into smaller children.

## Background

The substrate is 3D Gaussian Splatting (Kerbl, Kopanas, Leimkühler & Drettakis 2023). A primitive is a 3D Gaussian

```text
G_i(x) = exp(-1/2 (x - mu_i^3d)^T (Sigma_i^3d)^-1 (x - mu_i^3d))
```

with opacity `o_i` and spherical-harmonic color coefficients. For a camera view, the 3D Gaussian is
transformed and projected to a 2D Gaussian with image-plane center `mu_i` and covariance
`Sigma_i^2d`. The rendered pixel color is formed by front-to-back alpha compositing over the splats
that overlap the pixel:

```text
C(x) = sum_i c_i alpha_i prod_{q<i} (1 - alpha_q),
alpha_i = sigmoid(o_i) G_i^2d(x).
```

Training uses the differentiable rasterizer and a photometric objective combining L1 and a structural
similarity term. Backpropagation gives gradients for primitive positions, colors, opacity, and
covariance parameters. The density-control rule can reuse those gradients, but it should not alter the
meaning of the true optimization gradient used for parameter updates.

The point-based design is the reason density control matters. Neural radiance fields and grid/hash
variants can hide detail in a continuous field or fixed-resolution structure, but this representation
is an explicit finite set of primitives. If the controller leaves a large primitive in a region that
needs several smaller ones, the final image averages the detail; if it creates too many unnecessary
primitives, memory and training cost grow.

## Baselines

The inherited density controller (Kerbl et al. 2023) runs after an optimization warm-up and then refines periodically,
typically every 100 iterations until roughly the midpoint of a 30k-step training run. For every
primitive visible in the recent views, it accumulates the magnitude of the loss gradient with respect to
the primitive's projected 2D center:

```text
nabla_mu_i L = (1/M) sum_{k=1}^M || dL^k / dmu_i^k ||_2
             = (1/M) sum_{k=1}^M sqrt((dL^k/dmu_{i,x}^k)^2
                                      + (dL^k/dmu_{i,y}^k)^2).
```

If this average image-plane positional-gradient magnitude exceeds a threshold, the primitive is
densified. A small primitive is cloned, usually by copying it and letting optimization separate the
copies. A large primitive is split, replacing it with smaller children sampled from its own Gaussian
support and reducing their scales. Transparent primitives are pruned, opacities are periodically reset
low so redundant primitives can disappear, and oversized primitives may be culled.

The rule is a good first proxy: when the optimizer strongly wants to move a primitive in the image
plane, the primitive is probably sitting on a region that is not yet well represented. It works
especially naturally for under-covered small geometry, where only a few pixels pull the primitive and
the summed positional gradient remains large. The hard case is the large primitive over fine texture.
There the photometric errors can be high, but the average signed movement request for the projected
center can be small, so the split threshold is not reached.

This gap is the starting point. The controller needs a statistic that better reflects the representation
quality of all pixels covered by a primitive, while preserving the existing actions, schedules, opacity
reset, pruning, and training loop.

## Evaluation settings

The natural setting is the same real-scene novel-view-synthesis protocol used by the 3D Gaussian
Splatting baseline. Scenes include unbounded indoor and outdoor captures with high-frequency details,
large-scale scenes such as Tanks and Temples, and indoor Deep Blending scenes. Held-out testing follows
the usual train/test split convention, often taking every eighth image for test.

Quality is measured by PSNR, SSIM, and LPIPS. LPIPS is particularly relevant because the visible
artifact is blur from over-large primitives averaging detail. Model size and memory are also required
metrics: a criterion that recovers detail only by admitting many unnecessary primitives is not a
practical density-control improvement.

The controlled comparison keeps the renderer, loss, optimizer, data, schedule, and primitive operations
fixed. Only the densification statistic and the thresholds that interpret it should change. This
isolation is important because otherwise improved detail could be attributed to unrelated changes in
training or rendering rather than to the density-control criterion.

## Code framework

The implementation surface is the strategy object around the differentiable rasterizer. The rasterizer
returns rendered pixels plus an `info` dictionary containing projected 2D means, visibility/radii,
image dimensions, camera count, and primitive IDs. Before backward, the strategy retains gradients on
the projected means. After backward, the strategy updates per-primitive running statistics and, at
refinement steps, applies duplicate, split, remove, and opacity-reset operations.

```python
class DensityControl:
    prune_opa = 0.005
    grow_grad2d = 0.0002
    grow_scale3d = 0.01
    prune_scale3d = 0.1
    refine_start_iter = 500
    refine_stop_iter = 15_000
    reset_every = 3000
    refine_every = 100

    def initialize_state(self, scene_scale=1.0):
        return {
            "grad2d": None,
            "count": None,
            "scene_scale": scene_scale,
        }

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info):
        if step >= self.refine_stop_iter:
            return
        # Open slot:
        # 1. read a per-primitive statistic from the projected-center gradients,
        # 2. accumulate and average it over visibility,
        # 3. route high-statistic small primitives to duplicate,
        # 4. route high-statistic large primitives to split,
        # 5. prune and reset opacity using the inherited operations.
```

The open problem is the statistic in that post-backward hook. It must use information available during
backpropagation through the rasterizer, remain an auxiliary density-control signal rather than a
replacement for the true signed optimization gradient, and plug into the existing duplicate/split/prune
machinery.
